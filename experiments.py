"""All cross-validation experiments reported in the paper.

Writes CSV tables, the confusion matrix figure and out-of-fold predictions to results/.
Usage: python experiments.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from common import (SEED, ConcatTfidf, augment, duplicate_labels, load_train, make_model,
                    predict_with_exclusion, scores, seed_everything, without_leak)

OUT = 'results'
MODELS = {'cnb': 'Complement NB', 'svm': 'Linear SVM', 'logreg': 'Logistic Regression'}
AUGS = ['none', 'ros', 'dropout', 'crossover']
SHORT = {'Cardiovascular-Pulmonary': 'Cardio-Pulm', 'Dermatology': 'Derm',
         'Gastroenterology': 'Gastro', 'General Medicine': 'Gen Med', 'Neurology': 'Neuro',
         'Neurosurgery': 'Neurosurg', 'Obstetrics-Gynecology': 'Ob-Gyn',
         'Ophthalmology': 'Ophthalm', 'Orthopedic': 'Ortho', 'Psychiatry-Psychology': 'Psych',
         'Radiology': 'Radiology', 'Surgery': 'Surgery'}


def metrics(y, p):
    f1 = f1_score(y, p, average=None)
    return {'accuracy': accuracy_score(y, p), 'macro_f1': f1.mean(), 'min_f1': f1.min()}


# ---------------------------------------------------------------- baseline

_stem = PorterStemmer().stem


def stem_tokenizer(text):
    return [_stem(w) for w in text.split()]


def baseline_fold(df, a, b):
    vec = TfidfVectorizer(lowercase=True, tokenizer=stem_tokenizer, token_pattern=None)
    X = vec.fit_transform(df['description'].iloc[a])
    m = MultinomialNB().fit(X, df['medical_specialty'].iloc[a])
    return b, m.predict(vec.transform(df['description'].iloc[b]))


# ---------------------------------------------------------------- our systems

def system_fold(full, a, b, model, aug, leak):
    """Train one classifier on fold `a`, return every prediction variant for fold `b`.

    leak=True : the classifier sees the keywords field as given (specialty term included)
    leak=False: the specialty term is removed from keywords, so only real text is used
    """
    data = full if leak else without_leak(full)
    tr, va = augment(data.iloc[a], aug), data.iloc[b]
    feats = ConcatTfidf()
    clf = make_model(model).fit(feats.fit_transform(tr), tr['medical_specialty'].values)
    S = scores(clf, feats.transform(va))
    plain = clf.classes_[S.argmax(1)]
    dedup, hit = predict_with_exclusion(clf.classes_, S, va['raw_transcription'].tolist(),
                                        duplicate_labels(full.iloc[a]))
    kw = full['kw_specialty'].iloc[b].values
    has_kw = pd.notna(kw)
    hybrid = np.where(has_kw, kw, dedup)
    return b, plain, dedup, hybrid, hit


def run_system(full, folds, model, aug, leak):
    out = {k: np.empty(len(full), dtype=object) for k in ('plain', 'dedup', 'hybrid')}
    hit = np.zeros(len(full), dtype=bool)
    res = Parallel(n_jobs=5)(delayed(system_fold)(full, a, b, model, aug, leak) for a, b in folds)
    for b, p, d, h, ht in res:
        out['plain'][b], out['dedup'][b], out['hybrid'][b], hit[b] = p, d, h, ht
    return out, hit


def main():
    seed_everything()
    os.makedirs(OUT, exist_ok=True)
    full = load_train()
    y = full['medical_specialty'].values
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(full, y))

    # ---- data statistics used in the paper
    dup_groups = full[full['raw_transcription'] != ''].groupby('raw_transcription')['medical_specialty'].agg(list)
    dup_groups = dup_groups[dup_groups.str.len() > 1]
    stats = {
        'n_train': len(full),
        'class_counts': full['medical_specialty'].value_counts().to_dict(),
        'kw_specialty_present': int(full['kw_specialty'].notna().sum()),
        'kw_specialty_matches_label': int((full['kw_specialty'] == full['medical_specialty']).sum()),
        'duplicate_groups': len(dup_groups),
        'duplicate_rows': int(dup_groups.str.len().sum()),
        'duplicate_groups_conflicting': int((dup_groups.apply(lambda l: len(set(l))) > 1).sum()),
    }
    pd.Series(stats).to_json(f'{OUT}/data_stats.json', indent=2)
    print(stats)

    rows = []

    # ---- baseline
    pb = np.empty(len(full), dtype=object)
    for b, p in Parallel(n_jobs=5)(delayed(baseline_fold)(full, a, b) for a, b in folds):
        pb[b] = p
    rows.append({'system': 'Baseline (MNB, description, stemming)', 'model': 'mnb',
                 'aug': 'none', 'setting': 'baseline', **metrics(y, pb)})

    # ---- main grid: model x augmentation x setting
    keep = {}
    for model in MODELS:
        for aug in AUGS:
            for leak in (True, False):
                out, hit = run_system(full, folds, model, aug, leak)
                variants = {'plain': out['plain'], 'dedup': out['dedup']}
                if not leak:
                    variants['hybrid'] = out['hybrid']
                for v, p in variants.items():
                    setting = ('full' if leak else 'text-only') + ('' if v == 'plain' else f'+{v}')
                    r = {'system': MODELS[model], 'model': model, 'aug': aug,
                         'setting': setting, **metrics(y, p)}
                    rows.append(r)
                    print(r, flush=True)
                keep[(model, aug, leak)] = (out, hit)

    table = pd.DataFrame(rows)
    table.to_csv(f'{OUT}/cv_results.csv', index=False)

    # ---- best system: hybrid, chosen by macro-F1 (accuracy as tie breaker)
    hyb = table[table['setting'] == 'text-only+hybrid'].sort_values(['macro_f1', 'accuracy'], ascending=False)
    best = hyb.iloc[0]
    print('\nBest system:', best.to_dict())
    out, hit = keep[(best['model'], best['aug'], False)]
    pred = out['hybrid']
    rep = classification_report(y, pred, digits=4, output_dict=True)
    pd.DataFrame(rep).T.to_csv(f'{OUT}/per_class_best.csv')
    print(classification_report(y, pred, digits=4))

    # accuracy by subset: specialty keyword present / duplicate in training fold / neither
    has_kw = full['kw_specialty'].notna().values
    subsets = {'keyword present': has_kw, 'no keyword, duplicate in train': ~has_kw & hit,
               'no keyword, unique note': ~has_kw & ~hit}
    sub_rows = []
    for name, mask in subsets.items():
        sub_rows.append({'subset': name, 'n': int(mask.sum()),
                         'text_only': accuracy_score(y[mask], out['plain'][mask]),
                         'text_only+dedup': accuracy_score(y[mask], out['dedup'][mask]),
                         'hybrid': accuracy_score(y[mask], pred[mask])})
    pd.DataFrame(sub_rows).to_csv(f'{OUT}/subset_accuracy.csv', index=False)
    print(pd.DataFrame(sub_rows))

    # ---- confusion matrix of the text-only classifier (the part that must read the text)
    labels = sorted(SHORT)
    cm = confusion_matrix(y, out['plain'], labels=labels)
    plt.figure(figsize=(6.4, 4.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 7},
                xticklabels=[SHORT[l] for l in labels], yticklabels=[SHORT[l] for l in labels])
    plt.xlabel('Predicted'); plt.ylabel('Gold')
    plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT}/confusion_matrix.png', dpi=250)
    plt.close()

    # ---- out-of-fold predictions for manual error analysis
    pd.DataFrame({
        'gold': y, 'text_only': out['plain'], 'text_only_dedup': out['dedup'], 'hybrid': pred,
        'kw_specialty': full['kw_specialty'], 'dup_in_train_fold': hit,
        'sample_name': full['sample_name'], 'description': full['description'],
        'transcription': full['raw_transcription'].str[:400],
    }).to_csv(f'{OUT}/oof_predictions.csv', index=False)


if __name__ == '__main__':
    main()
