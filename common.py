"""Shared data loading, pre-processing, augmentation and model definitions."""
import random
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

SEED = 42
FIELDS = ['sample_name', 'description', 'keywords', 'transcription']
VALID_CLASSES = [
    'Cardiovascular-Pulmonary', 'Dermatology', 'Gastroenterology',
    'General Medicine', 'Neurology', 'Neurosurgery',
    'Obstetrics-Gynecology', 'Ophthalmology', 'Orthopedic',
    'Psychiatry-Psychology', 'Radiology', 'Surgery'
]


def seed_everything(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s\-\/]', ' ', t)
    return " ".join(t.split())


def load_train(path="train.csv"):
    df = pd.read_csv(path, sep=';', engine='python')
    df['medical_specialty'] = df['medical_specialty'].astype(str).str.strip()
    # 3 malformed rows (unescaped line breaks) put text in the label column
    df = df[df['medical_specialty'].isin(VALID_CLASSES)].reset_index(drop=True)
    return _prepare(df)


def load_test(path="test_no_labels.csv"):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        has_header = f.readline().strip().lower().startswith('description;')
    if has_header:
        df = pd.read_csv(path, sep=';', engine='python')
    else:
        df = pd.read_csv(path, sep=';', header=None, engine='python',
                         names=['description', 'sample_name', 'transcription', 'keywords'])
    return _prepare(df)


SPECIALTY_KEYWORDS = {
    'cardiovascular / pulmonary': 'Cardiovascular-Pulmonary', 'dermatology': 'Dermatology',
    'gastroenterology': 'Gastroenterology', 'general medicine': 'General Medicine',
    'neurology': 'Neurology', 'neurosurgery': 'Neurosurgery',
    'obstetrics / gynecology': 'Obstetrics-Gynecology', 'ophthalmology': 'Ophthalmology',
    'orthopedic': 'Orthopedic', 'psychiatry / psychology': 'Psychiatry-Psychology',
    'radiology': 'Radiology', 'surgery': 'Surgery',
}


def _prepare(df):
    df = df.copy()
    # In the source (MTSamples) the first keyword is the specialty the note was filed
    # under. We keep it apart so that experiments can run with or without it.
    terms = df['keywords'].fillna('').astype(str).str.split(',')
    first = terms.str[0].str.strip().str.lower()
    df['kw_specialty'] = first.map(SPECIALTY_KEYWORDS)
    df['keywords_noleak'] = [
        ', '.join(t[1:]) if k is not None and k == k else ', '.join(t)
        for t, k in zip(terms, df['kw_specialty'])]
    df['keywords_noleak'] = df['keywords_noleak'].apply(clean_text)
    # raw transcription is kept to detect exact duplicates between notes
    df['raw_transcription'] = df['transcription'].fillna('').astype(str).str.strip()
    for c in FIELDS:
        df[c] = df[c].fillna('').apply(clean_text)
    df['all_text'] = df[FIELDS].agg(' '.join, axis=1)
    return df


def without_leak(df):
    """Copy of df where the specialty term at the head of `keywords` is removed."""
    df = df.copy()
    df['keywords'] = df['keywords_noleak']
    df['all_text'] = df[FIELDS].agg(' '.join, axis=1)
    return df


# ---------------------------------------------------------------- augmentation

def drop_words(text, p=0.15, rng=random):
    words = text.split()
    if len(words) < 5:
        return text
    return " ".join(w for w in words if rng.random() > p)


def augment(df, strategy, min_count=150, rng=None):
    """Oversample every class with fewer than `min_count` notes.

    strategy:
      none     - no augmentation
      ros      - random oversampling (exact copies)
      dropout  - copies with 15% random word dropout in description and transcription
      crossover- new note made of the first half of one note and the second half of
                 another note of the same class (description/keywords kept from the first)
    """
    if strategy == 'none':
        return df
    rng = rng or random.Random(SEED)
    new_rows = []
    for cat, count in df['medical_specialty'].value_counts().items():
        if count >= min_count:
            continue
        sub = df[df['medical_specialty'] == cat]
        for _ in range(min_count - count):
            row = sub.iloc[rng.randrange(len(sub))].copy()
            if strategy == 'dropout':
                row['transcription'] = drop_words(row['transcription'], rng=rng)
                row['description'] = drop_words(row['description'], rng=rng)
            elif strategy == 'crossover':
                other = sub.iloc[rng.randrange(len(sub))]
                a, b = row['transcription'].split(), other['transcription'].split()
                row['transcription'] = " ".join(a[:len(a) // 2] + b[len(b) // 2:])
            row['all_text'] = " ".join(row[c] for c in FIELDS)
            new_rows.append(row)
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


# ---------------------------------------------------------------- features

class FieldTfidf:
    """One TF-IDF per field, horizontally stacked with a weight per field.

    Short fields (sample_name, keywords) are very informative but are drowned by
    the long transcription when everything is concatenated into one string.
    """

    def __init__(self, weights=None):
        self.weights = weights or {'sample_name': 1.0, 'description': 1.0,
                                   'keywords': 1.0, 'transcription': 1.0}
        self.vecs = {f: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                                        max_features=25000 if f == 'transcription' else 10000)
                     for f in self.weights}

    def fit_transform(self, df):
        return sp.hstack([self.vecs[f].fit_transform(df[f]) * w
                          for f, w in self.weights.items()]).tocsr()

    def transform(self, df):
        return sp.hstack([self.vecs[f].transform(df[f]) * w
                          for f, w in self.weights.items()]).tocsr()


class ConcatTfidf:
    """Single TF-IDF over the concatenation of the four fields."""

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)

    def fit_transform(self, df):
        return self.vec.fit_transform(df['all_text'])

    def transform(self, df):
        return self.vec.transform(df['all_text'])


def make_model(name):
    if name == 'cnb':
        return ComplementNB(alpha=0.3)
    if name == 'svm':
        return LinearSVC(C=0.1, class_weight='balanced', random_state=SEED)
    if name == 'logreg':
        return LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)
    raise ValueError(name)


def scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    return model.decision_function(X)


# ---------------------------------------------------------------- duplicate rule

def duplicate_labels(train_df):
    """Map raw transcription -> set of labels it already has in the training data."""
    d = train_df[train_df['raw_transcription'] != '']
    return d.groupby('raw_transcription')['medical_specialty'].agg(set).to_dict()


def predict_with_exclusion(classes, S, raw_texts, dup_map):
    """argmax over classes, skipping the labels a duplicated note already has in train.

    In this dataset a note that appears more than once (e.g. an operative report
    filed under both Orthopedic and Surgery) carries a *different* label in each
    copy (701/702 duplicate groups in train.csv), so the label already seen is
    almost never the label of the other copy.
    """
    S = np.array(S, dtype=float, copy=True)
    idx = {c: i for i, c in enumerate(classes)}
    hit = np.zeros(len(raw_texts), dtype=bool)
    for i, t in enumerate(raw_texts):
        seen = dup_map.get(t)
        if seen and len(seen) < len(classes):
            hit[i] = True
            for lab in seen:
                S[i, idx[lab]] = -np.inf
    return np.asarray(classes)[S.argmax(1)], hit
