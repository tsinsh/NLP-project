"""Train the final system on train.csv and write results.txt for test_no_labels.csv.

Final system (best macro-F1 in 5-fold CV, see experiments.py):
  1. keyword rule: if the first keyword of a note is a specialty name, predict it;
  2. otherwise a Linear SVM over TF-IDF of the four text fields (specialty term removed
     from keywords), trained with random oversampling of the minority classes;
  3. duplicate rule: if the same transcription is in train.csv, the labels it already
     has there are excluded from the SVM decision.

Usage: python main.py [--no-keyword-rule] [--no-duplicate-rule]
"""
import argparse

import numpy as np

from common import (ConcatTfidf, augment, duplicate_labels, load_test, load_train,
                    make_model, predict_with_exclusion, scores, seed_everything,
                    without_leak)

MODEL, AUG = 'svm', 'ros'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-keyword-rule', action='store_true')
    ap.add_argument('--no-duplicate-rule', action='store_true')
    ap.add_argument('--out', default='results.txt')
    args = ap.parse_args()

    seed_everything()
    train, test = load_train(), load_test()

    train_text = augment(without_leak(train), AUG)
    feats = ConcatTfidf()
    clf = make_model(MODEL).fit(feats.fit_transform(train_text), train_text['medical_specialty'].values)
    S = scores(clf, feats.transform(without_leak(test)))

    if args.no_duplicate_rule:
        pred, hit = clf.classes_[S.argmax(1)], np.zeros(len(test), dtype=bool)
    else:
        pred, hit = predict_with_exclusion(clf.classes_, S, test['raw_transcription'].tolist(),
                                           duplicate_labels(train))

    has_kw = test['kw_specialty'].notna().values
    if not args.no_keyword_rule:
        pred = np.where(has_kw, test['kw_specialty'].values, pred)

    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        for p in pred:
            f.write(p + '\n')

    print(f'{len(pred)} predictions written to {args.out}')
    print(f'  keyword rule applicable: {has_kw.sum()}  |  duplicate in train: {hit.sum()}  |  '
          f'classifier only: {(~has_kw & ~hit).sum()}')


if __name__ == '__main__':
    main()
