# NLP Project 2026 — Help the Doctor

Classification of clinical notes (MTSamples) into 12 medical specialties.

## Files

| File | Purpose |
|---|---|
| `common.py` | data loading, pre-processing, augmentation strategies, models, duplicate rule |
| `experiments.py` | all 5-fold CV experiments of the paper → `results/` |
| `main.py` | trains the final system on `train.csv` and writes `results.txt` |
| `33.tex`, `biblio.bib` | paper (compile with `tectonic 33.tex` or pdflatex + bibtex) |
| `results.txt` | predictions for `test_no_labels.csv`, one label per line, same order, no header |

## Reproduce

Python 3.11.

```bash
pip install -r requirements.txt
python experiments.py     # ~4 min, writes results/*.csv, results/confusion_matrix.png
python main.py            # writes results.txt
```

`main.py --no-keyword-rule --no-duplicate-rule` produces the text-only predictions.

## Final system

1. **Keyword rule** — if the first keyword is a specialty name (true for 83% of the training notes and
   identical to the label in 2171/2172 cases), predict it.
2. Otherwise a **Linear SVM** (C=0.1, balanced class weights) on word 1–2-gram TF-IDF of
   `sample_name + description + keywords + transcription`, with the specialty term removed from
   `keywords`, trained with random oversampling of classes with < 150 notes.
3. **Duplicate rule** — if the same transcription appears in `train.csv`, the labels of those copies are
   excluded from the SVM decision (in 701/702 duplicate groups the copies have different labels).

5-fold CV: 94.9% accuracy, 94.6% macro-F1, 81.6% minimum per-class F1
(text-only SVM: 47.2% / 50.6% / 29.7%; task baseline: 37.5% / 11.4% / 0%).
