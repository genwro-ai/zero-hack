# Baseline comparison

Models: ngram, most_frequent

### Task 1 — Next-step prediction

| Model | top-1 | top-3 | top-5 | MRR |
|---|---|---|---|---|
| ngram | 0.7133 | 0.9967 | 1.0000 | 0.8536 |
| most_frequent | 0.7067 | 0.9950 | 1.0000 | 0.8501 |

### Task 2 — Sequence completion

| Model | exact | norm-edit-dist | token-acc | block-acc |
|---|---|---|---|---|
| ngram | 0.0050 | 0.2243 | 0.3844 | 0.9641 |
| most_frequent | 0.0017 | 0.2458 | 0.4040 | 0.9605 |

### Task 3 — Anomaly detection

| Model | accuracy | precision | recall | F1 | ROC-AUC | rule-attr | detected |
|---|---|---|---|---|---|---|---|
| ngram | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 150 |
| most_frequent | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0 | 150 |
