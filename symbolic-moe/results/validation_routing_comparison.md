# Validation Metrics Tables

## Overall Comparison

| Setting | Overall Acc | Routing Top-1 | Gold Top-1 | Gold Top-2 | Gold Top-3 | Macro F1 | Micro F1 |
|---|---|---|---|---|---|---|---|
| Baseline | 0.8581 | 0.6353 | 0.3519 | 0.5076 | 0.7500 | 0.9097 | 0.9448 |
| NB Top-1 | 0.6816 | 0.6816 | 0.4461 | 0.4461 | 0.4461 | 0.9446 | 0.9606 |
| NB Top-2 | 0.8031 | 0.6816 | 0.4461 | 0.7130 | 0.7130 | 0.9300 | 0.9390 |

## Dataset Accuracy

| Dataset | Baseline | NB Top-1 | NB Top-2 |
|---|---|---|---|
| DynaHate | 0.9412 | 0.7375 | 0.8333 |
| Jigsaw Threat | 0.9619 | 0.8268 | 0.8725 |
| SOSNet Cyberbullying | 0.9401 | 0.5664 | 0.7800 |
| TweetEval Offensive | 0.5893 | 0.5959 | 0.7266 |

## Expert Metrics

| Expert | Setting | Accuracy | F1 |
|---|---|---|---|
| DynaHate Expert | Baseline | 0.9412 | 0.9463 |
| DynaHate Expert | NB Top-1 | 0.9482 | 0.9644 |
| DynaHate Expert | NB Top-2 | 0.9409 | 0.9581 |
| Offense Expert | Baseline | 0.8530 | 0.7663 |
| Offense Expert | NB Top-1 | 0.7725 | 0.8491 |
| Offense Expert | NB Top-2 | 0.8218 | 0.8083 |
| Cyberbullying Expert | Baseline | 0.9401 | 0.9634 |
| Cyberbullying Expert | NB Top-1 | 0.9829 | 0.9912 |
| Cyberbullying Expert | NB Top-2 | 0.9719 | 0.9855 |
| Threat Expert | Baseline | 0.9619 | 0.9629 |
| Threat Expert | NB Top-1 | 0.9762 | 0.9738 |
| Threat Expert | NB Top-2 | 0.9694 | 0.9682 |
