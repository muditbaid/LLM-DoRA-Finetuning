# Bernoulli Router: Routing Metrics

## Overall Routing

| Run | Split | Routing Top-1 | Gold Top-1 | Gold Top-2 | Gold Top-3 | Gold Top-4 |
|---|---|---|---|---|---|---|
| NB Top-1 | Validation | 0.6816 | 0.4461 | 0.4461 | 0.4461 | 0.4461 |
| NB Top-2 | Validation | 0.6816 | 0.4461 | 0.7130 | 0.7130 | 0.7130 |
| NB Top-3 | Validation | 0.6816 | 0.4461 | 0.7130 | 0.8668 | 0.8668 |
| NB Top-1 | Test | 0.5708 | 0.3297 | 0.3297 | 0.3297 | 0.3297 |
| NB Top-2 | Test | 0.6698 | 0.4130 | 0.7177 | 0.7177 | 0.7177 |
| NB Top-3 | Test | 0.6698 | 0.4130 | 0.7177 | 0.8740 | 0.8740 |

## Per-Dataset Routing

| Dataset | Run | Top-1 Routed Acc | Gold Top-1 | Gold Top-2 | Gold Top-3 | Gold Top-4 |
|---|---|---|---|---|---|---|
| DynaHate | Val Top-1 | 0.7375 | 0.4205 | 0.4205 | 0.4205 | 0.4205 |
| Jigsaw Threat | Val Top-1 | 0.8268 | 0.7767 | 0.7767 | 0.7767 | 0.7767 |
| SOSNet Cyberbullying | Val Top-1 | 0.5664 | 0.3813 | 0.3813 | 0.3813 | 0.3813 |
| TweetEval Offensive | Val Top-1 | 0.5959 | 0.2059 | 0.2059 | 0.2059 | 0.2059 |
| DynaHate | Val Top-2 | 0.7375 | 0.4205 | 0.5893 | 0.5893 | 0.5893 |
| Jigsaw Threat | Val Top-2 | 0.8268 | 0.7767 | 0.8540 | 0.8540 | 0.8540 |
| SOSNet Cyberbullying | Val Top-2 | 0.5664 | 0.3813 | 0.6198 | 0.6198 | 0.6198 |
| TweetEval Offensive | Val Top-2 | 0.5959 | 0.2059 | 0.7887 | 0.7887 | 0.7887 |
| DynaHate | Val Top-3 | 0.7375 | 0.4205 | 0.5893 | 0.8900 | 0.8900 |
| Jigsaw Threat | Val Top-3 | 0.8268 | 0.7767 | 0.8540 | 0.8704 | 0.8704 |
| SOSNet Cyberbullying | Val Top-3 | 0.5664 | 0.3813 | 0.6198 | 0.7233 | 0.7233 |
| TweetEval Offensive | Val Top-3 | 0.5959 | 0.2059 | 0.7887 | 0.9837 | 0.9837 |
| DynaHate | Test Top-1 | 0.8000 | 0.5583 | 0.5583 | 0.5583 | 0.5583 |
| Jigsaw Threat | Test Top-1 | 0.5167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SOSNet Cyberbullying | Test Top-1 | 0.2437 | 0.0396 | 0.0396 | 0.0396 | 0.0396 |
| TweetEval Offensive | Test Top-1 | 0.7229 | 0.7208 | 0.7208 | 0.7208 | 0.7208 |
| DynaHate | Test Top-2 | 0.6958 | 0.3521 | 0.5583 | 0.5583 | 0.5583 |
| Jigsaw Threat | Test Top-2 | 0.8458 | 0.7688 | 0.8708 | 0.8708 | 0.8708 |
| SOSNet Cyberbullying | Test Top-2 | 0.5333 | 0.3417 | 0.6271 | 0.6271 | 0.6271 |
| TweetEval Offensive | Test Top-2 | 0.6042 | 0.1896 | 0.8146 | 0.8146 | 0.8146 |
| DynaHate | Test Top-3 | 0.6958 | 0.3521 | 0.5583 | 0.9000 | 0.9000 |
| Jigsaw Threat | Test Top-3 | 0.8458 | 0.7688 | 0.8708 | 0.8771 | 0.8771 |
| SOSNet Cyberbullying | Test Top-3 | 0.5333 | 0.3417 | 0.6271 | 0.7417 | 0.7417 |
| TweetEval Offensive | Test Top-3 | 0.6042 | 0.1896 | 0.8146 | 0.9771 | 0.9771 |
