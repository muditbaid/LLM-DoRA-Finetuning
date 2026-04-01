# NB Top-k Calibration

Validation-only calibration of the Bernoulli router execution budget.

## Selection Rule

- Minimum gold-expert recall@k: `0.70`
- Maximum routing top-1 drop from the best candidate: `0.02`
- Among eligible candidates, choose the smallest `k`.
- If no candidate is eligible, fall back to the smallest `k` with available gold-expert recall.

## Candidate Summary

| Run | Gold Recall@k | Routing Top-1 | Overall Acc | Macro F1 | Micro F1 | Eligible |
|---|---|---|---|---|---|---|
| Top-1 | 0.4461 | 0.6816 | 0.6816 | 0.9446 | 0.9606 | no |
| Top-2 | 0.7130 | 0.6816 | 0.8031 | 0.9300 | 0.9390 | yes |
| Top-3 | 0.8668 | 0.6816 | 0.8584 | 0.9253 | 0.9280 | yes |

## Decision

- Best validation routing top-1: `0.6816`
- Selected execution budget: `top-2`
- Selected validation gold-expert recall@k: `0.7130`
- Selected validation routing top-1: `0.6816`
- Selected validation overall accuracy: `0.8031`

## Interpretation

This calibration prioritizes routing quality over permissive end-to-end accuracy alone. Overall accuracy and F1 are retained as secondary indicators of deployment behavior rather than the primary selection rule.
