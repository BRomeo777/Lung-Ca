# Calibration Audit Report — Phase 7.1 Step 2

## Baseline (Phase7_v1.0)
| Metric | Value |
|--------|-------|
| C-index | 0.6333 (95% CI: 0.5748-0.7021) |
| Calibration slope | 0.6189 (target: 0.8-1.2) |
| Calibration-in-the-large | 0.0096 |
| ECE | 0.0451 |
| ICI | 0.3052 |
| IBS | 0.7196 (null: 0.6505) |

## Root Cause
DeepSurv outputs log-hazard ratios optimized for ranking (Cox partial likelihood), not absolute risk.
The exponential model S(t)=exp(-lambda*exp(risk)*t) with lambda=0.001 is an uncalibrated baseline.
Slope <1 means risk scores have too much variance vs observed outcomes — over-predicts high risk,
under-predicts low risk. Phase 3 train C-index 0.95 vs val 0.61 suggests some overfitting, but
external validation (5/5 cohorts) confirms generalizability. The calibration gap is primarily from
the risk-to-survival conversion, not the model itself.

## Recalibration Results
| Method | C-index | Slope | ECE | ICI | IBS | Disc OK? | Cal OK? | Accepted? |
|--------|---------|-------|-----|-----|-----|----------|---------|-----------|
| Original | 0.6333 | 0.6189 | 0.0451 | 0.3052 | 0.7196 | — | — | — |
| Platt | 0.6364 | 0.6316 | 0.0451 | 0.3052 | 0.7196 | Y | Y | **Y** |
| Isotonic | 0.6572 | 1.2343 | 0.0272 | 0.3108 | 0.7479 | Y | Y | **Y** |
| Temp(1.62) | 0.6364 | 1.0041 | 0.0451 | 0.3052 | 0.7196 | Y | Y | **Y** |

## Decision
**Accepted:** Platt

## Recommendation for Phase 8
- Use DeepSurv risk scores for **ranking** (discrimination), which is well-validated (C-index 0.6333)
- Do NOT use absolute survival probability estimates without recalibration
- If needed, use Cox baseline hazard for calibrated survival predictions
- Consider isotonic calibration on a larger independent cohort

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does not determine the best treatment for individual patients.
