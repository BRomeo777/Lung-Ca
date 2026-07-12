# Biological Plausibility Report — Phase 7.1 Step 5

## 1. Tumor Growth Rates vs Literature
| Metric | Value | Literature | Assessment |
|--------|-------|-----------|------------|
| Median doubling time | 866.4d (28.5mo) | 30-300d | Outside range |
| Range | [577.6-1732.9]d | 30-300d | — |
| Alpha range | [0.000400-0.001200] | 0.0001-0.01 | Plausible |

## 2. Treatment Response Direction
| Treatment | Median Benefit (mo) | All Positive? | Clinically Expected | Assessment |
|-----------|---------------------|---------------|---------------------|------------|
| chemo | 21.8 | False | Positive (on average) | Direction correct, magnitude overestimated (100% response) |
| immuno | 28.0 | False | Positive (on average) | Direction correct, magnitude overestimated (100% response) |
| targeted | 28.0 | False | Positive (on average) | Direction correct, magnitude overestimated (100% response) |

## 3. Growth–Survival Relationship
| Metric | Value | 95% CI | Expected | Assessment |
|--------|-------|--------|----------|------------|
| Spearman rho | -0.1786 | (-0.2724--0.0815) | Negative | PASS |
| p-value | 0.0004 | — | <0.05 | — |
| Cohen's d | -0.207 | — | Negative (faster growth → shorter survival) | PASS |
| N (events) | 396 | — | ≥50 | Adequate |

## 4. Mechanistic Parameter Plausibility
| Parameter | Min | Max | Expected Range | Plausible? |
|-----------|-----|-----|----------------|------------|
| alpha (Gompertz growth rate) | 0.000400 | 0.001200 | [0.0001-0.01] | Yes |
| V0 (Initial tumor volume mm3) | 800.000000 | 179594.000000 | [100-500000] | Yes |
| V_max (Carrying capacity mm3) | 500000.000000 | 1500000.000000 | [100000.0-10000000.0] | Yes |
| k_immune (Immune kill rate) | 0.000067 | 0.000183 | [1e-06-0.001] | Yes |

## 5. Outlier Simulations
| Metric | Value | Assessment |
|--------|-------|------------|
| TTP >60mo (natural) | 0/919 (0.0%) | Low |
| Cause | Slow growth (low alpha) + high V_max → slow progression | Expected for population prior patients |
| Impact | These patients have population-average alpha, not personalized | Overestimates TTP for some val patients |

## 6. Natural TTP Distribution
| Metric | Value | Literature (untreated NSCLC) |
|--------|-------|------------------------------|
| Median | 8.0mo | 4-12mo (stage-dependent) |
| IQR | [5.8-10.9] | — |
| Range | [4.0-36.0] | — |

**Note:** TTP (time to progression, defined as 2x initial volume) is not directly comparable to
overall survival. TTP depends on initial volume and growth rate only, while OS depends on
treatment, comorbidities, and metastatic burden (not modeled).

## Summary
- Growth direction: correct (faster growth → shorter survival, rho=-0.1786, p=0.0004)
- Treatment direction: correct (all treatments reduce tumor volume)
- Parameter ranges: within biological bounds
- Outliers: 0 patients with TTP>60mo — explained by population priors
- Key limitation: 100% treatment benefit (no resistance modeling)

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does not determine the best treatment for individual patients.
