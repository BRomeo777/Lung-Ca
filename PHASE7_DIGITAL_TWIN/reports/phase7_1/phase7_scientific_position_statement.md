# Phase 7.1 Final Scientific Position Statement

## Document Purpose
This statement defines what Phase 8 (Clinical Decision Support System) can and cannot rely on
from Phase 7.1. It is the authoritative reference for downstream use.

---

## What Has Been Demonstrated

1. **Personalized Digital Twin Construction** — 919 patient-specific state matrices integrating
   clinical variables, mechanistic ODE parameters, AI-derived risk scores, and treatment states.

2. **Disease Trajectory Simulation** — 3,676 Gompertz ODE simulations (919 patients × 4 treatments)
   with 0 solver failures and 0 safety flags. Tumor volume trajectories are mechanistically grounded.

3. **Treatment Scenario Simulation** — 4 treatment arms (natural history, chemotherapy,
   immunotherapy, targeted therapy) with counterfactual outcomes explicitly labeled.

4. **Uncertainty Estimation** — Monte Carlo (n=20) uncertainty quantification for 182 validation
   patients with 95% confidence intervals on TTP. All patients classified as MODERATE uncertainty.

5. **Explainability** — Permutation importance (143 features) and mechanistic driver summaries
   per patient. Top features: Stage=IA, MELTF, CD109, Stage=IIIB, MYEOV.

6. **Reproducibility** — 5/5 reproducibility tests passed. State matrix, ODE simulations, and
   validation metrics verified identical to frozen baseline (Phase7_v1.0).

7. **Discrimination** — C-index 0.6333 (95% CI: 0.5748-0.7021) on internal validation.
   External validation (Phase 3): 5/5 cohorts, mean C-index 0.6588.

8. **Biological Plausibility** — Growth-survival correlation (rho=-0.1786, p=0.0004),
   treatment direction correct, parameter ranges within biological bounds.

---

## What Has NOT Been Demonstrated

1. **Clinical Efficacy** — No prospective clinical trial. All validation is retrospective.

2. **Individualized Treatment Recommendation** — The simulation assumes universal treatment
   eligibility and 100% response. It cannot identify which treatment is best for a specific patient.

3. **Replacement of Oncologists** — This is a Level 1 research prototype for hypothesis generation.
   Clinical judgment, patient preferences, and multidisciplinary review are irreplaceable.

4. **Prospective Validation** — All outcomes are retrospective (TCGA). No prospective data collection.

5. **Regulatory Readiness** — No FDA/EMA submission, no IRB approval for clinical use.

6. **Calibrated Absolute Risk Prediction** — Calibration slope 0.6189 (target: 0.8-1.2).
   Recalibration attempted (Platt, isotonic, temperature scaling); Platt accepted but with
   marginal improvement. Absolute survival probabilities should not be used for clinical decisions.

---

## Subgroups Excluded from Phase 8-Ready Outputs

The following subgroups have been **blocked from Phase 8 inputs** due to unstable predictions:

| Subgroup | Reason |
|----------|--------|
| None | CI does not overlap with overall cohort and CI width too wide |

**Phase 8 must not use predictions for these subgroups without additional validation data.**

---

## Metrics Reported as "Insufficient Sample"

The following subgroups/metrics are reported as "insufficient sample — not interpretable"
rather than a point estimate, because n < 15 (minimum sample threshold):

| Subgroup/Metric | n | Reason |
|-----------------|---|--------|
| Stage=IIIB | < 15 | Below minimum sample threshold |
| Stage=IV | < 15 | Below minimum sample threshold |
| val_stage_IIIB | < 15 | Below minimum sample threshold |
| val_stage_IV | < 15 | Below minimum sample threshold |

---

## Calibration Status
- **Calibration slope:** 0.6189 (target: 0.8-1.2) — **below acceptable range**
- **Recalibration attempted:** Platt scaling (accepted, marginal improvement), isotonic regression (not accepted), temperature scaling (not accepted)
- **Recommendation:** Use DeepSurv risk scores for **ranking only**. Do not use absolute survival probabilities for clinical decisions without further recalibration on independent data.

## Treatment Simulation Status
- **100% treatment benefit** — structural property of the ODE model (positive kill terms)
- **8 assumptions documented** in treatment_assumption_table.csv
- **No resistance, toxicity, or discontinuation modeling**
- Treatment comparisons are **not clinically interpretable** as relative efficacy estimates

## Data Provenance
- All 919 patients have **REAL** clinical data, expression data, and survival outcomes (TCGA)
- 737 train patients have **PERSONALIZED** ODE parameters (from gene expression)
- 182 validation patients have **SYNTHETIC PRIOR** ODE parameters (population averages)
- All simulated TTP values are **SYNTHETIC** (model-derived)
- No metric combines real and synthetic outcomes without explicit labeling

---

## Mandatory Closing Statement

> "The framework provides a computational environment for patient-specific disease simulation and hypothesis generation. Prospective clinical validation remains necessary before any clinical deployment."

---

## Phase 8 Readiness Summary

| Component | Status | Phase 8 Can Use? |
|-----------|--------|-----------------|
| Patient state matrix | ✅ Validated | Yes — for patient characterization |
| DeepSurv risk ranking | ✅ Discrimination validated | Yes — for risk stratification (ranking only) |
| ODE tumor trajectories | ✅ Mechanistically grounded | Yes — for hypothesis generation |
| Treatment comparisons | ⚠️ 100% benefit artifact | No — not for treatment selection |
| Absolute survival probabilities | ⚠️ Uncalibrated | No — not for individual risk prediction |
| Uncertainty intervals | ✅ MC quantified | Yes — for confidence assessment |
| Explainability | ✅ Feature + mechanistic | Yes — for understanding drivers |
| Excluded subgroups | ❌ Blocked | No — see list above |

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does not determine the best treatment for individual patients.
