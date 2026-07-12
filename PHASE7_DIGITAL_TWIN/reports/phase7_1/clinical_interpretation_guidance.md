# Clinical Interpretation Guidance — Phase 7.1 Step 7

## Purpose
This document defines what should and should not be inferred from the Digital Twin outputs.
It is mandatory reading before any use of Phase 7 results.

---

## ✅ Correct Inferences

1. **"The Digital Twin estimates disease trajectories under explicit modeling assumptions."**
   - The Gompertz ODE with treatment kill terms produces mechanistically-grounded trajectories
   - Assumptions are documented in treatment_assumption_table.csv

2. **"The Digital Twin ranks patients by predicted risk using a validated neural survival model."**
   - DeepSurv C-index 0.6333 (95% CI: 0.5748-0.7021) on internal validation
   - External validation: 5/5 cohorts, mean C-index 0.6588

3. **"The Digital Twin quantifies uncertainty in simulated outcomes via Monte Carlo sampling."**
   - MC (n=20) over ODE parameters with 95% confidence intervals
   - Uncertainty levels: MODERATE for all patients

4. **"The Digital Twin identifies mechanistic drivers of tumor progression per patient."**
   - Permutation importance (143 features) + mechanistic parameter summaries
   - Top features: Stage=IA, MELTF, CD109, Stage=IIIB, MYEOV

5. **"The Digital Twin provides a reproducible computational framework for hypothesis generation."**
   - 5/5 reproducibility tests passed
   - Deterministic outputs with fixed seed

---

## ❌ Incorrect Inferences (Forbidden)

1. **"The Digital Twin determines the best treatment."**
   - The simulation assumes all patients are eligible for all treatments
   - No resistance, toxicity, or discontinuation modeling
   - 100% treatment benefit is a structural artifact, not a clinical finding

2. **"The Digital Twin predicts individual patient survival."**
   - Calibration slope 0.6189 (target: 0.8-1.2) — absolute probabilities are not calibrated
   - Use risk scores for ranking only, not absolute risk prediction

3. **"The Digital Twin can replace oncologist judgment."**
   - This is a Level 1 research prototype
   - No prospective validation, no regulatory approval
   - Outputs are computational simulations, not clinical recommendations

4. **"Simulated treatment outcomes represent what would actually happen."**
   - Counterfactual simulations are model-based projections under idealized conditions
   - They do not account for patient-specific factors (comorbidities, performance status, genomics)
   - Every counterfactual outcome is labeled as such in counterfactual_audit.csv

5. **"The Digital Twin is validated for clinical deployment."**
   - Validated for computational reproducibility and internal consistency only
   - External validation of DeepSurv (Phase 3) does not validate the full Digital Twin pipeline
   - Prospective clinical validation remains necessary

---

## Context for Each Output

| Output | What It Shows | What It Does NOT Show |
|--------|--------------|----------------------|
| State matrix | Patient variables at baseline | Real-time patient state |
| Trajectory simulations | Tumor volume over time under ODE model | Actual tumor behavior |
| Treatment scenarios | TTP under each treatment (idealized) | Realistic treatment outcomes |
| Validation metrics | Model discrimination (C-index) | Clinical accuracy |
| Uncertainty analysis | MC confidence intervals on TTP | Full uncertainty (no model uncertainty) |
| Explainability | Feature importance for risk prediction | Causal mechanisms |

---

## Mandatory Disclaimer

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does not determine the best treatment for individual patients.

**All simulated outcomes are counterfactual projections under explicit modeling assumptions.
They must not be presented as clinical facts or used for individualized treatment recommendations.**
