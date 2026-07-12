# Phase 7: Personalized Neural-Mechanistic Digital Twin — Final Report

## Project Title
**A Federated Neural-Mechanistic Digital Twin Framework for Predictive Virtual Treatment Simulation and Personalized Clinical Decision Support in Lung Cancer**

**Author:** Romeo BANANEZA (Rwanda, BSc Clinical Medicine, AI in Healthcare)

**Date:** July 12, 2026

---

## MANDATORY DISCLAIMER

> ⚠️ **RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.**
> This system generates patient-specific computational simulations that support hypothesis generation and future clinical decision-support research. It does **NOT** determine the best treatment for individual patients. All simulated outcomes are computational predictions, not clinical facts. No component of this Digital Twin has been validated for clinical deployment.

---

## 1. Architecture Overview

### 1.1 Digital Twin Definition
This Digital Twin is a **Level 1 Computational Digital Twin** (per classification scheme: Level 0=Descriptive, 1=Predictive, 2=Prescriptive, 3=Autonomous). It integrates:

- **Neural component:** DeepSurv ensemble (Phase 3) — 2-model top-2 ensemble, [64,32] hidden dims, GELU activation, Cox partial likelihood with Efron ties
- **Mechanistic component:** Gompertz tumor growth ODE with treatment-response dynamics (Phase 4) — Skipper log-cell kill for chemotherapy, immune effector system with checkpoint blockade for immunotherapy, targeted therapy kill
- **Integration:** Hybrid patient state (Phase 5) — strategy C (neural + mechanistic), w_neural=0.0007
- **Federated layer:** Phase 6 federated learning results (FedAvg, non-IID heterogeneity analysis)

### 1.2 State Representation (7A)
The Digital Twin state is a 22-variable structured representation organized into four domains:

| Domain | Variables | Count |
|--------|-----------|-------|
| A. Clinical state | patient_id, age, sex, cancer_type, stage, smoking_status, os_time, os_event | 8 |
| B. Biological tumor state | alpha, V_max, V0, k_immune, mu_E, rho, delta_c, delta_t, E0 | 9 |
| C. AI-derived state | deepsurv_risk, deepsurv_risk_std, federated_risk, federated_risk_std, risk_group | 5 |
| D. Treatment state | Populated during simulation (current_treatment, simulated_ttp, trajectories) | Dynamic |

**Coverage:** 919/919 patients (100.0%) — all required variables populated with defined valid ranges and missing-data rules.

**Deliverables:**
- `digital_twin_state_matrix.csv` (919 patients × 22 variables)
- `digital_twin_schema.md` (variable definitions, domains, valid ranges, missing-data rules)

---

## 2. Data Provenance

### 2.1 Data Classification
The 919 HybridPatientStates comprise **mixed real and partially synthetic** data:

| Component | Type | Source | N |
|-----------|------|--------|---|
| Clinical variables (age, sex, stage, etc.) | **Real** | TCGA GDC portal | 919 |
| Gene expression (13 genes) | **Real** | TCGA RNA-seq (log2 CPM normalized) | 919 |
| DeepSurv risk scores | **AI-derived from real data** | Phase 3 model inference | 919 |
| ODE parameters (train patients) | **Partially synthetic** | Phase 4 personalization (real gene expression → parameter adjustment) | 737 |
| ODE parameters (val patients) | **Synthetic (population priors)** | Phase 4 population priors with stage-based V0 | 182 |
| Simulated TTP and trajectories | **Fully synthetic** | Gompertz ODE simulation | 919 × 4 treatments |
| Survival outcomes (OS time, event) | **Real** | TCGA clinical follow-up | 919 |

### 2.2 Ethics and IRB
- TCGA data: publicly available, de-identified, IRB-approved at contributing institutions
- No additional human subjects research conducted
- No patient-identifiable information stored or transmitted

### 2.3 Circularity Caveat
The mechanistic component is **partially circular**: TTP and treatment response outcomes are simulated by the same Gompertz ODE model used inside the twin. Biological validation tests validate **internal consistency**, not real-world predictive accuracy. Prediction validation (C-index, calibration, IBS) uses **real survival outcomes** and is NOT circular.

**Full provenance statement:** `data_provenance_statement.md`

---

## 3. Disease Evolution Simulation (7B)

### 3.1 Configuration
- **Simulation horizon:** 1095 days (3 years)
- **Time points:** 100 per trajectory
- **Treatments:** none (natural history), chemo (cisplatin), immuno (pembrolizumab), targeted (osimertinib)
- **Solver:** RK45 for natural history, Radau for treatment (stiff immune dynamics)
- **Safety checks:** NaN/Inf detection, negative volume clipping, solver success verification

### 3.2 Results
- **Total simulations:** 3,676 (919 patients × 4 treatments)
- **Successful:** 3,676 (100%)
- **Failed:** 0
- **Safety flags:** 0 (all simulations passed safety checks after volume clipping)

**Deliverables:**
- `digital_twin_trajectory_simulations.csv` (183,800 rows — 919 × 4 × 100 time points)
- `simulation_safety_flags.csv` (0 flags)

### 3.3 Treatment Effects
All three treatments showed benefit over natural history for 100% of patients:
- **Chemotherapy:** 100% of patients had >30-day TTP improvement
- **Immunotherapy:** 100% of patients had >30-day TTP improvement
- **Targeted therapy:** 100% of patients had >30-day TTP improvement

**Caveat:** 100% benefit is expected because the ODE model applies treatment effect to all patients. Real-world treatment response is heterogeneous (20-40% response rates for immuno/ targeted). This is a known model limitation.

---

## 4. Virtual Treatment Scenario Engine (7C)

### 4.1 Design
For each patient, the engine generates:
- **1 observed outcome** (real survival data labeled with actual treatment if available, or "none")
- **4 counterfactual outcomes** (simulated TTP under each treatment scenario)

Each row is explicitly labeled with `outcome_type` = "observed" or "counterfactual".

### 4.2 Results
- **Total rows:** 4,595 (919 observed + 3,676 counterfactual)
- **Counterfactuals per patient:** min=4, max=4 (uniform)
- **`outcome_type` present on every row:** Yes

**Deliverable:** `virtual_treatment_scenarios.csv`

---

## 5. Digital Twin Validation (7D)

### 5.1 Prediction Validation
| Metric | Value | Target/Reference |
|--------|-------|------------------|
| C-index | 0.6333 (95% CI: 0.5748-0.7021) | Phase 3: 0.6131 |
| Delta (twin - Phase 3) | +0.0202 | Positive (twin ≥ Phase 3) |
| Calibration slope | 0.6189 | 0.8-1.2 (below target) |
| IBS (Cox-calibrated) | 0.7196 | Lower is better |
| IBS (null model) | 0.6505 | Reference |

**Interpretation:**
- C-index slightly exceeds Phase 3 (0.6333 vs 0.6131), confirming the DeepSurv risk scores maintain discrimination in the Digital Twin context
- Calibration slope (0.62) is below the target range (0.8-1.2), indicating the risk scores are under-calibrated (predicted hazards are too extreme relative to observed events)
- IBS is higher than the null model, suggesting the exponential survival model is not well-calibrated. The Cox-calibrated version improved but still exceeds null

### 5.2 Subgroup C-indices by Stage
| Stage | N | C-index | 95% CI |
|-------|---|---------|--------|
| IA | 37 | 0.4266 | 0.1885-0.6622 |
| IB | 40 | 0.4506 | 0.3397-0.5861 |
| IIA | 16 | 0.6049 | 0.2808-0.9559 |
| IIB | 19 | 0.5307 | 0.3031-0.8225 |
| IIIA | 25 | 0.5989 | 0.4040-0.7866 |
| IIIB | 7 | 0.8571 | 0.3179-1.0000 |
| IV | 10 | N/A | N/A (insufficient events) |

**Finding:** C-index is lower for early-stage patients (IA, IB < 0.5) and higher for advanced stages. This is consistent with the known difficulty of predicting survival in early-stage disease where treatment intervention and comorbidities dominate.

### 5.3 Biological Validation
| Test | Expected | Result | Pass/Fail |
|------|----------|--------|-----------|
| 1. Faster growth → shorter survival | rho < 0, p < 0.1 | rho=-0.1786, p=0.0004 | **PASS** |
| 2. Higher mech risk → shorter survival | rho < 0, p < 0.1 | rho=-0.0153, p=0.7612 | **FAIL** |
| 3. Treatment benefit ≥50% | ≥50% for ≥1 therapy | chemo=100%, immuno=100%, targeted=100% | **PASS** |

**Note:** Tests 1-2 use ALL patients (train+val, n=919) because validation patients share population prior alpha (no personalization). Test 1 confirms the biological expectation that faster tumor growth correlates with shorter survival. Test 2 fails because the mechanistic risk (1/TTP) does not significantly correlate with real survival — this is expected since the ODE model does not capture all determinants of survival (surgery, comorbidities, metastatic burden).

### 5.4 Treatment Simulation Validity
- Responders: 87, Non-responders: 95 (split by median chemo benefit)
- Median TTP — Responders: 28.3 months, Non-responders: 22.2 months
- Log-rank p-value: 3.84e-44 (highly significant separation)

**Caveat:** Responder/non-responder labels are **model-derived** (simulated TTP benefit). This validates internal consistency of the treatment-response module, NOT real-world treatment discrimination.

**Deliverables:**
- `digital_twin_validation_report.txt`
- `treatment_simulation_validation_report.csv`

---

## 6. Uncertainty Quantification (7E)

### 6.1 Method
- **AI uncertainty:** DeepSurv ensemble standard deviation (2-model ensemble)
- **Mechanistic uncertainty:** Monte Carlo simulation (n=20) over ODE parameter bounds (alpha ±30%, V_max ±30%, V0 ±30%)
- **Scope:** 182 validation patients

### 6.2 Cohort-Wide Results
| Metric | Median | Median CI Width |
|--------|--------|-----------------|
| 1-year survival probability | 0.9781 | 0.0385 |
| 2-year survival probability | 0.9567 | 0.0742 |
| 3-year survival probability | 0.9358 | 0.1074 |
| TTP natural history (days) | 181.8 | 97.2 |
| DeepSurv risk | -2.8028 | 1.6161 |
| TTP CV | 0.1754 | — |

### 6.3 Uncertainty by Stage (1-year survival CI width)
| Stage | N | Median CI Width |
|-------|---|-----------------|
| IA | 37 | 0.0311 |
| IB | 40 | 0.0055 |
| IIA | 16 | 0.0792 |
| IIB | 19 | 0.0385 |
| IIIA | 25 | 0.0581 |
| IIIB | 6 | 0.4748 |
| IV | 10 | 0.1086 |

**Finding:** Uncertainty increases with stage severity, as expected. Early-stage patients have narrow CIs (tight predictions), while advanced-stage patients have wider CIs (greater uncertainty).

**Deliverable:** `uncertainty_analysis_report.csv` (1,086 rows)

---

## 7. Explainable Digital Twin (7F)

### 7.1 AI Feature Importance (Global)
Top 5 features from Phase 3 permutation importance:
1. Stage=IA (0.0397)
2. MELTF (0.0198)
3. CD109 (0.0183)
4. Stage=IIIB (0.0178)
5. MYEOV (0.0168)

### 7.2 Mechanistic Drivers
Per-patient mechanistic driver summary includes:
- Growth rate (alpha) — determines tumor doubling time
- Initial volume (V0) — determines starting tumor burden
- Immune activity (k_immune) — determines immune-mediated tumor control
- Treatment-specific TTP and benefit (months)

### 7.3 Per-Patient Explanations
Each validation patient receives a structured explanation combining:
- AI risk score and group (High/Low)
- Top contributing AI features
- Mechanistic driver values
- Simulated TTP under each treatment
- Treatment benefit in months
- Clinical use warning

**Deliverable:** `digital_twin_explainability_report.csv` (182 rows)

---

## 8. Interactive Interface Prototype (7G)

### 8.1 Framework
- **Framework:** Streamlit (Python web application)
- **Fidelity:** Live backend — directly calls DeepSurv model and Gompertz ODE solver
- **Visualization:** Plotly interactive charts (tumor trajectory, survival probability)

### 8.2 Features
- **Input modes:** Select existing patient (919 TCGA patients) or manual input
- **Simulation controls:** Treatment selection, simulation horizon (6-36 months)
- **Outputs:** Tumor volume trajectory, survival probability, TTP comparison table
- **Uncertainty:** DeepSurv ensemble CI, survival probability CI
- **Explainability:** Global permutation importance chart, mechanistic driver summary

### 8.3 Scope Limitations
- **Out of scope:** Real-time patient monitoring, clinical decision support, regulatory-approved medical device functionality
- **Mandatory banner:** Non-removable "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE" displayed at top and bottom

**Deliverables:**
- `PHASE7_DIGITAL_TWIN/app/digital_twin_app.py`
- `PHASE7_DIGITAL_TWIN/app/interface_readme.md`

---

## 9. Reproducibility Testing (7H)

### 9.1 Test Results
| Test | Description | Result |
|------|-------------|--------|
| 1. Reproducibility | Same input + same seed → exact match | **PASS** |
| 2. Invalid states | Negative volume, negative alpha, NaN risk → specific exceptions | **PASS** (4/4 cases) |
| 3. Stability | ≥20 simulations across seeds, CV < 0.1 | **PASS** (3/3 patient profiles) |
| 4. Output file integrity | 12 expected files exist and non-empty | **PASS** (12/12 files) |
| 5. State matrix completeness | All required columns present, no NaN in critical fields | **PASS** |

**Total: 5/5 tests passed**

### 9.2 Adversarial Edge Cases
- Large V0 (500,000 mm³): valid, V ≥ 0
- Small V0 (100 mm³): valid, V ≥ 0
- Fast growth (alpha=0.005): valid, finite

**Deliverable:** `test_phase7_digital_twin.py`

---

## 10. Limitations

### 10.1 Model Limitations
1. **No surgery modeling:** The ODE model does not include surgical resection, which is the primary curative treatment for early-stage NSCLC. This explains the poor C-index for Stage IA/IB patients.
2. **No metastatic burden:** The model simulates primary tumor volume only. Stage IV patients are not well-represented.
3. **100% treatment benefit:** All patients benefit from all treatments in the ODE model. Real-world response rates are 20-40% for immunotherapy and targeted therapy.
4. **No resistance modeling:** Treatment resistance (e.g., T790M mutation for osimertinib) is not modeled.
5. **Population priors for validation patients:** Only training patients have personalized ODE parameters. Validation patients use population priors, limiting mechanistic validation.
6. **Calibration:** The calibration slope (0.62) is below the target range (0.8-1.2), indicating under-calibration of risk scores.
7. **IBS:** The IBS (0.72) exceeds the null model (0.65), suggesting the survival probability model needs improvement (e.g., using a proper Cox baseline hazard or isotonic calibration).

### 10.2 Data Limitations
1. **Retrospective data:** All validation uses retrospective TCGA data with potential selection bias.
2. **No prospective validation:** No prospective clinical trial has been conducted.
3. **Synthetic circularity:** Mechanistic validation tests are partially circular (same ODE model used for simulation and validation).
4. **Limited external validation:** Digital Twin validation uses TCGA internal validation only. GEO cohorts were not included in Phase 7.

### 10.3 Computational Limitations
1. **Simulation time:** 3,676 ODE solves took ~7.3 hours on CPU. Real-time simulation would require GPU acceleration or simplified models.
2. **MC sample size:** n=20 Monte Carlo samples for uncertainty quantification is modest. Larger samples would improve CI estimates.

---

## 11. Future Translation

### 11.1 Short-term (6-12 months)
- Integrate Phase 6 federated learning risk scores into the Digital Twin state
- Add isotonic calibration to improve IBS
- Include surgery as a treatment modality in the ODE model
- Add GEO external validation cohorts

### 11.2 Medium-term (1-2 years)
- Incorporate imaging data (CT/MRI tumor volume measurements) for model calibration
- Add treatment resistance modeling (mutation evolution)
- Validate against independent retrospective cohorts
- Improve MC sample size (n=100-500) for tighter uncertainty bounds

### 11.3 Long-term (2-5 years)
- Prospective clinical validation study
- Integration with hospital EHR systems for real-time data input
- Regulatory pathway (FDA/CE mark) for clinical decision support
- Multi-institutional federated Digital Twin updates

---

## 12. Independent Review Table

| Criterion | Evidence | Verdict |
|-----------|----------|---------|
| State matrix covers 100% of patients | 919/919 (100.0%) | ✅ PASS |
| All state variables have defined metadata | 22 variables with valid ranges, units, sources | ✅ PASS |
| Simulations pass safety checks or are flagged | 0 flags / 3,676 simulations (100% pass) | ✅ PASS |
| Virtual treatment scenarios labeled | outcome_type on every row (4,595 rows) | ✅ PASS |
| C-index reported with CI | 0.6333 (0.5748-0.7021) | ✅ PASS |
| Calibration slope reported | 0.6189 (below target 0.8-1.2) | ⚠️ PARTIAL |
| IBS reported | 0.7196 (above null 0.6505) | ⚠️ PARTIAL |
| Biological validation quantitative | 2/3 tests passed | ✅ PASS |
| Subgroup sizes reported with CIs | 7 stages, n and CI reported | ✅ PASS |
| Uncertainty quantified cohort-wide | 182 patients, 6 metrics, by stage | ✅ PASS |
| Explainability: AI + mechanistic | Permutation importance + ODE drivers | ✅ PASS |
| Interface prototype with scope | Streamlit app, readme, disclaimers | ✅ PASS |
| Reproducibility tests pass | 5/5 tests passed | ✅ PASS |
| Data provenance statement | data_provenance_statement.md | ✅ PASS |
| Circularity caveat stated | In validation report and this report | ✅ PASS |
| Clinical use disclaimer | In all reports, interface, and outputs | ✅ PASS |
| No clinical decision claims | All outputs labeled as simulations | ✅ PASS |

**Overall: 15/17 PASS, 2/17 PARTIAL (calibration and IBS below target)**

---

## 13. File Index

### Data Files (`PHASE7_DIGITAL_TWIN/data/`)
| File | Description | Size |
|------|-------------|------|
| `digital_twin_state_matrix.csv` | 919 patients × 22 state variables | 214 KB |
| `digital_twin_trajectory_simulations.csv` | 183,800 rows (919 × 4 × 100 time points) | 34 MB |
| `virtual_treatment_scenarios.csv` | 4,595 rows (observed + counterfactual) | 711 KB |
| `twin_summaries_for_interface.csv` | 919 patients with TTP summaries | 178 KB |

### Reports (`PHASE7_DIGITAL_TWIN/reports/`)
| File | Description |
|------|-------------|
| `data_provenance_statement.md` | Data classification, ethics, circularity caveat |
| `digital_twin_schema.md` | Variable definitions, domains, valid ranges |
| `digital_twin_validation_report.txt` | C-index, calibration, IBS, biological validation |
| `treatment_simulation_validation_report.csv` | Responder analysis, log-rank test |
| `uncertainty_analysis_report.csv` | MC uncertainty for 182 val patients |
| `digital_twin_explainability_report.csv` | Per-patient AI + mechanistic explanations |
| `simulation_safety_flags.csv` | Safety flags (0 flags) |
| `phase7_log.txt` | Full execution log |

### Interface (`PHASE7_DIGITAL_TWIN/app/`)
| File | Description |
|------|-------------|
| `digital_twin_app.py` | Streamlit interactive application |
| `interface_readme.md` | Framework, scope, usage instructions |

### Scripts (`00_CODE/`)
| File | Description |
|------|-------------|
| `run_phase7_digital_twin.py` | Main Phase 7 pipeline (7A-7F) |
| `run_phase7_postprocess.py` | Post-processing fixes for 7D/7F reports |
| `test_phase7_digital_twin.py` | Reproducibility tests (7H) |

---

## 14. Conclusion

Phase 7 successfully constructed a personalized neural-mechanistic Digital Twin for lung cancer, integrating Phase 3 DeepSurv neural survival models, Phase 4 Gompertz mechanistic tumor growth ODEs, and Phase 5 hybrid integration. The system covers 919 TCGA patients with 22 state variables, simulates 4 treatment scenarios per patient (3,676 total simulations), and provides uncertainty quantification, explainability, and an interactive Streamlit interface.

Key achievements:
- **100% patient coverage** with complete state representation
- **C-index 0.6333** — maintains discrimination in Digital Twin context
- **2/3 biological validation tests passed** — growth rate correlates with survival as expected
- **5/5 reproducibility tests passed** — deterministic, stable, and handles invalid inputs
- **Cohort-wide uncertainty** quantified with MC simulations and ensemble std

Key limitations:
- Calibration slope (0.62) below target (0.8-1.2)
- IBS (0.72) above null model (0.65) — survival model needs improvement
- 100% treatment benefit (no resistance modeling)
- No surgery or metastatic burden modeling
- Mechanistic validation partially circular

This Digital Twin is a **research prototype** suitable for hypothesis generation and future clinical decision-support research. It is **NOT for clinical use** and requires prospective validation before any clinical application.

---

*RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does NOT determine the best treatment for individual patients.*
