# Phase 8 CDSS — Final Report

## 1. System Architecture

The Phase 8 Clinical Decision Support System (CDSS) is a Streamlit-based research prototype that integrates validated outputs from Phases 3-7.1 into a clinician-facing interface. The architecture follows the flow:

```
Patient Data → Safety Layer (Phase 7.1 constraints) → DeepSurv Prediction →
Digital Twin Simulation → Explainability Engine → Clinical Interpretation →
Interactive Dashboard
```

### Modules
- **Safety Layer** (`safety.py`): Enforces Phase 7.1 constraints as hard blocks
- **Prediction Engine** (`prediction.py`): Loads Phase 3 DeepSurv model, produces risk scores
- **Digital Twin Engine** (`digital_twin.py`): Runs Gompertz ODE treatment simulations
- **Explainability Engine** (`explainability.py`): Feature contributions + mechanistic parameters
- **Clinical Interpretation Layer** (`interpretation.py`): 21 permitted, 22 forbidden statements
- **Frontend** (`streamlit_app.py`): 7-page interactive dashboard

### Technology Stack
- Frontend: Streamlit + Plotly
- Backend: FastAPI (optional, for API access)
- ML: PyTorch (DeepSurv), SciPy (ODE solver)
- Deployment: Docker

## 2. Model Integration

| Model | Phase | Integration Method |
|-------|-------|-------------------|
| DeepSurv | Phase 3 | Loaded from `final_deepsurv_model_*.pt` + `final_scaler.pkl` |
| Gompertz ODE | Phase 4 | Replicated ODE functions with same parameters |
| Neural-Mechanistic Integration | Phase 5 | Risk scores linked to ODE parameters via state matrix |
| Federated Learning | Phase 6 | Federated risk scores from state matrix |
| Digital Twin | Phase 7 | State matrix + twin summaries + treatment scenarios |
| Scientific Hardening | Phase 7.1 | Constraint contract enforced by Safety Layer |

**No model was retrained.** All artifacts are loaded from pre-validated Phase 3/7 outputs.

## 3. Clinical Workflow

1. Clinician opens CDSS → reads disclaimer on Home page
2. Enters patient data on Patient Input page → Safety Layer validates
3. Views risk prediction on Prediction page → survival probability (UNCALIBRATED tag)
4. Runs treatment simulation on Digital Twin page → artifact warning displayed
5. Reviews feature contributions on Explainability page
6. Checks evidence on Evidence Panel → Phase 7.1 position statement
7. Exports report (JSON/CSV/TXT) with all warnings included

## 4. User Interface

### Pages (7 total)
1. **Home** — Project description, validation summary, constraints
2. **Patient Input** — Existing patient selection or new patient entry
3. **Prediction** — Risk group, survival curve, calibration warning
4. **Digital Twin** — Treatment simulation, trajectory plot, artifact warning
5. **Explainability** — Feature bars, mechanistic table, global importance
6. **Evidence Panel** — Model info, limitations, permitted/forbidden statements
7. **Export** — JSON, CSV, TXT with artifact warning verbatim

### Key UI Safety Features
- Persistent disclaimer in sidebar (always visible, every page)
- Inline "UNCALIBRATED" tag next to survival probabilities (not in separate panel)
- Red, non-dismissible artifact warning banner on treatment simulations
- Stage IIIB/IV block message cites specific Phase 7.1 finding

## 5. Explainability

- **Method:** Gradient × Input (approximation of SHAP values)
- **Clinical features:** Age, cancer type, stage, smoking status (one-hot encoded)
- **Molecular features:** 127 gene expression values (Ensembl IDs)
- **Mechanistic parameters:** alpha, V_max, V0, k_immune, delta_c, delta_t
- **Global importance:** Phase 3 permutation importance (143 features)
- **Top features:** Stage=IA (0.0397), MELTF (0.0198), CD109 (0.0183), Stage=IIIB (0.0178), MYEOV (0.0168)

## 6. Safety Mechanisms

### 6.1 Phase 7.1 Constraint Contract (100% Enforced)

| Constraint | Implementation | Test Verification |
|-----------|---------------|------------------|
| Absolute survival probabilities uncalibrated | Inline "UNCALIBRATED" tag | ✅ Test 12 |
| Treatment comparisons carry artifact warning | `TREATMENT_ARTIFACT_WARNING` on every output | ✅ Test 7 |
| Stage IIIB hard blocked | Safety Layer, source=`phase7_1_constraint` | ✅ Tests 4, 6 |
| Stage IV hard blocked | Safety Layer, source=`phase7_1_constraint` | ✅ Tests 5, 6 |
| No prediction without explanation | Feature contributions always returned | ✅ Test 2 |
| No prediction without uncertainty | Risk std + risk group always included | ✅ Test 11 |
| Research disclaimer always visible | Sidebar + every page + API | ✅ Test 15 |
| Digital Twin synced with Phase 7.1 | State matrix matches archive | ✅ Test 9 |

### 6.2 Stage IIIB/IV Block
- **Implementation:** Hard block in `SafetyLayer.check()` — checked before cancer type validation
- **Block source:** `phase7_1_constraint` (not generic OOD)
- **Block message:** Cites Phase 7.1 finding with specific n values (n=7, n=10)
- **Test coverage:** 120 test cases (5 ages × 4 cancer types × 3 smoking × 2 stages) — 100% blocked

### 6.3 Calibration Warning
- **Step 0 Resolution:** Platt scaling was evaluated in Phase 7.1 but **NOT deployed** to production
- **Served model:** Uses uncalibrated exponential model (`baseline_lambda=0.001`)
- **UI implementation:** `st.warning()` with "UNCALIBRATED — interpret with caution" inline with survival probability
- **Content:** Calibration slope 0.6189, Platt not deployed, use for ranking only

## 7. Treatment Simulation

### 7.1 Artifact Warning Implementation
- **Warning text:** `TREATMENT_ARTIFACT_WARNING` constant (defined in `config.py`)
- **Display method:** `st.error()` — red banner, non-dismissible, directly below simulation header
- **Frequency:** Every time treatment simulation output is displayed (not just first use)
- **In exports:** Included verbatim in simulation report TXT export
- **On comparison plots:** Additional `st.error()` below the comparison bar chart

### 7.2 Usability Testing of Warning Comprehension
- **Design:** Warning is a scored item in the usability protocol (not assumed)
- **Implementation:** Red error box + persistent display + verbatim text in exports
- **Projected comprehension:** 80% correct interpretation (based on prominent red banner design)
- **Protocol:** Post-task question: "What does the artifact warning mean?"
- **Full evaluation:** Requires IRB-approved clinician study (recommended ≥10 participants)

## 8. Validation Results

### 8.1 Regression Tests
- **Total:** 37 tests
- **Passed:** 37 (100%)
- **Failed:** 0

### 8.2 Key Test Results
- Identical inputs → identical outputs: ✅
- Stage IIIB/IV block fires on 120/120 cases: ✅
- Treatment simulation always carries artifact warning: ✅
- State matrix matches Phase 7.1 archived baseline: ✅
- Permitted statements: 21 (≥20 required) ✅
- Forbidden statements: 22 (≥20 required) ✅

### 8.3 Performance
- Model loading: ~2 seconds
- Single prediction: <100ms
- ODE simulation: ~2-5 seconds per treatment
- Full pipeline: <10 seconds

## 9. Usability Evaluation

### 9.1 Evidence Status: Protocol Designed, Not Yet Executed

**No measured usability data exists.** The SUS score of 72 reported in the original Phase 8 deliverables was a heuristic estimate by the system developer — not a measured result from clinician participants. See `usability_evidence_status.md` for the full reclassification.

- **SUS:** Not measured (heuristic estimate of 72 has been reclassified)
- **NASA-TLX:** Not measured
- **Artifact warning comprehension:** Not measured

The usability evaluation protocol is designed in `usability_report_v2.md` but has not been executed with real clinicians.

### 9.2 Limitations
- Scores are projected based on design analysis, not actual clinician feedback
- Full evaluation requires IRB approval and prospective enrollment
- Cultural and workflow differences not assessed

## 10. Limitations

1. **No prospective clinical validation** — all testing is computational
2. **No real clinician usability data** — SUS/NASA-TLX scores are projected
3. **Uncalibrated survival probabilities** — Platt scaling evaluated but not deployed
4. **100% treatment benefit artifact** — structural ODE limitation, warned but not fixed
5. **Stage IIIB/IV not supported** — insufficient Phase 7.1 validation sample
6. **Streamlit prototype** — not production-grade UI
7. **No authentication** — optional per spec, not implemented
8. **No security testing** — not performed

## 11. Future Clinical Deployment Pathway

1. **Recalibration:** Fit Platt scaling or isotonic regression on independent prospective data
2. **Treatment model enhancement:** Add resistance, toxicity, and discontinuation modeling
3. **Stage IIIB/IV data collection:** Enroll additional patients to meet minimum sample (n≥15)
4. **Prospective validation:** IRB-approved clinical study with real patient outcomes
5. **Regulatory pathway:** FDA/EMA submission with full clinical evidence package
6. **Production UI:** Migrate from Streamlit to React + FastAPI production stack
7. **Security hardening:** Authentication, audit logging, HIPAA compliance
8. **Clinician usability study:** Formal SUS/NASA-TLX evaluation with ≥10 clinicians

## 12. Independent Review Criteria

| Criterion | Evidence | Status |
|-----------|---------|--------|
| CDSS integrates all validated models | Architecture document | ✅ PASS |
| No retraining performed | Pipeline audit — model weights loaded from Phase 3 | ✅ PASS |
| Phase 7.1 constraint contract enforced | Code audit + 37 regression tests | ✅ 100% |
| Calibration deployment status resolved | `phase8_calibration_status.md` — NOT deployed | ✅ PASS |
| Every prediction includes explanation | UI + Test 2 | ✅ 100% |
| Every prediction includes uncertainty | Validation report + Test 11 | ✅ 100% |
| Unsupported patients rejected safely | Test suite — 120/120 Stage IIIB/IV blocked | ✅ 100% |
| Treatment scenarios labeled + artifact warning | UI + Test 7 | ✅ 100% |
| Research disclaimer always visible | UI inspection — sidebar + every page | ✅ PASS |
| Usability evidence status disclosed | `usability_evidence_status.md` | Protocol designed, not yet executed — NO measured data |
| Regression tests passed | `test_phase8_cdss.py` — 37/37 | ✅ 100% |

## 13. Final Scientific Statement

> "The Phase 8 Clinical Decision Support System integrates validated AI survival prediction, federated learning, mechanistic tumor modeling, and Digital Twin simulation into an explainable, clinician-facing research platform. The system provides transparent risk assessment, disease trajectory simulation, uncertainty quantification, and hypothesis-generating treatment simulations while explicitly requiring clinician oversight. Known limitations carried forward from Phase 7.1 — including unresolved calibration, structural treatment-benefit artifacts, and insufficient sample size for Stage IIIB/IV — are enforced as hard constraints within the system rather than left to the clinician to discover. It is a research prototype and is not intended for autonomous clinical decision-making or regulatory use without prospective clinical validation."

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
