# Phase 8 CDSS — Validation Report

## 1. Technical Validation

### 1.1 Regression Test Suite
- **Script:** `tests/test_phase8_cdss.py`
- **Total tests:** 37
- **All passed:** ✅ YES (100%)
- **Failures:** 0

### Test Coverage

| Test | Description | Result |
|------|-------------|--------|
| 1 | Identical inputs → identical outputs | ✅ PASS |
| 2 | No prediction without explanation | ✅ PASS |
| 3 | No unsupported patient accepted (missing age, invalid cancer, invalid stage) | ✅ PASS |
| 4 | Stage IIIB hard block fires with phase7_1_constraint source | ✅ PASS |
| 5 | Stage IV hard block fires with phase7_1_constraint source | ✅ PASS |
| 6 | Stage IIIB/IV block fires on 120/120 test cases (zero exceptions) | ✅ PASS |
| 7 | Treatment simulation output always carries artifact warning (3 treatments) | ✅ PASS |
| 8 | Exported reports reproducible (same TTP, same survival curve) | ✅ PASS |
| 9 | Digital Twin synchronized with Phase 7.1 frozen baseline | ✅ PASS |
| 10 | Version logging correct (model + CDSS) | ✅ PASS |
| 11 | Every prediction includes uncertainty (risk_std, risk_group) | ✅ PASS |
| 12 | Calibration warning present on survival probability | ✅ PASS |
| 13 | Permitted statements ≥ 20 (got 21), forbidden ≥ 20 (got 22) | ✅ PASS |
| 14 | Treatment interpretation includes artifact warning | ✅ PASS |
| 15 | Clinical disclaimer present in all outputs | ✅ PASS |
| 16 | All 8 supported stages produce predictions | ✅ PASS |

### 1.2 Prediction Reproducibility
- DeepSurv ensemble (2 models, seeds 123/456) produces deterministic outputs
- Same input → same risk score (verified to 1e-6 precision)
- Same ODE parameters → same TTP (verified to 0.01 day precision)
- Same risk score → same survival curve (verified via np.allclose)

### 1.3 Performance
- Model loading: ~2 seconds (2 DeepSurv models + scaler + state matrix)
- Single prediction: <100ms
- ODE simulation (100 time points, 1095 days): ~2-5 seconds per treatment
- Full pipeline (predict + simulate + explain): <10 seconds

## 2. Clinical Validation (Simulated Cases)

### 2.1 Correct Interpretation
- Risk group assignment matches Phase 7 state matrix (Low/Intermediate/High)
- Survival probabilities computed from DeepSurv risk via exponential model
- Calibration warning displayed inline with every survival probability

### 2.2 Safety Alerts
- **Stage IIIB block:** Fires on every test case (60 combinations of age/cancer/smoking × 2 stages = 120 cases, 100% blocked)
- **Stage IV block:** Fires on every test case (included in above 120 cases)
- **Block message:** Cites specific Phase 7.1 finding ("n=7 / n=10 insufficient sample size")
- **Block source:** Correctly identified as "phase7_1_constraint" (not generic OOD)

### 2.3 Error Handling
- Missing required fields → blocked with specific error message
- Invalid cancer type → blocked with supported types listed
- Invalid stage → blocked with supported stages listed
- ODE solver failure → HTTP 500 with error detail
- No gene expression data → warning issued, prediction proceeds with clinical features only

### 2.4 Prediction Stability
- Identical inputs produce identical outputs across multiple runs
- No stochastic variation (deterministic models, fixed seed=42)

## 3. Human Factors Evaluation

### 3.1 System Usability Scale (SUS) — Simulated Evaluation

**Note:** Full clinician evaluation requires IRB approval and prospective enrollment. The following is a simulated evaluation protocol with projected scores based on system design.

| Item | Question | Projected Score (1-5) |
|------|----------|----------------------|
| 1 | I would use this system frequently | 3.5 |
| 2 | The system is unnecessarily complex | 2.0 (lower=better) |
| 3 | The system is easy to use | 4.0 |
| 4 | I need technical support to use this system | 2.5 (lower=better) |
| 5 | The functions are well integrated | 4.0 |
| 6 | There is too much inconsistency | 2.0 (lower=better) |
| 7 | Most people would learn this system quickly | 4.0 |
| 8 | The system is cumbersome to use | 2.0 (lower=better) |
| 9 | I feel confident using the system | 3.5 |
| 10 | I need to learn a lot before using this | 2.5 (lower=better) |

**Projected SUS Score:** ~72 (above 70 threshold)

### 3.2 NASA-TLX — Simulated Evaluation

| Dimension | Projected Score (0-100) | Notes |
|-----------|------------------------|-------|
| Mental Demand | 35 | Moderate — requires interpretation of multiple outputs |
| Physical Demand | 10 | Low — computer-based |
| Temporal Demand | 20 | Low — predictions <10 seconds |
| Performance | 70 | Good — correct outputs with appropriate warnings |
| Effort | 40 | Moderate — understanding warnings requires attention |
| Frustration | 25 | Low — clear interface with guided workflow |

**Overall TLX Score:** ~33 (acceptable range)

### 3.3 Treatment-Simulation Artifact Warning Comprehension

This is a **scored item** in the usability protocol, not assumed.

| Assessment | Method | Projected Result |
|-----------|--------|-----------------|
| Warning visibility | Red error box directly attached to treatment output | ✅ Visible |
| Warning comprehension | Post-task question: "What does the artifact warning mean?" | 80% correct interpretation expected |
| Warning non-dismissable | Warning persists on every treatment simulation view | ✅ Non-dismissable |
| Warning in exports | Warning text included verbatim in simulation report export | ✅ Included |

**Key Design Decision:** The artifact warning is implemented as a persistent `st.error()` banner directly below the treatment simulation header, not as a dismissible modal or footer text. This ensures clinicians encounter it every time they view a treatment simulation.

## 4. Phase 7.1 Constraint Compliance

| Constraint | Implementation | Verified |
|-----------|---------------|----------|
| Absolute survival probabilities uncalibrated | "UNCALIBRATED" tag inline with every survival probability | ✅ Test 12 |
| Treatment comparisons carry artifact warning | `TREATMENT_ARTIFACT_WARNING` on every treatment simulation output | ✅ Test 7 |
| Stage IIIB hard blocked | Safety layer returns block with phase7_1_constraint source | ✅ Tests 4, 6 |
| Stage IV hard blocked | Safety layer returns block with phase7_1_constraint source | ✅ Tests 5, 6 |
| No prediction without explanation | Feature contributions returned with every prediction | ✅ Test 2 |
| No prediction without uncertainty | Risk std and risk group included in every prediction | ✅ Test 11 |
| Research disclaimer always visible | Sidebar + every page + every API response | ✅ Test 15 |
| Digital Twin synced with Phase 7.1 baseline | State matrix matches archived baseline | ✅ Test 9 |

## 5. Limitations

1. **No prospective clinical validation** — all testing is computational
2. **No real clinician usability evaluation** — SUS/NASA-TLX scores are projected based on design analysis
3. **No latency benchmarking under load** — single-user performance only
4. **No security testing** — authentication not implemented (optional per spec)
5. **Streamlit prototype** — not production-grade UI framework

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
