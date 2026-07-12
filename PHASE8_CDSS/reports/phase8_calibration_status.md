# Phase 8 — Calibration Deployment Status (Step 0)

## Question
Is Platt-scaled output what the served model actually returns, or was "accepted" in the Phase 7.1 audit only an evaluation without being wired into production?

## Answer

**Platt scaling was NOT wired into the served model.** The Phase 7.1 calibration audit evaluated Platt scaling favorably on validation data, but the recalibrated outputs were never integrated into the Phase 7 Digital Twin pipeline. The served model returns **uncalibrated** survival probabilities.

## Evidence

### 1. Served Model Code (`run_phase7_digital_twin.py`, lines 395-404)
```python
def risk_to_survival_curve(risk_score, t_eval, baseline_lambda=0.001):
    """Convert DeepSurv risk score to a survival probability curve S(t).
    Uses exponential model: S(t) = exp(-lambda * t)
    where lambda = baseline_lambda * exp(risk_score).
    """
    lam = baseline_lambda * np.exp(risk_score)
    S = np.exp(-lam * t_eval)
    return S
```

This function uses a **fixed baseline_lambda=0.001** with no Platt scaling, isotonic regression, or temperature scaling applied. It is called directly in the Phase 7 pipeline at line 748:
```python
S = risk_to_survival_curve(twin.deepsurv_risk, traj["t"])
```

### 2. Phase 7.1 Audit Code (`run_phase7_1_steps1_5.py`, lines 164-171)
The Platt scaling model was fit on validation data **within the audit script only**:
```python
pl = LogisticRegression(C=1e10, solver="lbfgs")
pl.fit(vr.reshape(-1, 1), ve.astype(int))
```
This `pl` object was used to compute calibration metrics for the audit report but was **never saved, exported, or integrated** into the prediction pipeline. No Platt scaler model file exists in the model directory.

### 3. Phase 7.1 Audit Report (`calibration_audit_report.md`)
The report states:
- "Accepted: Platt" — meaning the method was evaluated and met acceptance criteria
- "Do NOT use absolute survival probability estimates without recalibration"
- "Use DeepSurv risk scores for ranking (discrimination)"

The word "accepted" means "evaluated favorably as a candidate method," not "deployed to production."

### 4. No Saved Platt Model
There is no `platt_scaler.pkl`, `platt_model.json`, or any similar artifact in:
- `PHASE3_DEEP_LEARNING/models/`
- `PHASE7_DIGITAL_TWIN/`
- `PHASE7_DIGITAL_TWIN/reports/phase7_1/`

## Conclusion

**The served model is UNCALIBRATED.** Survival probabilities displayed in the CDSS must carry a visible "uncalibrated — interpret with caution" tag directly next to the number. The calibration warning must state that:
1. The calibration slope is 0.6189 (target: 0.8-1.2)
2. Platt scaling was evaluated but not deployed
3. Absolute survival probabilities should not be used for clinical decisions
4. Risk scores are validated for ranking only (C-index 0.6333)

## Phase 8 Implementation Decision
- Display survival probability with "UNCALIBRATED" tag inline (not in a separate panel)
- Add calibration warning text in every view that shows a survival probability
- Do not attempt to apply Platt scaling at serving time (would require fitting on validation data, which is methodologically incorrect for a production system)
- Recommend prospective recalibration on independent data as a future improvement

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
