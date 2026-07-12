# Phase 8 CDSS — Clinical Workflow

## 1. Workflow Overview

```
Clinician opens CDSS
        ↓
[Home Page] — Reads disclaimer, validation summary, Phase 7.1 constraints
        ↓
[Patient Input] — Selects existing patient OR enters new patient data
        ↓
    ┌───────────────────────────────────┐
    │ Safety Layer Validation           │
    │ • Stage IIIB/IV? → BLOCKED        │
    │ • Missing fields? → BLOCKED       │
    │ • Unsupported type? → BLOCKED     │
    │ • All checks pass? → PROCEED      │
    └───────────────────────────────────┘
        ↓
[Prediction Page]
    • Risk group (Low/Intermediate/High)
    • DeepSurv risk score + ensemble uncertainty
    • Survival probability (⚠️ UNCALIBRATED — inline tag)
    • Survival curve (Plotly interactive)
    • Confidence interval + uncertainty level
        ↓
[Digital Twin Page]
    • Select treatment scenario (none/chemo/immuno/targeted)
    • ⚠️ ARTIFACT WARNING displayed (non-dismissible, red banner)
    • Tumor growth trajectory (Plotly)
    • TTP (days/months)
    • Counterfactual comparison (if existing patient)
    • Clinical interpretation (permitted statement + warning)
        ↓
[Explainability Page]
    • Top clinical feature contributions (bar chart)
    • Top molecular feature contributions (bar chart)
    • Mechanistic parameter contributions (table)
    • Global cohort importance (Phase 3 permutation importance)
        ↓
[Evidence Panel]
    • Model version, training cohort, validation cohorts
    • Known limitations (from Phase 7.1 deliverables)
    • Permitted and forbidden statements (21/22)
    • Phase 7.1 scientific position statement (full text)
        ↓
[Export]
    • JSON report (full prediction + simulation + explanation)
    • CSV summary (patient + prediction + simulation)
    • Simulation report TXT (with artifact warning verbatim)
```

## 2. Decision Points

### 2.1 Safety Block
- **When:** Patient stage is IIIB or IV
- **Action:** System refuses prediction, displays:
  > "Prediction unavailable for Stage IIIB/IV patients: Phase 7.1 validation found insufficient sample size (n=7 / n=10) to support reliable estimates for this subgroup."
- **Source:** `phase7_1_constraint` (not generic OOD)

### 2.2 Calibration Warning
- **When:** Any survival probability is displayed
- **Action:** Inline "UNCALIBRATED — interpret with caution" tag displayed directly next to the number
- **Content:** Calibration slope 0.6189, Platt scaling evaluated but not deployed, use for ranking only

### 2.3 Treatment Artifact Warning
- **When:** Any treatment simulation output is displayed
- **Action:** Red banner with full warning text, non-dismissible
- **Content:** Model does not represent resistance, toxicity, or discontinuation; differences between therapies not clinically meaningful

## 3. Clinician Responsibilities

1. **Review the disclaimer** on the Home page before proceeding
2. **Verify patient eligibility** — the system blocks Stage IIIB/IV automatically
3. **Interpret survival probabilities with caution** — uncalibrated, use for ranking only
4. **Read the artifact warning** before interpreting treatment comparisons
5. **Review feature contributions** to understand what drives the prediction
6. **Use exports** for research documentation, not for clinical decision-making
7. **Exercise independent clinical judgment** — the CDSS is a research tool

## 4. What the CDSS Does NOT Do

- Does not recommend treatments
- Does not predict individual patient survival accurately (uncalibrated)
- Does not account for treatment resistance, toxicity, or discontinuation
- Does not support Stage IIIB/IV patients
- Does not replace oncologist judgment
- Does not claim regulatory readiness

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
