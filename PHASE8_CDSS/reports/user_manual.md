# Phase 8 CDSS — User Manual

## 1. Getting Started

### 1.1 Installation
```bash
cd PHASE8_CDSS
pip install -r requirements.txt
```

### 1.2 Running the CDSS
```bash
streamlit run app/frontend/streamlit_app.py
```
The application will open in your browser at `http://localhost:8501`.

### 1.3 Running the Backend API (Optional)
```bash
cd app/backend
uvicorn main:app --reload --port 8000
```
API documentation available at `http://localhost:8000/docs`.

### 1.4 Running Tests
```bash
python tests/test_phase8_cdss.py
```
All 37 tests must pass.

## 2. Using the CDSS

### 2.1 Home Page
- Read the research disclaimer (red banner in sidebar — always visible)
- Review validation summary (C-index, calibration, reproducibility)
- Review Phase 7.1 constraints enforced by the system

### 2.2 Patient Input
- **Option 1:** Select an existing patient from the TCGA cohort (dropdown)
- **Option 2:** Enter new patient data (age, cancer type, stage, smoking status, optional gene expression)
- Click "Validate & Save Patient"
- If blocked (e.g., Stage IIIB/IV), the system will display the specific block reason

### 2.3 Prediction Page
- Click "Generate Prediction"
- View risk group (color-coded: green=Low, orange=Intermediate, red=High)
- View DeepSurv risk score and ensemble uncertainty
- View survival probability at 12/24/36 months (⚠️ UNCALIBRATED — inline tag)
- View interactive survival curve
- Read the risk interpretation (permitted statement)

### 2.4 Digital Twin Page
- Select a treatment scenario from the dropdown
- Click "Run Simulation"
- **⚠️ Read the artifact warning** (red banner — appears every time)
- View tumor growth trajectory (interactive Plotly chart)
- View TTP (time-to-progression) in days and months
- View counterfactual comparison (if existing patient)
- Read the clinical interpretation (permitted statement + warning)

### 2.5 Explainability Page
- Click "Generate Explanation"
- View top clinical feature contributions (horizontal bar chart)
- View top molecular feature contributions (horizontal bar chart)
- View mechanistic parameter contributions (table with literature ranges)
- View global cohort importance (Phase 3 permutation importance)

### 2.6 Evidence Panel
- Review model information (training cohort, external validation)
- Review known limitations (from Phase 7.1 — not re-summarized)
- Review permitted and forbidden statements (21 permitted, 22 forbidden)
- Read the full Phase 7.1 scientific position statement

### 2.7 Export
- **JSON Export:** Full prediction + simulation + explanation data
- **CSV Export:** Summary table with patient, prediction, and simulation fields
- **Simulation Report (TXT):** Includes artifact warning text verbatim

## 3. Understanding the Outputs

### 3.1 Risk Score
- DeepSurv log-hazard ratio (higher = worse prognosis)
- Validated for **ranking** (C-index 0.6333)
- **NOT** calibrated for absolute risk prediction

### 3.2 Survival Probability
- Computed via uncalibrated exponential model: S(t) = exp(-0.001 × exp(risk) × t)
- **⚠️ UNCALIBRATED** — displayed with inline warning tag
- Use for relative comparison only, not absolute risk

### 3.3 Treatment Simulation
- ODE-based tumor growth simulation under treatment
- **⚠️ ARTIFACT:** Shows benefit for ALL therapies (structural model limitation)
- No resistance, toxicity, or discontinuation modeled
- Differences between therapies are **NOT** clinically meaningful

### 3.4 Uncertainty
- Ensemble standard deviation (across 2 DeepSurv models)
- All patients classified as MODERATE uncertainty
- 95% confidence intervals from Monte Carlo sampling (Phase 7)

### 3.5 Feature Contributions
- Gradient × Input approximation of SHAP values
- Indicates direction and relative magnitude, not causal mechanisms
- Separated into clinical (demographics, stage, smoking) and molecular (gene expression)

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| "Please select or enter a patient first" | Go to Patient Input page and select/enter a patient |
| Prediction blocked for Stage IIIB/IV | This is intentional — Phase 7.1 found insufficient sample (n=7/n=10) |
| No gene expression data | Prediction will use clinical features only (warning issued) |
| Simulation failed | Check console for ODE solver error; try different treatment scenario |
| Tests failing | Ensure all Phase 7 artifacts exist and paths are correct |

## 5. Safety Features Summary

| Feature | Implementation |
|---------|---------------|
| Stage IIIB/IV block | Hard block in Safety Layer with phase7_1_constraint source |
| Calibration warning | Inline "UNCALIBRATED" tag on every survival probability |
| Treatment artifact warning | Red banner on every treatment simulation (non-dismissible) |
| No prediction without explanation | Feature contributions always returned |
| No prediction without uncertainty | Risk std and group always included |
| Research disclaimer | Sidebar (always visible) + every page + every API response |

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
