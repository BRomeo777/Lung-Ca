# Digital Twin Interface — README

## Framework
**Streamlit** (Python web framework for data science applications)

## Fidelity Level
**Live backend** — the application directly calls the Digital Twin simulation engine:
- DeepSurv model (Phase 3) for AI risk prediction
- Gompertz ODE solver (Phase 4) for mechanistic tumor simulation
- Treatment-response ODE for chemotherapy, immunotherapy, and targeted therapy
- Real-time Plotly visualizations of tumor trajectories and survival curves

## What is In Scope
- Patient input: clinical variables (age, cancer type, stage, smoking status)
- Tumor biology parameters: growth rate, initial volume, carrying capacity
- Simulation controls: treatment selection, simulation horizon
- Output display: Digital Twin state, disease trajectory, treatment comparison,
  survival prediction with CI, uncertainty, explainability
- Patient selection from existing TCGA cohort (919 patients)

## What is Out of Scope
- Real-time patient monitoring (requires longitudinal imaging/biomarker updates)
- Clinical decision support (requires prospective trials and regulatory approval)
- Determination of "the best treatment" for an individual patient
- Regulatory-approved medical device functionality
- Real-time data integration with hospital EHR systems

## How to Run

```bash
# Install dependencies (if not already installed)
pip install streamlit plotly

# Run the app
streamlit run PHASE7_DIGITAL_TWIN/app/digital_twin_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Mandatory Banner
The application displays a non-removable banner on every page:
**"⚠️ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE."**

This banner appears at the top and bottom of the interface and cannot be
removed by the user.

## Input Modes

### 1. Select Existing Patient
Choose from 919 TCGA patients with pre-computed Digital Twin states.
Displays their clinical variables, AI risk scores, and ODE parameters.

### 2. Manual Input
Enter clinical variables and tumor biology parameters manually.
The DeepSurv model predicts risk in real-time.

## Outputs

### Tumor Volume Trajectory
Plotly line chart showing tumor volume (mm³) over time (months) for each
selected treatment scenario.

### Survival Probability
Plotly line chart showing S(t) for each treatment scenario, derived from
the DeepSurv risk score via an exponential model.

### TTP Comparison Table
Time to progression (days and months) for each treatment scenario, with
benefit compared to no treatment.

### Uncertainty
- DeepSurv ensemble std for AI risk uncertainty
- 95% CI for 1, 2, and 3-year survival probabilities

### Explainability
- Global permutation importance bar chart (top 10 features from Phase 3)
- Mechanistic driver summary (growth rate, initial volume, immune activity)

## Dependencies
- streamlit
- plotly
- torch
- numpy
- pandas
- scipy
- scikit-learn (for scaler)

## Data Sources
- Phase 3 DeepSurv model (`PHASE3_DEEP_LEARNING/models/`)
- Phase 4 population priors (`PHASE4_MECHANISTIC_MODELING/data/`)
- Phase 7 twin summaries (`PHASE7_DIGITAL_TWIN/data/twin_summaries_for_interface.csv`)
