# Digital Twin State Schema

## Variable Definitions

| Variable | Domain | Description | Source | Unit | Valid Range | Missing-Data Rule |
|----------|--------|-------------|--------|------|-------------|-------------------|
| patient_id | Identity | Unique patient identifier | TCGA GDC | string | TCGA-XX-XXXX | Required, no imputation |
| age | Clinical | Patient age at diagnosis | TCGA clinical | years | 30-90 | Impute with cohort median |
| sex | Clinical | Biological sex | TCGA clinical | category | male/female/not_available | 'unknown' |
| cancer_type | Clinical | Histological subtype | TCGA clinical | category | LUAD_Adenocarcinoma/LUSC_SquamousCell | 'unknown' |
| stage | Clinical | Disease stage at diagnosis | TCGA clinical | category | I/IA/IB/II/IIA/IIB/III/IIIA/IIIB/IV/not_available | 'not_available' |
| smoking_status | Clinical | Smoking history | TCGA clinical | category | Current/Former/Never/not_available | 'unknown' |
| os_time | Clinical | Overall survival time | TCGA clinical | days | 0-5000 | Required, drop if missing |
| os_event | Clinical | Death event (1=yes, 0=censored) | TCGA clinical | binary | 0 or 1 | Required, drop if missing |
| alpha | Biological | Gompertz growth rate | Phase 4 ODE | day^-1 | 1e-5 to 1e-2 | Population prior 0.0008 |
| V_max | Biological | Carrying capacity (max tumor volume) | Phase 4 ODE | mm^3 | 1e5 to 1e7 | Population prior 1e6 |
| V0 | Biological | Initial tumor volume | Phase 4 ODE | mm^3 | 100 to 500000 | Stage-based lookup |
| k_immune | Biological | Fractional immune kill rate | Phase 4 ODE | day^-1 | 1e-6 to 1e-3 | Population prior 1.22e-4 |
| mu_E | Biological | Immune effector death rate | Phase 4 ODE | day^-1 | 0.01 to 0.1 | Population prior 0.041 |
| rho | Biological | Immune recruitment rate | Phase 4 ODE | day^-1 | 0.005 to 0.05 | Population prior 0.02 |
| delta_c | Biological | Cytotoxic kill rate (chemo) | Phase 4 ODE | (mg/L)^-1 day^-1 | 0.01 to 0.05 | Population prior 0.028 |
| delta_t | Biological | Targeted therapy kill rate | Phase 4 ODE | (mg/L)^-1 day^-1 | 0.05 to 0.3 | Population prior 0.15 |
| E0 | Biological | Initial immune effector count | Phase 4 ODE | cells | 1e4 to 1e6 | Population prior 1.3e5 |
| deepsurv_risk | AI-derived | DeepSurv ensemble mean risk score | Phase 3 model | unitless (log hazard) | -5 to +5 | 0.0 if expression missing |
| deepsurv_risk_std | AI-derived | DeepSurv ensemble std (uncertainty) | Phase 3 model | unitless | 0 to 2 | 0.0 if single model |
| federated_risk | AI-derived | Phase 6 federated model risk score | Phase 5/6 hybrid | unitless | -5 to +5 | NaN if not available |
| federated_risk_std | AI-derived | Federated model uncertainty | Phase 5/6 | unitless | 0 to 2 | NaN if not available |
| risk_group | AI-derived | Binary risk classification | Derived | category | High/Low | Based on sign of deepsurv_risk |

## Domain Summary

- **A. Clinical state (8 variables):** age, sex, cancer_type, stage, smoking_status, os_time, os_event, patient_id
- **B. Biological tumor state (9 variables):** alpha, V_max, V0, k_immune, mu_E, rho, delta_c, delta_t, E0
- **C. AI-derived state (5 variables):** deepsurv_risk, deepsurv_risk_std, federated_risk, federated_risk_std, risk_group
- **D. Treatment state:** Populated during simulation (current_treatment, simulated_ttp, simulated_trajectories)

## Missing-Data Handling Rules

1. Age: impute with cohort median (age_median from Phase 3 config)
2. Stage: 'not_available' category retained; V0 imputed with stage-based lookup (IIB default)
3. Gene expression: set to 0.0 if Ensembl ID not found (DeepSurv handles via scaler)
4. ODE parameters: population priors for validation patients (no personalization)
5. Federated risk: NaN if patient not in Phase 5 hybrid states
6. Survival outcomes: required — patients with missing OS time or event are dropped

## Validation

- State matrix populated for 919 / 919 patients (100.0%)
- All 22 variables have defined valid ranges
- No undefined valid ranges