# Phase 8 CDSS — Architecture Document

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │ Home │ │ Patient  │ │Predict-  │ │ Digital  │ │Explain│ │
│  │      │ │ Input    │ │ion       │ │ Twin     │ │ability│ │
│  └──────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
│  ┌──────────────┐  ┌─────────┐                              │
│  │ Evidence     │  │ Export  │                              │
│  │ Panel        │  │         │                              │
│  └──────────────┘  └─────────┘                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  FastAPI    │
                    │  Backend    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────┴──────┐ ┌──────┴──────┐ ┌───────┴───────┐
   │ Safety      │ │ Prediction  │ │ Digital Twin  │
   │ Layer       │ │ Engine      │ │ Engine        │
   │             │ │             │ │               │
   │ • Stage     │ │ • DeepSurv  │ │ • Gompertz    │
   │   IIIB/IV   │ │ • Ensemble  │ │   ODE         │
   │   block     │ │ • Risk→S(t) │ │ • Treatment   │
   │ • OOD       │ │ • Features  │ │   response    │
   │ • Validation│ │ • Uncert.   │ │ • PK models   │
   └──────┬──────┘ └──────┬──────┘ └───────┬───────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────┴──────┐ ┌──────┴──────┐ ┌───────┴───────┐
   │ Explain-    │ │ Interpre-   │ │ Phase 7.1     │
   │ ability     │ │ tation      │ │ Deliverables  │
   │ Engine      │ │ Layer       │ │ (read-only)   │
   │             │ │             │ │               │
   │ • SHAP approx│ │ 21 permitted│ │ • Position    │
   │ • Mech.     │ │ 22 forbidden│ │   statement   │
   │   params    │ │ • Risk interp│ │ • Calibration │
   │ • Global    │ │ • Tx interp │ │   audit       │
   │   importance│ │             │ │ • Subgroup    │
   └─────────────┘ └─────────────┘ │   robustness  │
                                    └───────────────┘
```

## 2. Module Descriptions

### 2.1 Safety Layer (`safety.py`)
- **Purpose:** Enforce Phase 7.1 constraint contract as hard blocks
- **Key Features:**
  - Stage IIIB/IV hard block with `phase7_1_constraint` source
  - Missing variable detection
  - Impossible biological value detection
  - Unsupported subtype/stage detection
  - Generic OOD detection for unusual combinations
- **Input:** Patient demographics, cancer type, stage, smoking status, gene expression
- **Output:** `SafetyResult(allowed, blocked, warnings, errors, block_reason, block_source)`

### 2.2 Prediction Engine (`prediction.py`)
- **Purpose:** Load Phase 3 DeepSurv model and produce risk predictions
- **Key Features:**
  - Top-2 ensemble (seeds 123, 456), 143 features, [64, 32] architecture
  - Existing patient lookup from Phase 7 state matrix
  - New patient prediction with feature vector construction
  - Risk-to-survival curve (uncalibrated exponential model)
  - Risk group classification (Low/Intermediate/High)
  - Feature contributions via Gradient × Input
- **No retraining** — uses pre-validated Phase 3 model weights

### 2.3 Digital Twin Engine (`digital_twin.py`)
- **Purpose:** Run ODE tumor growth simulations for treatment scenarios
- **Key Features:**
  - Gompertz growth ODE (natural history)
  - Treatment response ODE with Skipper log-cell kill (chemo)
  - Immune effector + checkpoint blockade (immuno)
  - Targeted therapy kill (targeted)
  - One-compartment PK models for all drugs
  - Radau stiff solver for treatment ODE
  - NaN/Inf guards and parameter sanitization
  - Every output carries `TREATMENT_ARTIFACT_WARNING`

### 2.4 Explainability Engine (`explainability.py`)
- **Purpose:** Provide feature-level and mechanistic-level explainability
- **Key Features:**
  - Global cohort importance from Phase 3 permutation importance
  - Per-patient feature contributions (Gradient × Input approximation of SHAP)
  - Mechanistic parameter contributions with literature ranges
  - Clinical and molecular feature separation

### 2.5 Clinical Interpretation Layer (`interpretation.py`)
- **Purpose:** Ensure all outputs use permitted language and forbid dangerous interpretations
- **Key Features:**
  - 21 permitted statements (5 from Phase 7.1 + 16 new)
  - 22 forbidden statements (5 from Phase 7.1 + 17 new)
  - Treatment interpretation generator with mandatory artifact warning
  - Risk interpretation generator with calibration caveat

### 2.6 Frontend (`streamlit_app.py`)
- **Purpose:** Clinician-facing interactive dashboard
- **Pages:** Home, Patient Input, Prediction, Digital Twin, Explainability, Evidence Panel, Export
- **Key Features:**
  - Persistent disclaimer in sidebar (always visible)
  - Inline "UNCALIBRATED" tag on survival probabilities
  - Persistent artifact warning on treatment simulations (non-dismissible)
  - Plotly interactive plots (survival curves, tumor trajectories, feature bars)
  - JSON, CSV, and TXT export with artifact warning included verbatim

## 3. Data Flow

```
Patient Input
    ↓
Safety Layer (Phase 7.1 constraint check)
    ↓ [blocked? → refuse with specific reason]
Prediction Engine (DeepSurv risk score)
    ↓
[Existing patient? → load from state matrix]
[New patient? → construct features → ensemble prediction]
    ↓
Survival curve (uncalibrated exponential model)
    ↓
Digital Twin Engine (ODE simulation)
    ↓ [treatment != none? → attach artifact warning]
Explainability Engine (feature + mechanistic contributions)
    ↓
Clinical Interpretation Layer (permitted/forbidden statements)
    ↓
Frontend display (with all warnings, disclaimers, uncertainty)
```

## 4. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Streamlit | ≥1.28.0 |
| Backend API | FastAPI | ≥0.104.0 |
| Plotting | Plotly | ≥5.18.0 |
| ML Framework | PyTorch | ≥2.0.0 |
| Data | NumPy, Pandas | ≥1.24, ≥2.0 |
| ODE Solver | SciPy | ≥1.11.0 |
| Survival Analysis | lifelines, scikit-survival | ≥0.27, ≥0.22 |
| Deployment | Docker | — |

## 5. Independent Testability

Every module is independently testable:
- `safety.py` — `SafetyLayer.check()` returns structured `SafetyResult`
- `prediction.py` — `PredictionEngine.predict_new()` / `predict_existing()`
- `digital_twin.py` — `DigitalTwinEngine.simulate_treatment()`
- `explainability.py` — `ExplainabilityEngine.get_patient_explanation()`
- `interpretation.py` — `get_all_statements()`, `get_treatment_interpretation()`

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
