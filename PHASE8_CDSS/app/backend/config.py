"""
Phase 8 CDSS — Configuration and Constants
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
from pathlib import Path

# ── PATHS ──
CODE_DIR = Path(__file__).resolve().parent
P8_DIR = CODE_DIR.parent.parent  # PHASE8_CDSS/

# Deployment-aware path resolution:
# - If artifacts/ directory exists (Render deployment), use it
# - Otherwise, fall back to sibling project directories (local development)
ARTIFACTS_DIR = P8_DIR / "artifacts"

if ARTIFACTS_DIR.exists():
    # Deployment mode — self-contained
    P3_MODELS = ARTIFACTS_DIR / "models"
    P4_DATA = ARTIFACTS_DIR / "data"
    P7_DIR = ARTIFACTS_DIR
    P7_DATA = ARTIFACTS_DIR / "data"
    P7_REPORTS = ARTIFACTS_DIR / "phase7_1"
    P7_1_DIR = ARTIFACTS_DIR / "phase7_1"
else:
    # Development mode — sibling directories
    PROJECT_ROOT = CODE_DIR.parent.parent.parent
    P3_MODELS = PROJECT_ROOT / "PHASE3_DEEP_LEARNING" / "models"
    P4_DATA = PROJECT_ROOT / "PHASE4_MECHANISTIC_MODELING" / "data"
    P7_DIR = PROJECT_ROOT / "PHASE7_DIGITAL_TWIN"
    P7_DATA = P7_DIR / "data"
    P7_REPORTS = P7_DIR / "reports"
    P7_1_DIR = P7_REPORTS / "phase7_1"

P8_REPORTS = P8_DIR / "reports"

# Phase 3.1 recalibration artifact (single-covariate Cox recalibration).
# Resolves in both deployment (artifacts/models) and dev (PHASE3.../models) modes.
CALIBRATOR_PATH = P3_MODELS / "calibrator.json"

# ── CONSTANTS ──
SEED = 42
SIM_HORIZON_DAYS = 1095
SIM_N_POINTS = 100
MIN_N = 15

CLINICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. "
    "This system generates computational simulations for hypothesis generation. "
    "It does not determine the best treatment for individual patients."
)

TREATMENT_ARTIFACT_WARNING = (
    "This simulation currently shows a predicted benefit for all therapies due to a known "
    "modeling limitation — the underlying model does not yet represent treatment resistance, "
    "toxicity, or discontinuation. Differences between therapies shown here should not be "
    "interpreted as clinically meaningful."
)

CALIBRATION_WARNING = (
    "UNCALIBRATED — interpret with caution. "
    "Survival curves now use a Phase 3.1 single-covariate Cox recalibration fit on the internal "
    "validation cohort (an improvement over the prior ad-hoc exponential). Calibration remains "
    "imperfect: internal slope 0.40, external slopes 0.61-0.99 (ideal = 1.0; best GSE30219 = 0.99). "
    "Absolute survival probabilities remain approximate and should not drive clinical decisions. "
    "Risk scores are validated for ranking only (mean external C-index 0.659)."
)

STAGE_BLOCK_MESSAGE = (
    "Prediction unavailable for Stage IIIB/IV patients: "
    "Phase 7.1 validation found insufficient sample size (n=7 / n=10) "
    "to support reliable estimates for this subgroup."
)

# ── PHASE 7.1 CONSTRAINT CONTRACT ──
BLOCKED_STAGES = ["IIIB", "IV"]
SUPPORTED_STAGES = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA"]
SUPPORTED_CANCER_TYPES = ["LUAD_Adenocarcinoma", "LUSC_SquamousCell"]
SUPPORTED_SMOKING = ["Current", "Former", "Never"]

# ── TREATMENTS ──
TREATMENTS = ["none", "chemo", "immuno", "targeted"]
TREATMENT_LABELS = {
    "none": "Natural History (no treatment)",
    "chemo": "Chemotherapy (Cisplatin)",
    "immuno": "Immunotherapy (Pembrolizumab)",
    "targeted": "Targeted Therapy (Osimertinib)",
}

# ── MODEL VERSION ──
MODEL_VERSION = "Phase7_v1.0"
PHASE8_VERSION = "Phase8_v1.0"
