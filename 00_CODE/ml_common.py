"""Shared helpers for Phase 1B baseline survival ML."""
from __future__ import annotations
import os, json, pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
READY_DIR = PROJECT_ROOT / "03_ANALYSIS_READY_DATA"
ML_DIR = PROJECT_ROOT / "ML_RESULTS"
MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"
FIGURES_DIR = ML_DIR / "figures"

for d in [ML_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

PROVENANCE_COLS = [
    "Patient_ID", "Dataset_Source", "Cohort_Label", "Analysis_Role",
    "Platform", "GPL_ID", "Modality_Available", "Sample_Type", "Selection_Method",
]
OUTCOME_COLS = ["Overall_Survival_Time", "Survival_Status"]

def log(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(REPORTS_DIR / "ml_pipeline_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def reset_log():
    p = REPORTS_DIR / "ml_pipeline_log.txt"
    if p.exists(): p.unlink()

def save_model(obj, name: str):
    path = MODELS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path

def load_expr(path: Path) -> pd.DataFrame:
    """Load gene x patient expression matrix."""
    return pd.read_csv(path, sep="\t", index_col=0)

def normalize_counts_log2cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """CPM normalize then log2(x+1) transform. Fit per-sample (no train info needed)."""
    lib = counts.sum(axis=0)
    lib[lib == 0] = 1
    cpm = counts.div(lib, axis=1) * 1e6
    return np.log2(cpm + 1)

def make_surv_array(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))
