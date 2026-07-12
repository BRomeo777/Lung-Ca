"""
dt_common.py
Shared configuration, paths, and harmonization helpers for the
Lung Cancer Digital Twin - Phase 1 data engineering pipeline.

This module contains NO side effects on import beyond path definitions.
All cohorts are kept SEPARATE by design (no cross-cohort patient merging).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# 00_CODE/dt_common.py  ->  PROJECT_ROOT = Lung_Cancer_Digital_Twin_Project/
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
# Existing verified raw download lives next to this project.
RAW_SRC = PROJECT_ROOT.parent / "Lung_Cancer_AI_Datasets"

RAW_DIR = PROJECT_ROOT / "01_RAW_DATA"
PROC_DIR = PROJECT_ROOT / "02_PROCESSED_DATA"
READY_DIR = PROJECT_ROOT / "03_ANALYSIS_READY_DATA"
DOC_DIR = PROJECT_ROOT / "04_DOCUMENTATION"

PROC_TCGA = PROC_DIR / "TCGA_DISCOVERY"
PROC_CPTAC = PROC_DIR / "CPTAC_CLINICAL_VALIDATION"
PROC_GEO = PROC_DIR / "GEO_EXTERNAL_VALIDATION"
PROC_REPORTS = PROC_DIR / "REPORTS"

GEO_SERIES = ["GSE31210", "GSE30219", "GSE50081", "GSE68465", "GSE72094"]

ALL_DIRS = [
    RAW_DIR, RAW_DIR / "TCGA_LUAD", RAW_DIR / "TCGA_LUSC", RAW_DIR / "GEO",
    RAW_DIR / "CPTAC_LUNG",
    *[RAW_DIR / "GEO" / g for g in GEO_SERIES],
    PROC_DIR, PROC_TCGA, PROC_CPTAC, PROC_GEO, PROC_REPORTS,
    *[PROC_GEO / g for g in GEO_SERIES],
    READY_DIR, DOC_DIR, CODE_DIR,
]

# --------------------------------------------------------------------------- #
# Canonical ML-ready schema (one row = one patient)
# --------------------------------------------------------------------------- #
# Provenance schema (never blank; use 'not_applicable' where a field
# genuinely does not apply to a cohort).
PROVENANCE_COLS = [
    "Patient_ID", "Dataset_Source", "Cohort_Label", "Analysis_Role",
    "Platform", "GPL_ID", "Modality_Available", "Sample_Type",
    "Selection_Method",
]
CLINICAL_COLS = [
    "Age", "Sex", "Cancer_Type", "Stage", "Smoking_Status", "Treatment",
    "Overall_Survival_Time", "Survival_Status",
]
FINAL_COLS = PROVENANCE_COLS + CLINICAL_COLS

NA = "not_applicable"
NAV = "not_available"

# --------------------------------------------------------------------------- #
# GEO per-series configuration (derived by inspecting the series matrices)
# --------------------------------------------------------------------------- #
GEO_CONFIG = {
    "GSE31210": {
        "gpl": "GPL570",
        "platform": "Affymetrix Human Genome U133 Plus 2.0 Array",
        "technology": "in situ oligonucleotide microarray",
        "probes": 54675,
        "age": ["age (years)", "age"],
        "sex": ["gender"],
        "smoking": ["smoking status"],
        "stage": ["pathological stage"],
        "histology": ["tissue"],
        "normal_flag": ("tissue", "normal"),
        "os_event": ["death"],
        "os_time": ["days before death/censor"],
        "os_time_unit": "days",
    },
    "GSE30219": {
        "gpl": "GPL570",
        "platform": "Affymetrix Human Genome U133 Plus 2.0 Array",
        "technology": "in situ oligonucleotide microarray",
        "probes": 54675,
        "age": ["age at surgery"],
        "sex": ["gender"],
        "smoking": [],
        "stage": ["pt stage", "pn stage", "pm stage"],
        "histology": ["histology"],
        "normal_flag": ("histology", "non tumoral"),
        "os_event": ["status"],
        "os_time": ["follow-up time (months)"],
        "os_time_unit": "months",
    },
    "GSE50081": {
        "gpl": "GPL570",
        "platform": "Affymetrix Human Genome U133 Plus 2.0 Array",
        "technology": "in situ oligonucleotide microarray",
        "probes": 54675,
        "age": ["age"],
        "sex": ["sex"],
        "smoking": ["smoking"],
        "stage": ["stage"],
        "histology": ["histology"],
        "normal_flag": None,
        "os_event": ["status"],
        "os_time": ["survival time"],
        "os_time_unit": "years",
    },
    "GSE68465": {
        "gpl": "GPL96",
        "platform": "Affymetrix Human Genome U133A Array",
        "technology": "in situ oligonucleotide microarray",
        "probes": 22283,
        "age": ["age"],
        "sex": ["sex"],
        "smoking": ["smoking_history"],
        "stage": ["disease_stage"],
        "histology": ["disease_state"],
        "normal_flag": ("disease_state", "normal"),
        "os_event": ["vital_status"],
        "os_time": ["months_to_last_contact_or_death"],
        "os_time_unit": "months",
    },
    "GSE72094": {
        "gpl": "GPL15048",
        "platform": "Rosetta/Merck Human RSTA Custom Affymetrix 2.0 microarray",
        "technology": "in situ oligonucleotide microarray",
        "probes": 60607,
        "age": ["age_at_diagnosis"],
        "sex": ["gender"],
        "smoking": ["smoking_status"],
        "stage": ["stage"],
        "histology": [],
        "normal_flag": None,
        "os_event": ["vital_status"],
        "os_time": ["survival_time_in_days"],
        "os_time_unit": "days",
        "patient_id": ["patient_id"],
    },
}

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_PATH = DOC_DIR / "preprocessing_log.txt"


def log(msg: str, echo: bool = True):
    """Append a timestamped line to preprocessing_log.txt."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if echo:
        print(line)


def reset_log():
    if _LOG_PATH.exists():
        _LOG_PATH.unlink()
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Value harmonizers
# --------------------------------------------------------------------------- #
def std_sex(v) -> str:
    if v is None:
        return NAV
    s = str(v).strip().lower()
    if s in ("", "nan", "na", "unknown", "--", "missing", "not reported"):
        return NAV
    if s.startswith("m"):
        return "Male"
    if s.startswith("f"):
        return "Female"
    return NAV


def std_smoking(v) -> str:
    if v is None:
        return NAV
    s = str(v).strip().lower()
    if s in ("", "nan", "na", "unknown", "--", "missing", "not reported"):
        return NAV
    if "never" in s or "non-smoker" in s or s == "no":
        return "Never"
    if "former" in s or "ex-" in s or "ever" in s or "reformed" in s:
        return "Former"
    if "current" in s or "smoker" in s or s == "yes":
        return "Current"
    return str(v).strip()


def std_vital_to_event(v):
    """Return 1 (death event), 0 (alive/censored), or None (unknown)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "nan", "na", "unknown", "--", "missing", "not reported"):
        return None
    if s in ("1", "1.0"):
        return 1
    if s in ("0", "0.0"):
        return 0
    if any(t in s for t in ("dead", "decease", "death", "expired")):
        return 1
    if any(t in s for t in ("alive", "living", "censor")):
        return 0
    return None


_ROMAN = {"i": "I", "ii": "II", "iii": "III", "iv": "IV"}


def std_stage(v) -> str:
    """Normalise a free-text stage value to I/II/III/IV (+ optional A/B) or raw."""
    if v is None:
        return NAV
    s = str(v).strip()
    if s.lower() in ("", "nan", "na", "unknown", "--", "missing", "not reported"):
        return NAV
    low = s.lower().replace("stage", "").strip()
    # Combined TNM like 'pN1pT2' -> keep raw upper
    m = re.match(r"^(iv|iii|ii|i)\s*([ab]?)", low)
    if m:
        roman = _ROMAN[m.group(1)]
        sub = m.group(2).upper()
        return roman + sub
    m2 = re.match(r"^([1-4])\s*([ab]?)$", low)
    if m2:
        roman = {"1": "I", "2": "II", "3": "III", "4": "IV"}[m2.group(1)]
        return roman + m2.group(2).upper()
    return s.upper()


def os_time_to_days(value, unit: str):
    """Convert a survival-time value to days given its source unit."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("", "nan", "na", "unknown", "--", "missing", "not reported"):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if x < 0:
        return None
    factor = {"days": 1.0, "months": 30.4375, "years": 365.25}.get(unit, 1.0)
    return round(x * factor, 2)


# --------------------------------------------------------------------------- #
# TCGA barcode helpers
# --------------------------------------------------------------------------- #
def tcga_patient_from_barcode(barcode: str) -> str:
    """TCGA-44-2655-01A-... -> TCGA-44-2655"""
    parts = str(barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode)


def tcga_portion_number(barcode: str):
    """Return the integer portion identifier from a full aliquot barcode.

    Barcode layout: TCGA-TSS-Participant-Sample+Vial-Portion+Analyte-Plate-Center
    e.g. TCGA-44-2655-01A-11R-1758-07 -> portion = 11
    Returns a large sentinel if it cannot be parsed (sorts last).
    """
    parts = str(barcode).split("-")
    if len(parts) >= 5:
        m = re.match(r"(\d+)", parts[4])
        if m:
            return int(m.group(1))
    return 9999


def parse_geo_characteristics(meta_df):
    """Turn a GEO sample_metadata frame into a list of per-sample key->value dicts.

    Each 'Sample_characteristics_ch1*' cell is 'key: value'.
    Returns dict keyed by GSM accession.
    """
    char_cols = [c for c in meta_df.columns if "characteristics" in c.lower()]
    out = {}
    acc_col = "geo_accession" if "geo_accession" in meta_df.columns else meta_df.columns[0]
    for _, row in meta_df.iterrows():
        acc = str(row[acc_col])
        kv = {}
        for c in char_cols:
            val = row[c]
            if val is None:
                continue
            sval = str(val)
            if ":" in sval:
                k, v = sval.split(":", 1)
                kv[k.strip().lower()] = v.strip()
        out[acc] = kv
    return out


def first_key(kv: dict, keys: list):
    """Return the first present value among candidate keys (case-insensitive)."""
    for k in keys:
        if k in kv and str(kv[k]).strip() not in ("", "--", "NA", "nan"):
            return kv[k]
    return None
