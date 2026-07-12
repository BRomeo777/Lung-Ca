"""
STEP 1 Diagnostic: Root-cause the Phase 3 vs Phase 5 DeepSurv C-index discrepancy.

Phase 3 reported DeepSurv C-index=0.6131 on TCGA validation.
Phase 5 reported DeepSurv C-index=0.3869 on the same set.
This script reproduces both data loading paths and compares them.
"""
from __future__ import annotations
import sys, json, pickle, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READY_DIR = PROJECT_ROOT / "03_ANALYSIS_READY_DATA"
ML_DIR = PROJECT_ROOT / "ML_RESULTS"
P3_DIR = PROJECT_ROOT / "PHASE3_DEEP_LEARNING"
P3_MODELS = P3_DIR / "models"

INVALID_PATIENTS = ["TCGA-05-4395", "TCGA-77-A5G6"]

# ── DeepSurv model (same architecture in both pipelines) ──
ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}

class DeepSurv(nn.Module):
    def __init__(self, n_features, hidden_dims=[64, 32], dropout=0.2, activation="gelu"):
        super().__init__()
        act_fn = ACTIVATIONS[activation]
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(in_dim))
            linear = nn.Linear(in_dim, h_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        out_layer = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

def normalize_counts_log2cpm(counts):
    lib = counts.sum(axis=0); lib[lib == 0] = 1
    cpm = counts.div(lib, axis=1) * 1e6
    return np.log2(cpm + 1)

def map_cancer_type(v):
    s = str(v).lower()
    if 'adc' in s or 'adenocarcinoma' in s: return 'LUAD_Adenocarcinoma'
    if 'sqc' in s or 'squamous' in s or 'scc' in s: return 'LUSC_SquamousCell'
    return None

def tnm_to_stage(tnm):
    import re
    s = str(tnm).upper().strip()
    if 'NTL' in s: return None
    t_m = re.search(r'T(\d|X)', s)
    n_m = re.search(r'N(\d|X)', s)
    m_m = re.search(r'M(\d|X)', s)
    t = int(t_m.group(1)) if t_m and t_m.group(1).isdigit() else None
    n = int(n_m.group(1)) if n_m and n_m.group(1).isdigit() else None
    m = int(m_m.group(1)) if m_m and m_m.group(1).isdigit() else None
    if m == 1: return 'IV'
    if n == 3: return 'IIIB'
    if n == 2: return 'IIIB' if t and t >= 3 else 'IIIA'
    if n == 1:
        if t == 1: return 'IIA'
        if t == 2: return 'IIB'
        if t and t >= 3: return 'III'
    if n == 0:
        if t == 1: return 'IA'
        if t == 2: return 'IB'
        if t == 3: return 'IIB'
        if t and t >= 4: return 'III'
    return None

# Phase 3 map_stage (with TNM fallback)
def map_stage_p3(v):
    s = str(v).strip()
    if s in ['I','IA','IB','II','IIA','IIB','III','IIIA','IIIB','IV']: return s
    if s.startswith('T') and ('N' in s or 'M' in s): return tnm_to_stage(s)
    return None

# Phase 5 map_stage (without TNM fallback)
def map_stage_p5(v):
    s = str(v).strip()
    if s in ["I","IA","IB","II","IIA","IIB","III","IIIA","IIIB","IV"]: return s
    return None

def map_smoking(v):
    s = str(v).lower()
    if "former" in s: return "Former"
    if "current" in s: return "Current"
    if "never" in s: return "Never"
    return None

def build_features(clin_df, expr_norm, gene_features, feature_names, age_median, scaler, map_stage_fn):
    """Build scaled feature matrix — parametrized by map_stage function."""
    expr_cols = [pid for pid in clin_df['Patient_ID'].astype(str) if pid in expr_norm.columns]
    clin_matched = clin_df[clin_df['Patient_ID'].astype(str).isin(expr_cols)].reset_index(drop=True)
    if len(clin_matched) == 0:
        return None, None
    available_ensg = [g for g in gene_features if g in expr_norm.index]
    expr_sel = expr_norm.loc[available_ensg][clin_matched['Patient_ID'].astype(str).tolist()].T
    expr_sel.index = clin_matched.index
    clin_enc = pd.DataFrame(index=clin_matched.index)
    clin_enc['Age'] = pd.to_numeric(clin_matched['Age'], errors='coerce').fillna(age_median)
    for col_name in ['Cancer_Type', 'Stage', 'Smoking_Status']:
        for fname in feature_names:
            if fname.startswith(f"{col_name}="):
                cat = fname.split("=", 1)[1]
                if col_name == 'Cancer_Type':
                    mapped = clin_matched[col_name].apply(map_cancer_type)
                    clin_enc[fname] = (mapped == cat).astype(int)
                elif col_name == 'Stage':
                    mapped = clin_matched[col_name].apply(map_stage_fn)
                    clin_enc[fname] = (mapped == cat).astype(int)
                elif col_name == 'Smoking_Status':
                    mapped = clin_matched[col_name].apply(map_smoking)
                    clin_enc[fname] = (mapped == cat).astype(int)
    X = pd.concat([clin_enc, expr_sel], axis=1)
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[feature_names]
    X_scaled = scaler.transform(X.values)
    return X_scaled, clin_matched

def checksum(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def cindex(y_event, y_time, risk):
    return concordance_index_censored(y_event.astype(bool), y_time, risk)[0]

def main():
    print("=" * 70)
    print("DIAGNOSTIC: Phase 3 vs Phase 5 DeepSurv Discrepancy")
    print("=" * 70)

    # ── Load Phase 3 config and scaler ──
    with open(P3_MODELS / "final_config.json", "r") as f:
        p3_config = json.load(f)
    with open(P3_MODELS / "final_scaler.pkl", "rb") as f:
        scaler_p5 = pickle.load(f)  # This is the scaler Phase 5 uses

    feature_names = p3_config["feature_names"]
    age_median = p3_config["age_median"]
    gene_features = [f for f in feature_names if "=" not in f and f != "Age"]
    print(f"Feature names: {len(feature_names)} total ({len(gene_features)} genes, {len(feature_names) - len(gene_features)} clinical)")

    # ── Also load CoxPH model to get Phase 3's feature_names source ──
    with open(ML_DIR / "models" / "cox_ph_model.pkl", "rb") as f:
        cox_model = pickle.load(f)
    cox_feature_names = list(cox_model.params_.index)
    print(f"CoxPH feature_names: {len(cox_feature_names)}")
    print(f"Config feature_names: {len(feature_names)}")
    print(f"Feature names match: {cox_feature_names == feature_names}")

    # ── Load TCGA validation data ──
    val_df = pd.read_csv(READY_DIR / "TCGA_internal_validation.csv")
    val_df = val_df[~val_df['Patient_ID'].isin(INVALID_PATIENTS)]
    val_df = val_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    expr_val = pd.read_csv(READY_DIR / "TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0)
    expr_val_norm = normalize_counts_log2cpm(expr_val)
    val_expr_cols = [pid for pid in val_df['Patient_ID'].astype(str) if pid in expr_val_norm.columns]
    val_df = val_df[val_df['Patient_ID'].astype(str).isin(val_expr_cols)].reset_index(drop=True)

    print(f"\nValidation patients: {len(val_df)}")
    print(f"Validation events: {int(pd.to_numeric(val_df['Survival_Status']).sum())}")

    # ── Build features with Phase 3 map_stage (TNM fallback) ──
    X_val_p3, clin_val_p3 = build_features(
        val_df, expr_val_norm, gene_features, feature_names, age_median, scaler_p5, map_stage_p3)

    # ── Build features with Phase 5 map_stage (no TNM fallback) ──
    X_val_p5, clin_val_p5 = build_features(
        val_df, expr_val_norm, gene_features, feature_names, age_median, scaler_p5, map_stage_p5)

    # ── Compare feature matrices ──
    print("\n" + "=" * 70)
    print("FEATURE MATRIX COMPARISON")
    print("=" * 70)
    print(f"Phase 3 map_stage shape: {X_val_p3.shape}")
    print(f"Phase 5 map_stage shape: {X_val_p5.shape}")
    print(f"Matrices identical: {np.array_equal(X_val_p3, X_val_p5)}")
    print(f"Checksum P3: {checksum(X_val_p3)}")
    print(f"Checksum P5: {checksum(X_val_p5)}")
    print(f"Max abs diff: {np.max(np.abs(X_val_p3 - X_val_p5))}")
    print(f"Num differing elements: {np.sum(X_val_p3 != X_val_p5)}")

    # ── Check stage mapping differences ──
    stage_p3 = val_df['Stage'].apply(map_stage_p3)
    stage_p5 = val_df['Stage'].apply(map_stage_p5)
    diffs = (stage_p3 != stage_p5).sum()
    print(f"\nStage mapping differences: {diffs}/{len(val_df)}")
    if diffs > 0:
        diff_mask = stage_p3 != stage_p5
        print("Differing patients:")
        for idx in val_df.index[diff_mask]:
            print(f"  {val_df.loc[idx, 'Patient_ID']}: Stage='{val_df.loc[idx, 'Stage']}' -> P3={stage_p3[idx]}, P5={stage_p5[idx]}")

    # ── Also check: does Phase 3 fit its own scaler vs using saved scaler? ──
    # Phase 3 run_final_deepsurv.py fits scaler on training data
    train_df = pd.read_csv(READY_DIR / "TCGA_train.csv")
    train_df = train_df[~train_df['Patient_ID'].isin(INVALID_PATIENTS)]
    train_df = train_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    expr_train = pd.read_csv(READY_DIR / "TCGA_train_expression.tsv.gz", sep="\t", index_col=0)
    expr_train_norm = normalize_counts_log2cpm(expr_train)
    train_expr_cols = [pid for pid in train_df['Patient_ID'].astype(str) if pid in expr_train_norm.columns]
    train_df = train_df[train_df['Patient_ID'].astype(str).isin(train_expr_cols)].reset_index(drop=True)

    # Build training features with Phase 3 map_stage and fit new scaler
    expr_cols_train = [pid for pid in train_df['Patient_ID'].astype(str) if pid in expr_train_norm.columns]
    clin_matched_train = train_df[train_df['Patient_ID'].astype(str).isin(expr_cols_train)].reset_index(drop=True)
    available_ensg = [g for g in gene_features if g in expr_train_norm.index]
    expr_sel_train = expr_train_norm.loc[available_ensg][clin_matched_train['Patient_ID'].astype(str).tolist()].T
    expr_sel_train.index = clin_matched_train.index
    clin_enc_train = pd.DataFrame(index=clin_matched_train.index)
    clin_enc_train['Age'] = pd.to_numeric(clin_matched_train['Age'], errors='coerce').fillna(age_median)
    for col_name in ['Cancer_Type', 'Stage', 'Smoking_Status']:
        for fname in feature_names:
            if fname.startswith(f"{col_name}="):
                cat = fname.split("=", 1)[1]
                if col_name == 'Cancer_Type':
                    mapped = clin_matched_train[col_name].apply(map_cancer_type)
                    clin_enc_train[fname] = (mapped == cat).astype(int)
                elif col_name == 'Stage':
                    mapped = clin_matched_train[col_name].apply(map_stage_p3)
                    clin_enc_train[fname] = (mapped == cat).astype(int)
                elif col_name == 'Smoking_Status':
                    mapped = clin_matched_train[col_name].apply(map_smoking)
                    clin_enc_train[fname] = (mapped == cat).astype(int)
    X_train_raw = pd.concat([clin_enc_train, expr_sel_train], axis=1)
    for feat in feature_names:
        if feat not in X_train_raw.columns:
            X_train_raw[feat] = 0.0
    X_train_raw = X_train_raw[feature_names]

    scaler_fresh = StandardScaler()
    X_train_scaled = scaler_fresh.fit_transform(X_train_raw.values)

    # Compare fresh scaler vs saved scaler
    print("\n" + "=" * 70)
    print("SCALER COMPARISON")
    print("=" * 70)
    print(f"Saved scaler mean (first 5): {scaler_p5.mean_[:5]}")
    print(f"Fresh scaler mean (first 5): {scaler_fresh.mean_[:5]}")
    print(f"Saved scaler scale (first 5): {scaler_p5.scale_[:5]}")
    print(f"Fresh scaler scale (first 5): {scaler_fresh.scale_[:5]}")
    print(f"Scaler means identical: {np.allclose(scaler_p5.mean_, scaler_fresh.mean_)}")
    print(f"Scaler scales identical: {np.allclose(scaler_p5.scale_, scaler_fresh.scale_)}")

    # Build val features with fresh scaler
    X_val_fresh, _ = build_features(
        val_df, expr_val_norm, gene_features, feature_names, age_median, scaler_fresh, map_stage_p3)
    print(f"\nVal features (saved scaler) checksum: {checksum(X_val_p3)}")
    print(f"Val features (fresh scaler) checksum: {checksum(X_val_fresh)}")
    print(f"Val features identical (saved vs fresh): {np.array_equal(X_val_p3, X_val_fresh)}")

    # ── Load DeepSurv models and predict ──
    print("\n" + "=" * 70)
    print("DEEPSURV PREDICTIONS")
    print("=" * 70)

    y_val_time = pd.to_numeric(clin_val_p3['Overall_Survival_Time'], errors='coerce').values
    y_val_event = pd.to_numeric(clin_val_p3['Survival_Status'], errors='coerce').fillna(0).astype(int).values

    deepsurv_models = []
    for i in range(p3_config["n_models"]):
        m = DeepSurv(len(feature_names),
                     hidden_dims=p3_config["arch"]["hidden_dims"],
                     dropout=p3_config["arch"]["dropout"],
                     activation=p3_config["arch"]["activation"])
        m.load_state_dict(torch.load(P3_MODELS / f"final_deepsurv_model_{i}.pt", map_location="cpu"))
        m.eval()
        deepsurv_models.append(m)

    def predict(X):
        preds = []
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            for m in deepsurv_models:
                preds.append(m(X_t).cpu().numpy())
        return np.mean(preds, axis=0)

    # C-index with Phase 3 features (TNM fallback + saved scaler)
    risk_p3 = predict(X_val_p3)
    ci_p3 = cindex(y_val_event, y_val_time, risk_p3)
    print(f"C-index (Phase 3 map_stage + saved scaler): {ci_p3:.4f}")

    # C-index with Phase 5 features (no TNM fallback + saved scaler)
    risk_p5 = predict(X_val_p5)
    ci_p5 = cindex(y_val_event, y_val_time, risk_p5)
    print(f"C-index (Phase 5 map_stage + saved scaler): {ci_p5:.4f}")

    # C-index with fresh scaler
    risk_fresh = predict(X_val_fresh)
    ci_fresh = cindex(y_val_event, y_val_time, risk_fresh)
    print(f"C-index (Phase 3 map_stage + fresh scaler): {ci_fresh:.4f}")

    # ── Also check: is BatchNorm1d causing issues in eval mode? ──
    print("\n" + "=" * 70)
    print("BATCHNORM CHECK")
    print("=" * 70)
    for i, m in enumerate(deepsurv_models):
        # Check if model has BatchNorm layers
        for name, module in m.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                print(f"Model {i}: BatchNorm1d found: {name}")
                print(f"  running_mean shape: {module.running_mean.shape}")
                print(f"  running_mean[:5]: {module.running_mean[:5].numpy()}")
                print(f"  running_var[:5]: {module.running_var[:5].numpy()}")
                print(f"  num_batches_tracked: {module.num_batches_tracked.item()}")
                break

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print(f"Phase 3 reported C-index: 0.6131")
    print(f"Phase 5 reported C-index: 0.3869")
    print(f"Diagnostic C-index (P3 map_stage + saved scaler): {ci_p3:.4f}")
    print(f"Diagnostic C-index (P5 map_stage + saved scaler): {ci_p5:.4f}")
    print(f"Diagnostic C-index (P3 map_stage + fresh scaler): {ci_fresh:.4f}")
    print()
    if abs(ci_p3 - 0.6131) < 0.01:
        print("ROOT CAUSE: Phase 5's map_stage is missing TNM fallback.")
        print("  Fix: Add tnm_to_stage() fallback to Phase 5's map_stage function.")
    elif abs(ci_p5 - 0.3869) < 0.01:
        print("ROOT CAUSE: Something beyond map_stage. Investigate further.")
    else:
        print("ROOT CAUSE: Neither map_stage nor scaler explains the discrepancy.")
        print("  The issue may be in model loading, label vectors, or data filtering.")
        print(f"  Diagnostic reproduces P3={ci_p3:.4f}, P5={ci_p5:.4f}")
        print(f"  Expected: P3=0.6131, P5=0.3869")

if __name__ == "__main__":
    main()
