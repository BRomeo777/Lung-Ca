"""
Phase 3 DeepSurv — Diagnostic Audit (STEP 1)

Evaluates every ensemble member individually on all cohorts,
checks risk direction consistency, and compares ensemble strategies.

No new training. Only loads existing models and evaluates.
"""
from __future__ import annotations
import os, sys, pickle, json, random, warnings, copy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

# ── PATHS ──
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
READY_DIR = PROJECT_ROOT / "03_ANALYSIS_READY_DATA"
ML_DIR = PROJECT_ROOT / "ML_RESULTS"
MODELS_DIR = ML_DIR / "models"
PHASE1C_DIR = PROJECT_ROOT / "PHASE1C_GEO_HARMONIZATION"
PHASE1D_DIR = PROJECT_ROOT / "PHASE1D_GEO_SCALE_ALIGNMENT"
PHASE3_DIR = PROJECT_ROOT / "PHASE3_DEEP_LEARNING"
P3_MODELS = PHASE3_DIR / "models"
P3_REPORTS = PHASE3_DIR / "reports"
P3_TABLES = PHASE3_DIR / "tables"

INVALID_PATIENTS = ["TCGA-05-4395", "TCGA-77-A5G6"]
GEO_COHORTS = ["GSE30219", "GSE50081", "GSE72094", "GSE31210"]
GEO_EXPR = {
    "GSE30219": PHASE1C_DIR/"processed_data"/"GSE30219"/"expression_gene_mapped.csv",
    "GSE50081": PHASE1C_DIR/"processed_data"/"GSE50081"/"expression_gene_mapped.csv",
    "GSE72094": PHASE1C_DIR/"processed_data"/"GSE72094"/"expression_gene_mapped.csv",
    "GSE31210": PHASE1D_DIR/"processed_data"/"GSE31210"/"GSE31210_expression_gene_mapped_log2.csv",
}
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]

def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(P3_REPORTS/"diagnostic_audit_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── UTILITIES (copied from main script) ──
def normalize_counts_log2cpm(counts):
    lib = counts.sum(axis=0); lib[lib == 0] = 1
    cpm = counts.div(lib, axis=1) * 1e6
    return np.log2(cpm + 1)

def make_surv(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))

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

def map_stage(v):
    s = str(v).strip()
    if s in ['I','IA','IB','II','IIA','IIB','III','IIIA','IIIB','IV']: return s
    if s.startswith('T') and ('N' in s or 'M' in s): return tnm_to_stage(s)
    return None

def map_smoking(v):
    s = str(v).lower()
    if 'former' in s: return 'Former'
    if 'current' in s: return 'Current'
    if 'never' in s: return 'Never'
    return None

def is_normal_sample(row):
    ct = str(row.get('Cancer_Type','')).lower()
    st = str(row.get('Stage','')).lower()
    return 'ntl' in ct or 'normal' in ct or 'ntl' in st

def build_features(clin_df, expr_norm, gene_features, feature_names, ensg2sym,
                   age_median, scaler=None, fit_scaler=False):
    expr_cols = [pid for pid in clin_df['Patient_ID'].astype(str) if pid in expr_norm.columns]
    clin_matched = clin_df[clin_df['Patient_ID'].astype(str).isin(expr_cols)].reset_index(drop=True)
    if len(clin_matched) == 0:
        return None, None, scaler
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
                    mapped = clin_matched[col_name].apply(map_stage)
                    clin_enc[fname] = (mapped == cat).astype(int)
                elif col_name == 'Smoking_Status':
                    mapped = clin_matched[col_name].apply(map_smoking)
                    clin_enc[fname] = (mapped == cat).astype(int)
    X = pd.concat([clin_enc, expr_sel], axis=1)
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[feature_names]
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = scaler.transform(X.values)
    return X_scaled, clin_matched, scaler

# ── DEEPSURV MODEL (same architecture) ──
ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
}

class DeepSurv(nn.Module):
    def __init__(self, n_features, hidden_dims=[64, 32], dropout=0.3, activation='relu'):
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


def predict_single(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        return model(X_t).cpu().numpy()


def cindex(y_event, y_time, risk):
    return concordance_index_censored(y_event.astype(bool), y_time, risk)[0]


def main():
    log_file = P3_REPORTS / "diagnostic_audit_log.txt"
    if log_file.exists():
        log_file.unlink()

    log("=" * 70)
    log("DIAGNOSTIC AUDIT — Phase 3 DeepSurv Ensemble")
    log("=" * 70)

    device = 'cpu'

    # ── LOAD DATA ──
    log("Loading CoxPH model and data...")

    with open(MODELS_DIR / "cox_ph_model.pkl", "rb") as f:
        cox_model = pickle.load(f)
    feature_names = list(cox_model.params_.index)
    gene_features = [f for f in feature_names if '=' not in f and f != 'Age']

    ensg_map = pd.read_csv(PHASE1C_DIR / "reports" / "phase1B_ensg_to_symbol_mapping.csv")
    ensg2sym = dict(zip(ensg_map['Ensembl_ID'], ensg_map['Gene_Symbol']))

    # TCGA train
    train_df = pd.read_csv(READY_DIR / "TCGA_train.csv")
    train_df = train_df[~train_df['Patient_ID'].isin(INVALID_PATIENTS)]
    train_df = train_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    expr_train = pd.read_csv(READY_DIR / "TCGA_train_expression.tsv.gz", sep="\t", index_col=0)
    expr_train_norm = normalize_counts_log2cpm(expr_train)
    train_expr_cols = [pid for pid in train_df['Patient_ID'].astype(str) if pid in expr_train_norm.columns]
    train_df = train_df[train_df['Patient_ID'].astype(str).isin(train_expr_cols)].reset_index(drop=True)

    # TCGA val
    val_df = pd.read_csv(READY_DIR / "TCGA_internal_validation.csv")
    val_df = val_df[~val_df['Patient_ID'].isin(INVALID_PATIENTS)]
    val_df = val_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    expr_val = pd.read_csv(READY_DIR / "TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0)
    expr_val_norm = normalize_counts_log2cpm(expr_val)
    val_expr_cols = [pid for pid in val_df['Patient_ID'].astype(str) if pid in expr_val_norm.columns]
    val_df = val_df[val_df['Patient_ID'].astype(str).isin(val_expr_cols)].reset_index(drop=True)

    age_median = pd.to_numeric(train_df['Age'], errors='coerce').median()

    # Build features
    X_train, train_matched, scaler = build_features(
        train_df, expr_train_norm, gene_features, feature_names, ensg2sym,
        age_median, fit_scaler=True)
    X_val, val_matched, _ = build_features(
        val_df, expr_val_norm, gene_features, feature_names, ensg2sym,
        age_median, scaler=scaler)

    y_train_time = pd.to_numeric(train_matched['Overall_Survival_Time'], errors='coerce').values
    y_train_event = pd.to_numeric(train_matched['Survival_Status'], errors='coerce').fillna(0).astype(int).values
    y_val_time = pd.to_numeric(val_matched['Overall_Survival_Time'], errors='coerce').values
    y_val_event = pd.to_numeric(val_matched['Survival_Status'], errors='coerce').fillna(0).astype(int).values

    log(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    log(f"  Train events: {y_train_event.sum()}/{len(y_train_event)}")
    log(f"  Val events: {y_val_event.sum()}/{len(y_val_event)}")

    # ── LOAD ENSEMBLE MODELS ──
    log("Loading 5 ensemble models...")
    n_features = X_train.shape[1]
    # Architecture from best HP: [64, 32], GELU, dropout=0.4
    ensemble_models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        model = DeepSurv(n_features, hidden_dims=[64, 32], dropout=0.4, activation='gelu')
        state = torch.load(P3_MODELS / f"deepsurv_ensemble_member_{i}.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        ensemble_models.append(model)
        log(f"  Loaded member {i} (seed={seed})")

    # ── BUILD GEO COHORT DATA ──
    log("Building GEO cohort features...")
    geo_data = {}
    for cohort in GEO_COHORTS:
        geo_clin = pd.read_csv(READY_DIR / f"{cohort}_external_validation.csv")
        geo_expr = pd.read_csv(GEO_EXPR[cohort], index_col=0)
        normal_mask = geo_clin.apply(is_normal_sample, axis=1)
        geo_clin = geo_clin[~normal_mask].reset_index(drop=True)
        geo_clin = geo_clin.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
        X_geo, geo_matched, _ = build_features(
            geo_clin, geo_expr, gene_features, feature_names, ensg2sym,
            age_median, scaler=scaler)
        if X_geo is None or len(geo_matched) == 0:
            log(f"  {cohort}: No matched patients, skipping")
            continue
        y_geo_time = pd.to_numeric(geo_matched['Overall_Survival_Time'], errors='coerce').values
        y_geo_event = pd.to_numeric(geo_matched['Survival_Status'], errors='coerce').fillna(0).astype(int).values
        geo_data[cohort] = {
            'X': X_geo,
            'y_time': y_geo_time,
            'y_event': y_geo_event,
            'n': len(geo_matched),
        }
        log(f"  {cohort}: {len(geo_matched)} patients, {y_geo_event.sum()} events")

    # ═══════════════════════════════════════════════════════════
    # STEP 1A: EVALUATE EACH ENSEMBLE MEMBER INDIVIDUALLY
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 1A: Individual ensemble member evaluation")
    log("=" * 70)

    # Get predictions from each model on each cohort
    all_cohorts = [("TCGA_val", X_val, y_val_time, y_val_event)]
    for cohort in GEO_COHORTS:
        if cohort in geo_data:
            all_cohorts.append((cohort, geo_data[cohort]['X'],
                               geo_data[cohort]['y_time'], geo_data[cohort]['y_event']))

    member_preds = {}  # {model_idx: {cohort_name: risk_scores}}
    member_cis = {}    # {model_idx: {cohort_name: c_index}}

    header = f"{'Model':>12} | " + " | ".join(f"{name:>12}" for name, _, _, _ in all_cohorts)
    log(header)
    log("-" * len(header))

    for i, model in enumerate(ensemble_models):
        member_preds[i] = {}
        member_cis[i] = {}
        row_parts = [f"Member_{i}({ENSEMBLE_SEEDS[i]:>4})"]
        for name, X, yt, ye in all_cohorts:
            risk = predict_single(model, X, device)
            ci = cindex(ye, yt, risk)
            member_preds[i][name] = risk
            member_cis[i][name] = ci
            row_parts.append(f"{ci:>12.4f}")
        log(" | ".join(row_parts))

    # Also get train predictions for correlation analysis
    train_preds = {}
    for i, model in enumerate(ensemble_models):
        train_preds[i] = predict_single(model, X_train, device)

    # ═══════════════════════════════════════════════════════════
    # STEP 1B: RISK DIRECTION ANALYSIS
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 1B: Risk direction correlation analysis")
    log("=" * 70)

    # Use member 0 as reference (or best internal val model)
    best_member = max(member_cis.keys(), key=lambda k: member_cis[k]["TCGA_val"])
    log(f"  Best internal val model: Member_{best_member} (seed={ENSEMBLE_SEEDS[best_member]}), C-index={member_cis[best_member]['TCGA_val']:.4f}")

    ref_train = train_preds[best_member]
    ref_val = member_preds[best_member]["TCGA_val"]

    log("")
    log("  Correlation with best model (on TCGA train):")
    for i in range(len(ensemble_models)):
        corr_train = scipy.stats.spearmanr(ref_train, train_preds[i])[0]
        corr_val = scipy.stats.spearmanr(ref_val, member_preds[i]["TCGA_val"])[0]
        flip = "FLIP!" if corr_train < 0 else "ok"
        log(f"    Member_{i} (seed={ENSEMBLE_SEEDS[i]:>4}): rho_train={corr_train:+.4f}, rho_val={corr_val:+.4f}  [{flip}]")

    # Check correlation on each GEO cohort
    log("")
    log("  Correlation with best model (on each GEO cohort):")
    for cohort in GEO_COHORTS:
        if cohort not in geo_data:
            continue
        ref_geo = member_preds[best_member][cohort]
        log(f"    {cohort}:")
        for i in range(len(ensemble_models)):
            corr = scipy.stats.spearmanr(ref_geo, member_preds[i][cohort])[0]
            flip = "FLIP!" if corr < 0 else "ok"
            log(f"      Member_{i}: rho={corr:+.4f}  [{flip}]")

    # ═══════════════════════════════════════════════════════════
    # STEP 1C: COMPARE ENSEMBLE STRATEGIES
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 1C: Compare ensemble strategies on all cohorts")
    log("=" * 70)

    strategies = {}

    # Strategy 1: Single best model
    strategies["single_best"] = {name: member_preds[best_member][name] for name, _, _, _ in all_cohorts}

    # Strategy 2: Z-score ensemble (current v2 approach)
    def zscore_ensemble(X_input, models_list):
        preds = []
        for m in models_list:
            raw = predict_single(m, X_input, device)
            raw_std = (raw - raw.mean()) / (raw.std() + 1e-8)
            preds.append(raw_std)
        return np.mean(preds, axis=0)

    strategies["zscore_ensemble"] = {}
    for name, X, yt, ye in all_cohorts:
        strategies["zscore_ensemble"][name] = zscore_ensemble(X, ensemble_models)

    # Strategy 3: Rank-based ensemble
    def rank_ensemble(X_input, models_list):
        preds = []
        for m in models_list:
            raw = predict_single(m, X_input, device)
            ranks = scipy.stats.rankdata(raw)
            preds.append(ranks)
        return np.mean(preds, axis=0)

    strategies["rank_ensemble"] = {}
    for name, X, yt, ye in all_cohorts:
        strategies["rank_ensemble"][name] = rank_ensemble(X, ensemble_models)

    # Strategy 4: Sign-consistent z-score ensemble
    # Flip models with negative correlation to best model, then z-score average
    consistent_models = []
    flipped_models = []
    ref_train_risk = train_preds[best_member]
    for i, model in enumerate(ensemble_models):
        corr = scipy.stats.spearmanr(ref_train_risk, train_preds[i])[0]
        if corr < 0:
            log(f"  Flipping Member_{i} (rho={corr:+.4f})")
            flipped_models.append(i)
            # Create a flipped version by negating output
            consistent_models.append(("flip", model))
        else:
            consistent_models.append(("keep", model))

    def sign_consistent_zscore(X_input, model_specs):
        preds = []
        for sign, m in model_specs:
            raw = predict_single(m, X_input, device)
            if sign == "flip":
                raw = -raw
            raw_std = (raw - raw.mean()) / (raw.std() + 1e-8)
            preds.append(raw_std)
        return np.mean(preds, axis=0)

    strategies["sign_consistent_zscore"] = {}
    for name, X, yt, ye in all_cohorts:
        strategies["sign_consistent_zscore"][name] = sign_consistent_zscore(X, consistent_models)

    # Strategy 5: Sign-consistent rank ensemble
    def sign_consistent_rank(X_input, model_specs):
        preds = []
        for sign, m in model_specs:
            raw = predict_single(m, X_input, device)
            if sign == "flip":
                raw = -raw
            ranks = scipy.stats.rankdata(raw)
            preds.append(ranks)
        return np.mean(preds, axis=0)

    strategies["sign_consistent_rank"] = {}
    for name, X, yt, ye in all_cohorts:
        strategies["sign_consistent_rank"][name] = sign_consistent_rank(X, consistent_models)

    # Strategy 6: Median ensemble (raw)
    def median_ensemble(X_input, models_list):
        preds = []
        for m in models_list:
            raw = predict_single(m, X_input, device)
            preds.append(raw)
        return np.median(preds, axis=0)

    strategies["median_raw"] = {}
    for name, X, yt, ye in all_cohorts:
        strategies["median_raw"][name] = median_ensemble(X, ensemble_models)

    # Strategy 7: Mean raw ensemble (v1 style — no normalization)
    def mean_raw_ensemble(X_input, models_list):
        preds = []
        for m in models_list:
            raw = predict_single(m, X_input, device)
            preds.append(raw)
        return np.mean(preds, axis=0)

    strategies["mean_raw"] = {}
    for name, X, yt, ye in all_cohorts:
        strategies["mean_raw"][name] = mean_raw_ensemble(X, ensemble_models)

    # Evaluate all strategies
    log("")
    header = f"{'Strategy':>25} | " + " | ".join(f"{name:>12}" for name, _, _, _ in all_cohorts) + f" | {'Mean_Ext':>10}"
    log(header)
    log("-" * len(header))

    results_table = []
    for strat_name, preds_dict in strategies.items():
        row_parts = [f"{strat_name:>25}"]
        strat_cis = {}
        ext_cis = []
        for name, X, yt, ye in all_cohorts:
            ci = cindex(ye, yt, preds_dict[name])
            strat_cis[name] = ci
            row_parts.append(f"{ci:>12.4f}")
            if name != "TCGA_val":
                ext_cis.append(ci)
        mean_ext = np.mean(ext_cis) if ext_cis else 0
        row_parts.append(f"{mean_ext:>10.4f}")
        log(" | ".join(row_parts))
        results_table.append({
            "strategy": strat_name,
            **strat_cis,
            "mean_external": mean_ext,
        })

    # Also evaluate CoxPH for reference
    cox_cis = {}
    cox_parts = [f"{'CoxPH':>25}"]
    cox_ext = []
    for name, X, yt, ye in all_cohorts:
        if name == "TCGA_val":
            cox_risk = cox_model.predict_partial_hazard(pd.DataFrame(X, columns=feature_names)).values.ravel()
        else:
            cox_risk = cox_model.predict_partial_hazard(pd.DataFrame(X, columns=feature_names)).values.ravel()
        ci = cindex(ye, yt, cox_risk)
        cox_cis[name] = ci
        cox_parts.append(f"{ci:>12.4f}")
        if name != "TCGA_val":
            cox_ext.append(ci)
    cox_parts.append(f"{np.mean(cox_ext):>10.4f}")
    log(" | ".join(cox_parts))

    # ═══════════════════════════════════════════════════════════
    # SUMMARY AND RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("DIAGNOSTIC SUMMARY")
    log("=" * 70)

    # Best strategy by external mean
    best_strat = max(results_table, key=lambda x: x["mean_external"])
    log(f"  Best ensemble strategy by mean external C-index: {best_strat['strategy']} ({best_strat['mean_external']:.4f})")

    # Check if any individual model is better than any ensemble
    best_individual_ext = max(
        (np.mean([member_cis[i][c] for c in GEO_COHORTS if c in geo_data]) for i in range(len(ensemble_models))),
    )
    best_individual_idx = max(range(len(ensemble_models)),
                              key=lambda i: np.mean([member_cis[i][c] for c in GEO_COHORTS if c in geo_data]))
    log(f"  Best individual model by mean external C-index: Member_{best_individual_idx} ({best_individual_ext:.4f})")

    # GSE31210 specific analysis
    log("")
    log("  GSE31210 detailed analysis:")
    for i in range(len(ensemble_models)):
        risk_gse = member_preds[i].get("GSE31210")
        if risk_gse is not None:
            ci = member_cis[i]["GSE31210"]
            # Check if risk direction is inverted (negative correlation with survival time)
            y_t = geo_data["GSE31210"]["y_time"]
            y_e = geo_data["GSE31210"]["y_event"]
            corr_time = scipy.stats.spearmanr(risk_gse[y_e == 1], y_t[y_e == 1])[0] if y_e.sum() > 2 else np.nan
            log(f"    Member_{i}: C-index={ci:.4f}, corr(risk, time|event)={corr_time:+.4f} {'(INVERTED!)' if corr_time > 0 else ''}")

    # Save results
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(P3_TABLES / "diagnostic_audit_ensemble_strategies.csv", index=False)
    log(f"\n  Results saved to: {P3_TABLES / 'diagnostic_audit_ensemble_strategies.csv'}")

    # Save individual member results
    member_rows = []
    for i in range(len(ensemble_models)):
        row = {"member": i, "seed": ENSEMBLE_SEEDS[i]}
        for name, _, _, _ in all_cohorts:
            row[f"{name}_cindex"] = member_cis[i][name]
        member_rows.append(row)
    member_df = pd.DataFrame(member_rows)
    member_df.to_csv(P3_TABLES / "diagnostic_audit_individual_members.csv", index=False)
    log(f"  Individual member results saved to: {P3_TABLES / 'diagnostic_audit_individual_members.csv'}")

    log("")
    log("=" * 70)
    log("DIAGNOSTIC AUDIT COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
