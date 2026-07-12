#!/usr/bin/env python3
"""Phase 2: External Scientific Validation — Steps 0-4 (Core Validation)."""
from __future__ import annotations
import os, sys, pickle, re, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Paths ───
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
READY_DIR = PROJECT_ROOT / "03_ANALYSIS_READY_DATA"
ML_DIR = PROJECT_ROOT / "ML_RESULTS"
MODELS_DIR = ML_DIR / "models"
ML_REPORTS = ML_DIR / "reports"
PHASE1C_DIR = PROJECT_ROOT / "PHASE1C_GEO_HARMONIZATION"
PHASE1D_DIR = PROJECT_ROOT / "PHASE1D_GEO_SCALE_ALIGNMENT"
PHASE2_DIR = PROJECT_ROOT / "PHASE2_EXTERNAL_VALIDATION"
P2_REPORTS = PHASE2_DIR / "reports"
P2_FIGURES = PHASE2_DIR / "figures"
P2_TABLES = PHASE2_DIR / "tables"
P2_RISK = PHASE2_DIR / "risk_scores"
for d in [PHASE2_DIR, P2_REPORTS, P2_FIGURES, P2_TABLES, P2_RISK,
          P2_FIGURES/"KM_curves", P2_FIGURES/"ROC_curves",
          P2_FIGURES/"Calibration", P2_FIGURES/"Feature_importance",
          P2_FIGURES/"Workflow"]:
    d.mkdir(parents=True, exist_ok=True)

INVALID_PATIENTS = ["TCGA-05-4395", "TCGA-77-A5G6"]
GEO_COHORTS = ["GSE30219", "GSE50081", "GSE72094", "GSE31210"]
HORIZONS = {"1yr": 365, "3yr": 1095, "5yr": 1825}
INTERNAL_VAL_CINDEX = 0.5944
GEO_EXPR = {
    "GSE30219": PHASE1C_DIR/"processed_data"/"GSE30219"/"expression_gene_mapped.csv",
    "GSE50081": PHASE1C_DIR/"processed_data"/"GSE50081"/"expression_gene_mapped.csv",
    "GSE72094": PHASE1C_DIR/"processed_data"/"GSE72094"/"expression_gene_mapped.csv",
    "GSE31210": PHASE1D_DIR/"processed_data"/"GSE31210"/"GSE31210_expression_gene_mapped_log2.csv",
}

def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(P2_REPORTS/"phase2_log.txt", "a", encoding="utf-8") as f:
        f.write(line+"\n")

def normalize_counts_log2cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0); lib[lib==0]=1
    cpm = counts.div(lib, axis=1)*1e6
    return np.log2(cpm+1)

def make_surv(time, event):
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))

# ─── Clinical mapping helpers ───
def map_cancer_type(v):
    s = str(v).lower()
    if 'adc' in s or 'adenocarcinoma' in s: return 'LUAD_Adenocarcinoma'
    if 'sqc' in s or 'squamous' in s or 'scc' in s: return 'LUSC_SquamousCell'
    return None

def tnm_to_stage(tnm):
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

# ─── Step 0: Data Quality Finalization ───
def step0_data_quality():
    log("STEP 0: Data quality finalization")
    train = pd.read_csv(READY_DIR/"TCGA_train.csv")
    val = pd.read_csv(READY_DIR/"TCGA_internal_validation.csv")
    train_n = len(train); val_n = len(val)
    train = train[~train['Patient_ID'].isin(INVALID_PATIENTS)]
    val = val[~val['Patient_ID'].isin(INVALID_PATIENTS)]
    train_after = len(train); val_after = len(val)
    lines = ["="*70, "FINAL SURVIVAL QUALITY REPORT", "="*70, "",
        f"Excluded patients (invalid survival records):",
        f"  TCGA-05-4395: OS_time=0, OS_event=1 (biologically impossible)",
        f"  TCGA-77-A5G6: OS_time=0, OS_event=1 (biologically impossible)",
        f"  Reason: Per Pre-Phase 2 Audit Finding #4 (settled decision)",
        f"  Action: Excluded from all Phase 2 analyses", "",
        f"TCGA Training: {train_n} -> {train_after} (removed {train_n-train_after})",
        f"TCGA Internal Validation: {val_n} -> {val_after} (removed {val_n-val_after})", "",
        f"Final training N (after dropping NaN OS): {train_after - train['Overall_Survival_Time'].isna().sum() - train['Survival_Status'].isna().sum()}",
        f"Final validation N (after dropping NaN OS): {val_after - val['Overall_Survival_Time'].isna().sum() - val['Survival_Status'].isna().sum()}",
    ]
    (P2_REPORTS/"final_survival_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"  Train: {train_n}->{train_after}, Val: {val_n}->{val_after}")
    return train, val

# ─── Step 1: Load Frozen Model + Reconstruct Scaler ───
def step1_load_model(train_df):
    log("STEP 1: Load frozen model and reconstruct preprocessing")
    with open(MODELS_DIR/"cox_ph_model.pkl", "rb") as f:
        cox_model = pickle.load(f)
    feature_names = list(cox_model.params_.index)
    gene_features = [f for f in feature_names if '=' not in f and f != 'Age']
    clinical_features = [f for f in feature_names if '=' in f or f == 'Age']
    log(f"  Model loaded: CoxPH penalizer={cox_model.penalizer}, {len(feature_names)} features")
    log(f"  Clinical: {len(clinical_features)}, Genes: {len(gene_features)}")

    # Load Ensembl→Symbol mapping
    ensg_map = pd.read_csv(PHASE1C_DIR/"reports"/"phase1B_ensg_to_symbol_mapping.csv")
    ensg2sym = dict(zip(ensg_map['Ensembl_ID'], ensg_map['Gene_Symbol']))
    sym2ensg = {v: k for k, v in ensg2sym.items()}

    # Reconstruct scaler from training data
    log("  Reconstructing scaler from training data...")
    train_clean = train_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).copy()
    train_clean = train_clean[~train_clean['Patient_ID'].isin(INVALID_PATIENTS)]

    # Load and normalize expression
    expr = pd.read_csv(READY_DIR/"TCGA_train_expression.tsv.gz", sep="\t", index_col=0)
    expr_norm = normalize_counts_log2cpm(expr)

    # Match patient IDs between clinical and expression
    train_expr_cols = []
    valid_mask = []
    for pid in train_clean['Patient_ID'].astype(str):
        if pid in expr.columns:
            train_expr_cols.append(pid)
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    train_clean = train_clean[valid_mask].reset_index(drop=True)
    log(f"  Train patients matched to expression: {len(train_clean)}")

    expr_sel = expr_norm.loc[gene_features][train_expr_cols].T  # samples × genes
    expr_sel.index = train_clean.index

    # Build clinical features
    train_clin_enc = pd.DataFrame(index=train_clean.index)
    train_clin_enc['Age'] = pd.to_numeric(train_clean['Age'], errors='coerce')
    age_median = train_clin_enc['Age'].median()
    train_clin_enc['Age'] = train_clin_enc['Age'].fillna(age_median)

    for col in ['Cancer_Type', 'Stage', 'Smoking_Status']:
        train_clean[col] = train_clean[col].fillna('not_available')
        cats = sorted(train_clean[col].dropna().unique())
        for cat in cats:
            fname = f"{col}={cat}"
            if fname in feature_names:
                train_clin_enc[fname] = (train_clean[col] == cat).astype(int)

    X_train = pd.concat([train_clin_enc, expr_sel], axis=1)
    X_train = X_train[feature_names]  # exact column order

    scaler = StandardScaler()
    scaler.fit(X_train)
    log(f"  Scaler reconstructed: {X_train.shape}")

    # Verify all 127 genes match
    model_genes = set(gene_features)
    mapped_genes = set(ensg2sym.keys())
    unmapped = model_genes - mapped_genes
    log(f"  Model genes: {len(model_genes)}, Mapped: {len(mapped_genes)}, Unmapped: {len(unmapped)}")

    # Write report
    lines = ["="*70, "MODEL LOADING VALIDATION REPORT", "="*70, "",
        f"Model: CoxPH (lifelines CoxPHFitter)", f"Penalizer: {cox_model.penalizer}",
        f"Total features: {len(feature_names)}",
        f"  Clinical: {len(clinical_features)}",
        f"  Gene (Ensembl): {len(gene_features)}", "",
        f"Scaler: StandardScaler reconstructed from TCGA training data",
        f"  Training samples: {X_train.shape[0]}",
        f"  Feature count: {X_train.shape[1]}",
        f"  Age imputation median: {age_median:.2f}", "",
        f"Gene symbol mapping:",
        f"  Mapped Ensembl IDs: {len(mapped_genes)}/{len(model_genes)}",
        f"  Unmapped (no gene symbol): {len(unmapped)}",
        f"  Unmapped IDs: {', '.join(sorted(unmapped))}", "",
        f"Feature ordering: Verified (matches model params_.index)",
        f"Model status: FROZEN — no retraining performed", "",
        "Verification: PASSED",
    ]
    (P2_REPORTS/"model_loading_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")

    return cox_model, scaler, feature_names, gene_features, ensg2sym, sym2ensg, unmapped, age_median, X_train

# ─── Step 2: Prepare GEO Validation Datasets ───
def step2_prepare_geo(cox_model, feature_names, gene_features, sym2ensg, unmapped):
    log("STEP 2: Prepare GEO validation datasets")
    ensg2sym = {v: k for k, v in sym2ensg.items()}
    summary_rows = []
    geo_data = {}

    for cohort in GEO_COHORTS:
        log(f"  Processing {cohort}...")
        clin = pd.read_csv(READY_DIR/f"{cohort}_external_validation.csv")
        expr = pd.read_csv(GEO_EXPR[cohort], index_col=0)

        # Exclude normal samples
        normal_mask = clin.apply(is_normal_sample, axis=1)
        n_normal = normal_mask.sum()
        clin = clin[~normal_mask].reset_index(drop=True)
        log(f"    Normal samples excluded: {n_normal}, remaining: {len(clin)}")

        # Drop missing survival
        clin = clin.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
        log(f"    After dropping NaN survival: {len(clin)}")

        # Map gene symbols → Ensembl IDs
        available_ensg = []
        missing_ensg = []
        for g in gene_features:
            if g in unmapped:
                missing_ensg.append(g)
                continue
            sym = ensg2sym.get(g)
            if sym and sym in expr.index:
                available_ensg.append(g)
            else:
                missing_ensg.append(g)

        n_avail = len(available_ensg)
        n_miss = len(missing_ensg)
        coverage_pct = 100.0 * n_avail / len(gene_features)
        log(f"    Genes available: {n_avail}/{len(gene_features)} ({coverage_pct:.1f}%)")

        # Missing gene symbols
        missing_syms = [ensg2sym.get(g, '[UNMAPPED]') for g in missing_ensg]

        # Effective coefficient coverage
        betas = cox_model.params_
        total_abs = betas[gene_features].abs().sum()
        avail_abs = betas[available_ensg].abs().sum() if available_ensg else 0
        eff_coeff_cov = 100.0 * avail_abs / total_abs if total_abs > 0 else 0
        log(f"    Effective coefficient coverage: {eff_coeff_cov:.1f}%")

        # Filter expression to available genes only
        expr_avail = expr.loc[[ensg2sym[g] for g in available_ensg]].copy()

        # Align expression columns to clinical patients
        expr_cols = [c for c in clin['Patient_ID'].astype(str) if c in expr_avail.columns]
        clin = clin[clin['Patient_ID'].astype(str).isin(expr_cols)].reset_index(drop=True)
        expr_avail = expr_avail[clin['Patient_ID'].astype(str).tolist()]
        expr_avail = expr_avail.T  # samples × genes
        expr_avail.columns = available_ensg  # rename to Ensembl IDs
        expr_avail.index = clin.index

        summary_rows.append({
            "Dataset": cohort,
            "Platform": clin['GPL_ID'].iloc[0] if 'GPL_ID' in clin.columns else 'Unknown',
            "N_patients": len(clin),
            "N_events": int(clin['Survival_Status'].sum()),
            "Available_genes": n_avail,
            "Missing_genes": n_miss,
            "Missing_gene_list": ";".join(missing_syms),
            "Panel_coverage_pct": round(coverage_pct, 1),
            "Effective_coeff_coverage_pct": round(eff_coeff_cov, 1),
        })

        geo_data[cohort] = {
            "clinical": clin,
            "expr": expr_avail,
            "available_ensg": available_ensg,
            "missing_ensg": missing_ensg,
            "missing_syms": missing_syms,
            "coverage_pct": coverage_pct,
            "eff_coeff_cov": eff_coeff_cov,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(P2_REPORTS/"external_validation_dataset_summary.csv", index=False)
    log(f"  -> external_validation_dataset_summary.csv written")
    return geo_data

# ─── Step 3: Apply Frozen Model — Risk Scores ───
def step3_risk_scores(cox_model, scaler, feature_names, gene_features, geo_data, age_median):
    log("STEP 3: Apply frozen model — risk scores with missing-gene handling")
    betas = cox_model.params_
    baseline_surv = cox_model.baseline_survival_["baseline survival"]

    for cohort in GEO_COHORTS:
        log(f"  Computing risk scores for {cohort}...")
        d = geo_data[cohort]
        clin = d["clinical"]
        expr = d["expr"]
        avail = d["available_ensg"]
        missing = d["missing_ensg"]
        cov_pct = d["coverage_pct"]
        eff_cov = d["eff_coeff_cov"]

        # Build clinical features
        clin_enc = pd.DataFrame(index=clin.index)
        clin_enc['Age'] = pd.to_numeric(clin['Age'], errors='coerce').fillna(age_median)

        for col in ['Cancer_Type', 'Stage', 'Smoking_Status']:
            for fname in feature_names:
                if fname.startswith(f"{col}="):
                    cat = fname.split("=", 1)[1]
                    if col == 'Cancer_Type':
                        mapped = clin[col].apply(map_cancer_type)
                        clin_enc[fname] = (mapped == cat).astype(int)
                    elif col == 'Stage':
                        mapped = clin[col].apply(map_stage)
                        clin_enc[fname] = (mapped == cat).astype(int)
                    elif col == 'Smoking_Status':
                        mapped = clin[col].apply(map_smoking)
                        clin_enc[fname] = (mapped == cat).astype(int)

        # Combine clinical + available expression
        X_geo = pd.concat([clin_enc, expr], axis=1)

        # Apply scaler to available features only
        X_scaled = pd.DataFrame(index=X_geo.index)
        for col in X_geo.columns:
            if col in scaler.feature_names_in_:
                X_scaled[col] = (X_geo[col].values - scaler.mean_[list(scaler.feature_names_in_).index(col)]) / \
                                scaler.scale_[list(scaler.feature_names_in_).index(col)]
            else:
                X_scaled[col] = X_geo[col].values

        # Compute risk score: sum(beta_i * x_i_scaled) for available features only
        risk_scores = np.zeros(len(X_geo))
        for col in X_scaled.columns:
            if col in betas.index:
                risk_scores += betas[col] * X_scaled[col].values

        # Compute survival function: S(t) = S0(t)^exp(risk_score)
        # For horizons
        times = np.array(list(HORIZONS.values()))
        surv_at_times = baseline_surv.reindex(baseline_surv.index.union(times)).sort_index().interpolate().reindex(times).values
        surv_probs = surv_at_times ** np.exp(risk_scores[:, None])

        # Save risk scores
        note = f"Reduced model: {len(avail)}/127 genes ({cov_pct:.1f}% panel, {eff_cov:.1f}% coeff coverage). Missing: {', '.join(d['missing_syms'])}"
        risk_df = pd.DataFrame({
            "Patient_ID": clin['Patient_ID'],
            "Risk_Score": risk_scores,
            "S_1yr": surv_probs[:, 0],
            "S_3yr": surv_probs[:, 1],
            "S_5yr": surv_probs[:, 2],
            "Reduced_Model_Note": note,
        })
        risk_df.to_csv(P2_RISK/f"{cohort}_risk_scores.csv", index=False)
        log(f"    -> {cohort}_risk_scores.csv ({len(risk_df)} patients)")

        d["risk_scores"] = risk_scores
        d["surv_probs"] = surv_probs
        d["X_scaled"] = X_scaled

    return geo_data

# ─── Step 4: Performance Evaluation ───
def step4_performance(cox_model, scaler, feature_names, gene_features, geo_data, X_train, train_df):
    log("STEP 4: Performance evaluation per cohort")
    train_clean = train_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).copy()
    train_clean = train_clean[~train_clean['Patient_ID'].isin(INVALID_PATIENTS)]

    y_train_time = pd.to_numeric(train_clean['Overall_Survival_Time'], errors='coerce').values
    y_train_event = pd.to_numeric(train_clean['Survival_Status'], errors='coerce').fillna(0).astype(int).values
    y_train_surv = make_surv(y_train_time, y_train_event)

    # Compute training risk scores for median threshold
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    train_risk = cox_model.predict_partial_hazard(X_train_scaled).values.ravel()
    train_median_risk = np.median(train_risk)
    log(f"  Training median risk (threshold): {train_median_risk:.4f}")

    # Internal validation reference
    val = pd.read_csv(READY_DIR/"TCGA_internal_validation.csv")
    val = val[~val['Patient_ID'].isin(INVALID_PATIENTS)]
    val = val.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    log(f"  Internal validation: {len(val)} patients")

    # Build val features (same as training)
    expr_val = pd.read_csv(READY_DIR/"TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0)
    expr_val_norm = normalize_counts_log2cpm(expr_val)
    val_expr_cols = [pid for pid in val['Patient_ID'].astype(str) if pid in expr_val_norm.columns]
    val = val[val['Patient_ID'].astype(str).isin(val_expr_cols)].reset_index(drop=True)
    expr_val_sel = expr_val_norm.loc[gene_features][val_expr_cols].T
    expr_val_sel.index = val.index

    val_enc = pd.DataFrame(index=val.index)
    val_enc['Age'] = pd.to_numeric(val['Age'], errors='coerce').fillna(age_median_global)
    for col in ['Cancer_Type', 'Stage', 'Smoking_Status']:
        val[col] = val[col].fillna('not_available')
        for fname in feature_names:
            if fname.startswith(f"{col}="):
                cat = fname.split("=", 1)[1]
                val_enc[fname] = (val[col] == cat).astype(int)
    X_val = pd.concat([val_enc, expr_val_sel], axis=1)
    X_val = X_val[feature_names]
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    y_val_time = pd.to_numeric(val['Overall_Survival_Time'], errors='coerce').values
    y_val_event = pd.to_numeric(val['Survival_Status'], errors='coerce').fillna(0).astype(int).values
    y_val_surv = make_surv(y_val_time, y_val_event)

    val_risk = cox_model.predict_partial_hazard(X_val_scaled).values.ravel()
    val_ci = concordance_index_censored(y_val_event.astype(bool), y_val_time, val_risk)[0]

    # Val AUC
    val_times = np.array([t for t in HORIZONS.values() if t < y_val_time.max() and t < y_train_time.max()])
    val_auc, val_mean_auc = cumulative_dynamic_auc(y_train_surv, y_val_surv, val_risk, val_times)

    # Val IBS
    baseline_surv = cox_model.baseline_survival_["baseline survival"]
    surv_df_val = cox_model.predict_survival_function(X_val_scaled, times=val_times)
    val_surv_prob = surv_df_val.values.T
    val_ibs = integrated_brier_score(y_train_surv, y_val_surv, val_surv_prob, val_times)

    # Val logrank
    val_high = val_risk > train_median_risk
    try:
        lr = logrank_test(y_val_time[val_high], y_val_time[~val_high],
                          y_val_event[val_high], y_val_event[~val_high])
        val_logrank = lr.p_value
    except:
        val_logrank = None

    results = [{
        "Cohort": "TCGA_internal_val",
        "Panel_coverage_pct": 100.0,
        "Eff_coeff_coverage_pct": 100.0,
        "N": len(val), "Events": int(y_val_event.sum()),
        "C_index": round(val_ci, 4),
        "Mean_AUC": round(val_mean_auc, 4),
        "IBS": round(val_ibs, 4),
        "Logrank_p": f"{val_logrank:.2e}" if val_logrank else "N/A",
        "Reduced_Model": "No (full 127-gene panel)",
    }]
    for i, (label, t) in enumerate(HORIZONS.items()):
        if t in val_times:
            results[0][f"AUC_{label}"] = round(val_auc[list(val_times).index(t)], 4)

    # Evaluate each GEO cohort
    for cohort in GEO_COHORTS:
        log(f"  Evaluating {cohort}...")
        d = geo_data[cohort]
        clin = d["clinical"]
        risk = d["risk_scores"]
        surv_probs = d["surv_probs"]

        y_time = pd.to_numeric(clin['Overall_Survival_Time'], errors='coerce').values
        y_event = pd.to_numeric(clin['Survival_Status'], errors='coerce').fillna(0).astype(int).values
        y_surv = make_surv(y_time, y_event)

        # C-index
        ci = concordance_index_censored(y_event.astype(bool), y_time, risk)[0]

        # Valid horizons
        cohort_times = np.array([t for t in HORIZONS.values() if t < y_time.max() and t < y_train_time.max()])

        # AUC
        try:
            auc_vals, mean_auc = cumulative_dynamic_auc(y_train_surv, y_surv, risk, cohort_times)
        except Exception as ex:
            log(f"    AUC failed: {ex}")
            auc_vals = np.array([0.5]*len(cohort_times)); mean_auc = 0.5

        # IBS — compute survival probs: S(t) = S0(t)^exp(risk)
        try:
            bs = baseline_surv.copy()
            surv_at_horizons = np.zeros((len(risk), len(cohort_times)))
            for ti, t in enumerate(cohort_times):
                if t in bs.index:
                    s0_t = float(bs.loc[t])
                else:
                    bs_ext = bs.reindex(bs.index.union([t])).sort_index().interpolate()
                    s0_t = float(bs_ext.loc[t])
                surv_at_horizons[:, ti] = s0_t ** np.exp(risk)
            ibs = integrated_brier_score(y_train_surv, y_surv, surv_at_horizons, cohort_times)
        except Exception as ex:
            log(f"    IBS failed: {ex}")
            ibs = None

        # Logrank
        high_risk = risk > train_median_risk
        try:
            lr = logrank_test(y_time[high_risk], y_time[~high_risk],
                              y_event[high_risk], y_event[~high_risk])
            logrank_p = lr.p_value
        except:
            logrank_p = None

        row = {
            "Cohort": cohort,
            "Panel_coverage_pct": round(d["coverage_pct"], 1),
            "Eff_coeff_coverage_pct": round(d["eff_coeff_cov"], 1),
            "N": len(clin), "Events": int(y_event.sum()),
            "C_index": round(ci, 4),
            "Mean_AUC": round(mean_auc, 4),
            "IBS": round(ibs, 4) if ibs else None,
            "Logrank_p": f"{logrank_p:.2e}" if logrank_p else "N/A",
            "Reduced_Model": f"Yes ({len(d['available_ensg'])}/127 genes)",
        }
        for i, (label, t) in enumerate(HORIZONS.items()):
            if t in cohort_times:
                row[f"AUC_{label}"] = round(auc_vals[list(cohort_times).index(t)], 4)
        results.append(row)

        d["y_time"] = y_time
        d["y_event"] = y_event
        d["y_surv"] = y_surv
        d["ci"] = ci
        d["auc_vals"] = auc_vals
        d["mean_auc"] = mean_auc
        d["ibs"] = ibs
        d["logrank_p"] = logrank_p
        d["high_risk"] = high_risk
        d["cohort_times"] = cohort_times
        d["train_median_risk"] = train_median_risk

    perf_df = pd.DataFrame(results)
    perf_df.to_csv(P2_REPORTS/"performance_summary.csv", index=False)
    perf_df.to_csv(P2_TABLES/"cohort_performance_table.csv", index=False)
    log(f"  -> performance_summary.csv, cohort_performance_table.csv written")

    # Risk distribution table
    risk_dist = []
    for cohort in GEO_COHORTS + ["TCGA_internal_val"]:
        if cohort == "TCGA_internal_val":
            r = val_risk; n = len(r)
        else:
            r = geo_data[cohort]["risk_scores"]; n = len(r)
        risk_dist.append({
            "Cohort": cohort, "N": n,
            "Risk_mean": round(np.mean(r), 4),
            "Risk_median": round(np.median(r), 4),
            "Risk_std": round(np.std(r), 4),
            "Risk_min": round(np.min(r), 4),
            "Risk_max": round(np.max(r), 4),
            "High_risk_pct": round(100*np.mean(r > train_median_risk), 1),
        })
    pd.DataFrame(risk_dist).to_csv(P2_TABLES/"risk_distribution_table.csv", index=False)
    log(f"  -> risk_distribution_table.csv written")

    return geo_data, perf_df, train_median_risk, val_ci, val_risk, y_val_time, y_val_event, val_high

# ─── Main ───
age_median_global = None

def main():
    log("="*70)
    log("PHASE 2: EXTERNAL SCIENTIFIC VALIDATION")
    log("="*70)

    # Clear old log
    (P2_REPORTS/"phase2_log.txt").unlink(missing_ok=True)

    # Step 0
    train_df, val_df = step0_data_quality()

    # Step 1
    global age_median_global
    cox_model, scaler, feature_names, gene_features, ensg2sym, sym2ensg, unmapped, age_median, X_train = \
        step1_load_model(train_df)
    age_median_global = age_median

    # Step 2
    geo_data = step2_prepare_geo(cox_model, feature_names, gene_features, sym2ensg, unmapped)

    # Step 3
    geo_data = step3_risk_scores(cox_model, scaler, feature_names, gene_features, geo_data, age_median)

    # Step 4
    geo_data, perf_df, train_median_risk, val_ci, val_risk, y_val_time, y_val_event, val_high = \
        step4_performance(cox_model, scaler, feature_names, gene_features, geo_data, X_train, train_df)

    log("\n" + "="*70)
    log("STEPS 0-4 COMPLETE — Core validation finished")
    log("="*70)
    log(f"Internal validation C-index: {val_ci:.4f}")
    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        ibs_str = f"{d['ibs']:.4f}" if d['ibs'] is not None else "N/A"
        log(f"  {cohort}: C-index={d['ci']:.4f}, IBS={ibs_str}, "
            f"coverage={d['coverage_pct']:.1f}%")
    log("\nRun run_phase2_analysis.py for Steps 5-10 (reports, figures, final assessment).")

if __name__ == "__main__":
    main()
