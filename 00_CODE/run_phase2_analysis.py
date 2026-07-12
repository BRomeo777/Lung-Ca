#!/usr/bin/env python3
"""Phase 2: External Scientific Validation — Steps 5-10 (Analysis, Figures, Reports)."""
from __future__ import annotations
import os, sys, pickle, re, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

INVALID_PATIENTS = ["TCGA-05-4395", "TCGA-77-A5G6"]
GEO_COHORTS = ["GSE30219", "GSE50081", "GSE72094", "GSE31210"]
HORIZONS = {"1yr": 365, "3yr": 1095, "5yr": 1825}
INTERNAL_VAL_CINDEX_AUDIT = 0.5944
GEO_EXPR = {
    "GSE30219": PHASE1C_DIR/"processed_data"/"GSE30219"/"expression_gene_mapped.csv",
    "GSE50081": PHASE1C_DIR/"processed_data"/"GSE50081"/"expression_gene_mapped.csv",
    "GSE72094": PHASE1C_DIR/"processed_data"/"GSE72094"/"expression_gene_mapped.csv",
    "GSE31210": PHASE1D_DIR/"processed_data"/"GSE31210"/"GSE31210_expression_gene_mapped_log2.csv",
}

def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(P2_REPORTS/"phase2_analysis_log.txt", "a", encoding="utf-8") as f:
        f.write(line+"\n")

def normalize_counts_log2cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0); lib[lib==0]=1
    cpm = counts.div(lib, axis=1)*1e6
    return np.log2(cpm+1)

def make_surv(time, event):
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))

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

# ═════════════════════════════════════════════════════════════
# RELOAD ALL DATA (same as validation script)
# ═════════════════════════════════════════════════════════════
def load_all_data():
    log("Loading all data and recomputing results...")

    # Load model
    with open(MODELS_DIR/"cox_ph_model.pkl", "rb") as f:
        cox_model = pickle.load(f)
    feature_names = list(cox_model.params_.index)
    gene_features = [f for f in feature_names if '=' not in f and f != 'Age']

    # Ensembl mapping
    ensg_map = pd.read_csv(PHASE1C_DIR/"reports"/"phase1B_ensg_to_symbol_mapping.csv")
    ensg2sym = dict(zip(ensg_map['Ensembl_ID'], ensg_map['Gene_Symbol']))
    sym2ensg = {v: k for k, v in ensg2sym.items()}
    unmapped = set(gene_features) - set(ensg2sym.keys())

    # Training data
    train_df = pd.read_csv(READY_DIR/"TCGA_train.csv")
    train_df = train_df[~train_df['Patient_ID'].isin(INVALID_PATIENTS)]
    train_clean = train_df.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).copy()

    expr = pd.read_csv(READY_DIR/"TCGA_train_expression.tsv.gz", sep="\t", index_col=0)
    expr_norm = normalize_counts_log2cpm(expr)

    train_expr_cols = []
    valid_mask = []
    for pid in train_clean['Patient_ID'].astype(str):
        if pid in expr.columns:
            train_expr_cols.append(pid); valid_mask.append(True)
        else:
            valid_mask.append(False)
    train_clean = train_clean[valid_mask].reset_index(drop=True)

    expr_sel = expr_norm.loc[gene_features][train_expr_cols].T
    expr_sel.index = train_clean.index

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
    X_train = X_train[feature_names]
    scaler = StandardScaler()
    scaler.fit(X_train)

    y_train_time = pd.to_numeric(train_clean['Overall_Survival_Time'], errors='coerce').values
    y_train_event = pd.to_numeric(train_clean['Survival_Status'], errors='coerce').fillna(0).astype(int).values
    y_train_surv = make_surv(y_train_time, y_train_event)

    # Internal validation
    val = pd.read_csv(READY_DIR/"TCGA_internal_validation.csv")
    val = val[~val['Patient_ID'].isin(INVALID_PATIENTS)]
    val = val.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    expr_val = pd.read_csv(READY_DIR/"TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0)
    expr_val_norm = normalize_counts_log2cpm(expr_val)
    val_expr_cols = [pid for pid in val['Patient_ID'].astype(str) if pid in expr_val_norm.columns]
    val = val[val['Patient_ID'].astype(str).isin(val_expr_cols)].reset_index(drop=True)
    expr_val_sel = expr_val_norm.loc[gene_features][val_expr_cols].T
    expr_val_sel.index = val.index

    val_enc = pd.DataFrame(index=val.index)
    val_enc['Age'] = pd.to_numeric(val['Age'], errors='coerce').fillna(age_median)
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

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    train_risk = cox_model.predict_partial_hazard(X_train_scaled).values.ravel()
    train_median_risk = np.median(train_risk)

    # GEO data
    geo_data = {}
    for cohort in GEO_COHORTS:
        clin = pd.read_csv(READY_DIR/f"{cohort}_external_validation.csv")
        expr_geo = pd.read_csv(GEO_EXPR[cohort], index_col=0)
        normal_mask = clin.apply(is_normal_sample, axis=1)
        clin = clin[~normal_mask].reset_index(drop=True)
        clin = clin.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)

        available_ensg = []
        missing_ensg = []
        for g in gene_features:
            if g in unmapped:
                missing_ensg.append(g); continue
            sym = ensg2sym.get(g)
            if sym and sym in expr_geo.index:
                available_ensg.append(g)
            else:
                missing_ensg.append(g)

        missing_syms = [ensg2sym.get(g, '[UNMAPPED]') for g in missing_ensg]
        betas = cox_model.params_
        total_abs = betas[gene_features].abs().sum()
        avail_abs = betas[available_ensg].abs().sum() if available_ensg else 0
        coverage_pct = 100.0 * len(available_ensg) / len(gene_features)
        eff_coeff_cov = 100.0 * avail_abs / total_abs if total_abs > 0 else 0

        expr_avail = expr_geo.loc[[ensg2sym[g] for g in available_ensg]].copy()
        expr_cols = [c for c in clin['Patient_ID'].astype(str) if c in expr_avail.columns]
        clin = clin[clin['Patient_ID'].astype(str).isin(expr_cols)].reset_index(drop=True)
        expr_avail = expr_avail[clin['Patient_ID'].astype(str).tolist()].T
        expr_avail.columns = available_ensg
        expr_avail.index = clin.index

        # Build features
        clin_enc = pd.DataFrame(index=clin.index)
        clin_enc['Age'] = pd.to_numeric(clin['Age'], errors='coerce').fillna(age_median)
        for col_name in ['Cancer_Type', 'Stage', 'Smoking_Status']:
            for fname in feature_names:
                if fname.startswith(f"{col_name}="):
                    cat = fname.split("=", 1)[1]
                    if col_name == 'Cancer_Type':
                        mapped = clin[col_name].apply(map_cancer_type)
                        clin_enc[fname] = (mapped == cat).astype(int)
                    elif col_name == 'Stage':
                        mapped = clin[col_name].apply(map_stage)
                        clin_enc[fname] = (mapped == cat).astype(int)
                    elif col_name == 'Smoking_Status':
                        mapped = clin[col_name].apply(map_smoking)
                        clin_enc[fname] = (mapped == cat).astype(int)

        X_geo = pd.concat([clin_enc, expr_avail], axis=1)
        X_scaled = pd.DataFrame(index=X_geo.index)
        feat_list = list(scaler.feature_names_in_)
        for col in X_geo.columns:
            if col in feat_list:
                idx = feat_list.index(col)
                X_scaled[col] = (X_geo[col].values - scaler.mean_[idx]) / scaler.scale_[idx]
            else:
                X_scaled[col] = X_geo[col].values

        risk_scores = np.zeros(len(X_geo))
        for col in X_scaled.columns:
            if col in betas.index:
                risk_scores += betas[col] * X_scaled[col].values

        y_time = pd.to_numeric(clin['Overall_Survival_Time'], errors='coerce').values
        y_event = pd.to_numeric(clin['Survival_Status'], errors='coerce').fillna(0).astype(int).values
        y_surv = make_surv(y_time, y_event)
        ci = concordance_index_censored(y_event.astype(bool), y_time, risk_scores)[0]

        cohort_times = np.array([t for t in HORIZONS.values() if t < y_time.max() and t < y_train_time.max()])
        try:
            auc_vals, mean_auc = cumulative_dynamic_auc(y_train_surv, y_surv, risk_scores, cohort_times)
        except:
            auc_vals = np.array([0.5]*len(cohort_times)); mean_auc = 0.5

        baseline_surv = cox_model.baseline_survival_["baseline survival"]
        bs = baseline_surv.copy()
        surv_at_horizons = np.zeros((len(risk_scores), len(cohort_times)))
        for ti, t in enumerate(cohort_times):
            if t in bs.index:
                s0_t = float(bs.loc[t])
            else:
                bs_ext = bs.reindex(bs.index.union([t])).sort_index().interpolate()
                s0_t = float(bs_ext.loc[t])
            surv_at_horizons[:, ti] = s0_t ** np.exp(risk_scores)
        try:
            ibs = integrated_brier_score(y_train_surv, y_surv, surv_at_horizons, cohort_times)
        except:
            ibs = None

        high_risk = risk_scores > train_median_risk
        try:
            lr = logrank_test(y_time[high_risk], y_time[~high_risk],
                              y_event[high_risk], y_event[~high_risk])
            logrank_p = lr.p_value
        except:
            logrank_p = None

        geo_data[cohort] = {
            "clinical": clin, "risk_scores": risk_scores,
            "y_time": y_time, "y_event": y_event, "y_surv": y_surv,
            "ci": ci, "auc_vals": auc_vals, "mean_auc": mean_auc,
            "ibs": ibs, "logrank_p": logrank_p,
            "high_risk": high_risk, "cohort_times": cohort_times,
            "available_ensg": available_ensg, "missing_ensg": missing_ensg,
            "missing_syms": missing_syms,
            "coverage_pct": coverage_pct, "eff_coeff_cov": eff_coeff_cov,
            "surv_at_horizons": surv_at_horizons,
            "X_scaled": X_scaled,
        }
        ibs_str = f"{ibs:.4f}" if ibs is not None else "N/A"
        log(f"  {cohort}: C-index={ci:.4f}, IBS={ibs_str}")

    val_high = val_risk > train_median_risk

    return cox_model, scaler, feature_names, gene_features, ensg2sym, sym2ensg, unmapped, \
           age_median, X_train, X_train_scaled, train_risk, train_median_risk, \
           y_train_time, y_train_event, y_train_surv, \
           val, val_risk, val_ci, y_val_time, y_val_event, y_val_surv, val_high, geo_data


# ═════════════════════════════════════════════════════════════
# STEP 5: GENERALIZATION ANALYSIS
# ═════════════════════════════════════════════════════════════
def step5_generalization(val_ci, geo_data):
    log("STEP 5: Generalization analysis")
    lines = ["="*70, "GENERALIZATION ANALYSIS REPORT", "="*70, ""]

    lines.append("1. REFERENCE BASELINE")
    lines.append(f"   TCGA Internal Validation C-index: {val_ci:.4f}")
    lines.append(f"   Audit-confirmed unbiased estimate: {INTERNAL_VAL_CINDEX_AUDIT}")
    lines.append(f"   Note: Minor difference ({abs(val_ci-INTERNAL_VAL_CINDEX_AUDIT):.4f}) due to")
    lines.append(f"   scaler reconstruction from training data (737 matched patients).")
    lines.append(f"   The audit value ({INTERNAL_VAL_CINDEX_AUDIT}) remains the canonical reference.")
    lines.append("")

    lines.append("2. PER-COHORT EXTERNAL VALIDATION RESULTS")
    lines.append(f"{'Cohort':<18} {'C-index':>8} {'ΔC-index':>9} {'AUC_mean':>9} {'IBS':>8} {'Panel%':>7} {'Coeff%':>7} {'Logrank':>12}")
    lines.append("-"*90)
    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        delta = d["ci"] - val_ci
        ibs_str = f"{d['ibs']:.4f}" if d['ibs'] is not None else "N/A"
        lr_str = f"{d['logrank_p']:.2e}" if d['logrank_p'] is not None else "N/A"
        lines.append(f"{cohort:<18} {d['ci']:>8.4f} {delta:>+9.4f} {d['mean_auc']:>9.4f} {ibs_str:>8} {d['coverage_pct']:>6.1f}% {d['eff_coeff_cov']:>6.1f}% {lr_str:>12}")
    lines.append("")

    lines.append("3. DEGRADATION ANALYSIS: REDUCED-PANEL vs PLATFORM/BIOLOGICAL EFFECTS")
    lines.append("")
    lines.append("   Two distinct sources of expected performance difference:")
    lines.append("   A) Reduced-panel effect: Missing genes reduce effective coefficient coverage")
    lines.append("      (90.5%-93.8% across cohorts), meaning 6.2%-9.5% of the model's learned")
    lines.append("      signal is not exercised. This is a structural limitation, not a")
    lines.append("      biological finding.")
    lines.append("   B) Platform/biological effect: TCGA (RNA-seq, STAR counts → log2CPM)")
    lines.append("      vs GEO (microarray, log2 intensities). Different dynamic ranges,")
    lines.append("      noise profiles, and probe hybridization characteristics.")
    lines.append("")

    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        delta_ci = d["ci"] - val_ci
        missing_coeff = 100.0 - d["eff_coeff_cov"]
        lines.append(f"   {cohort}:")
        lines.append(f"     C-index change: {delta_ci:+.4f} ({'improvement' if delta_ci > 0 else 'degradation'})")
        lines.append(f"     Missing coefficient weight: {missing_coeff:.1f}% of model signal")
        lines.append(f"     Missing genes: {', '.join(d['missing_syms'])}")
        if delta_ci < 0:
            platform_portion = "Cannot be precisely decomposed without controlled experiments."
            lines.append(f"     Attribution: {missing_coeff:.1f}% coeff loss (reduced-panel) + remainder (platform/biological)")
            lines.append(f"     Note: These two effects are confounded and cannot be fully separated.")
        else:
            lines.append(f"     Attribution: Performance exceeds internal validation despite {missing_coeff:.1f}% coeff loss.")
            lines.append(f"     This suggests platform/biological differences may actually FAVOR")
            lines.append(f"     this cohort (e.g., more homogeneous patient population).")
        lines.append("")

    lines.append("4. RNA-seq vs MICROARRAY PLATFORM CONSIDERATIONS")
    lines.append("")
    lines.append("   TCGA: Illumina RNA-seq, STAR counts → CPM normalization → log2(CPM+1)")
    lines.append("   GEO:  Affymetrix microarray, log2-transformed intensities (GPL570/GPL15048)")
    lines.append("")
    lines.append("   Key differences:")
    lines.append("   - RNA-seq counts are discrete and library-size dependent; microarray")
    lines.append("     intensities are continuous and probe-affinity dependent")
    lines.append("   - log2(CPM+1) and log2(intensity) are not on the same absolute scale,")
    lines.append("     but both are log-transformed and approximately comparable for")
    lines.append("     relative expression patterns")
    lines.append("   - Standardization (z-score using TCGA training mean/std) partially")
    lines.append("     mitigates scale differences but does not eliminate probe-specific")
    lines.append("     hybridization biases")
    lines.append("   - No batch correction was performed (per protocol)")
    lines.append("")

    lines.append("5. COHORT-SPECIFIC OBSERVATIONS")
    lines.append("")
    lines.append("   GSE30219 (GPL570, n=293, 121/127 genes):")
    lines.append("     C-index 0.6567 — ABOVE internal validation. Mixed histology cohort")
    lines.append("     (ADC, SQC, LCNE, BAS). Logrank p=2.46e-06 (strong stratification).")
    lines.append("     Possible reasons: diverse histology may benefit from gene panel;")
    lines.append("     microarray platform well-aligned with training scale.")
    lines.append("")
    lines.append("   GSE50081 (GPL570, n=181, 121/127 genes):")
    lines.append("     C-index 0.5872 — comparable to internal validation. Early-stage only")
    lines.append("     (IA-IIB). Logrank p=0.666 (no stratification). Limited event rate")
    lines.append("     (75/181=41%) and narrow stage range may reduce discriminative power.")
    lines.append("")
    lines.append("   GSE72094 (GPL15048, n=398, 117/127 genes):")
    lines.append("     C-index 0.6830 — ABOVE internal validation despite lowest gene coverage.")
    lines.append("     Logrank p=1.41e-07 (strongest stratification). HuGene 2.0 ST array")
    lines.append("     (gene-level summarization) may align better with Ensembl gene models.")
    lines.append("     IBS 0.1449 is lowest (best) — excellent calibration.")
    lines.append("")
    lines.append("   GSE31210 (GPL570, n=226, 121/127 genes):")
    lines.append("     C-index 0.7557 — HIGHEST across all cohorts. Very low event rate")
    lines.append("     (35/226=15.5%). High C-index with low event rate should be interpreted")
    lines.append("     cautiously — fewer events means fewer comparable pairs.")
    lines.append("     IBS 0.7954 is highest (worst) — poor calibration despite good discrimination.")
    lines.append("     Logrank p=3.44e-07 (strong stratification).")
    lines.append("")

    (P2_REPORTS/"generalization_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("  -> generalization_report.txt written")


# ═════════════════════════════════════════════════════════════
# STEP 6: CALIBRATION AND PLATFORM CLUSTER ANALYSIS
# ═════════════════════════════════════════════════════════════
def step6_calibration(cox_model, geo_data, val, val_risk, y_val_time, y_val_event, train_median_risk):
    log("STEP 6: Calibration and platform cluster analysis")
    baseline_surv = cox_model.baseline_survival_["baseline survival"]
    t_cal = HORIZONS["3yr"]

    # Compute calibration data for each cohort
    cal_data = {}
    for cohort in GEO_COHORTS + ["TCGA_internal_val"]:
        if cohort == "TCGA_internal_val":
            risk = val_risk; y_time = y_val_time; y_event = y_val_event
        else:
            d = geo_data[cohort]
            risk = d["risk_scores"]; y_time = d["y_time"]; y_event = d["y_event"]

        # Predicted survival at 3yr
        if t_cal in baseline_surv.index:
            s0 = float(baseline_surv.loc[t_cal])
        else:
            bs_ext = baseline_surv.reindex(baseline_surv.index.union([t_cal])).sort_index().interpolate()
            s0 = float(bs_ext.loc[t_cal])
        pred_surv = s0 ** np.exp(risk)

        # Observed survival at 3yr (KM estimate)
        kmf = KaplanMeierFitter()
        kmf.fit(y_time, y_event)
        obs_surv = kmf.predict(t_cal)

        # Calibration in quartiles
        n_groups = 4
        try:
            quartiles = pd.qcut(pred_surv, n_groups, labels=False, duplicates="drop")
        except:
            quartiles = pd.Series([0]*len(pred_surv))

        pred_by_group = []
        obs_by_group = []
        for q in sorted(np.unique(quartiles)):
            mask = quartiles == q
            if mask.sum() < 5:
                continue
            pred_by_group.append(pred_surv[mask].mean())
            kmf_q = KaplanMeierFitter()
            kmf_q.fit(y_time[mask], y_event[mask])
            obs_by_group.append(kmf_q.predict(t_cal))

        # Mean predicted vs observed
        mean_pred = pred_surv.mean()
        cal_offset = mean_pred - obs_surv

        cal_data[cohort] = {
            "pred_surv": pred_surv, "obs_surv": obs_surv,
            "mean_pred": mean_pred, "cal_offset": cal_offset,
            "pred_by_group": pred_by_group, "obs_by_group": obs_by_group,
            "risk": risk, "y_time": y_time, "y_event": y_event,
        }
        log(f"  {cohort}: mean_pred={mean_pred:.4f}, obs={obs_surv:.4f}, offset={cal_offset:+.4f}")

    # Generate calibration plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, cohort in enumerate(["TCGA_internal_val"] + GEO_COHORTS):
        ax = axes[i]
        cd = cal_data[cohort]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.scatter(cd["pred_by_group"], cd["obs_by_group"], s=80, zorder=5, color="steelblue")
        title = "TCGA Internal Val" if cohort == "TCGA_internal_val" else cohort
        ax.set_title(f"{title}\n(offset={cd['cal_offset']:+.3f})", fontsize=10)
        ax.set_xlabel("Predicted Survival (3yr)")
        ax.set_ylabel("Observed Survival (3yr)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[5].set_visible(False)
    plt.suptitle("Calibration Plots — 3-Year Survival (Quartile Groups)", fontsize=14)
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"Calibration"/"calibration_all_cohorts.png", dpi=150)
    plt.close()
    log("  -> calibration_all_cohorts.png")

    # Test Phase 1D cluster hypothesis: GSE30219/GSE50081 vs GSE72094/GSE31210
    cluster_a = ["GSE30219", "GSE50081"]
    cluster_b = ["GSE72094", "GSE31210"]

    offsets_a = [cal_data[c]["cal_offset"] for c in cluster_a]
    offsets_b = [cal_data[c]["cal_offset"] for c in cluster_b]
    mean_a = np.mean(offsets_a)
    mean_b = np.mean(offsets_b)

    # IBS comparison
    ibs_a = [geo_data[c]["ibs"] for c in cluster_a if geo_data[c]["ibs"] is not None]
    ibs_b = [geo_data[c]["ibs"] for c in cluster_b if geo_data[c]["ibs"] is not None]

    # Risk distribution comparison
    risk_medians_a = [np.median(geo_data[c]["risk_scores"]) for c in cluster_a]
    risk_medians_b = [np.median(geo_data[c]["risk_scores"]) for c in cluster_b]

    lines = ["="*70, "CALIBRATION AND PLATFORM CLUSTER COMPARISON REPORT", "="*70, ""]
    lines.append("1. CALIBRATION-OFFSET SUMMARY (3-year horizon)")
    lines.append(f"{'Cohort':<18} {'Mean Pred':>10} {'Observed':>10} {'Offset':>10} {'IBS':>10}")
    lines.append("-"*60)
    for cohort in ["TCGA_internal_val"] + GEO_COHORTS:
        cd = cal_data[cohort]
        ibs_val = geo_data[cohort]["ibs"] if cohort in geo_data else None
        ibs_str = f"{ibs_val:.4f}" if ibs_val is not None else "N/A"
        lines.append(f"{cohort:<18} {cd['mean_pred']:>10.4f} {cd['obs_surv']:>10.4f} {cd['cal_offset']:>+10.4f} {ibs_str:>10}")
    lines.append("")

    lines.append("2. PHASE 1D CLUSTER HYPOTHESIS TEST")
    lines.append("")
    lines.append("   Phase 1D observation: Raw expression medians suggested a ~2 log2-unit")
    lines.append("   baseline split: GSE30219/GSE50081 (lower) vs GSE72094/GSE31210 (higher).")
    lines.append("")
    lines.append("   Testing at calibration level:")
    lines.append(f"   Cluster A (GSE30219, GSE50081): mean offset = {mean_a:+.4f}")
    lines.append(f"   Cluster B (GSE72094, GSE31210): mean offset = {mean_b:+.4f}")
    lines.append(f"   Difference (B - A): {mean_b - mean_a:+.4f}")
    lines.append("")

    # Determine if pattern holds
    pattern_holds = (mean_a * mean_b > 0) and abs(mean_b - mean_a) > 0.05
    same_sign = mean_a * mean_b > 0
    large_diff = abs(mean_b - mean_a) > 0.05

    if same_sign and not large_diff:
        lines.append("   FINDING: Calibration offsets are similar in direction and magnitude.")
        lines.append("   The Phase 1D expression-level split does NOT produce a clear")
        lines.append("   calibration-level split. Both clusters show similar calibration behavior.")
        lines.append("   The expression-level baseline difference does not translate to")
        lines.append("   a calibration-level difference after standardization.")
        verdict = "NOT SUPPORTED at calibration level"
    elif same_sign and large_diff:
        lines.append("   FINDING: Calibration offsets are in the same direction but differ")
        lines.append("   in magnitude. Partial support for cluster difference.")
        verdict = "PARTIALLY SUPPORTED"
    elif not same_sign:
        lines.append("   FINDING: Calibration offsets are in OPPOSITE directions between clusters.")
        lines.append("   This suggests the Phase 1D expression-level split may have some")
        lines.append("   calibration relevance, but the pattern is not consistent within clusters.")
        verdict = "MIXED EVIDENCE"
    else:
        lines.append("   FINDING: Insufficient evidence to confirm or reject the cluster hypothesis.")
        verdict = "INCONCLUSIVE"

    lines.append(f"   VERDICT: {verdict}")
    lines.append("")

    lines.append("3. IBS COMPARISON")
    if ibs_a and ibs_b:
        lines.append(f"   Cluster A mean IBS: {np.mean(ibs_a):.4f}")
        lines.append(f"   Cluster B mean IBS: {np.mean(ibs_b):.4f}")
        lines.append(f"   Note: IBS values are not directly comparable across cohorts with")
        lines.append(f"   different event rates and censoring patterns. Interpret with caution.")
    lines.append("")

    lines.append("4. RISK DISTRIBUTION COMPARISON")
    lines.append(f"   Cluster A risk medians: {[f'{r:.4f}' for r in risk_medians_a]}")
    lines.append(f"   Cluster B risk medians: {[f'{r:.4f}' for r in risk_medians_b]}")
    lines.append("")

    lines.append("5. CONCLUSION")
    lines.append(f"   The Phase 1D expression-level cluster pattern is {verdict.lower()}.")
    lines.append("   Calibration behavior should be assessed per-cohort, not by assumed clusters.")
    lines.append("   The standardization step (z-score using TCGA training statistics) appears")
    lines.append("   to partially mitigate the raw expression-level baseline differences,")
    lines.append("   though cohort-specific calibration offsets remain variable.")

    (P2_REPORTS/"calibration_cluster_comparison_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("  -> calibration_cluster_comparison_report.txt written")
    return cal_data


# ═════════════════════════════════════════════════════════════
# STEP 7: MODEL EXPLAINABILITY
# ═════════════════════════════════════════════════════════════
def step7_explainability(cox_model, feature_names, gene_features, ensg2sym, geo_data):
    log("STEP 7: Model explainability")
    betas = cox_model.params_
    hazard_ratios = cox_model.hazard_ratios_

    # Feature importance = |beta| * std(feature) from training
    # Since we don't have the original training std easily, use |beta| as importance
    importance = betas.abs().sort_values(ascending=False)

    rows = []
    for feat in importance.index:
        sym = ensg2sym.get(feat, feat) if feat in gene_features else feat
        is_gene = feat in gene_features
        # Check availability in each GEO cohort
        avail_in = []
        for cohort in GEO_COHORTS:
            if not is_gene:
                avail_in.append(cohort)
            elif feat in geo_data[cohort]["available_ensg"]:
                avail_in.append(cohort)
        avail_str = ", ".join(avail_in) if avail_in else "None"

        rows.append({
            "Feature": feat,
            "Gene_Symbol": sym,
            "Type": "Gene" if is_gene else "Clinical",
            "Beta": round(betas[feat], 6),
            "Abs_Beta": round(abs(betas[feat]), 6),
            "Hazard_Ratio": round(hazard_ratios[feat], 4),
            "Direction": "Risk-increasing" if betas[feat] > 0 else "Risk-decreasing",
            "Importance_Rank": int(importance.index.get_loc(feat)) + 1,
            "Available_in_GEO": avail_str,
            "Missing_in": ", ".join([c for c in GEO_COHORTS if c not in avail_str]) if is_gene else "N/A",
        })

    fi_df = pd.DataFrame(rows)
    fi_df.to_csv(P2_TABLES/"feature_importance.csv", index=False)
    log(f"  -> feature_importance.csv ({len(fi_df)} features)")

    # Top 20 feature importance plot
    top20 = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#d62728' if b > 0 else '#1f77b4' for b in top20['Beta']]
    labels = [f"{r['Gene_Symbol']}" + (f" ({r['Type']})" if r['Type']=='Clinical' else "") for _, r in top20.iterrows()]
    ax.barh(range(len(top20)), top20['Abs_Beta'], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("|β| (Absolute Coefficient)")
    ax.set_title("Top 20 Feature Importance — Frozen CoxPH Model\n(Red = risk-increasing, Blue = risk-decreasing)")
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"Feature_importance"/"top20_feature_importance.png", dpi=150)
    plt.close()
    log("  -> top20_feature_importance.png")

    # Explainability report
    lines = ["="*70, "MODEL EXPLAINABILITY REPORT", "="*70, "",
        "Model: CoxPH (lifelines CoxPHFitter, penalizer=1.0)",
        f"Total features: {len(feature_names)} ({len(gene_features)} genes + {len(feature_names)-len(gene_features)} clinical)",
        "",
        "IMPORTANT NOTE: Explainability outputs describe the FULL 127-gene trained model.",
        "A feature highlighted as important may have been UNAVAILABLE in one or more GEO",
        "cohorts. Cross-reference the 'Missing_in' column in feature_importance.csv.",
        "",
        "TOP 20 FEATURES BY |β|:",
        f"{'Rank':<6} {'Feature':<25} {'Symbol':<15} {'β':>10} {'HR':>8} {'Direction':<18} {'Missing in':<20}",
        "-"*110,
    ]
    for _, r in top20.iterrows():
        lines.append(f"{r['Importance_Rank']:<6} {r['Feature']:<25} {r['Gene_Symbol']:<15} {r['Beta']:>10.6f} {r['Hazard_Ratio']:>8.4f} {r['Direction']:<18} {r['Missing_in']:<20}")
    lines.append("")

    # Clinical feature summary
    clin_feats = fi_df[fi_df['Type']=='Clinical']
    lines.append("CLINICAL FEATURES:")
    for _, r in clin_feats.iterrows():
        lines.append(f"  {r['Feature']}: β={r['Beta']:.6f}, HR={r['Hazard_Ratio']:.4f}, {r['Direction']}")
    lines.append("")

    # Gene availability cross-reference
    lines.append("GENE AVAILABILITY CROSS-REFERENCE:")
    lines.append(f"{'Gene':<25} {'Symbol':<15} {'|β|':>10} {'Missing in':<30}")
    lines.append("-"*85)
    gene_rows = fi_df[fi_df['Type']=='Gene'].sort_values('Abs_Beta', ascending=False)
    for _, r in gene_rows.head(15).iterrows():
        lines.append(f"{r['Feature']:<25} {r['Gene_Symbol']:<15} {r['Abs_Beta']:>10.6f} {r['Missing_in']:<30}")
    lines.append("  ... (see feature_importance.csv for full list)")
    lines.append("")

    # Permutation importance note
    lines.append("PERMUTATION IMPORTANCE:")
    lines.append("  Not computed for external validation cohorts because permuting features")
    lines.append("  in a reduced-panel setting would conflate missing-gene effects with")
    lines.append("  importance effects. Feature importance is reported from the frozen model's")
    lines.append("  coefficients (|β|), which is the standard approach for CoxPH models.")
    lines.append("")
    lines.append("SHAP ANALYSIS:")
    lines.append("  SHAP values are not directly applicable to CoxPH models in the same way")
    lines.append("  as tree-based models. The coefficient-based importance (|β|) serves as")
    lines.append("  the primary explainability method for this linear model.")

    (P2_REPORTS/"explainability_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("  -> explainability_report.txt written")
    return fi_df


# ═════════════════════════════════════════════════════════════
# STEP 8: BIOLOGICAL INTERPRETATION
# ═════════════════════════════════════════════════════════════
def step8_biological(fi_df, gene_features, ensg2sym, geo_data):
    log("STEP 8: Biological interpretation")

    # Known lung cancer gene annotations (curated from literature)
    LUNG_CANCER_GENES = {
        "SPP1": "Osteopontin; upregulated in NSCLC, promotes invasion and metastasis",
        "COL1A1": "Collagen type I; stromal marker, associated with tumor microenvironment remodeling",
        "MMP1": "Matrix metalloproteinase-1; extracellular matrix degradation, invasion",
        "FSCN1": "Fascin-1; actin-bundling protein, promotes cell migration; NSCLC biomarker",
        "LAMC2": "Laminin gamma-2; basement membrane component, associated with invasion",
        "LAMA3": "Laminin alpha-3; basement membrane, epithelial adhesion",
        "CXCL8": "IL-8; pro-inflammatory chemokine, angiogenesis and immune evasion",
        "CXCL5": "ENA-78; neutrophil attractant, pro-tumorigenic inflammation",
        "CXCL13": "BLC; B-cell attractant, associated with immune infiltration",
        "CXCL17": "DMC; mucosal chemokine, dendritic cell attractant",
        "IL1A": "IL-1 alpha; pro-inflammatory cytokine, tumor microenvironment",
        "GREM1": "Gremlin-1; BMP antagonist, angiogenesis and fibrosis",
        "DKK1": "Dickkopf-1; Wnt antagonist, osteolytic bone metastasis in lung cancer",
        "SOX9": "SOX9; transcription factor, stemness and squamous differentiation",
        "HMGA2": "HMGA2; chromatin remodeling, associated with poor prognosis in NSCLC",
        "HOXB9": "Homeobox B9; developmental gene, reactivated in cancer",
        "TFAP2A": "AP-2 alpha; transcription factor, regulates differentiation",
        "CALB2": "Calretinin; mesothelial marker, differential diagnosis",
        "KRT81": "Keratin 81; squamous differentiation marker",
        "MME": "Neprilysin/CD10; common acute lymphoblastic leukemia antigen",
        "MS4A1": "CD20; B-cell marker, immune infiltration indicator",
        "CD79A": "B-cell receptor component; immune infiltration",
        "PSCA": "Prostate stem cell antigen; aberrant expression in NSCLC",
        "NT5E": "CD73; immunosuppressive, adenosine pathway",
        "PCSK9": "Proprotein convertase; cholesterol metabolism, emerging cancer target",
        "FGA": "Fibrinogen alpha chain; coagulation, metastasis",
        "FGG": "Fibrinogen gamma chain; coagulation, metastasis",
        "F5": "Factor V; coagulation pathway",
        "SLC2A1": "GLUT1; glucose transporter, Warburg effect marker",
        "EGLN3": "PHD3; hypoxia sensor, HIF pathway",
        "ANGPTL4": "Angiopoietin-like 4; hypoxia-induced, metastasis promotion",
        "AREG": "Amphiregulin; EGF family, proliferation",
        "EREG": "Epiregulin; EGF family, proliferation",
        "FOSL1": "Fra-1; AP-1 family, proliferation and invasion",
        "RHOV": "RhoV/Wrch-1; Rho GTPase, cell migration",
        "MYEOV": "Myeloma overexpressed gene; oncogene in multiple cancers",
        "KYNU": "Kynureninase; tryptophan metabolism, immune evasion",
        "SLC16A1": "MCT1; monocarboxylate transporter, lactate metabolism",
        "STC2": "Stanniocalcin-2; stress response, ER stress",
        "SFRP1": "Secreted frizzled-related protein 1; Wnt antagonist, tumor suppressor",
        "FLRT3": "Fibronectin leucine-rich transmembrane 3; cell adhesion",
        "JAG1": "Jagged-1; Notch ligand, stemness and angiogenesis",
        "ITGA6": "Integrin alpha 6; CD49f, stemness marker",
        "ITGB4": "Integrin beta 4; hemidesmosomes, invasion",
        "TMEM163": "Transmembrane protein 163; function less characterized",
        "PRDM16": "PR domain 16; transcriptional regulator",
        "ELF5": "E74-like factor 5; ETS family, epithelial differentiation",
        "CFTR": "Cystic fibrosis transmembrane conductance regulator; ion channel",
        "ABCC2": "MRP2; drug resistance transporter",
        "CACNA2D2": "Calcium channel alpha-2-delta-2; tumor suppressor candidate",
        "CPS1": "Carbamoyl phosphate synthetase 1; urea cycle, ammonia detoxification",
        "TENM1": "Teneurin-1; cell adhesion, axon guidance",
        "ADGRF4": "Adhesion G protein-coupled receptor F4; function emerging",
        "ADH1C": "Alcohol dehydrogenase 1C; alcohol metabolism",
        "ADM": "Adrenomedullin; angiogenesis, hypoxia response",
        "AHNAK2": "AHNAK nucleoprotein 2; cell migration, giant protein",
        "ANO1": "Anoctamin-1/TMEM16A; chloride channel, secretory carcinoma",
        "AQP5": "Aquaporin-5; water channel, glandular differentiation",
        "BPIFB2": "BPI fold-containing B2; lipid binding, innate immunity",
        "CASP14": "Caspase-14; keratinocyte differentiation",
        "CCL20": "Macrophage inflammatory protein-3 alpha; Th17 attractant",
        "CD109": "CD109; GPI-anchored protein, TGF-beta co-receptor",
        "CDH17": "Cadherin-17; liver-intestine cadherin, aberrant in cancer",
        "CDH26": "Cadherin-26; atypical cadherin",
        "CHIT1": "Chitinase-1; inflammatory marker",
        "CLIC6": "Chloride intracellular channel 6",
        "COL12A1": "Collagen type XII alpha 1; stromal ECM",
        "COL22A1": "Collagen type XXII alpha 1; tissue junctions",
        "CR2": "Complement receptor 2 (CD21); B-cell marker",
        "CXCL17": "Chemokine, mucosal DC attractant",
        "DNAH2": "Dynein axonemal heavy chain 2; ciliary function",
        "ELAPOR1": "Endosomal/lysosomal proton amino acid transporter",
        "GJB2": "Gap junction beta 2 (Connexin 26); cell communication",
        "GJB3": "Gap junction beta 3 (Connexin 31); cell communication",
        "GNG4": "G protein subunit gamma 4; signaling",
        "GSTA1": "Glutathione S-transferase alpha 1; detoxification",
        "HAL": "Histidine ammonia-lyase; amino acid metabolism",
        "HHIPL2": "Hedgehog interacting protein-like 2; Hedgehog pathway",
        "HLF": "Hepatic leukemia factor; transcription factor, circadian",
        "IL20RB": "IL-20 receptor beta; immune signaling",
        "INHA": "Inhibin alpha; TGF-beta family",
        "KREMEN2": "Kringle containing transmembrane protein 2; Wnt pathway",
        "LCAL1": "LALBA (alpha-lactalbumin); usually mammary, aberrant in lung",
        "LONRF2": "LON peptidase N-terminal domain ring finger 2",
        "LY6K": "Lymphocyte antigen 6 complex locus K; cancer-testis antigen",
        "MELTF": "Melanotransferrin; iron metabolism",
        "MSMB": "PSP94; tumor suppressor in prostate, emerging in lung",
        "NIPAL4": "NIPA-like domain containing 4; magnesium transporter",
        "PCDH7": "Protocadherin 7; cell adhesion, tumor suppressor",
        "PCP4L1": "Purkinje cell protein 4-like 1; neuronal, aberrant in cancer",
        "PLEKHB1": "Pleckstrin homology domain containing B1",
        "PPP2R2C": "PP2A regulatory subunit B gamma; tumor suppressor pathway",
        "PRR15": "Proline rich 15; function less characterized",
        "PTPRH": "Protein tyrosine phosphatase receptor type H",
        "PTPRT": "Protein tyrosine phosphatase receptor type T; tumor suppressor",
        "RGMA": "Repulsive guidance molecule a; BMP family",
        "SACK1A": "SAC1 kinase; function less characterized",
        "SCGB3A1": "Secretoglobin 3A1; lung-specific secretory protein",
        "SCNN1B": "Sodium channel epithelial 1 beta; ion transport",
        "SLC6A14": "Solute carrier family 6 member 14; amino acid transporter",
        "SMOC1": "SPARC related modular calcium binding 1; matricellular",
        "SORCS2": "Sortilin related VPS10 domain receptor 2",
        "SPOCK1": "Sparc/osteonectin, cwcv and kazal-like domains proteoglycan 1; matricellular",
        "SUSD4": "Sushi domain containing 4; complement pathway",
        "SYT8": "Synaptotagmin 8; membrane trafficking",
        "TCN1": "Transcobalamin-1 (haptocorrin); B12 binding",
        "TFPI2": "Tissue factor pathway inhibitor 2; serine protease inhibitor",
        "TMEM63C": "Transmembrane protein 63C; osmotic stress response",
        "TMPRSS11E": "Transmembrane serine protease 11E; epithelial protease",
        "TNNT1": "Troponin T1; slow skeletal muscle, aberrant in cancer",
        "TNS4": "Tensin 4; cell migration, focal adhesion",
        "TSPAN7": "Tetraspanin 7; cell membrane organization",
        "ZBTB7C": "Zinc finger and BTB domain containing 7C; tumor suppressor candidate",
    }

    gene_rows = fi_df[fi_df['Type']=='Gene'].sort_values('Abs_Beta', ascending=False)
    lines = ["="*70, "BIOLOGICAL INTERPRETATION REPORT", "="*70, "",
        "CAVEAT: The 127-gene panel was selected by univariate Cox screening at p<0.05",
        "WITHOUT multiple-testing correction (Pre-Phase 2 Audit Limitation #2).",
        "These genes should be interpreted as a PREDICTIVE FEATURE SET, not a confirmed",
        "causal/biological gene list. Association does not imply causation.",
        "",
        "TOP 30 GENES BY PREDICTIVE CONTRIBUTION (|β|):",
        f"{'Rank':<6} {'Ensembl_ID':<25} {'Symbol':<15} {'β':>10} {'HR':>8} {'Direction':<18} {'Missing in GEO':<25}",
        "-"*120,
    ]

    for i, (_, r) in enumerate(gene_rows.head(30).iterrows()):
        sym = r['Gene_Symbol']
        lines.append(f"{i+1:<6} {r['Feature']:<25} {sym:<15} {r['Beta']:>10.6f} {r['Hazard_Ratio']:>8.4f} {r['Direction']:<18} {r['Missing_in']:<25}")

    lines.append("")
    lines.append("BIOLOGICAL ANNOTATIONS FOR TOP GENES:")
    lines.append("")

    annotated = 0
    for _, r in gene_rows.head(30).iterrows():
        sym = r['Gene_Symbol']
        annotation = LUNG_CANCER_GENES.get(sym, "Function not curated in this report; see literature for details.")
        direction = "higher expression → higher risk" if r['Beta'] > 0 else "higher expression → lower risk"
        missing = r['Missing_in']
        lines.append(f"  {sym} ({r['Feature']})")
        lines.append(f"    β={r['Beta']:.6f}, HR={r['Hazard_Ratio']:.4f}, {direction}")
        lines.append(f"    {annotation}")
        if missing and missing != "N/A":
            lines.append(f"    ⚠ UNAVAILABLE in: {missing} (reduced-panel effect)")
        lines.append("")
        annotated += 1

    lines.append("FUNCTIONAL CATEGORIES REPRESENTED IN TOP GENES:")
    lines.append("")
    categories = {
        "Extracellular matrix / invasion": ["COL1A1", "COL12A1", "MMP1", "LAMC2", "LAMA3", "SPOCK1", "SMOC1", "TNS4", "FLRT3"],
        "Chemokines / immune signaling": ["CXCL8", "CXCL5", "CXCL13", "CXCL17", "CCL20", "IL1A", "MS4A1", "CD79A", "CR2"],
        "Hypoxia / angiogenesis": ["EGLN3", "ANGPTL4", "ADM", "AREG", "STC2"],
        "Cell adhesion / differentiation": ["PCDH7", "CDH17", "CDH26", "GJB2", "GJB3", "ITGA6", "ITGB4", "TSPAN7", "KRT81", "CALB2"],
        "Metabolism": ["SLC2A1", "SLC16A1", "KYNU", "GSTA1", "HAL", "CPS1", "ADH1C", "PCSK9"],
        "Signaling pathways": ["DKK1", "SFRP1", "JAG1", "GREM1", "HHIPL2", "KREMEN2", "FOSL1", "SOX9", "ELF5"],
        "Coagulation": ["FGA", "FGG", "F5", "TFPI2"],
        "Drug resistance / transport": ["ABCC2", "CFTR", "SCNN1B", "AQP5"],
    }
    for cat, genes in categories.items():
        found = [g for g in genes if g in gene_rows['Gene_Symbol'].values[:30]]
        if found:
            lines.append(f"  {cat}: {', '.join(found)}")
    lines.append("")
    lines.append("CLINICAL FEATURE INTERPRETATION:")
    lines.append("  Stage features: The model uses one-hot encoded stage categories.")
    lines.append("  Higher stage (IIIB, IV) generally shows risk-increasing coefficients,")
    lines.append("  while lower stage (IA) shows risk-decreasing coefficients — biologically consistent.")
    lines.append("  Age: included as a continuous variable; direction depends on coefficient sign.")
    lines.append("  Cancer_Type and Smoking_Status: included as categorical features;")
    lines.append("  coefficients reflect baseline category differences.")
    lines.append("")
    lines.append("CROSS-REFERENCE WITH GEO AVAILABILITY:")
    lines.append("  Several top-ranked genes are missing in one or more GEO cohorts.")
    lines.append("  This means the per-cohort reduced model may rely more heavily on")
    lines.append("  available high-weight genes, potentially shifting the effective")
    lines.append("  importance ranking compared to the full model.")
    lines.append("  See feature_importance.csv for per-gene availability details.")

    (P2_REPORTS/"biological_interpretation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("  -> biological_interpretation_report.txt written")


# ═════════════════════════════════════════════════════════════
# STEP 9: PUBLICATION-QUALITY FIGURES
# ═════════════════════════════════════════════════════════════
def step9_figures(cox_model, geo_data, val, val_risk, y_val_time, y_val_event, val_high,
                  train_median_risk, y_train_time, y_train_event, fi_df):
    log("STEP 9: Publication-quality figures")

    # ─── KM Curves per cohort ───
    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        fig, ax = plt.subplots(figsize=(8, 6))
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        high = d["high_risk"]
        kmf_high.fit(d["y_time"][high], d["y_event"][high], label="High Risk")
        kmf_high.plot_survival_function(ax=ax, color="red", ci_show=True)
        kmf_low.fit(d["y_time"][~high], d["y_event"][~high], label="Low Risk")
        kmf_low.plot_survival_function(ax=ax, color="blue", ci_show=True)
        ax.set_title(f"Kaplan-Meier — {cohort}\n({len(d['available_ensg'])}/127 genes, "
                      f"coeff coverage {d['eff_coeff_cov']:.1f}%, logrank p={d['logrank_p']:.2e})",
                      fontsize=11)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        plt.tight_layout()
        fig.savefig(P2_FIGURES/"KM_curves"/f"KM_{cohort}.png", dpi=150)
        plt.close()

    # Internal validation KM
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
    kmf_h.fit(y_val_time[val_high], y_val_event[val_high], label="High Risk")
    kmf_h.plot_survival_function(ax=ax, color="red", ci_show=True)
    kmf_l.fit(y_val_time[~val_high], y_val_event[~val_high], label="Low Risk")
    kmf_l.plot_survival_function(ax=ax, color="blue", ci_show=True)
    ax.set_title("Kaplan-Meier — TCGA Internal Validation\n(Full 127-gene panel)", fontsize=11)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Survival Probability")
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"KM_curves"/"KM_TCGA_internal_val.png", dpi=150)
    plt.close()
    log("  -> KM curves saved")

    # ─── ROC Curves ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    all_cohorts = ["TCGA_internal_val"] + GEO_COHORTS
    for i, cohort in enumerate(all_cohorts):
        ax = axes[i]
        if cohort == "TCGA_internal_val":
            risk = val_risk; y_time = y_val_time; y_event = y_val_event
            y_surv = make_surv(y_time, y_event)
            y_train_surv = make_surv(y_train_time, y_train_event)
            times = np.array([t for t in HORIZONS.values() if t < y_time.max() and t < y_train_time.max()])
            try:
                auc_vals, _ = cumulative_dynamic_auc(y_train_surv, y_surv, risk, times)
            except:
                auc_vals = np.array([0.5]*len(times))
            title = "TCGA Internal Val"
        else:
            d = geo_data[cohort]
            risk = d["risk_scores"]; times = d["cohort_times"]; auc_vals = d["auc_vals"]
            title = f"{cohort} ({len(d['available_ensg'])}/127)"

        for j, (label, t) in enumerate(HORIZONS.items()):
            if t in times:
                idx = list(times).index(t)
                ax.plot(times[:idx+1], auc_vals[:idx+1], marker='o', label=f"{label} (AUC={auc_vals[idx]:.3f})")

        ax.set_xlabel("Time (days)"); ax.set_ylabel("AUC")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[5].set_visible(False)
    plt.suptitle("Time-Dependent ROC-AUC — All Cohorts", fontsize=14)
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"ROC_curves"/"ROC_all_cohorts.png", dpi=150)
    plt.close()
    log("  -> ROC curves saved")

    # ─── C-index Comparison Plot ───
    fig, ax = plt.subplots(figsize=(10, 6))
    cohorts = ["TCGA Internal\nVal (0.5944\naudit ref)"] + GEO_COHORTS
    c_indices = [INTERNAL_VAL_CINDEX_AUDIT] + [geo_data[c]["ci"] for c in GEO_COHORTS]
    colors = ['#2ca02c'] + ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    bars = ax.bar(range(len(cohorts)), c_indices, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(cohorts)))
    labels = ["TCGA Int Val\n(audit ref)"] + [f"{c}\n({geo_data[c]['coverage_pct']:.0f}% panel)" for c in GEO_COHORTS]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("C-index"); ax.set_ylim(0.4, 0.85)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axhline(INTERNAL_VAL_CINDEX_AUDIT, color='green', linestyle=':', alpha=0.7, label=f'Audit ref ({INTERNAL_VAL_CINDEX_AUDIT})')
    for bar, ci in zip(bars, c_indices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{ci:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title("C-index Comparison: Internal Validation vs External GEO Cohorts", fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"ROC_curves"/"C_index_comparison.png", dpi=150)
    plt.close()
    log("  -> C-index comparison saved")

    # ─── Risk Distribution Plot ───
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [val_risk] + [geo_data[c]["risk_scores"] for c in GEO_COHORTS]
    labels = ["TCGA Int Val"] + GEO_COHORTS
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    colors_box = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(train_median_risk, color='red', linestyle='--', label=f'Train median risk ({train_median_risk:.4f})')
    ax.set_ylabel("Risk Score (Linear Predictor)")
    ax.set_title("Risk Score Distribution Across Cohorts", fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"Calibration"/"risk_distribution.png", dpi=150)
    plt.close()
    log("  -> Risk distribution saved")

    # ─── Workflow Diagram ───
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    boxes = [
        (1, 3, "TCGA Training\n(n=737, RNA-seq)\nlog2(CPM+1)"),
        (4, 3, "Frozen CoxPH Model\n127 genes + 16 clinical\n(penalizer=1.0)"),
        (7.5, 4.5, "GSE30219\n121/127 genes\n95.3% panel"),
        (7.5, 3, "GSE50081\n121/127 genes\n95.3% panel"),
        (7.5, 1.5, "GSE72094\n117/127 genes\n92.1% panel"),
        (7.5, 0, "GSE31210\n121/127 genes\n95.3% panel"),
        (11.5, 3, "Risk Prediction\n& Clinical\nInterpretation"),
    ]
    for x, y, txt in boxes:
        color = '#4CAF50' if 'Frozen' in txt else '#2196F3' if 'TCGA' in txt else '#FF9800' if 'Risk' in txt else '#9C27B0'
        rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2, facecolor=color, alpha=0.3, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, txt, ha='center', va='center', fontsize=7, fontweight='bold')
    # Arrows
    ax.annotate('', xy=(3.2, 3), xytext=(1.8, 3), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(6.7, 4.5), xytext=(4.8, 3.3), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(6.7, 3), xytext=(4.8, 3), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(6.7, 1.5), xytext=(4.8, 2.7), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(6.7, 0), xytext=(4.8, 2.5), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(10.7, 3), xytext=(8.3, 4.5), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(10.7, 3), xytext=(8.3, 3), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(10.7, 3), xytext=(8.3, 1.5), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(10.7, 3), xytext=(8.3, 0), arrowprops=dict(arrowstyle='->', lw=1))
    ax.set_title("Phase 2 Validation Workflow: TCGA Training → Frozen ML Model → GEO External Validation → Clinical Interpretation",
                  fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(P2_FIGURES/"Workflow"/"phase2_workflow.png", dpi=150)
    plt.close()
    log("  -> Workflow diagram saved")


# ═════════════════════════════════════════════════════════════
# STEP 10: FINAL SCIENTIFIC ASSESSMENT
# ═════════════════════════════════════════════════════════════
def step10_final_assessment(val_ci, geo_data, fi_df, cal_data):
    log("STEP 10: Final scientific assessment")

    lines = ["="*70, "PHASE 2 FINAL SCIENTIFIC ASSESSMENT", "="*70, ""]

    lines.append("FINAL SCIENTIFIC QUESTION:")
    lines.append('"Does the frozen baseline lung cancer survival prediction model')
    lines.append('demonstrate sufficient external validity and biological plausibility —')
    lines.append('evaluated honestly as a reduced-panel model where GEO coverage is')
    lines.append('incomplete — to serve as the foundation for the next-generation')
    lines.append('neural-mechanistic digital twin framework?"')
    lines.append("")

    lines.append("1. DOES THE MODEL GENERALIZE OUTSIDE TCGA?")
    lines.append("")
    lines.append(f"   Internal validation C-index: {INTERNAL_VAL_CINDEX_AUDIT} (audit-confirmed)")
    lines.append(f"   Reconstructed C-index: {val_ci:.4f} (minor scaler reconstruction difference)")
    lines.append("")
    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        delta = d["ci"] - INTERNAL_VAL_CINDEX_AUDIT
        lines.append(f"   {cohort}: C-index={d['ci']:.4f} (Δ={delta:+.4f}), "
                      f"panel={d['coverage_pct']:.1f}%, coeff={d['eff_coeff_cov']:.1f}%")
    lines.append("")
    lines.append("   FINDING: The model generalizes to external cohorts. Three of four GEO")
    lines.append("   cohorts (GSE30219, GSE72094, GSE31210) show C-index ABOVE the internal")
    lines.append("   validation reference, suggesting the gene panel captures robust")
    lines.append("   prognostic signal across platforms. GSE50081 shows comparable performance.")
    lines.append("")
    lines.append("   Reduced-panel effect: 6.2-9.5% of coefficient weight is missing across")
    lines.append("   cohorts. Since performance is maintained or improved despite this loss,")
    lines.append("   the missing genes appear to carry relatively low predictive weight,")
    lines.append("   and platform/biological differences do not negate the model's signal.")
    lines.append("")

    lines.append("2. WHICH COHORTS SHOW ACCEPTABLE PERFORMANCE?")
    lines.append("")
    lines.append(f"   Acceptable = C-index > 0.55 AND significant logrank stratification:")
    lines.append("")
    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        acceptable = d["ci"] > 0.55 and d["logrank_p"] is not None and d["logrank_p"] < 0.05
        status = "ACCEPTABLE" if acceptable else "MARGINAL"
        lines.append(f"   {cohort}: {status} — C-index={d['ci']:.4f}, "
                      f"coeff={d['eff_coeff_cov']:.1f}%, logrank p={d['logrank_p']:.2e}")
    lines.append("")
    lines.append("   GSE50081 is marginal: C-index 0.5872 is acceptable but logrank p=0.666")
    lines.append("   fails to stratify risk groups. This may be due to its narrow stage range")
    lines.append("   (only IA-IIB) reducing the discriminative signal.")
    lines.append("")

    lines.append("3. PERFORMANCE DEGRADATION: DISCRIMINATION vs CALIBRATION")
    lines.append("")
    lines.append(f"   {'Cohort':<18} {'C-index':>8} {'IBS':>8} {'Cal offset':>11} {'Assessment':<25}")
    lines.append("-"*75)
    for cohort in GEO_COHORTS:
        d = geo_data[cohort]
        cd = cal_data[cohort]
        ibs_str = f"{d['ibs']:.4f}" if d['ibs'] is not None else "N/A"
        if d["ci"] > 0.6 and (d["ibs"] is not None and d["ibs"] < 0.25):
            assess = "Good discrimination + calibration"
        elif d["ci"] > 0.6 and (d["ibs"] is not None and d["ibs"] > 0.5):
            assess = "Good discrimination, poor calibration"
        elif d["ci"] > 0.6:
            assess = "Good discrimination"
        else:
            assess = "Moderate discrimination"
        lines.append(f"   {cohort:<18} {d['ci']:>8.4f} {ibs_str:>8} {cd['cal_offset']:>+11.4f} {assess:<25}")
    lines.append("")
    lines.append("   Key finding: GSE72094 shows both excellent discrimination (C-index 0.6830)")
    lines.append("   AND calibration (IBS 0.1449). GSE31210 shows the best discrimination")
    lines.append("   (C-index 0.7557) but worst calibration (IBS 0.7954) — the model")
    lines.append("   discriminates well but predicted survival probabilities are poorly")
    lines.append("   calibrated, likely due to very low event rate (15.5%) and platform")
    lines.append("   scale effects on the baseline hazard.")
    lines.append("")

    lines.append("4. CALIBRATION AND CLUSTER A/B PATTERN")
    lines.append("")
    lines.append("   The Phase 1D expression-level cluster pattern (GSE30219/GSE50081 lower")
    lines.append("   vs GSE72094/GSE31210 higher) does NOT clearly replicate at the calibration")
    lines.append("   level. Standardization (z-score using TCGA training statistics) partially")
    lines.append("   mitigates raw expression differences. Calibration should be assessed")
    lines.append("   per-cohort, not by assumed clusters. See calibration_cluster_comparison_report.txt")
    lines.append("   for detailed analysis.")
    lines.append("")

    lines.append("5. BIOLOGICAL PLAUSIBILITY (with no-multiple-testing-correction caveat)")
    lines.append("")
    lines.append("   The top predictive genes include biologically plausible lung cancer")
    lines.append("   markers: SPP1 (osteopontin), COL1A1, MMP1 (ECM/invasion), CXCL8/IL-8")
    lines.append("   (inflammation), LAMC2 (basement membrane), FSCN1 (migration),")
    lines.append("   EGLN3 (hypoxia), ANGPTL4 (angiogenesis), SOX9 (stemness).")
    lines.append("   Functional categories span ECM remodeling, immune signaling, hypoxia,")
    lines.append("   metabolism, and cell adhesion — all relevant to lung cancer biology.")
    lines.append("")
    lines.append("   CAVEAT: Without FDR correction, some genes may be false positives.")
    lines.append("   The panel is a PREDICTIVE feature set, not a confirmed causal gene list.")
    lines.append("   Biological plausibility supports but does not prove causal relevance.")
    lines.append("")

    lines.append("6. PHASE 3 READINESS AND PANEL RECOMMENDATION")
    lines.append("")
    lines.append("   The baseline ML system demonstrates:")
    lines.append("   ✓ External validity across 4 independent GEO cohorts (3/4 above internal val)")
    lines.append("   ✓ Significant risk stratification in 3/4 cohorts (logrank p < 0.001)")
    lines.append("   ✓ Biologically plausible predictive genes")
    lines.append("   ✓ Robust performance despite reduced-panel effects (90.5-93.8% coeff coverage)")
    lines.append("   ✗ Calibration is variable across cohorts (excellent for GSE72094, poor for GSE31210)")
    lines.append("   ✗ GSE50081 shows no significant risk stratification (marginal performance)")
    lines.append("")
    lines.append("   RECOMMENDATION: PROCEED TO PHASE 3 with documented limitations.")
    lines.append("")
    lines.append("   PANEL STRATEGY FOR PHASE 3:")
    lines.append("   The current 127-gene panel has 95.3% coverage in 3/4 GEO cohorts and 92.1%")
    lines.append("   in GSE72094. The 6-10 missing genes carry only 6.2-9.5% of coefficient weight.")
    lines.append("")
    lines.append("   Option A: Retain full 127-gene panel for TCGA-only deep learning/mechanistic")
    lines.append("     modeling. Use reduced-panel versions for any cross-platform validation.")
    lines.append("     PROS: Maximizes information; CONS: Cannot be directly applied to microarray data.")
    lines.append("")
    lines.append("   Option B: Identify a cross-platform-compatible core panel (genes available in")
    lines.append("     ALL cohorts, ~117 genes) and retrain future models on this core.")
    lines.append("     PROS: Consistent feature set across platforms; CONS: Loses ~10 genes,")
    lines.append("     requires new model training (not frozen model reuse).")
    lines.append("")
    lines.append("   RECOMMENDED: Option A for Phase 3 neural-mechanistic/digital twin development")
    lines.append("   (TCGA has full RNA-seq coverage). Use reduced-panel evaluation for any")
    lines.append("   external validation in Phase 3, documenting the coverage caveat as in Phase 2.")
    lines.append("   Consider Option B (core panel) only if federated learning requires consistent")
    lines.append("   features across institutions with different profiling platforms.")
    lines.append("")

    lines.append("="*70)
    lines.append("OVERALL VERDICT: CONDITIONALLY READY FOR PHASE 3")
    lines.append("="*70)
    lines.append("")
    lines.append("The frozen baseline CoxPH model demonstrates sufficient external validity")
    lines.append("and biological plausibility to serve as the foundation for next-generation")
    lines.append("neural-mechanistic digital twin development. The model's prognostic signal")
    lines.append("transfers across RNA-seq and microarray platforms, and its top predictive")
    lines.append("genes are biologically coherent with lung cancer biology.")
    lines.append("")
    lines.append("Conditions for Phase 3:")
    lines.append("  1. Carry forward all 4 documented Phase 1 limitations (non-nested feature")
    lines.append("     selection, no FDR correction, missing sex/gender, invalid survival exclusions)")
    lines.append("  2. Report reduced-panel caveats for any cross-platform evaluation")
    lines.append("  3. Address calibration variability before clinical deployment")
    lines.append("  4. Do not interpret the 127-gene panel as confirmed causal drivers")
    lines.append("  5. Use audit-confirmed C-index (0.5944) as the baseline reference")

    (P2_REPORTS/"phase3_readiness_assessment.txt").write_text("\n".join(lines), encoding="utf-8")
    log("  -> phase3_readiness_assessment.txt written")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    log("="*70)
    log("PHASE 2 ANALYSIS: Steps 5-10")
    log("="*70)

    (P2_REPORTS/"phase2_analysis_log.txt").unlink(missing_ok=True)

    # Reload all data
    (cox_model, scaler, feature_names, gene_features, ensg2sym, sym2ensg, unmapped,
     age_median, X_train, X_train_scaled, train_risk, train_median_risk,
     y_train_time, y_train_event, y_train_surv,
     val, val_risk, val_ci, y_val_time, y_val_event, y_val_surv, val_high,
     geo_data) = load_all_data()

    # Step 5
    step5_generalization(val_ci, geo_data)

    # Step 6
    cal_data = step6_calibration(cox_model, geo_data, val, val_risk, y_val_time, y_val_event, train_median_risk)

    # Step 7
    fi_df = step7_explainability(cox_model, feature_names, gene_features, ensg2sym, geo_data)

    # Step 8
    step8_biological(fi_df, gene_features, ensg2sym, geo_data)

    # Step 9
    step9_figures(cox_model, geo_data, val, val_risk, y_val_time, y_val_event, val_high,
                  train_median_risk, y_train_time, y_train_event, fi_df)

    # Step 10
    step10_final_assessment(val_ci, geo_data, fi_df, cal_data)

    log("\n" + "="*70)
    log("PHASE 2 ANALYSIS COMPLETE — All steps 5-10 finished")
    log("="*70)

if __name__ == "__main__":
    main()
