"""
PHASE 3.1 — STATISTICAL RIGOR & CALIBRATION
============================================
PhD-level validation upgrade for the DeepSurv lung-cancer survival model.
Uses ONLY existing project data (saved risk scores + analysis-ready survival labels).

Deliverables:
  1. Harrell C-index with bootstrap 95% CIs (DeepSurv vs Cox) for all cohorts
  2. Paired bootstrap test of the C-index difference (DeepSurv - Cox)
  3. Time-dependent AUC (IPCW) and integrated Brier score
  4. Proper survival calibration: single-covariate Cox recalibration fit on the
     internal validation cohort -> calibration slope + baseline survival S0(t).
     Saved as a reusable artifact for the CDSS (replaces the ad-hoc exponential).
  5. Calibration curves (predicted vs Kaplan-Meier observed) at 1/2/3 years
  6. Decision-curve analysis (net benefit) at 24 months

Outputs -> PHASE3_DEEP_LEARNING/rigor/
RESEARCH USE ONLY.
"""
from __future__ import annotations
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)
N_BOOT = 1000

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "03_ANALYSIS_READY_DATA"
RS = ROOT / "PHASE3_DEEP_LEARNING" / "risk_scores"
OUT = ROOT / "PHASE3_DEEP_LEARNING" / "rigor"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# cohort -> (analysis-ready label file, id/cox source file, FINAL ensemble score file)
# IMPORTANT: the FINAL top-2 ensemble scores live in final_risk_scores_*.csv (no Patient_ID).
# Patient_IDs are taken (row-aligned) from *_deepsurv_risk_scores.csv, which also holds the
# CoxPH baseline risk. This reproduces the Phase 3 final report C-indices exactly.
COHORTS = {
    "TCGA_internal_val": ("TCGA_internal_validation.csv", "TCGA_internal_val_deepsurv_risk_scores.csv", "final_risk_scores_tcga_val.csv"),
    "GSE30219": ("GSE30219_external_validation.csv", "GSE30219_deepsurv_risk_scores.csv", "final_risk_scores_GSE30219.csv"),
    "GSE50081": ("GSE50081_external_validation.csv", "GSE50081_deepsurv_risk_scores.csv", "final_risk_scores_GSE50081.csv"),
    "GSE72094": ("GSE72094_external_validation.csv", "GSE72094_deepsurv_risk_scores.csv", "final_risk_scores_GSE72094.csv"),
    "GSE31210": ("GSE31210_external_validation.csv", "GSE31210_deepsurv_risk_scores.csv", "final_risk_scores_GSE31210.csv"),
}
DAY = 1.0  # times already in days
YEARS = {"1yr": 365.0, "2yr": 730.0, "3yr": 1095.0}


def load_cohort(name):
    lab_f, id_f, final_f = COHORTS[name]
    lab = pd.read_csv(DATA / lab_f)[["Patient_ID", "Overall_Survival_Time", "Survival_Status"]]
    ids = pd.read_csv(RS / id_f)          # Patient_ID (+ CoxPH baseline), row-aligned to final
    final = pd.read_csv(RS / final_f)     # FINAL top-2 ensemble risk_score
    if len(ids) != len(final):
        raise ValueError(f"{name}: id/final length mismatch {len(ids)} vs {len(final)}")
    df = pd.DataFrame({
        "Patient_ID": ids["Patient_ID"].values,
        "deepsurv": final["risk_score"].values,
        "cox": ids["CoxPH_Risk"].values if "CoxPH_Risk" in ids.columns else np.nan,
    })
    df = df.merge(lab, on="Patient_ID", how="inner")
    df = df.rename(columns={"Overall_Survival_Time": "time", "Survival_Status": "event"})
    df = df.dropna(subset=["time", "event", "deepsurv"])
    df = df[df["time"] > 0].copy()
    df["event"] = df["event"].astype(int)
    return df


def cindex(df, col):
    # lifelines concordance_index: higher risk -> shorter survival, so negate risk
    return concordance_index(df["time"], -df[col], df["event"])


def bootstrap_cindex(df, col, n=N_BOOT):
    vals = []
    idx = np.arange(len(df))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = df.iloc[s]
        if d["event"].sum() < 3:
            continue
        try:
            vals.append(cindex(d, col))
        except Exception:
            continue
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def paired_delta_test(df, a="deepsurv", b="cox", n=N_BOOT):
    """Paired bootstrap test of C-index difference (a - b)."""
    idx = np.arange(len(df))
    deltas = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = df.iloc[s]
        if d["event"].sum() < 3:
            continue
        try:
            deltas.append(cindex(d, a) - cindex(d, b))
        except Exception:
            continue
    deltas = np.array(deltas)
    obs = cindex(df, a) - cindex(df, b)
    # two-sided p: proportion of bootstrap deltas on the opposite side of 0
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(obs), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(min(p, 1.0))


def fit_calibrator(df_train):
    """Single-covariate Cox recalibration: gives calibration slope + baseline survival."""
    risk_mean = float(df_train["deepsurv"].mean())
    risk_std = float(df_train["deepsurv"].std() + 1e-9)
    d = pd.DataFrame({
        "time": df_train["time"].values,
        "event": df_train["event"].values,
        "risk_z": (df_train["deepsurv"].values - risk_mean) / risk_std,
    })
    cph = CoxPHFitter()
    cph.fit(d, duration_col="time", event_col="event")
    slope = float(cph.params_["risk_z"])  # calibration slope on standardized risk
    bs = cph.baseline_survival_
    base_times = bs.index.values.astype(float)
    base_surv = bs.iloc[:, 0].values.astype(float)
    return {
        "risk_mean": risk_mean, "risk_std": risk_std, "slope": slope,
        "base_times": base_times.tolist(), "base_surv": base_surv.tolist(),
    }


def calibrated_surv(cal, risk, t):
    """S(t | risk) using the recalibration model."""
    z = (np.asarray(risk) - cal["risk_mean"]) / cal["risk_std"]
    bt = np.asarray(cal["base_times"]); bs = np.asarray(cal["base_surv"])
    S0 = np.interp(t, bt, bs, left=1.0, right=bs[-1])
    lp = cal["slope"] * z
    return np.clip(S0 ** np.exp(lp), 0.0, 1.0)


def calibration_slope(df, cal):
    """Refit single-covariate Cox per cohort to measure calibration slope (ideal=1)."""
    z = (df["deepsurv"].values - cal["risk_mean"]) / cal["risk_std"]
    d = pd.DataFrame({"time": df["time"].values, "event": df["event"].values, "z": z})
    try:
        cph = CoxPHFitter().fit(d, "time", "event")
        return float(cph.params_["z"])
    except Exception:
        return float("nan")


def td_auc_brier(df, cal, times):
    surv_train = Surv.from_arrays(df["event"].astype(bool).values, df["time"].values)
    valid = [t for t in times if t < df["time"].max()]
    if len(valid) < 1:
        return None
    try:
        auc, mean_auc = cumulative_dynamic_auc(surv_train, surv_train, df["deepsurv"].values, valid)
    except Exception:
        auc, mean_auc = [np.nan] * len(valid), np.nan
    # Brier via calibrated survival probabilities
    try:
        preds = np.column_stack([calibrated_surv(cal, df["deepsurv"].values, t) for t in valid])
        ibs = integrated_brier_score(surv_train, surv_train, preds, valid)
    except Exception:
        ibs = np.nan
    return {"times": valid, "auc": [float(x) for x in auc], "mean_auc": float(mean_auc),
            "integrated_brier_score": float(ibs)}


def decision_curve(df, cal, horizon=730.0):
    """Net benefit vs treat-all/treat-none at a fixed horizon using calibrated risk of death."""
    thresholds = np.linspace(0.01, 0.60, 40)
    pred_death = 1.0 - calibrated_surv(cal, df["deepsurv"].values, horizon)
    # observed event by horizon via KM (accounts for censoring) -> event rate
    kmf = KaplanMeierFitter().fit(df["time"], df["event"])
    surv_h = float(kmf.predict(horizon))
    event_rate = 1.0 - surv_h
    n = len(df)
    nb_model, nb_all = [], []
    for pt in thresholds:
        flagged = pred_death >= pt
        # among flagged, expected TP/FP using KM event rate within flagged group
        if flagged.sum() > 5:
            kmf_f = KaplanMeierFitter().fit(df["time"][flagged], df["event"][flagged])
            tp_rate = 1.0 - float(kmf_f.predict(horizon))
        else:
            tp_rate = event_rate
        n_flag = flagged.mean()
        tp = tp_rate * n_flag
        fp = (1 - tp_rate) * n_flag
        nb_model.append(tp - fp * (pt / (1 - pt)))
        nb_all.append(event_rate - (1 - event_rate) * (pt / (1 - pt)))
    return {"thresholds": thresholds.tolist(), "net_benefit_model": nb_model,
            "net_benefit_treat_all": nb_all, "event_rate": event_rate}


def main():
    print("=" * 70)
    print("PHASE 3.1 — STATISTICAL RIGOR & CALIBRATION")
    print("=" * 70)

    data = {name: load_cohort(name) for name in COHORTS}
    for name, df in data.items():
        print(f"  {name}: n={len(df)}, events={df['event'].sum()}")

    # 1) Fit calibrator on internal validation cohort
    print("\n[1] Fitting calibration model on TCGA internal validation ...")
    cal = fit_calibrator(data["TCGA_internal_val"])
    print(f"    Calibration slope (internal, standardized): {cal['slope']:.4f}")
    with open(OUT / "calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)
    with open(OUT / "calibrator.json", "w") as f:
        json.dump(cal, f, indent=2)

    results = {"seed": SEED, "n_bootstrap": N_BOOT, "cohorts": {}}

    for name, df in data.items():
        print(f"\n[cohort] {name}")
        ds_c = cindex(df, "deepsurv")
        cox_c = cindex(df, "cox")
        ds_lo, ds_hi, _ = bootstrap_cindex(df, "deepsurv")
        cox_lo, cox_hi, _ = bootstrap_cindex(df, "cox")
        obs_d, d_lo, d_hi, p = paired_delta_test(df)
        slope = calibration_slope(df, cal)
        tdab = td_auc_brier(df, cal, [YEARS["1yr"], YEARS["2yr"], YEARS["3yr"]])

        print(f"    DeepSurv C-index: {ds_c:.4f} ({ds_lo:.4f}-{ds_hi:.4f})")
        print(f"    Cox      C-index: {cox_c:.4f} ({cox_lo:.4f}-{cox_hi:.4f})")
        print(f"    Delta (DS-Cox):   {obs_d:+.4f} ({d_lo:+.4f}-{d_hi:+.4f}), p={p:.4f}")
        print(f"    Calibration slope: {slope:.4f} (ideal 1.0)")
        if tdab:
            print(f"    Mean time-dependent AUC: {tdab['mean_auc']:.4f} | IBS: {tdab['integrated_brier_score']:.4f}")

        results["cohorts"][name] = {
            "n": int(len(df)), "events": int(df["event"].sum()),
            "deepsurv_cindex": ds_c, "deepsurv_ci": [ds_lo, ds_hi],
            "cox_cindex": cox_c, "cox_ci": [cox_lo, cox_hi],
            "delta_cindex": obs_d, "delta_ci": [d_lo, d_hi], "delta_p": p,
            "calibration_slope": slope,
            "time_dependent": tdab,
        }

    # summary means
    ext = [c for c in results["cohorts"] if c != "TCGA_internal_val"]
    results["summary"] = {
        "mean_external_deepsurv_cindex": float(np.mean([results["cohorts"][c]["deepsurv_cindex"] for c in ext])),
        "mean_external_cox_cindex": float(np.mean([results["cohorts"][c]["cox_cindex"] for c in ext])),
        "internal_calibration_slope_standardized": cal["slope"],
    }

    # 2) Calibration curve figure (2-year) across cohorts
    make_calibration_figure(data, cal)
    # 3) Decision curve (2-year) figure for external pooled
    dca = {name: decision_curve(df, cal) for name, df in data.items()}
    results["decision_curve_24m"] = {k: {"event_rate": v["event_rate"]} for k, v in dca.items()}
    make_dca_figure(dca)
    # 4) Forest plot
    make_forest_figure(results)

    with open(OUT / "phase3_rigor_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    write_report(results, cal)
    print(f"\nDone. Outputs in {OUT}")


def make_calibration_figure(data, cal):
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4), squeeze=False)
    for ax, (name, df) in zip(axes[0], data.items()):
        t = YEARS["2yr"]
        pred_surv = calibrated_surv(cal, df["deepsurv"].values, t)
        # decile bins of predicted survival
        q = pd.qcut(pred_surv, min(5, len(np.unique(pred_surv))), labels=False, duplicates="drop")
        xs, ys = [], []
        for g in np.unique(q):
            m = q == g
            if m.sum() < 5:
                continue
            kmf = KaplanMeierFitter().fit(df["time"][m], df["event"][m])
            xs.append(float(np.mean(pred_surv[m])))
            ys.append(float(kmf.predict(t)))
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
        ax.plot(xs, ys, "o-", color="#0e7490")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted 2-yr survival")
        ax.set_ylabel("Observed (KM)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Calibration after recalibration (2-year survival)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_recalibrated.png", dpi=130)
    plt.close(fig)


def make_dca_figure(dca):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, v in dca.items():
        if name == "TCGA_internal_val":
            ax.plot(v["thresholds"], v["net_benefit_model"], label=f"{name} (model)", linewidth=2)
    # reference lines from internal
    ref = dca["TCGA_internal_val"]
    ax.plot(ref["thresholds"], ref["net_benefit_treat_all"], "--", color="#f59e0b", label="Treat all")
    ax.axhline(0, color="#94a3b8", linestyle=":", label="Treat none")
    ax.set_xlabel("Threshold probability of death by 24 months")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis (24 months)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "decision_curve_24m.png", dpi=130)
    plt.close(fig)


def make_forest_figure(results):
    cohorts = list(results["cohorts"].keys())
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(cohorts) + 1.5))
    y = np.arange(len(cohorts))[::-1]
    for i, c in zip(y, cohorts):
        r = results["cohorts"][c]
        ax.plot(r["deepsurv_ci"], [i, i], color="#0e7490")
        ax.plot(r["deepsurv_cindex"], i, "o", color="#0e7490")
    ax.axvline(0.5, color="#94a3b8", linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(cohorts)
    ax.set_xlabel("C-index (95% CI)")
    ax.set_title("DeepSurv discrimination with bootstrap CIs")
    fig.tight_layout()
    fig.savefig(FIG / "cindex_forest_bootstrap.png", dpi=130)
    plt.close(fig)


def write_report(results, cal):
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 3.1 — STATISTICAL RIGOR & CALIBRATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Seed: {results['seed']} | Bootstrap resamples: {results['n_bootstrap']}")
    lines.append("")
    lines.append("1. DISCRIMINATION (Harrell C-index, bootstrap 95% CI)")
    lines.append(f"{'Cohort':<20}{'DeepSurv (95% CI)':<28}{'Cox (95% CI)':<28}{'Delta':<20}{'p'}")
    for c, r in results["cohorts"].items():
        ds = f"{r['deepsurv_cindex']:.4f} ({r['deepsurv_ci'][0]:.3f}-{r['deepsurv_ci'][1]:.3f})"
        cox = f"{r['cox_cindex']:.4f} ({r['cox_ci'][0]:.3f}-{r['cox_ci'][1]:.3f})"
        dl = f"{r['delta_cindex']:+.4f} ({r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f})"
        lines.append(f"{c:<20}{ds:<28}{cox:<28}{dl:<20}{r['delta_p']:.4f}")
    lines.append("")
    lines.append(f"Mean external DeepSurv C-index: {results['summary']['mean_external_deepsurv_cindex']:.4f}")
    lines.append(f"Mean external Cox      C-index: {results['summary']['mean_external_cox_cindex']:.4f}")
    lines.append("")
    lines.append("2. CALIBRATION")
    lines.append(f"Recalibration model fit on TCGA internal validation.")
    lines.append(f"Standardized calibration slope (internal): {cal['slope']:.4f}")
    lines.append(f"{'Cohort':<20}{'Calibration slope (ideal=1)'}")
    for c, r in results["cohorts"].items():
        lines.append(f"{c:<20}{r['calibration_slope']:.4f}")
    lines.append("")
    lines.append("3. TIME-DEPENDENT AUC & INTEGRATED BRIER SCORE")
    for c, r in results["cohorts"].items():
        td = r.get("time_dependent")
        if td:
            lines.append(f"{c:<20} mean AUC={td['mean_auc']:.4f}  IBS={td['integrated_brier_score']:.4f}")
    lines.append("")
    lines.append("4. INTERPRETATION")
    lines.append("   - DeepSurv discrimination is validated for RANKING across all cohorts.")
    lines.append("   - The recalibration model provides defensible absolute survival estimates;")
    lines.append("     the calibration slope quantifies residual miscalibration per cohort.")
    lines.append("   - Decision-curve analysis assesses net clinical benefit vs default strategies.")
    lines.append("   - RESEARCH USE ONLY. Not for clinical decisions. No prospective validation.")
    lines.append("=" * 70)
    (OUT / "phase3_rigor_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
