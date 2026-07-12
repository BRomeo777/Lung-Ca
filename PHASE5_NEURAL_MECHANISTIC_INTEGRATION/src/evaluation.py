"""
Evaluation suite for Phase 5 hybrid model.

Comprehensive evaluation comparing CoxPH, DeepSurv standalone, and Hybrid model.
Metrics: C-index, IBS, time-dependent AUC, calibration, NRI/IDI, consistency.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats
from typing import Dict, Tuple, List, Optional

from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def bootstrap_cindex(y_event, y_time, risk, n_bootstrap=500, rng=None):
    """Bootstrap 95% CI for C-index."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_event)
    cis = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            ci = concordance_index_censored(
                y_event[idx].astype(bool), y_time[idx], risk[idx])[0]
            if np.isfinite(ci):
                cis.append(ci)
        except Exception:
            continue
    if len(cis) < 10:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(cis, [2.5, 97.5])
    return float(np.mean(cis)), float(lo), float(hi)


def compute_cindex(y_event, y_time, risk):
    """Compute C-index. Higher risk = shorter survival (standard convention)."""
    return concordance_index_censored(y_event.astype(bool), y_time, risk)[0]


def compute_ibs(y_event, y_time, risk, train_y_event, train_y_time,
                horizons_days=(365, 1095, 1825)):
    """
    Integrated Brier Score using IPCW from training data.
    """
    try:
        train_surv = Surv.from_arrays(
            event=train_y_event.astype(bool), time=train_y_time)
        eval_surv = Surv.from_arrays(
            event=y_event.astype(bool), time=y_time)

        # Estimate survival function from risk scores using a simple model
        # Higher risk = lower survival probability
        # Use empirical estimation
        times = np.array(horizons_days, dtype=float)

        # Compute survival probabilities at each horizon
        # Use a monotonic transform: S(t) = exp(-exp(risk) * t / median_time)
        median_time = np.median(train_y_time[train_y_event == 1]) if np.any(train_y_event == 1) else 365.0
        surv_probs = np.zeros((len(y_time), len(times)))
        for j, t in enumerate(times):
            surv_probs[:, j] = np.exp(-np.exp(risk - np.median(risk)) * t / median_time)

        ibs = integrated_brier_score(train_surv, eval_surv, surv_probs, times)
        return float(ibs)
    except Exception as e:
        return float("nan")


def compute_time_dependent_auc(y_event, y_time, risk, train_y_event, train_y_time,
                                horizons_days=(365, 1095, 1825)):
    """Time-dependent AUC at 1, 3, 5 years."""
    try:
        train_surv = Surv.from_arrays(
            event=train_y_event.astype(bool), time=train_y_time)
        auc, times = cumulative_dynamic_auc(
            train_surv, y_event.astype(bool), y_time, risk, times=horizons_days)
        return {f"{h//365}yr": float(a) for h, a in zip(horizons_days, auc)}
    except Exception:
        return {f"{h//365}yr": float("nan") for h in horizons_days}


def compute_nri(
    risk_new: np.ndarray,
    risk_ref: np.ndarray,
    y_event: np.ndarray,
    y_time: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Net Reclassification Improvement of new model vs reference model.
    """
    if threshold is None:
        threshold = np.median(risk_ref)

    # Classify patients as high/low risk
    high_new = risk_new > threshold
    high_ref = risk_ref > threshold

    events = y_event.astype(bool)
    non_events = ~events

    # NRI among events: proportion correctly reclassified up
    n_events = events.sum()
    n_non_events = non_events.sum()

    if n_events == 0 or n_non_events == 0:
        return {"NRI": float("nan"), "NRI_events": float("nan"), "NRI_non_events": float("nan")}

    # Events should be classified as high risk
    reclassified_up_event = (high_new & ~high_ref & events).sum()
    reclassified_down_event = (~high_new & high_ref & events).sum()
    nri_events = (reclassified_up_event - reclassified_down_event) / n_events

    # Non-events should be classified as low risk
    reclassified_up_non = (high_new & ~high_ref & non_events).sum()
    reclassified_down_non = (~high_new & high_ref & non_events).sum()
    nri_non_events = (reclassified_down_non - reclassified_up_non) / n_non_events

    nri = nri_events + nri_non_events

    return {
        "NRI": float(nri),
        "NRI_events": float(nri_events),
        "NRI_non_events": float(nri_non_events),
    }


def compute_idi(
    risk_new: np.ndarray,
    risk_ref: np.ndarray,
    y_event: np.ndarray,
) -> Dict:
    """
    Integrated Discrimination Improvement.
    """
    events = y_event.astype(bool)
    non_events = ~events

    if events.sum() == 0 or non_events.sum() == 0:
        return {"IDI": float("nan"), "IDI_events": float("nan"), "IDI_non_events": float("nan")}

    mean_diff_events = risk_new[events].mean() - risk_ref[events].mean()
    mean_diff_non_events = risk_new[non_events].mean() - risk_ref[non_events].mean()
    idi = mean_diff_events - mean_diff_non_events

    return {
        "IDI": float(idi),
        "IDI_events": float(mean_diff_events),
        "IDI_non_events": float(mean_diff_non_events),
    }


def evaluate_model(
    risk_scores: np.ndarray,
    y_event: np.ndarray,
    y_time: np.ndarray,
    train_y_event: np.ndarray,
    train_y_time: np.ndarray,
    cohort_name: str,
    risk_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Comprehensive evaluation of a model on a cohort.
    """
    ci = compute_cindex(y_event, y_time, risk_scores)
    ci_mean, ci_lo, ci_hi = bootstrap_cindex(y_event, y_time, risk_scores, rng=rng)
    ibs = compute_ibs(y_event, y_time, risk_scores, train_y_event, train_y_time)
    aucs = compute_time_dependent_auc(y_event, y_time, risk_scores, train_y_event, train_y_time)

    # KM stratification
    if risk_threshold is None:
        risk_threshold = np.median(risk_scores)
    high_risk = risk_scores > risk_threshold
    if high_risk.sum() > 0 and (~high_risk).sum() > 0:
        try:
            lr = logrank_test(
                y_time[high_risk], y_time[~high_risk],
                y_event[high_risk], y_event[~high_risk])
            logrank_p = lr.p_value
        except Exception:
            logrank_p = float("nan")
    else:
        logrank_p = float("nan")

    return {
        "cohort": cohort_name,
        "n": len(y_event),
        "events": int(y_event.sum()),
        "cindex": ci,
        "cindex_lo": ci_lo,
        "cindex_hi": ci_hi,
        "ibs": ibs,
        "auc_1yr": aucs.get("1yr", float("nan")),
        "auc_3yr": aucs.get("3yr", float("nan")),
        "auc_5yr": aucs.get("5yr", float("nan")),
        "logrank_p": logrank_p,
    }


def evaluate_hybrid_vs_baselines(
    hybrid_risk: np.ndarray,
    deepsurv_risk: np.ndarray,
    coxph_risk: np.ndarray,
    y_event: np.ndarray,
    y_time: np.ndarray,
    train_y_event: np.ndarray,
    train_y_time: np.ndarray,
    cohort_name: str,
    train_risk_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Three-way comparison: CoxPH vs DeepSurv vs Hybrid.
    """
    results = {}
    for name, risk in [("CoxPH", coxph_risk), ("DeepSurv", deepsurv_risk), ("Hybrid", hybrid_risk)]:
        eval_result = evaluate_model(
            risk, y_event, y_time, train_y_event, train_y_time,
            cohort_name, risk_threshold=train_risk_threshold, rng=rng)
        results[name] = eval_result

    # NRI/IDI of hybrid vs CoxPH
    nri = compute_nri(hybrid_risk, coxph_risk, y_event, y_time)
    idi = compute_idi(hybrid_risk, coxph_risk, y_event)
    results["NRI_vs_CoxPH"] = nri
    results["IDI_vs_CoxPH"] = idi

    # NRI/IDI of hybrid vs DeepSurv
    nri_ds = compute_nri(hybrid_risk, deepsurv_risk, y_event, y_time)
    idi_ds = compute_idi(hybrid_risk, deepsurv_risk, y_event)
    results["NRI_vs_DeepSurv"] = nri_ds
    results["IDI_vs_DeepSurv"] = idi_ds

    return results
