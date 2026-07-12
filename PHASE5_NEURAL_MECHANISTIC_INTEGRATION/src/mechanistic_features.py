"""
Mechanistic Feature Extraction — Extract 16 scalar features using analytical Gompertz TTP.

Uses closed-form Gompertz TTP for natural history (instant, no ODE).
Treatment TTP approximated by analytical kill-rate scaling.
~100x faster than ODE-based extraction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys

P4_MODELS = Path(__file__).resolve().parent.parent.parent / "PHASE4_MECHANISTIC_MODELING" / "models"
sys.path.insert(0, str(P4_MODELS))

from treatment_response_model import TREATMENT_PARAMS
from personalization_function import STAGE_V0


MECH_FEATURE_NAMES = [
    "TTP_natural", "TTP_chemo", "TTP_immuno", "TTP_targeted",
    "TTP_benefit_chemo", "TTP_benefit_immuno", "TTP_benefit_targeted",
    "growth_rate_alpha", "carrying_capacity_log", "initial_volume_log",
    "k_immune_log", "mu_E", "rho", "immune_activity_score",
    "ttp_uncertainty_width", "mechanistic_risk_direction",
]


def _gompertz_ttp_analytical(alpha, V_max, V0, progression_factor=2.0):
    """
    Closed-form TTP for Gompertz growth: V(t) = V_max * exp(-exp(-alpha*(t-t0))).

    TTP = (1/alpha) * [ln(ln(V_max/V0)) - ln(ln(V_max/(V0*progression_factor)))]

    Returns t_span[1] (capped) if growth never reaches threshold.
    """
    if V0 <= 0 or V_max <= V0 * progression_factor or alpha <= 0:
        return 1825.0  # Never progresses within 5 years

    try:
        t0 = np.log(np.log(V_max / V0)) / alpha
        threshold = V0 * progression_factor
        t_ttp = t0 - np.log(np.log(V_max / threshold)) / alpha
        if np.isfinite(t_ttp) and t_ttp > 0:
            return min(t_ttp, 1825.0)
        return 1825.0
    except (ValueError, ZeroDivisionError, RuntimeWarning):
        return 1825.0


def _approximate_treatment_ttp(ttp_natural, params, treatment, alpha, V_max, V0):
    """
    Approximate treatment TTP by scaling natural history TTP.

    For cytotoxic/targeted therapy: TTP increases proportional to kill rate.
    For immunotherapy: TTP increases based on immune kill + checkpoint blockade.

    This is an analytical approximation — not a full ODE solve.
    Sufficient for feature extraction where relative ordering matters most.
    """
    if ttp_natural >= 1825.0:
        return 1825.0

    # Effective kill rate from treatment
    delta_c = params.get("delta_c", 0.028)
    delta_t = params.get("delta_t", 0.15)
    k_immune = params.get("k_immune", 1.22e-4)
    k_immuno = params.get("k_immuno", 2.5e-5)
    E0 = params.get("E0", 1.3e5)
    E_ref = params.get("E_ref", 1.3e5)

    # Growth rate parameter
    growth_rate = alpha * np.log(max(V_max / V0, 2.0))

    if treatment == "chemo":
        # Cisplatin: steady-state concentration ~ dose/Vd * k_el/(2*k_el) = dose/(2*Vd*k_el)
        # Approximate C_ss ~ 2.3 mg/L for standard dosing
        C_ss = 2.3
        kill_rate = delta_c * C_ss
        # TTP extends by factor: 1 + kill/growth
        extension = 1.0 + kill_rate / max(growth_rate, 1e-6)
        return min(ttp_natural * extension, 1825.0)

    elif treatment == "targeted":
        # Osimertinib: C_ss ~ 0.26 µM ~ 0.13 mg/L
        C_ss = 0.13
        kill_rate = delta_t * C_ss
        extension = 1.0 + kill_rate / max(growth_rate, 1e-6)
        return min(ttp_natural * extension, 1825.0)

    elif treatment == "immuno":
        # Pembrolizumab: immune checkpoint blockade
        # Boost immune kill + checkpoint blockade kill
        immune_ratio = E0 / max(E_ref, 1.0)
        immune_kill = k_immune * immune_ratio
        checkpoint_kill = k_immuno * 0.5  # Approximate average concentration effect
        total_kill = immune_kill + checkpoint_kill
        extension = 1.0 + total_kill / max(growth_rate, 1e-6)
        return min(ttp_natural * extension, 1825.0)

    return ttp_natural


def extract_mechanistic_features(
    patient_params_df: pd.DataFrame,
    population_priors: Dict,
    t_span: tuple = (0, 1825),
    t_eval: Optional[np.ndarray] = None,
    n_mc: int = 5,
    rng: Optional[np.random.Generator] = None,
    skip_mc: bool = False,
    n_workers: int = 0,
) -> pd.DataFrame:
    """Extract 16 scalar mechanistic features per patient. Analytical — very fast."""
    if rng is None:
        rng = np.random.default_rng(42)

    results = []

    for idx, row in patient_params_df.iterrows():
        pid = row["Patient_ID"]
        params = {}
        for col in patient_params_df.columns:
            if col != "Patient_ID":
                val = row[col]
                params[col] = float(val) if not np.isnan(val) else population_priors.get(col, 0.0)

        # 7 features directly from parameters
        alpha = params.get("alpha", 0.0008)
        V_max = params.get("V_max", 1e6)
        V0 = params.get("V0", 8000)
        k_immune = params.get("k_immune", 1.22e-4)
        mu_E = params.get("mu_E", 0.041)
        rho = params.get("rho", 0.02)
        e0 = params.get("E0", 1.3e5)
        pop_e0 = population_priors.get("E0", 1.3e5)
        immune_activity = e0 / pop_e0 if pop_e0 > 0 else 1.0

        # Analytical TTP (instant — no ODE solve)
        ttp_nat = _gompertz_ttp_analytical(alpha, V_max, V0)

        # Approximate treatment TTPs
        ttp_chemo = _approximate_treatment_ttp(ttp_nat, params, "chemo", alpha, V_max, V0)
        ttp_immuno = _approximate_treatment_ttp(ttp_nat, params, "immuno", alpha, V_max, V0)
        ttp_targeted = _approximate_treatment_ttp(ttp_nat, params, "targeted", alpha, V_max, V0)

        # MC uncertainty (analytical — just vary alpha and V_max)
        if skip_mc or n_mc <= 0:
            ttp_uncertainty_width = float("nan")
        else:
            ttp_samples = []
            for _ in range(n_mc):
                mc_alpha = rng.uniform(alpha * 0.7, alpha * 1.3)
                mc_V_max = rng.uniform(V_max * 0.7, V_max * 1.3)
                mc_V0 = rng.uniform(max(V0 * 0.7, 100), V0 * 1.3)
                ttp_mc = _gompertz_ttp_analytical(mc_alpha, mc_V_max, mc_V0)
                if ttp_mc < 1825.0:
                    ttp_samples.append(ttp_mc)
            if len(ttp_samples) > 3:
                ttp_uncertainty_width = np.percentile(ttp_samples, 97.5) - np.percentile(ttp_samples, 2.5)
            else:
                ttp_uncertainty_width = float("nan")

        features = {
            "Patient_ID": pid,
            "TTP_natural": ttp_nat,
            "TTP_chemo": ttp_chemo,
            "TTP_immuno": ttp_immuno,
            "TTP_targeted": ttp_targeted,
            "TTP_benefit_chemo": ttp_chemo - ttp_nat,
            "TTP_benefit_immuno": ttp_immuno - ttp_nat,
            "TTP_benefit_targeted": ttp_targeted - ttp_nat,
            "growth_rate_alpha": alpha,
            "carrying_capacity_log": np.log10(max(V_max, 1.0)),
            "initial_volume_log": np.log10(max(V0, 1.0)),
            "k_immune_log": np.log10(max(k_immune, 1e-10)),
            "mu_E": mu_E,
            "rho": rho,
            "immune_activity_score": immune_activity,
            "ttp_uncertainty_width": ttp_uncertainty_width,
            "mechanistic_risk_direction": 0,
        }
        results.append(features)

    feat_df = pd.DataFrame(results)

    if len(feat_df) > 0:
        pop_median_ttp = feat_df["TTP_natural"].median()
        feat_df["mechanistic_risk_direction"] = (
            feat_df["TTP_natural"] < pop_median_ttp).astype(int)
        for col in MECH_FEATURE_NAMES:
            if feat_df[col].isna().any():
                med = feat_df[col].median()
                if np.isnan(med):
                    med = 0.0
                feat_df[col] = feat_df[col].fillna(med)

    return feat_df


def extract_mechanistic_features_geo(
    geo_clinical_df: pd.DataFrame,
    population_priors: Dict,
    t_span: tuple = (0, 1825),
    t_eval: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    n_workers: int = 0,
) -> pd.DataFrame:
    """Extract mechanistic features for GEO patients using population-prior parameters."""
    if rng is None:
        rng = np.random.default_rng(42)

    params_list = []
    for _, row in geo_clinical_df.iterrows():
        params = dict(population_priors)
        stage = str(row.get("Stage", "IIB"))
        params["V0"] = STAGE_V0.get(stage, STAGE_V0.get("IIB", 8000.0))
        params_list.append({"Patient_ID": str(row["Patient_ID"]), **params})

    params_df = pd.DataFrame(params_list)
    return extract_mechanistic_features(
        params_df, population_priors, t_span=t_span, t_eval=t_eval,
        n_mc=0, rng=rng, skip_mc=True)
