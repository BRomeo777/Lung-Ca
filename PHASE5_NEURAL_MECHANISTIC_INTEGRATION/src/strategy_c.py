"""
Strategy C: Prediction-Level Fusion (Calibrated Ensemble).

Combines DeepSurv risk scores and mechanistic TTP predictions into a unified
risk estimate using multiple combination methods. No retraining of either
component. Preserves both frozen models.
"""
from __future__ import annotations
import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional


def normalize_ttp_to_risk(
    ttp_values: np.ndarray,
    method: str = "rank",
    reference_ttp: Optional[float] = None,
) -> np.ndarray:
    """
    Convert mechanistic TTP (days) to risk score space.

    Methods:
        'rank': rank-based normalization, shorter TTP = higher risk, [0, 1]
        'inverse_log': -log(TTP / reference_TTP), biologically motivated
    """
    ttp_safe = np.where(ttp_values > 0, ttp_values, 1.0)

    if method == "rank":
        ranks = scipy.stats.rankdata(-ttp_safe)
        return (ranks - 1) / max(len(ranks) - 1, 1)

    elif method == "inverse_log":
        if reference_ttp is None:
            reference_ttp = np.median(ttp_safe)
        return -np.log(ttp_safe / max(reference_ttp, 1.0) + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def fixed_weight_fusion(
    deepsurv_risk: np.ndarray,
    mech_risk: np.ndarray,
    w_neural: float = 0.6,
    w_mech: float = 0.4,
) -> np.ndarray:
    """Fixed-weight linear combination of neural and mechanistic risk scores."""
    return w_neural * deepsurv_risk + w_mech * mech_risk


def learned_weight_fusion(
    deepsurv_risk_train: np.ndarray,
    mech_risk_train: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
) -> Tuple[float, Dict]:
    """
    Learn optimal fusion weight via 1D optimization maximizing C-index.

    Finds w_neural such that w_neural * neural + (1-w_neural) * mech
    maximizes concordance.

    Returns
    -------
    optimal_w_neural : float
    cv_performance : dict with 'cindex'
    """
    from sksurv.metrics import concordance_index_censored

    def neg_cindex(w):
        combined = w * deepsurv_risk_train + (1 - w) * mech_risk_train
        ci = concordance_index_censored(y_event.astype(bool), y_time, combined)[0]
        return -ci

    result = minimize_scalar(neg_cindex, bounds=(0.0, 1.0), method="bounded")
    return result.x, {"cindex": -result.fun}


def consistency_weighted_fusion(
    deepsurv_risk: np.ndarray,
    mech_risk: np.ndarray,
    consistency_violations: np.ndarray,
    max_mech_weight: float = 0.4,
) -> np.ndarray:
    """
    Weight mechanistic contribution by per-patient consistency confidence.

    For patients where neural and mechanistic agree (low violation):
        give mechanistic component higher weight.
    For patients where they disagree (high violation):
        down-weight mechanistic, trust neural more.
    """
    violations_safe = np.where(consistency_violations > 0, consistency_violations, 0.0)
    mech_confidence = 1.0 / (1.0 + violations_safe)
    w_mech = max_mech_weight * mech_confidence
    w_neural = 1.0 - w_mech
    return w_neural * deepsurv_risk + w_mech * mech_risk


def compute_consistency_violations(
    deepsurv_risk: np.ndarray,
    mech_risk: np.ndarray,
) -> np.ndarray:
    """
    Compute per-patient consistency violation score.

    Violation = absolute difference in rank positions between neural and mechanistic.
    Higher = larger disagreement.
    """
    neural_ranks = scipy.stats.rankdata(deepsurv_risk)
    mech_ranks = scipy.stats.rankdata(mech_risk)
    n = len(deepsurv_risk)
    if n <= 1:
        return np.zeros_like(deepsurv_risk)
    return np.abs(neural_ranks - mech_ranks) / n


class PredictionLevelFusion:
    """
    Strategy C: Calibrated Prediction-Level Fusion.

    Combines DeepSurv risk scores and mechanistic TTP predictions
    into a unified risk estimate.
    """

    def __init__(self, normalization_method: str = "rank"):
        self.normalization_method = normalization_method
        self.optimal_w_neural = 0.6
        self.is_fitted = False

    def fit(
        self,
        deepsurv_risk_train: np.ndarray,
        ttp_train: np.ndarray,
        y_time: np.ndarray,
        y_event: np.ndarray,
    ) -> "PredictionLevelFusion":
        """Learn optimal fusion weight on training data."""
        mech_risk_train = normalize_ttp_to_risk(
            ttp_train, method=self.normalization_method)

        # Also normalize DeepSurv to [0,1] for consistent scale
        ds_norm = scipy.stats.rankdata(deepsurv_risk_train)
        ds_norm = (ds_norm - 1) / max(len(ds_norm) - 1, 1)

        w_opt, perf = learned_weight_fusion(
            ds_norm, mech_risk_train, y_time, y_event)
        self.optimal_w_neural = w_opt
        self.fusion_performance = perf
        self.is_fitted = True
        return self

    def predict(
        self,
        deepsurv_risk: np.ndarray,
        ttp_values: np.ndarray,
        use_consistency_weighting: bool = False,
    ) -> np.ndarray:
        """
        Generate fused risk scores.

        Returns
        -------
        hybrid_risk : np.ndarray — fused risk scores
        """
        if not self.is_fitted:
            w = 0.6
        else:
            w = self.optimal_w_neural

        mech_risk = normalize_ttp_to_risk(
            ttp_values, method=self.normalization_method)

        # Normalize DeepSurv to [0,1]
        ds_norm = scipy.stats.rankdata(deepsurv_risk)
        ds_norm = (ds_norm - 1) / max(len(ds_norm) - 1, 1)

        if use_consistency_weighting:
            violations = compute_consistency_violations(ds_norm, mech_risk)
            return consistency_weighted_fusion(ds_norm, mech_risk, violations)
        else:
            return fixed_weight_fusion(ds_norm, mech_risk, w_neural=w, w_mech=1 - w)
