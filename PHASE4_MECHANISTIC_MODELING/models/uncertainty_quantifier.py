"""
Phase 4 — Uncertainty Quantification

Implements:
- Monte Carlo trajectory simulation with parameter uncertainty
- Trajectory envelopes (median, 2.5th, 97.5th percentiles)
- Out-of-distribution detection (Mahalanobis distance)
- Patient-level confidence flags

RESEARCH PROTOTYPE ONLY. Simulated tumor trajectories are based on population-derived
ODE parameters personalized by genomic features. They represent biologically plausible
scenarios under stated assumptions, not clinically validated predictions. This system
must not be used for clinical decision-making without prospective validation with serial
imaging data and regulatory approval.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import mahalanobis
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ConfidenceFlag:
    """Patient-level uncertainty flag."""
    flag: bool
    reasons: List[str]
    confidence_level: str  # 'HIGH', 'MODERATE', 'LOW', 'DO_NOT_USE'
    trajectory_width_1yr: float
    max_param_uncertainty: float
    mahalanobis_distance: float


def compute_trajectory_envelope(
    V_all: np.ndarray,
    t: np.ndarray,
    percentiles: Tuple[float, float] = (2.5, 97.5),
) -> Dict:
    """
    Compute trajectory envelope from Monte Carlo simulations.

    Parameters
    ----------
    V_all : (n_mc, n_timepoints) array of tumor volumes
    t : time points
    percentiles : (lower, upper)

    Returns
    -------
    dict with t, V_median, V_lower, V_upper, V_mean, V_std
    """
    V_median = np.nanmedian(V_all, axis=0)
    V_lower = np.nanpercentile(V_all, percentiles[0], axis=0)
    V_upper = np.nanpercentile(V_all, percentiles[1], axis=0)
    V_mean = np.nanmean(V_all, axis=0)
    V_std = np.nanstd(V_all, axis=0)

    return {
        "t": t,
        "V_median": V_median,
        "V_lower": V_lower,
        "V_upper": V_upper,
        "V_mean": V_mean,
        "V_std": V_std,
    }


def mahalanobis_distance(
    patient_features: np.ndarray,
    training_mean: np.ndarray,
    training_cov_inv: np.ndarray,
    pca_transform=None,
) -> float:
    """
    Compute Mahalanobis distance of a patient from the training distribution.

    Parameters
    ----------
    patient_features : 1D array — patient feature vector (original space)
    training_mean : 1D array — mean of training features (PCA space)
    training_cov_inv : 2D array — inverse covariance matrix (PCA space)
    pca_transform : fitted PCA object or None — if provided, transforms patient_features

    Returns
    -------
    float — Mahalanobis distance (in standard deviations)
    """
    try:
        if pca_transform is not None:
            patient_features = pca_transform.transform(patient_features.reshape(1, -1))[0]
        diff = patient_features - training_mean
        dist = np.sqrt(diff @ training_cov_inv @ diff)
        return float(dist)
    except Exception:
        return float("inf")


def fit_training_distribution(X_train: np.ndarray, regularization: float = 1e-3,
                              n_components: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and inverse covariance of training features for OOD detection.

    Uses PCA dimensionality reduction before covariance estimation to avoid
    singularity issues with high-dimensional feature spaces (n_features > n_samples/10).

    Parameters
    ----------
    X_train : (n_samples, n_features) — training feature matrix
    regularization : float — added to diagonal for numerical stability
    n_components : int — number of PCA components for OOD detection

    Returns
    -------
    mean : 1D array — mean in PCA space
    cov_inv : 2D array — inverse covariance in PCA space
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train)

    mean = np.mean(X_pca, axis=0)
    cov = np.cov(X_pca, rowvar=False)
    cov += np.eye(cov.shape[0]) * regularization
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Store PCA transform for later use
    fit_training_distribution.pca = pca
    return mean, cov_inv


def uncertainty_flag(
    patient_params: Dict,
    uncertainty_bounds: Dict,
    trajectory_envelope: Dict,
    population_priors: Dict,
    mahalanobis_dist: float = 0.0,
    ml_risk_score: Optional[float] = None,
    simulated_progression_time: Optional[float] = None,
    n_pca_components: int = 20,
) -> ConfidenceFlag:
    """
    Flag patients with high simulation uncertainty.

    Criteria for flagging:
    1. Any parameter uncertainty > 150% of personalized value
    2. Trajectory envelope width at 1 year > 50% of personalized V0
    3. Patient features outside TCGA training distribution
       (Mahalanobis > chi2 97.5th percentile for n_pca_components)
    4. Conflicting ML risk score vs. mechanistic trajectory direction

    Returns
    -------
    ConfidenceFlag
    """
    from scipy.stats import chi2

    reasons = []

    # 1. Check parameter uncertainty (relative to personalized value, not prior)
    max_param_ratio = 0.0
    for key, (lo, hi) in uncertainty_bounds.items():
        if key in patient_params and patient_params[key] > 0:
            personalized = patient_params[key]
            width = hi - lo
            ratio = width / personalized
            if ratio > 1.5:  # uncertainty > 150% of personalized value
                reasons.append(f"High parameter uncertainty: {key} width={ratio:.1%} of personalized")
            max_param_ratio = max(max_param_ratio, ratio)

    # 2. Trajectory envelope width at 1 year (relative to V0)
    t = trajectory_envelope["t"]
    V_lower = trajectory_envelope["V_lower"]
    V_upper = trajectory_envelope["V_upper"]

    idx_1yr = np.argmin(np.abs(t - 365))
    width_1yr = V_upper[idx_1yr] - V_lower[idx_1yr]
    V0 = patient_params.get("V0", 8000.0)
    if width_1yr > 0.5 * V0:  # envelope wider than 50% of initial volume
        reasons.append(f"Wide trajectory envelope at 1yr: {width_1yr:.0f} mm^3 ({width_1yr/V0:.0%} of V0)")

    # 3. Mahalanobis distance (chi-square threshold for n_components)
    maha_threshold = np.sqrt(chi2.ppf(0.975, n_pca_components))
    if mahalanobis_dist > maha_threshold:
        reasons.append(f"Out-of-distribution: Mahalanobis distance={mahalanobis_dist:.2f} > {maha_threshold:.2f}")

    # 4. ML risk vs mechanistic consistency
    # Use simulated progression time for consistency check
    # High risk should correlate with short progression time
    if ml_risk_score is not None and simulated_progression_time is not None:
        # High risk should correlate with short time to progression
        if ml_risk_score > 0.5 and simulated_progression_time > 1095:  # >3yr
            reasons.append("ML risk score indicates high risk but simulated progression is slow (>3yr)")
        if ml_risk_score < -0.5 and simulated_progression_time < 365:  # <1yr
            reasons.append("ML risk score indicates low risk but simulated progression is fast (<1yr)")

    # Determine confidence level
    n_reasons = len(reasons)
    if mahalanobis_dist > maha_threshold * 1.5:  # very far OOD
        confidence = "DO_NOT_USE"
    elif n_reasons >= 3:
        confidence = "LOW"
    elif n_reasons >= 1:
        confidence = "MODERATE"
    else:
        confidence = "HIGH"

    return ConfidenceFlag(
        flag=n_reasons > 0,
        reasons=reasons if reasons else ["No issues detected"],
        confidence_level=confidence,
        trajectory_width_1yr=float(width_1yr),
        max_param_uncertainty=float(max_param_ratio),
        mahalanobis_distance=float(mahalanobis_dist),
    )
