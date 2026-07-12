"""
Cascade Calibrator — Two-stage calibration for hybrid risk scores.

Stage 1: Isotonic regression on DeepSurv risk scores (from Phase 3/4 calibration set)
Stage 2: Platt scaling on hybrid combined scores

Both stages fit on calibration set ONLY (seed=99 split).
"""
from __future__ import annotations
import numpy as np
import pickle
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, Optional
from pathlib import Path


class CascadeCalibrator:
    """
    Two-stage calibration for hybrid risk scores.
    """

    def __init__(self):
        self.isotonic = None
        self.platt_a = 0.0
        self.platt_b = 0.0
        self.is_fitted = False

    def fit(
        self,
        risk_scores: np.ndarray,
        y_event: np.ndarray,
        y_time: np.ndarray,
        horizon_days: int = 1095,
    ) -> "CascadeCalibrator":
        """
        Fit cascade calibrator on calibration set.

        Parameters
        ----------
        risk_scores : risk scores for calibration patients
        y_event : event indicators (1 = death/progression)
        y_time : survival times in days
        horizon_days : time horizon for binary event definition (default 3yr)
        """
        # Binary event: occurred within horizon
        binary_event = ((y_event == 1) & (y_time <= horizon_days)).astype(int)

        if binary_event.sum() < 5 or (1 - binary_event).sum() < 5:
            # Not enough events/non-events for calibration
            self.is_fitted = True
            return self

        # Stage 1: Isotonic regression
        self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso_calibrated = self.isotonic.fit_transform(risk_scores, binary_event)

        # Stage 2: Platt scaling (logistic regression on isotonic-calibrated scores)
        # p = 1 / (1 + exp(a * x + b))
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * iso_calibrated + b)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.sum(binary_event * np.log(p) + (1 - binary_event) * np.log(1 - p))

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        self.platt_a, self.platt_b = result.x
        self.is_fitted = True
        return self

    def transform(self, risk_scores: np.ndarray) -> np.ndarray:
        """Apply cascade calibration to risk scores."""
        if not self.is_fitted:
            return risk_scores
        if self.isotonic is not None:
            scores = self.isotonic.transform(risk_scores)
        else:
            scores = risk_scores
        # Platt scaling
        calibrated = 1.0 / (1.0 + np.exp(-(self.platt_a * scores + self.platt_b)))
        return calibrated

    def save(self, path: Path):
        """Save calibrator."""
        with open(path, "wb") as f:
            pickle.dump({
                "isotonic": self.isotonic,
                "platt_a": self.platt_a,
                "platt_b": self.platt_b,
                "is_fitted": self.is_fitted,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "CascadeCalibrator":
        """Load calibrator."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        cal = cls()
        cal.isotonic = data["isotonic"]
        cal.platt_a = data["platt_a"]
        cal.platt_b = data["platt_b"]
        cal.is_fitted = data["is_fitted"]
        return cal
