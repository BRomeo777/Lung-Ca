"""
Phase 8 CDSS — Prediction Module
Loads DeepSurv model and produces risk scores, survival curves, and risk groups.
Does NOT retrain any model. Uses pre-validated Phase 3 artifacts.
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import P3_MODELS, P7_DATA, MODEL_VERSION, CALIBRATION_WARNING

# ── MODEL DEFINITION (must match Phase 3) ──
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


class PredictionEngine:
    """Loads Phase 3 DeepSurv model and produces predictions for the CDSS."""

    def __init__(self):
        self._loaded = False
        self._models = []
        self._scaler = None
        self._config = None
        self._feature_names = []
        self._gene_features = []
        self._age_median = 67.0
        self._state_df = None
        self._risk_cache = {}

    def load(self):
        """Load model artifacts from Phase 3."""
        with open(P3_MODELS / "final_config.json", "r") as f:
            self._config = json.load(f)
        with open(P3_MODELS / "final_scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)

        self._feature_names = self._config["feature_names"]
        self._age_median = self._config["age_median"]
        arch = self._config["arch"]
        n_models = self._config["n_models"]
        self._gene_features = [f for f in self._feature_names if "=" not in f and f != "Age"]

        for i in range(n_models):
            m = DeepSurv(len(self._feature_names), hidden_dims=arch["hidden_dims"],
                         dropout=arch["dropout"], activation=arch["activation"])
            m.load_state_dict(torch.load(P3_MODELS / f"final_deepsurv_model_{i}.pt",
                                         map_location="cpu", weights_only=True))
            m.eval()
            self._models.append(m)

        # Load state matrix for existing patient lookup
        self._state_df = pd.read_csv(P7_DATA / "digital_twin_state_matrix.csv")
        self._loaded = True

    def predict_existing(self, patient_id: str) -> dict | None:
        """Get prediction for an existing patient from the Phase 7 state matrix."""
        if not self._loaded:
            self.load()
        row = self._state_df[self._state_df["patient_id"].astype(str) == str(patient_id)]
        if len(row) == 0:
            return None
        row = row.iloc[0]
        return {
            "patient_id": str(row["patient_id"]),
            "age": float(row["age"]),
            "sex": str(row["sex"]),
            "cancer_type": str(row["cancer_type"]),
            "stage": str(row["stage"]),
            "smoking_status": str(row["smoking_status"]),
            "deepsurv_risk": float(row["deepsurv_risk"]),
            "deepsurv_risk_std": float(row.get("deepsurv_risk_std", 0.0)),
            "federated_risk": float(row.get("federated_risk", row["deepsurv_risk"])),
            "federated_risk_std": float(row.get("federated_risk_std", 0.0)),
            "risk_group": str(row.get("risk_group", "Unknown")),
            "os_time": float(row.get("os_time", 0)),
            "os_event": int(row.get("os_event", 0)),
        }

    def predict_new(self, age: float, cancer_type: str, stage: str,
                    smoking_status: str, gene_expression: dict | None = None) -> dict:
        """Predict risk for a new patient using the DeepSurv ensemble."""
        if not self._loaded:
            self.load()

        # Build feature vector
        row = {}
        row["Age"] = float(age) if age is not None else self._age_median

        ct = self._map_cancer_type(cancer_type)
        st = self._map_stage(stage)
        sm = self._map_smoking(smoking_status)

        for fname in self._feature_names:
            if fname == "Age":
                continue
            elif fname.startswith("Cancer_Type="):
                cat = fname.split("=", 1)[1]
                row[fname] = 1.0 if ct == cat else 0.0
            elif fname.startswith("Stage="):
                cat = fname.split("=", 1)[1]
                row[fname] = 1.0 if st == cat else 0.0
            elif fname.startswith("Smoking_Status="):
                cat = fname.split("=", 1)[1]
                row[fname] = 1.0 if sm == cat else 0.0

        if gene_expression is not None:
            for g in self._gene_features:
                row[g] = float(gene_expression.get(g, 0.0))
        else:
            for g in self._gene_features:
                row[g] = 0.0

        X = np.array([[row.get(f, 0.0) for f in self._feature_names]])
        X_scaled = self._scaler.transform(X)

        preds = []
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled)
            for m in self._models:
                preds.append(m(X_t).cpu().numpy()[0])

        risk = float(np.mean(preds))
        risk_std = float(np.std(preds))

        # Determine risk group from validation cohort distribution
        val_risks = self._state_df["deepsurv_risk"].values
        q33 = np.percentile(val_risks, 33)
        q66 = np.percentile(val_risks, 66)
        if risk < q33:
            risk_group = "Low"
        elif risk < q66:
            risk_group = "Intermediate"
        else:
            risk_group = "High"

        return {
            "deepsurv_risk": risk,
            "deepsurv_risk_std": risk_std,
            "federated_risk": risk,
            "federated_risk_std": risk_std,
            "risk_group": risk_group,
            "calibration_warning": CALIBRATION_WARNING,
            "model_version": MODEL_VERSION,
        }

    def risk_to_survival_curve(self, risk_score: float, t_eval: np.ndarray,
                               baseline_lambda: float = 0.001) -> np.ndarray:
        """Convert DeepSurv risk score to survival probability curve.
        UNCALIBRATED — uses exponential model from Phase 7."""
        lam = baseline_lambda * np.exp(risk_score)
        S = np.exp(-lam * t_eval)
        return np.clip(S, 0, 1)

    def get_survival_at_timepoints(self, risk_score: float,
                                   timepoints_months: list[float] = [12, 24, 36]) -> dict:
        """Get survival probability at specific time points (months)."""
        t_eval = np.array(timepoints_months) * 30.44
        S = self.risk_to_survival_curve(risk_score, t_eval)
        return {
            f"{tp}m": float(s) for tp, s in zip(timepoints_months, S)
        }

    def get_feature_contributions(self, age, cancer_type, stage, smoking_status,
                                  gene_expression=None) -> dict:
        """Get feature contributions for explainability.
        Uses simple feature-value * model weight approximation (not full SHAP).
        """
        if not self._loaded:
            self.load()

        # Build features
        row = {}
        row["Age"] = float(age) if age is not None else self._age_median
        ct = self._map_cancer_type(cancer_type)
        st = self._map_stage(stage)
        sm = self._map_smoking(smoking_status)

        for fname in self._feature_names:
            if fname == "Age":
                continue
            elif fname.startswith("Cancer_Type="):
                cat = fname.split("=", 1)[1]
                row[fname] = 1.0 if ct == cat else 0.0
            elif fname.startswith("Stage="):
                cat = fname.split("=", 1)[1]
                row[fname] = 1.0 if st == cat else 0.0
            elif fname.startswith("Smoking_Status="):
                cat = fname.split("=", 1)[1]
                row[fname] = 1.0 if sm == cat else 0.0

        if gene_expression is not None:
            for g in self._gene_features:
                row[g] = float(gene_expression.get(g, 0.0))
        else:
            for g in self._gene_features:
                row[g] = 0.0

        X = np.array([[row.get(f, 0.0) for f in self._feature_names]])
        X_scaled = self._scaler.transform(X)

        # Get per-model feature contributions via gradient * input
        X_t = torch.FloatTensor(X_scaled)
        X_t.requires_grad_(True)
        contributions = {}
        for m in self._models:
            m.zero_grad()
            out = m(X_t)
            out.backward()
            grad = X_t.grad[0].cpu().numpy()
            for i, fname in enumerate(self._feature_names):
                if fname not in contributions:
                    contributions[fname] = []
                contributions[fname].append(float(grad[i] * X_scaled[0][i]))
            X_t.grad = None

        # Average across models
        avg_contrib = {}
        for fname, vals in contributions.items():
            avg_contrib[fname] = float(np.mean(vals))

        # Sort by absolute contribution
        sorted_contrib = sorted(avg_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
        return {
            "top_clinical": [(k, v) for k, v in sorted_contrib if "=" in k or k == "Age"][:10],
            "top_molecular": [(k, v) for k, v in sorted_contrib if "=" not in k and k != "Age"][:10],
            "all_contributions": avg_contrib,
        }

    @staticmethod
    def _map_cancer_type(v):
        s = str(v).lower()
        if "luad" in s or "adc" in s or "adenocarcinoma" in s:
            return "LUAD_Adenocarcinoma"
        if "lusc" in s or "sqc" in s or "squamous" in s or "scc" in s:
            return "LUSC_SquamousCell"
        return None

    @staticmethod
    def _map_stage(v):
        s = str(v).strip()
        valid = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV"]
        if s in valid:
            return s
        return None

    @staticmethod
    def _map_smoking(v):
        s = str(v).lower()
        if "former" in s:
            return "Former"
        if "current" in s:
            return "Current"
        if "never" in s:
            return "Never"
        return None
