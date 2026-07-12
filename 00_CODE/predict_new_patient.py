"""
DeepSurv Prediction Function for New Patients

Usage:
    from predict_new_patient import predict_patient, predict_batch
    
    # Single patient
    risk = predict_patient(
        age=65,
        cancer_type="LUAD",
        stage="IIA",
        smoking_status="Former",
        gene_expression=expr_df  # DataFrame: index=Ensembl IDs, values=expression
    )
    
    # Batch prediction
    risks = predict_batch(clinical_df, expression_df)
"""
import json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── PATHS ──
P3_MODELS = Path(__file__).resolve().parent.parent / "PHASE3_DEEP_LEARNING" / "models"

# ── MODEL DEFINITION ──
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

# ── LOAD CONFIG ──
with open(P3_MODELS / "final_config.json", "r") as f:
    _config = json.load(f)
with open(P3_MODELS / "final_scaler.pkl", "rb") as f:
    _scaler = pickle.load(f)

_feature_names = _config["feature_names"]
_age_median = _config["age_median"]
_arch = _config["arch"]
_n_models = _config["n_models"]
_gene_features = [f for f in _feature_names if "=" not in f and f != "Age"]

# ── LOAD MODELS ──
_models = []
for i in range(_n_models):
    m = DeepSurv(len(_feature_names), hidden_dims=_arch["hidden_dims"],
                dropout=_arch["dropout"], activation=_arch["activation"])
    m.load_state_dict(torch.load(P3_MODELS / f"final_deepsurv_model_{i}.pt", map_location="cpu"))
    m.eval()
    _models.append(m)


def _map_cancer_type(v):
    s = str(v).lower()
    if "adc" in s or "adenocarcinoma" in s: return "LUAD_Adenocarcinoma"
    if "sqc" in s or "squamous" in s or "scc" in s: return "LUSC_SquamousCell"
    return None

def _map_stage(v):
    s = str(v).strip()
    if s in ["I","IA","IB","II","IIA","IIB","III","IIIA","IIIB","IV"]: return s
    return None

def _map_smoking(v):
    s = str(v).lower()
    if "former" in s: return "Former"
    if "current" in s: return "Current"
    if "never" in s: return "Never"
    return None


def _build_single_features(age, cancer_type, stage, smoking_status, gene_expression):
    """Build feature vector for a single patient."""
    row = {}
    row["Age"] = float(age) if age is not None else _age_median
    
    ct = _map_cancer_type(cancer_type)
    st = _map_stage(stage)
    sm = _map_smoking(smoking_status)
    
    for fname in _feature_names:
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
    
    # Gene expression
    if gene_expression is not None:
        for g in _gene_features:
            if g in gene_expression.index:
                row[g] = float(gene_expression[g])
            else:
                row[g] = 0.0
    else:
        for g in _gene_features:
            row[g] = 0.0
    
    # Build in correct order
    X = np.array([[row.get(f, 0.0) for f in _feature_names]])
    X_scaled = _scaler.transform(X)
    return X_scaled


def predict_patient(age, cancer_type, stage, smoking_status, gene_expression=None):
    """
    Predict survival risk score for a single patient.
    
    Parameters:
        age: float or int
        cancer_type: str ("LUAD", "LUSC", "adenocarcinoma", "squamous", etc.)
        stage: str ("I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV")
        smoking_status: str ("Current", "Former", "Never")
        gene_expression: pd.Series or dict, indexed by Ensembl gene IDs
    
    Returns:
        float: risk score (higher = worse prognosis)
    """
    X = _build_single_features(age, cancer_type, stage, smoking_status, gene_expression)
    preds = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for m in _models:
            preds.append(m(X_t).cpu().numpy()[0])
    return float(np.mean(preds))


def predict_batch(clinical_df, expression_df):
    """
    Predict risk scores for a batch of patients.
    
    Parameters:
        clinical_df: DataFrame with columns Patient_ID, Age, Cancer_Type, Stage, Smoking_Status
        expression_df: DataFrame, index=Ensembl IDs, columns=Patient_IDs
    
    Returns:
        pd.Series: risk scores indexed by Patient_ID
    """
    results = {}
    for _, row in clinical_df.iterrows():
        pid = str(row["Patient_ID"])
        if pid in expression_df.columns:
            gene_expr = expression_df[pid]
        else:
            gene_expr = None
        risk = predict_patient(
            age=row.get("Age"),
            cancer_type=row.get("Cancer_Type", ""),
            stage=row.get("Stage", ""),
            smoking_status=row.get("Smoking_Status", ""),
            gene_expression=gene_expr,
        )
        results[pid] = risk
    return pd.Series(results)


def predict_with_uncertainty(age, cancer_type, stage, smoking_status, gene_expression=None):
    """
    Predict risk score with ensemble uncertainty (std across models).
    
    Returns:
        tuple: (mean_risk, std_risk)
    """
    X = _build_single_features(age, cancer_type, stage, smoking_status, gene_expression)
    preds = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for m in _models:
            preds.append(m(X_t).cpu().numpy()[0])
    return float(np.mean(preds)), float(np.std(preds))


if __name__ == "__main__":
    # Example usage
    print("DeepSurv Prediction Function")
    print(f"Models loaded: {_n_models}")
    print(f"Features: {len(_feature_names)}")
    print(f"Architecture: {_arch['hidden_dims']}")
    print()
    print("Usage:")
    print("  from predict_new_patient import predict_patient")
    print("  risk = predict_patient(age=65, cancer_type='LUAD', stage='IIA',")
    print("                        smoking_status='Former', gene_expression=expr_series)")
