"""
Strategy A: Feature-Level Fusion (Mechanistic Feature Augmentation).

Extracts scalar mechanistic features from Phase 4 ODE simulations and augments
the Phase 3 feature matrix. A new survival model is trained on the augmented set.

Uses identical architecture to Phase 3 best model ([64, 32] GELU, dropout=0.2, AdamW)
to ensure performance differences are attributable to feature augmentation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List

from mechanistic_features import MECH_FEATURE_NAMES


ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}


class DeepSurv(nn.Module):
    """DeepSurv architecture — identical to Phase 3 best model."""
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


def cox_partial_likelihood_loss(risk_scores, times, events):
    """
    Cox partial likelihood loss with Efron tie handling.
    Identical to Phase 3 implementation.
    """
    eps = 1e-7
    sorted_idx = torch.argsort(times, descending=True)
    risk = risk_scores[sorted_idx]
    t = times[sorted_idx]
    e = events[sorted_idx]

    log_cumsum = torch.logcumsumexp(risk, dim=0)

    unique_times = torch.unique(t[e == 1])
    loss = 0.0
    n_events = 0
    for ut in unique_times:
        mask_ut = (t == ut) & (e == 1)
        mask_at_risk = t >= ut
        risk_at_risk = risk[mask_at_risk]
        n_events_t = mask_ut.sum()
        n_ties = (t == ut).sum()

        log_risk_sum = torch.logsumexp(risk_at_risk, dim=0)
        tied_risk = risk[mask_ut | ((t == ut) & (e == 0))]

        for k in range(int(n_events_t.item())):
            if k == 0:
                log_denom = log_risk_sum
            else:
                mask_remaining = mask_at_risk.clone()
                idx_remove = torch.where(t == ut)[0][:k]
                mask_remaining[idx_remove] = False
                if mask_remaining.sum() > 0:
                    log_denom = torch.logsumexp(risk[mask_remaining], dim=0)
                else:
                    log_denom = log_risk_sum
            loss = loss - risk[mask_ut].sum() / n_events_t + log_denom
            n_events += 1

    if n_events > 0:
        loss = loss / n_events
    return loss


class FeatureLevelFusion:
    """
    Strategy A: Mechanistic Feature Augmentation.

    Trains a new DeepSurv model on augmented features (original 143 + 16 mechanistic = 159).
    """

    def __init__(self, hidden_dims=[64, 32], dropout=0.2, activation="gelu",
                 lr=0.001, wd=5e-5, n_models=2, seeds=(123, 456), device="cpu"):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.wd = wd
        self.n_models = n_models
        self.seeds = seeds
        self.device = device
        self.models = []
        self.mech_scaler = None
        self.is_fitted = False

    def fit(
        self,
        X_train_original: np.ndarray,
        mech_features_train: np.ndarray,
        y_time: np.ndarray,
        y_event: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 256,
        val_split: float = 0.15,
    ) -> "FeatureLevelFusion":
        """
        Train new DeepSurv models on augmented feature matrix.
        """
        # Scale mechanistic features
        self.mech_scaler = StandardScaler()
        mech_scaled = self.mech_scaler.fit_transform(mech_features_train)

        # Augment feature matrix
        X_aug = np.hstack([X_train_original, mech_scaled])
        n_features = X_aug.shape[1]

        # Split for internal validation during training
        n = len(X_aug)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_val = int(n * val_split)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_tr = torch.FloatTensor(X_aug[train_idx]).to(self.device)
        t_tr = torch.FloatTensor(y_time[train_idx]).to(self.device)
        e_tr = torch.FloatTensor(y_event[train_idx]).to(self.device)

        X_vl = torch.FloatTensor(X_aug[val_idx]).to(self.device)
        t_vl = torch.FloatTensor(y_time[val_idx]).to(self.device)
        e_vl = torch.FloatTensor(y_event[val_idx]).to(self.device)

        for model_idx in range(self.n_models):
            seed = self.seeds[model_idx]
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = DeepSurv(n_features, self.hidden_dims, self.dropout, self.activation)
            model = model.to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            best_val_ci = -1
            best_state = None

            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                risk = model(X_tr)
                loss = cox_partial_likelihood_loss(risk, t_tr, e_tr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        risk_val = model(X_vl).cpu().numpy()
                    from sksurv.metrics import concordance_index_censored
                    ci = concordance_index_censored(
                        y_event[val_idx].astype(bool),
                        y_time[val_idx], risk_val)[0]
                    scheduler.step(ci)
                    if ci > best_val_ci:
                        best_val_ci = ci
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, X_original: np.ndarray, mech_features: np.ndarray) -> np.ndarray:
        """Predict risk scores using ensemble of augmented models."""
        if not self.is_fitted:
            raise RuntimeError("FeatureLevelFusion not fitted. Call fit() first.")

        mech_scaled = self.mech_scaler.transform(mech_features)
        X_aug = np.hstack([X_original, mech_scaled])
        X_t = torch.FloatTensor(X_aug).to(self.device)

        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(X_t).cpu().numpy())
        return np.mean(preds, axis=0)
