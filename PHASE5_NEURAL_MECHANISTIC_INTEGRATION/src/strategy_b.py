"""
Strategy B: Physics-Informed DeepSurv (Loss-Level Fusion).

Adds a mechanistic consistency term to the DeepSurv loss function.
The neural network is penalized when its risk predictions are inconsistent
with the ODE's biological trajectory.

Architecture: identical to Phase 3 best model ([64, 32] GELU, dropout=0.2, wd=5e-5).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional

from strategy_a import DeepSurv, cox_partial_likelihood_loss


class PhysicsInformedDeepSurv(nn.Module):
    """
    Strategy B: Physics-Informed DeepSurv.

    Modified training objective: L_hybrid = L_Cox + lambda_mech * L_mech

    L_mech = mean((rank(risk) - rank(1/TTP))^2)

    Lambda schedule: lambda(t) = lambda_max * (1 - exp(-t/tau))
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2,
                 ttp_values: Optional[np.ndarray] = None,
                 lambda_max=0.1, tau=50):
        super().__init__()
        self.network = self._build_network(input_dim, hidden_dims, dropout)
        self.lambda_max = lambda_max
        self.tau = tau

        if ttp_values is not None:
            self.ttp_ranks = self._compute_ttp_ranks(ttp_values)
        else:
            self.ttp_ranks = None

    def _build_network(self, input_dim, hidden_dims, dropout):
        act_fn = nn.GELU()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(in_dim))
            linear = nn.Linear(in_dim, h_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        out_layer = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def _compute_ttp_ranks(self, ttp_values: np.ndarray) -> torch.Tensor:
        """Compute ranks of inverse TTP (higher 1/TTP = higher risk = higher rank)."""
        import scipy.stats
        inv_ttp = 1.0 / np.where(ttp_values > 0, ttp_values, 1e8)
        ranks = scipy.stats.rankdata(inv_ttp)
        n = len(ranks)
        return torch.FloatTensor((ranks - 1) / max(n - 1, 1))

    def _soft_rank(self, x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Differentiable approximation of rank function."""
        n = len(x)
        if n <= 1:
            return torch.zeros_like(x)
        pairs = x.unsqueeze(0) - x.unsqueeze(1)
        soft_ranks = torch.sigmoid(pairs / temperature).sum(dim=1)
        return soft_ranks / n

    def mechanistic_consistency_loss(
        self,
        risk_scores: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute rank-based mechanistic consistency loss.

        Returns: (loss, lambda_t, consistency_rho)
        """
        if self.ttp_ranks is None:
            return torch.tensor(0.0, device=risk_scores.device), 0.0, 0.0

        lambda_t = self.lambda_max * (1.0 - np.exp(-epoch / self.tau))

        risk_ranks = self._soft_rank(risk_scores)
        mech_ranks = self.ttp_ranks[:len(risk_scores)].to(risk_scores.device)

        rank_diff = risk_ranks - mech_ranks
        consistency_loss = lambda_t * torch.mean(rank_diff ** 2)

        with torch.no_grad():
            if len(risk_scores) > 2:
                rho = torch.corrcoef(
                    torch.stack([risk_scores, -self.ttp_ranks[:len(risk_scores)].to(risk_scores.device)])
                )[0, 1].item()
            else:
                rho = 0.0

        return consistency_loss, lambda_t, rho


class PhysicsInformedTrainer:
    """Trainer for Strategy B physics-informed model."""

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2,
                 lambda_max=0.1, tau=50, lr=0.001, wd=5e-5,
                 n_models=2, seeds=(123, 456), device="cpu"):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lambda_max = lambda_max
        self.tau = tau
        self.lr = lr
        self.wd = wd
        self.n_models = n_models
        self.seeds = seeds
        self.device = device
        self.models = []
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_time: np.ndarray,
        y_event: np.ndarray,
        ttp_values: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 256,
        val_split: float = 0.15,
    ) -> "PhysicsInformedTrainer":
        """Train physics-informed DeepSurv models."""
        n = len(X_train)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_val = int(n * val_split)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_tr = torch.FloatTensor(X_train[train_idx]).to(self.device)
        t_tr = torch.FloatTensor(y_time[train_idx]).to(self.device)
        e_tr = torch.FloatTensor(y_event[train_idx]).to(self.device)
        ttp_tr = ttp_values[train_idx]

        X_vl = torch.FloatTensor(X_train[val_idx]).to(self.device)

        for model_idx in range(self.n_models):
            seed = self.seeds[model_idx]
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = PhysicsInformedDeepSurv(
                self.input_dim, self.hidden_dims, self.dropout,
                ttp_values=ttp_tr, lambda_max=self.lambda_max, tau=self.tau)
            model = model.to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            best_val_ci = -1
            best_state = None

            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                risk = model(X_tr)
                cox_loss = cox_partial_likelihood_loss(risk, t_tr, e_tr)
                mech_loss, lambda_t, rho = model.mechanistic_consistency_loss(risk, epoch)
                total_loss = cox_loss + mech_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores using ensemble of physics-informed models."""
        if not self.is_fitted:
            raise RuntimeError("PhysicsInformedTrainer not fitted.")
        X_t = torch.FloatTensor(X).to(self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(X_t).cpu().numpy())
        return np.mean(preds, axis=0)
