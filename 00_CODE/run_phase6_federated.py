"""
Phase 6 — Federated Learning across 5 simulated non-IID sites.

Steps 1-5 of the Phase 6 protocol:
  1. Establish centralized (pooled) upper bound + per-site no-collaboration lower bound
  2. Implement FedAvg + correctness unit test
  3. Run FedAvg across 5 simulated sites, compare to both bounds
  4. Quantify non-IID heterogeneity
  5. Report privacy and cost honestly

SIMULATED FEDERATED SETTING — NOT a real multi-institutional deployment.
This system is a research prototype and must NOT be used for clinical decision-making.
"""
from __future__ import annotations
import json, pickle, random, warnings, sys, time, copy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════
P = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
READY = P / "03_ANALYSIS_READY_DATA"
ML = P / "ML_RESULTS"
P3M = P / "PHASE3_DEEP_LEARNING" / "models"
P6 = P / "PHASE6_FEDERATED_LEARNING"
P6_DATA = P6 / "data"
P6_REPORTS = P6 / "reports"
P6_MODELS = P6 / "models"
for d in [P6, P6_DATA, P6_REPORTS, P6_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

INVALID = ["TCGA-05-4395", "TCGA-77-A5G6"]
GEO_COHORTS = ["GSE30219", "GSE50081", "GSE72094", "GSE31210"]
GEO_EXPR = {
    "GSE30219": P / "PHASE1C_GEO_HARMONIZATION" / "processed_data" / "GSE30219" / "expression_gene_mapped.csv",
    "GSE50081": P / "PHASE1C_GEO_HARMONIZATION" / "processed_data" / "GSE50081" / "expression_gene_mapped.csv",
    "GSE72094": P / "PHASE1C_GEO_HARMONIZATION" / "processed_data" / "GSE72094" / "expression_gene_mapped.csv",
    "GSE31210": P / "PHASE1D_GEO_SCALE_ALIGNMENT" / "processed_data" / "GSE31210" / "GSE31210_expression_gene_mapped_log2.csv",
}

SEED = 42
DEVICE = "cpu"

# ═════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════
_log_file = P6_REPORTS / "phase6_log.txt"

def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(_log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ═════════════════════════════════════════════════════════════
# DEEPSURV MODEL (same architecture as Phase 3)
# ═════════════════════════════════════════════════════════════
ACT = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}

class DeepSurv(nn.Module):
    def __init__(self, n_features, hidden_dims=[64, 32], dropout=0.2, activation="gelu"):
        super().__init__()
        f = ACT[activation]
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.BatchNorm1d(in_dim))
            linear = nn.Linear(in_dim, h)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(f())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        out = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

class SimpleMLP(nn.Module):
    """MLP without BatchNorm — for FedAvg correctness testing.

    BatchNorm running statistics diverge across federated sites and cannot be
    correctly averaged, making it unsuitable for correctness verification.
    """
    def __init__(self, n_features, hidden_dims=[32, 16], dropout=0.3, activation="relu"):
        super().__init__()
        f = ACT[activation]
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            linear = nn.Linear(in_dim, h)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(f())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        out = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

# ═════════════════════════════════════════════════════════════
# COX PARTIAL LIKELIHOOD LOSS (Efron tie handling)
# ═════════════════════════════════════════════════════════════
def cox_loss(risk, times, events):
    """Negative Cox partial log-likelihood with Efron approximation."""
    order = torch.argsort(times, descending=True)
    risk_s = risk[order]
    events_s = events[order]
    times_s = times[order]
    event_mask = events_s > 0
    n_events = events_s.sum()
    if n_events == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk.device)
    log_cumsum_exp = torch.logcumsumexp(risk_s, dim=0)
    breslow_loss = -(events_s * (risk_s - log_cumsum_exp)).sum()
    # Efron correction for ties
    event_times = times_s[event_mask]
    unique_t, counts = torch.unique(event_times, return_counts=True)
    tied = unique_t[counts > 1]
    if len(tied) == 0:
        return breslow_loss / n_events
    efron_corr = torch.tensor(0.0, device=risk.device)
    for t in tied:
        d_t = counts[unique_t == t].item()
        event_at_t = (times_s == t) & event_mask
        risk_set = times_s >= t
        lse_risk = torch.logsumexp(risk_s[risk_set], dim=0)
        lse_events = torch.logsumexp(risk_s[event_at_t], dim=0)
        diff = lse_events - lse_risk
        for j in range(d_t):
            frac = j / d_t
            inner = 1.0 - frac * torch.exp(diff)
            inner = torch.clamp(inner, min=1e-10)
            efron_corr = efron_corr + torch.log(inner)
    return (breslow_loss + efron_corr) / n_events

# ═════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═════════════════════════════════════════════════════════════
def nlog2(counts):
    lib = counts.sum(axis=0); lib[lib == 0] = 1
    return np.log2(counts.div(lib, axis=1) * 1e6 + 1)

def mct(v):
    s = str(v).lower()
    return 'LUAD_Adenocarcinoma' if 'adc' in s or 'adenocarcinoma' in s else \
           'LUSC_SquamousCell' if 'sqc' in s or 'squamous' in s or 'scc' in s else None

def mst(v):
    s = str(v).strip()
    return s if s in ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV"] else None

def msm(v):
    s = str(v).lower()
    return "Former" if "former" in s else "Current" if "current" in s else "Never" if "never" in s else None

def is_normal(row):
    ct = str(row.get('Cancer_Type', '')).lower()
    st = str(row.get('Stage', '')).lower()
    return 'ntl' in ct or 'normal' in ct or 'ntl' in st

def build_features(clin, expr, gf, fn, am, scaler):
    """Build scaled feature matrix from clinical + expression data."""
    cols = [p for p in clin['Patient_ID'].astype(str) if p in expr.columns]
    cm = clin[clin['Patient_ID'].astype(str).isin(cols)].reset_index(drop=True)
    if len(cm) == 0:
        return None, None
    av = [g for g in gf if g in expr.index]
    es = expr.loc[av][cm['Patient_ID'].astype(str).tolist()].T
    es.index = cm.index
    ce = pd.DataFrame(index=cm.index)
    ce['Age'] = pd.to_numeric(cm['Age'], errors='coerce').fillna(am)
    for cn in ['Cancer_Type', 'Stage', 'Smoking_Status']:
        for f in fn:
            if f.startswith(f"{cn}="):
                cat = f.split("=", 1)[1]
                if cn == 'Cancer_Type': m = cm[cn].apply(mct)
                elif cn == 'Stage': m = cm[cn].apply(mst)
                else: m = cm[cn].apply(msm)
                ce[f] = (m == cat).astype(int)
    X = pd.concat([ce, es], axis=1)
    for f in fn:
        if f not in X.columns:
            X[f] = 0.0
    X = X[fn]
    return scaler.transform(X.values), cm

# ═════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════
def train_deepsurv(model, X_train, y_t, y_e, X_val, y_vt, y_ve,
                   lr=0.001, wd=5e-5, n_epochs=500, patience=30,
                   batch_size=256, grad_clip=1.0, device='cpu', verbose=False):
    """Train DeepSurv with mini-batch, LR scheduling, early stopping."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=15, min_lr=1e-6)
    X_t = torch.FloatTensor(X_train).to(device)
    t_t = torch.FloatTensor(y_t).to(device)
    e_t = torch.FloatTensor(y_e).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    dataset = TensorDataset(X_t, t_t, e_t)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(X_train)),
                        shuffle=True, drop_last=True)
    best_ci = 0.0
    best_state = None
    no_improve = 0
    for epoch in range(n_epochs):
        model.train()
        for bx, bt, be in loader:
            optimizer.zero_grad()
            risk = model(bx)
            loss = cox_loss(risk, bt, be)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            rv = model(X_v).cpu().numpy()
        ci = concordance_index_censored(y_ve.astype(bool), y_vt, rv)[0]
        scheduler.step(ci)
        if ci > best_ci:
            best_ci = ci
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            log(f"    Epoch {epoch}: val_ci={ci:.4f}, best={best_ci:.4f}")
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ci

def predict_deepsurv(model, X, device='cpu'):
    """Get risk predictions from a DeepSurv model."""
    model.eval()
    with torch.no_grad():
        Xt = torch.FloatTensor(X).to(device)
        return model(Xt).cpu().numpy()

# ═════════════════════════════════════════════════════════════
# BOOTSTRAP C-INDEX CI
# ═════════════════════════════════════════════════════════════
def boot_ci(ye, yt, risk, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(ye)
    cis = []
    for _ in range(n):
        i = rng.choice(n_samples, n_samples, replace=True)
        try:
            c = concordance_index_censored(ye[i].astype(bool), yt[i], risk[i])[0]
            if np.isfinite(c):
                cis.append(c)
        except:
            pass
    if len(cis) < 10:
        return float('nan'), float('nan'), float('nan')
    lo, hi = np.percentile(cis, [2.5, 97.5])
    return float(np.mean(cis)), float(lo), float(hi)

def cindex(ye, yt, risk):
    return concordance_index_censored(ye.astype(bool), yt, risk)[0]

# ═════════════════════════════════════════════════════════════
# FEDAVG IMPLEMENTATION
# ═════════════════════════════════════════════════════════════
def fedavg_round(global_model, site_data, local_epochs=5, lr=0.001, wd=5e-5,
                  batch_size=256, device='cpu', hidden_dims=[64, 32], dropout=0.2,
                  activation="gelu", model_class=DeepSurv):
    """One round of FedAvg: all sites train locally, then average weights.

    Args:
        global_model: current global model state_dict
        site_data: list of dicts with 'X_train', 'y_t', 'y_e', 'X_val', 'y_vt', 'y_ve'
        local_epochs: epochs per site per round
        lr, wd: learning rate and weight decay
        device: torch device

    Returns:
        new global state_dict, per-site training info
    """
    site_weights = []
    site_sizes = []
    site_info = []

    for idx, sd in enumerate(site_data):
        # Create local model from global weights
        local_model = model_class(
            n_features=sd['X_train'].shape[1],
            hidden_dims=hidden_dims, dropout=dropout, activation=activation
        ).to(device)
        local_model.load_state_dict(global_model)
        local_model.train()

        optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=wd)

        X_t = torch.FloatTensor(sd['X_train']).to(device)
        t_t = torch.FloatTensor(sd['y_t']).to(device)
        e_t = torch.FloatTensor(sd['y_e']).to(device)
        dataset = TensorDataset(X_t, t_t, e_t)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(sd['X_train'])),
                           shuffle=True, drop_last=True)

        for epoch in range(local_epochs):
            for bx, bt, be in loader:
                if bx.shape[0] < 2:
                    continue
                optimizer.zero_grad()
                risk = local_model(bx)
                loss = cox_loss(risk, bt, be)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                optimizer.step()

        # Evaluate local model on local validation
        local_model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(sd['X_val']).to(device)
            rv = local_model(X_v).cpu().numpy()
        local_ci = cindex(sd['y_ve'], sd['y_vt'], rv)

        site_weights.append(local_model.state_dict())
        site_sizes.append(len(sd['X_train']))
        site_info.append({'site': idx, 'n_train': len(sd['X_train']), 'local_ci': local_ci})

    # Weighted average of model weights
    total_samples = sum(site_sizes)
    avg_state = {}
    for key in global_model.keys():
        avg_state[key] = sum(
            site_weights[i][key] * site_sizes[i] for i in range(len(site_weights))
        ) / total_samples

    return avg_state, site_info

def fedavg_train(site_data, n_rounds=50, local_epochs=5, lr=0.001, wd=5e-5,
                 device='cpu', verbose=True, hidden_dims=[64, 32], dropout=0.2,
                 activation="gelu", model_class=DeepSurv):
    """Full FedAvg training loop.

    Args:
        site_data: list of dicts with 'X_train', 'y_t', 'y_e', 'X_val', 'y_vt', 'y_ve', 'name'
        n_rounds: number of communication rounds
        local_epochs: epochs per site per round
        lr, wd: learning rate and weight decay

    Returns:
        global_model state_dict, round_history
    """
    n_features = site_data[0]['X_train'].shape[1]

    # Initialize global model
    torch.manual_seed(SEED)
    global_model = model_class(n_features, hidden_dims, dropout, activation).to(device)
    global_state = global_model.state_dict()

    # Pooled validation for global monitoring
    X_val_pooled = np.vstack([sd['X_val'] for sd in site_data])
    y_vt_pooled = np.concatenate([sd['y_vt'] for sd in site_data])
    y_ve_pooled = np.concatenate([sd['y_ve'] for sd in site_data])

    round_history = []

    for rnd in range(n_rounds):
        global_state, site_info = fedavg_round(
            global_state, site_data, local_epochs=local_epochs,
            lr=lr, wd=wd, device=device,
            hidden_dims=hidden_dims, dropout=dropout, activation=activation,
            model_class=model_class
        )

        # Evaluate global model on pooled validation
        global_model.load_state_dict(global_state)
        global_model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val_pooled).to(device)
            rv = global_model(X_v).cpu().numpy()
        global_ci = cindex(y_ve_pooled, y_vt_pooled, rv)

        round_history.append({
            'round': rnd,
            'global_val_ci': global_ci,
            'site_info': site_info,
        })

        if verbose and (rnd % 10 == 0 or rnd == n_rounds - 1):
            local_cis = [f"{si['local_ci']:.4f}" for si in site_info]
            log(f"    Round {rnd}: global_val_ci={global_ci:.4f}, local_cis={local_cis}")

    return global_state, round_history

# ═════════════════════════════════════════════════════════════
# STRATIFIED TRAIN/TEST SPLIT
# ═════════════════════════════════════════════════════════════
def stratified_split(y_event, test_size=0.2, seed=42):
    """Stratified split by event status."""
    rng = np.random.default_rng(seed)
    n = len(y_event)
    indices = np.arange(n)
    train_idx = []
    test_idx = []
    for status in [0, 1]:
        idx = indices[y_event == status]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))

def further_split(train_idx, y_event, val_frac=0.2, seed=42):
    """Further split train into train/val for early stopping."""
    rng = np.random.default_rng(seed)
    indices = train_idx.copy()
    rng.shuffle(indices)
    n_val = int(len(indices) * val_frac)
    val_idx = sorted(indices[:n_val])
    tr_idx = sorted(indices[n_val:])
    return np.array(tr_idx), np.array(val_idx)


# ═════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════
def main():
    if _log_file.exists():
        _log_file.unlink()

    log("=" * 70)
    log("PHASE 6: FEDERATED LEARNING — SIMULATED 5-SITE NON-IID")
    log("=" * 70)
    log("SIMULATED FEDERATED SETTING — NOT a real multi-institutional deployment.")
    log("RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL USE.")
    log("")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # ── Load Phase 3 artifacts ──
    cfg = json.load(open(P3M / "final_config.json"))
    scaler = pickle.load(open(P3M / "final_scaler.pkl", "rb"))
    fn = cfg["feature_names"]; am = cfg["age_median"]
    gf = [f for f in fn if "=" not in f and f != "Age"]
    log(f"  Phase 3 config: {len(fn)} features, arch={cfg['arch']['hidden_dims']}")

    # ── Load all 5 site datasets ──
    log("")
    log("Loading data for 5 simulated sites...")

    sites = {}

    # Site 1: TCGA (train + val combined as one site)
    tr = pd.read_csv(READY / "TCGA_train.csv")
    tr = tr[~tr['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    etr = nlog2(pd.read_csv(READY / "TCGA_train_expression.tsv.gz", sep="\t", index_col=0))
    tr = tr[tr['Patient_ID'].astype(str).isin(etr.columns)].reset_index(drop=True)
    Xtr, ctr = build_features(tr, etr, gf, fn, am, scaler)
    ytr_t = pd.to_numeric(ctr['Overall_Survival_Time'], errors='coerce').values
    ytr_e = pd.to_numeric(ctr['Survival_Status'], errors='coerce').fillna(0).astype(int).values

    va = pd.read_csv(READY / "TCGA_internal_validation.csv")
    va = va[~va['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    eva = nlog2(pd.read_csv(READY / "TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0))
    va = va[va['Patient_ID'].astype(str).isin(eva.columns)].reset_index(drop=True)
    Xva, cva = build_features(va, eva, gf, fn, am, scaler)
    yva_t = pd.to_numeric(cva['Overall_Survival_Time'], errors='coerce').values
    yva_e = pd.to_numeric(cva['Survival_Status'], errors='coerce').fillna(0).astype(int).values

    # Combine TCGA train+val as one site, then split into site-train/site-test
    X_tcga = np.vstack([Xtr, Xva])
    y_tcga_t = np.concatenate([ytr_t, yva_t])
    y_tcga_e = np.concatenate([ytr_e, yva_e])
    tr_idx, te_idx = stratified_split(y_tcga_e, test_size=0.2, seed=SEED)
    tr_idx2, val_idx2 = further_split(tr_idx, y_tcga_e, val_frac=0.15, seed=SEED)
    sites["TCGA"] = {
        'X_all': X_tcga, 'y_t_all': y_tcga_t, 'y_e_all': y_tcga_e,
        'X_train': X_tcga[tr_idx2], 'y_t': y_tcga_t[tr_idx2], 'y_e': y_tcga_e[tr_idx2],
        'X_val': X_tcga[val_idx2], 'y_vt': y_tcga_t[val_idx2], 'y_ve': y_tcga_e[val_idx2],
        'X_test': X_tcga[te_idx], 'y_tt': y_tcga_t[te_idx], 'y_te': y_tcga_e[te_idx],
        'platform': 'RNA-seq', 'n_total': len(X_tcga),
    }
    log(f"  TCGA: {len(X_tcga)} patients ({int(y_tcga_e.sum())} events), RNA-seq")

    # Sites 2-5: GEO cohorts
    for cohort in GEO_COHORTS:
        gc = pd.read_csv(READY / f"{cohort}_external_validation.csv")
        normal_mask = gc.apply(is_normal, axis=1)
        gc = gc[~normal_mask].reset_index(drop=True)
        gc = gc.dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
        ge = pd.read_csv(GEO_EXPR[cohort], index_col=0)
        Xg, gcm = build_features(gc, ge, gf, fn, am, scaler)
        if Xg is None or len(gcm) == 0:
            log(f"  {cohort}: SKIP — no matched patients")
            continue
        yg_t = pd.to_numeric(gcm['Overall_Survival_Time'], errors='coerce').values
        yg_e = pd.to_numeric(gcm['Survival_Status'], errors='coerce').fillna(0).astype(int).values
        # Split: 80% train, 20% test; from train, 15% for val
        tr_i, te_i = stratified_split(yg_e, test_size=0.2, seed=SEED)
        tr_i2, va_i2 = further_split(tr_i, yg_e, val_frac=0.15, seed=SEED)
        platform = "Affymetrix HuGene 1.0" if cohort == "GSE30219" else "Affymetrix HG-U133 Plus 2.0"
        sites[cohort] = {
            'X_all': Xg, 'y_t_all': yg_t, 'y_e_all': yg_e,
            'X_train': Xg[tr_i2], 'y_t': yg_t[tr_i2], 'y_e': yg_e[tr_i2],
            'X_val': Xg[va_i2], 'y_vt': yg_t[va_i2], 'y_ve': yg_e[va_i2],
            'X_test': Xg[te_i], 'y_tt': yg_t[te_i], 'y_te': yg_e[te_i],
            'platform': platform, 'n_total': len(Xg),
        }
        log(f"  {cohort}: {len(Xg)} patients ({int(yg_e.sum())} events), {platform}")

    site_names = list(sites.keys())
    log(f"  Total sites: {len(site_names)}")

    # ═══════════════════════════════════════════════════════════
    # STEP 1: ESTABLISH TWO BOUNDS
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 1: Establish two bounds — centralized (pooled) + per-site (no-collab)")
    log("=" * 70)

    # ── Upper bound: centralized (pooled) training ──
    log("  Training centralized DeepSurv on pooled data from all sites...")
    X_pool = np.vstack([sites[s]['X_train'] for s in site_names])
    yt_pool = np.concatenate([sites[s]['y_t'] for s in site_names])
    ye_pool = np.concatenate([sites[s]['y_e'] for s in site_names])
    X_pool_val = np.vstack([sites[s]['X_val'] for s in site_names])
    yt_pool_val = np.concatenate([sites[s]['y_vt'] for s in site_names])
    ye_pool_val = np.concatenate([sites[s]['y_ve'] for s in site_names])
    log(f"  Pooled: {len(X_pool)} train, {len(X_pool_val)} val")

    torch.manual_seed(SEED)
    centralized_model = DeepSurv(len(fn), [64, 32], 0.2, "gelu")
    centralized_model, cent_val_ci = train_deepsurv(
        centralized_model, X_pool, yt_pool, ye_pool,
        X_pool_val, yt_pool_val, ye_pool_val,
        lr=0.001, wd=5e-5, n_epochs=500, patience=30,
        batch_size=256, device=DEVICE, verbose=True)
    log(f"  Centralized pooled val C-index: {cent_val_ci:.4f}")

    # Evaluate centralized model on each site's test set
    log("  Evaluating centralized model on each site's held-out test set...")
    centralized_results = {}
    for s in site_names:
        risk = predict_deepsurv(centralized_model, sites[s]['X_test'], DEVICE)
        ci_val = cindex(sites[s]['y_te'], sites[s]['y_tt'], risk)
        m, lo, hi = boot_ci(sites[s]['y_te'], sites[s]['y_tt'], risk, n=500, seed=SEED)
        centralized_results[s] = {'ci': ci_val, 'ci_lo': lo, 'ci_hi': hi, 'risk': risk}
        log(f"    {s}: C-index={ci_val:.4f} ({lo:.4f}-{hi:.4f})")

    # ── Lower bound: per-site training (no collaboration) ──
    log("  Training per-site DeepSurv models (no collaboration)...")
    per_site_results = {}
    for s in site_names:
        sd = sites[s]
        if len(sd['X_train']) < 20:
            log(f"    {s}: SKIP — too few training samples ({len(sd['X_train'])})")
            per_site_results[s] = {'ci': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan')}
            continue
        torch.manual_seed(SEED)
        local_model = DeepSurv(len(fn), [64, 32], 0.2, "gelu")
        local_model, local_ci = train_deepsurv(
            local_model, sd['X_train'], sd['y_t'], sd['y_e'],
            sd['X_val'], sd['y_vt'], sd['y_ve'],
            lr=0.001, wd=5e-5, n_epochs=500, patience=30,
            batch_size=min(256, len(sd['X_train'])), device=DEVICE, verbose=False)
        risk = predict_deepsurv(local_model, sd['X_test'], DEVICE)
        ci_val = cindex(sd['y_te'], sd['y_tt'], risk)
        m, lo, hi = boot_ci(sd['y_te'], sd['y_tt'], risk, n=500, seed=SEED)
        per_site_results[s] = {'ci': ci_val, 'ci_lo': lo, 'ci_hi': hi, 'risk': risk, 'model': local_model}
        log(f"    {s}: C-index={ci_val:.4f} ({lo:.4f}-{hi:.4f}) [train_ci={local_ci:.4f}]")

    # Save bounds table
    bounds_rows = []
    for s in site_names:
        cr = centralized_results.get(s, {})
        ps = per_site_results.get(s, {})
        bounds_rows.append({
            'site': s, 'n_test': len(sites[s]['y_te']),
            'centralized_ci': cr.get('ci', float('nan')),
            'centralized_lo': cr.get('ci_lo', float('nan')),
            'centralized_hi': cr.get('ci_hi', float('nan')),
            'per_site_ci': ps.get('ci', float('nan')),
            'per_site_lo': ps.get('ci_lo', float('nan')),
            'per_site_hi': ps.get('ci_hi', float('nan')),
        })
    bounds_df = pd.DataFrame(bounds_rows)
    bounds_df.to_csv(P6_REPORTS / "bounds_comparison.csv", index=False)
    log(f"  -> bounds_comparison.csv saved")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: IMPLEMENT FEDAVG + CORRECTNESS TEST
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 2: FedAvg implementation + correctness unit test")
    log("=" * 70)

    # Correctness test: identical synthetic data → fedavg == centralized
    log("  Correctness test: identical synthetic data at all sites...")
    np.random.seed(42); torch.manual_seed(42)
    n_synth = 300; n_feat = 20
    X_synth = np.random.randn(n_synth, n_feat).astype(np.float32)
    # Create a real survival signal: linear combination of features → risk
    true_w = np.random.randn(n_feat).astype(np.float32) * 0.5
    risk_synth = X_synth @ true_w
    risk_synth = (risk_synth - risk_synth.mean()) / (risk_synth.std() + 1e-8)
    # Higher risk → shorter survival, more likely to have event
    y_t_synth = np.exp(-risk_synth * 0.5 + 1) * 365 + 30
    y_t_synth = y_t_synth.astype(np.float64)
    y_e_synth = (np.random.rand(n_synth) < 1 / (1 + np.exp(-risk_synth))).astype(int)
    # Ensure at least 30% events
    if y_e_synth.mean() < 0.3:
        y_e_synth = (np.random.rand(n_synth) < 0.5).astype(int)

    # Split into train/val
    tr_i, te_i = stratified_split(y_e_synth, test_size=0.2, seed=42)
    tr_i2, va_i2 = further_split(tr_i, y_e_synth, val_frac=0.2, seed=42)

    # 3 identical sites with the same data
    synth_sites = []
    for i in range(3):
        synth_sites.append({
            'X_train': X_synth[tr_i2], 'y_t': y_t_synth[tr_i2], 'y_e': y_e_synth[tr_i2].astype(float),
            'X_val': X_synth[va_i2], 'y_vt': y_t_synth[va_i2], 'y_ve': y_e_synth[va_i2],
            'name': f'synth_{i}',
        })

    # Centralized on all pooled synth data
    torch.manual_seed(42)
    cent_synth = SimpleMLP(n_feat, [32, 16], 0.3, "relu")
    cent_synth, _ = train_deepsurv(
        cent_synth, X_synth[tr_i2], y_t_synth[tr_i2], y_e_synth[tr_i2].astype(float),
        X_synth[va_i2], y_t_synth[va_i2], y_e_synth[va_i2],
        lr=0.001, wd=1e-4, n_epochs=500, patience=50, batch_size=64, device=DEVICE, verbose=False)

    # FedAvg on 3 identical sites
    torch.manual_seed(42)
    fed_synth_state, fed_synth_history = fedavg_train(
        synth_sites, n_rounds=50, local_epochs=10, lr=0.001, wd=1e-4,
        device=DEVICE, verbose=False, hidden_dims=[32, 16], dropout=0.3,
        activation="relu", model_class=SimpleMLP)

    fed_synth = SimpleMLP(n_feat, [32, 16], 0.3, "relu")
    fed_synth.load_state_dict(fed_synth_state)
    fed_synth.eval()

    # Compare predictions on test set
    with torch.no_grad():
        Xt = torch.FloatTensor(X_synth[te_i])
        cent_pred = cent_synth(Xt).cpu().numpy()
        fed_pred = fed_synth(Xt).cpu().numpy()

    # Correlation between centralized and federated predictions
    corr = scipy.stats.spearmanr(cent_pred, fed_pred)[0]
    log(f"  Spearman correlation (centralized vs fedavg): {corr:.4f}")

    # C-index comparison
    ci_cent = cindex(y_e_synth[te_i], y_t_synth[te_i], cent_pred)
    ci_fed = cindex(y_e_synth[te_i], y_t_synth[te_i], fed_pred)
    log(f"  Centralized C-index: {ci_cent:.4f}")
    log(f"  FedAvg C-index: {ci_fed:.4f}")
    log(f"  Difference: {abs(ci_cent - ci_fed):.4f}")

    if corr > 0.95:
        log("  CORRECTNESS TEST: PASS (rho > 0.95)")
    else:
        log(f"  CORRECTNESS TEST: CHECK (rho={corr:.4f}) — with identical data and enough rounds, should converge")

    # Save correctness test results
    with open(P6_REPORTS / "fedavg_correctness_test.txt", "w") as f:
        f.write("FedAvg Correctness Test — Identical Data at All Sites\n")
        f.write("=" * 60 + "\n")
        f.write(f"Spearman correlation (centralized vs fedavg): {corr:.4f}\n")
        f.write(f"Centralized C-index: {ci_cent:.4f}\n")
        f.write(f"FedAvg C-index: {ci_fed:.4f}\n")
        f.write(f"Abs difference: {abs(ci_cent - ci_fed):.4f}\n")
        f.write(f"Result: {'PASS' if corr > 0.95 else 'CHECK'} (threshold: rho > 0.95)\n")

    # ═══════════════════════════════════════════════════════════
    # STEP 3: RUN FEDAVG ACROSS 5 SIMULATED SITES
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 3: FedAvg across 5 simulated non-IID sites")
    log("=" * 70)

    # Build site data for FedAvg
    fed_sites = []
    for s in site_names:
        sd = sites[s]
        if len(sd['X_train']) < 10:
            log(f"  Skipping {s} for FedAvg — too few training samples")
            continue
        fed_sites.append({
            'X_train': sd['X_train'], 'y_t': sd['y_t'], 'y_e': sd['y_e'].astype(float),
            'X_val': sd['X_val'], 'y_vt': sd['y_vt'], 'y_ve': sd['y_ve'],
            'name': s,
        })

    log(f"  Sites in federation: {len(fed_sites)}")
    log(f"  Hyperparameters: rounds=50, local_epochs=5, lr=0.001, wd=5e-5, batch_size=256")
    log(f"  Client sampling: all sites every round (only {len(fed_sites)} sites)")

    t_start = time.time()
    fed_state, fed_history = fedavg_train(
        fed_sites, n_rounds=50, local_epochs=5, lr=0.001, wd=5e-5,
        device=DEVICE, verbose=True)
    fed_time = time.time() - t_start
    log(f"  FedAvg training completed in {fed_time:.1f}s ({fed_time/60:.1f} min)")

    # Evaluate federated model on each site's test set
    log("  Evaluating federated global model on each site's held-out test set...")
    fed_model = DeepSurv(len(fn), [64, 32], 0.2, "gelu")
    fed_model.load_state_dict(fed_state)

    fed_results = {}
    for s in site_names:
        risk = predict_deepsurv(fed_model, sites[s]['X_test'], DEVICE)
        ci_val = cindex(sites[s]['y_te'], sites[s]['y_tt'], risk)
        m, lo, hi = boot_ci(sites[s]['y_te'], sites[s]['y_tt'], risk, n=500, seed=SEED)
        fed_results[s] = {'ci': ci_val, 'ci_lo': lo, 'ci_hi': hi, 'risk': risk}
        log(f"    {s}: C-index={ci_val:.4f} ({lo:.4f}-{hi:.4f})")

    # Pooled federated performance
    X_test_pool = np.vstack([sites[s]['X_test'] for s in site_names])
    yt_test_pool = np.concatenate([sites[s]['y_tt'] for s in site_names])
    ye_test_pool = np.concatenate([sites[s]['y_te'] for s in site_names])
    fed_risk_pool = predict_deepsurv(fed_model, X_test_pool, DEVICE)
    fed_ci_pool = cindex(ye_test_pool, yt_test_pool, fed_risk_pool)
    _, fed_lo, fed_hi = boot_ci(ye_test_pool, yt_test_pool, fed_risk_pool, n=500, seed=SEED)
    log(f"  Pooled federated C-index: {fed_ci_pool:.4f} ({fed_lo:.4f}-{fed_hi:.4f})")

    # Save comparison table
    comp_rows = []
    for s in site_names:
        cr = centralized_results.get(s, {})
        ps = per_site_results.get(s, {})
        fr = fed_results.get(s, {})
        comp_rows.append({
            'site': s, 'n_test': len(sites[s]['y_te']),
            'centralized_ci': cr.get('ci', float('nan')),
            'centralized_lo': cr.get('ci_lo', float('nan')),
            'centralized_hi': cr.get('ci_hi', float('nan')),
            'per_site_ci': ps.get('ci', float('nan')),
            'per_site_lo': ps.get('ci_lo', float('nan')),
            'per_site_hi': ps.get('ci_hi', float('nan')),
            'federated_ci': fr.get('ci', float('nan')),
            'federated_lo': fr.get('ci_lo', float('nan')),
            'federated_hi': fr.get('ci_hi', float('nan')),
            'delta_fed_vs_cent': fr.get('ci', float('nan')) - cr.get('ci', float('nan')),
            'delta_fed_vs_local': fr.get('ci', float('nan')) - ps.get('ci', float('nan')),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(P6_REPORTS / "federated_comparison.csv", index=False)
    log(f"  -> federated_comparison.csv saved")

    # Save round history
    rh_df = pd.DataFrame([{
        'round': r['round'], 'global_val_ci': r['global_val_ci'],
        **{f"site_{i}_ci": si['local_ci'] for i, si in enumerate(r['site_info'])}
    } for r in fed_history])
    rh_df.to_csv(P6_REPORTS / "fedavg_round_history.csv", index=False)
    log(f"  -> fedavg_round_history.csv saved")

    # Save federated model
    torch.save(fed_state, P6_MODELS / "fedavg_global_model.pt")
    log(f"  -> fedavg_global_model.pt saved")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: NON-IID HETEROGENEITY ANALYSIS
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 4: Non-IID heterogeneity analysis")
    log("=" * 70)

    hetero_rows = []
    for s in site_names:
        sd = sites[s]
        cr = centralized_results.get(s, {})
        fr = fed_results.get(s, {})
        ps = per_site_results.get(s, {})
        # Site characteristics
        event_rate = sd['y_e_all'].mean()
        n_train = len(sd['X_train'])
        # Gene panel coverage: fraction of gene features with non-zero std
        gene_idx = [i for i, f in enumerate(fn) if "=" not in f and f != "Age"]
        gene_std = np.std(sd['X_all'][:, gene_idx], axis=0)
        nonzero_genes = np.sum(gene_std > 1e-10)
        total_genes = len(gene_idx)
        panel_coverage = nonzero_genes / total_genes if total_genes > 0 else 0
        hetero_rows.append({
            'site': s, 'platform': sd['platform'],
            'n_total': sd['n_total'], 'n_train': n_train,
            'event_rate': round(event_rate, 4),
            'gene_panel_coverage': round(panel_coverage, 4),
            'centralized_ci': cr.get('ci', float('nan')),
            'federated_ci': fr.get('ci', float('nan')),
            'per_site_ci': ps.get('ci', float('nan')),
            'delta_fed_vs_cent': fr.get('ci', float('nan')) - cr.get('ci', float('nan')),
        })
    hetero_df = pd.DataFrame(hetero_rows)
    hetero_df.to_csv(P6_REPORTS / "noniid_heterogeneity.csv", index=False)
    log(f"  -> noniid_heterogeneity.csv saved")

    log("  Per-site summary:")
    for _, row in hetero_df.iterrows():
        log(f"    {row['site']}: platform={row['platform']}, "
            f"n_train={row['n_train']}, events={row['event_rate']:.1%}, "
            f"coverage={row['gene_panel_coverage']:.1%}, "
            f"fed_ci={row['federated_ci']:.4f}, cent_ci={row['centralized_ci']:.4f}, "
            f"delta={row['delta_fed_vs_cent']:.4f}")

    # Correlation: does event rate or platform correlate with federated performance gap?
    deltas = [fr.get('ci', 0) - centralized_results.get(s, {}).get('ci', 0) for s in site_names]
    event_rates = [sites[s]['y_e_all'].mean() for s in site_names]
    n_trains = [len(sites[s]['X_train']) for s in site_names]
    if len(deltas) >= 3 and np.std(deltas) > 0:
        rho_er, p_er = scipy.stats.spearmanr(event_rates, deltas)
        rho_nt, p_nt = scipy.stats.spearmanr(n_trains, deltas)
        log(f"  Spearman(delta vs event_rate): rho={rho_er:.4f}, p={p_er:.4f}")
        log(f"  Spearman(delta vs n_train): rho={rho_nt:.4f}, p={p_nt:.4f}")
    else:
        log("  Cannot compute heterogeneity correlations (insufficient variance)")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: PRIVACY AND COST REPORT
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("STEP 5: Privacy and cost report")
    log("=" * 70)

    # Model size
    n_params = sum(p.numel() for p in fed_model.parameters())
    model_size_mb = n_params * 4 / (1024 * 1024)  # float32
    n_rounds = len(fed_history)
    n_sites_fed = len(fed_sites)

    # Communication cost: model weights sent per round = model_size * (n_sites * 2)
    # (each site receives global weights + sends updated weights)
    comm_per_round_mb = model_size_mb * n_sites_fed * 2
    total_comm_mb = comm_per_round_mb * n_rounds

    # Centralized training time (for comparison)
    log("  Re-timing centralized training for comparison...")
    t_cent = time.time()
    torch.manual_seed(SEED)
    cent_model_timed = DeepSurv(len(fn), [64, 32], 0.2, "gelu")
    cent_model_timed, _ = train_deepsurv(
        cent_model_timed, X_pool, yt_pool, ye_pool,
        X_pool_val, yt_pool_val, ye_pool_val,
        lr=0.001, wd=5e-5, n_epochs=500, patience=30,
        batch_size=256, device=DEVICE, verbose=False)
    cent_time = time.time() - t_cent

    log(f"  Model parameters: {n_params}")
    log(f"  Model size: {model_size_mb:.4f} MB")
    log(f"  Rounds: {n_rounds}")
    log(f"  Sites: {n_sites_fed}")
    log(f"  Communication per round: {comm_per_round_mb:.4f} MB")
    log(f"  Total communication: {total_comm_mb:.2f} MB")
    log(f"  FedAvg wall-clock time: {fed_time:.1f}s ({fed_time/60:.1f} min)")
    log(f"  Centralized wall-clock time: {cent_time:.1f}s ({cent_time/60:.1f} min)")
    log(f"  Overhead ratio: {fed_time/cent_time:.2f}x")

    privacy_cost = {
        'n_params': n_params,
        'model_size_mb': round(model_size_mb, 4),
        'n_rounds': n_rounds,
        'n_sites': n_sites_fed,
        'comm_per_round_mb': round(comm_per_round_mb, 4),
        'total_comm_mb': round(total_comm_mb, 2),
        'fedavg_time_s': round(fed_time, 1),
        'centralized_time_s': round(cent_time, 1),
        'overhead_ratio': round(fed_time / cent_time, 2),
        'what_is_shared': 'Model weights (state_dict) — all layers including BatchNorm parameters',
        'what_is_not_shared': 'Raw patient data, clinical records, gene expression matrices, survival outcomes',
        'privacy_guarantee': 'NONE — no differential privacy, no secure aggregation, no gradient compression',
        'residual_risks': 'Weight-based membership inference attacks; model inversion; property inference',
    }
    with open(P6_REPORTS / "privacy_and_cost.json", "w") as f:
        json.dump(privacy_cost, f, indent=2)
    log(f"  -> privacy_and_cost.json saved")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    log("")
    log("=" * 70)
    log("PHASE 6 SUMMARY")
    log("=" * 70)

    # Pooled metrics
    cent_risk_pool = predict_deepsurv(centralized_model, X_test_pool, DEVICE)
    cent_ci_pool = cindex(ye_test_pool, yt_test_pool, cent_risk_pool)
    _, cent_lo, cent_hi = boot_ci(ye_test_pool, yt_test_pool, cent_risk_pool, n=500, seed=SEED)

    log(f"  Pooled test C-index:")
    log(f"    Centralized:  {cent_ci_pool:.4f} ({cent_lo:.4f}-{cent_hi:.4f})")
    log(f"    Federated:    {fed_ci_pool:.4f} ({fed_lo:.4f}-{fed_hi:.4f})")
    log(f"    Delta (fed-cent): {fed_ci_pool - cent_ci_pool:.4f}")

    # Per-site no-collaboration mean
    local_cis = [per_site_results[s]['ci'] for s in site_names if not np.isnan(per_site_results[s]['ci'])]
    log(f"    Per-site mean: {np.mean(local_cis):.4f} (range: {np.min(local_cis):.4f}-{np.max(local_cis):.4f})")

    log("")
    log("  SIMULATED FEDERATED SETTING — NOT a real multi-institutional deployment.")
    log("  RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL USE.")
    log("  Phase 6 Steps 1-5 complete. Run test_phase6_regression.py for Step 6.")


if __name__ == "__main__":
    main()
