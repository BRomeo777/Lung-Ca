"""
Phase 5 Regression Tests — prevents silent reintroduction of the sign-negation bug.

Tests:
1. DeepSurv C-index in Phase 5 matches Phase 3 (±0.01)
2. CoxPH C-index in Phase 5 matches Phase 2 (±0.01)
3. Fusion sign convention: higher risk → shorter survival (synthetic test)
4. Strategy C fusion: w_neural near 1.0 produces hybrid ≈ DeepSurv
5. compute_cindex does NOT negate risk internally
"""
import sys, json, pickle, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import pytest
from sksurv.metrics import concordance_index_censored

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))

READY = P / "03_ANALYSIS_READY_DATA"
ML = P / "ML_RESULTS"
P3M = P / "PHASE3_DEEP_LEARNING" / "models"
P5D = P / "PHASE5_NEURAL_MECHANISTIC_INTEGRATION" / "data"
P5R = P / "PHASE5_NEURAL_MECHANISTIC_INTEGRATION" / "reports"
P5S = P / "PHASE5_NEURAL_MECHANISTIC_INTEGRATION" / "src"
INVALID = ["TCGA-05-4395", "TCGA-77-A5G6"]

# Phase reference values (from Phase 2 and Phase 3 final reports)
P3_DS_CINDEX = 0.6131
P2_COX_CINDEX = 0.5881


class DeepSurv(nn.Module):
    def __init__(self, n, hd=[64, 32], do=0.2, act="gelu"):
        super().__init__()
        fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[act]
        ls = []; d = n
        for h in hd:
            ls += [nn.BatchNorm1d(d), nn.Linear(d, h), fn(), nn.Dropout(do)]; d = h
        o = nn.Linear(d, 1); nn.init.xavier_uniform_(o.weight); nn.init.zeros_(o.bias)
        ls.append(o); self.network = nn.Sequential(*ls)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _norm_log2cpm(c):
    lib = c.sum(axis=0); lib[lib == 0] = 1
    return np.log2(c.div(lib, axis=1) * 1e6 + 1)


def _map_ct(v):
    s = str(v).lower()
    return 'LUAD_Adenocarcinoma' if 'adc' in s or 'adenocarcinoma' in s else 'LUSC_SquamousCell' if 'sqc' in s or 'squamous' in s or 'scc' in s else None


def _map_st(v):
    s = str(v).strip()
    return s if s in ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV"] else None


def _map_sm(v):
    s = str(v).lower()
    return "Former" if "former" in s else "Current" if "current" in s else "Never" if "never" in s else None


def _build_features(clin, expr, gf, fn, am, sc):
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
                if cn == 'Cancer_Type': m = cm[cn].apply(_map_ct)
                elif cn == 'Stage': m = cm[cn].apply(_map_st)
                else: m = cm[cn].apply(_map_sm)
                ce[f] = (m == cat).astype(int)
    X = pd.concat([ce, es], axis=1)
    for f in fn:
        if f not in X.columns: X[f] = 0.0
    X = X[fn]
    return sc.transform(X.values), cm


@pytest.fixture(scope="module")
def val_data():
    """Load validation features and model predictions once for all tests."""
    cfg = json.load(open(P3M / "final_config.json"))
    scaler = pickle.load(open(P3M / "final_scaler.pkl", "rb"))
    cox = pickle.load(open(ML / "models" / "cox_ph_model.pkl", "rb"))
    fn = cfg["feature_names"]; am = cfg["age_median"]
    gf = [f for f in fn if "=" not in f and f != "Age"]
    models = []
    for i in range(cfg["n_models"]):
        m = DeepSurv(len(fn), cfg["arch"]["hidden_dims"], cfg["arch"]["dropout"], cfg["arch"]["activation"])
        m.load_state_dict(torch.load(P3M / f"final_deepsurv_model_{i}.pt", map_location="cpu"))
        models.append(m)

    va = pd.read_csv(READY / "TCGA_internal_validation.csv")
    va = va[~va['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
    eva = _norm_log2cpm(pd.read_csv(READY / "TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0))
    va = va[va['Patient_ID'].astype(str).isin(eva.columns)].reset_index(drop=True)
    Xva, cva = _build_features(va, eva, gf, fn, am, scaler)
    yva_t = pd.to_numeric(cva['Overall_Survival_Time'], errors='coerce').values
    yva_e = pd.to_numeric(cva['Survival_Status'], errors='coerce').fillna(0).astype(int).values

    with torch.no_grad():
        Xt = torch.FloatTensor(Xva)
        ds_risk = np.mean([m.eval()(Xt).cpu().numpy() for m in models], axis=0)
    cox_risk = cox.predict_partial_hazard(pd.DataFrame(Xva, columns=fn)).values.ravel()

    return {
        'X': Xva, 'y_event': yva_e, 'y_time': yva_t,
        'ds_risk': ds_risk, 'cox_risk': cox_risk, 'feature_names': fn,
    }


class TestCIndexReproduction:
    """Test 1 & 2: Phase 5 C-index matches Phase 2/3 reference values."""

    def test_deepsurv_matches_phase3(self, val_data):
        """DeepSurv C-index in Phase 5 must match Phase 3 (0.6131 ± 0.01)."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.evaluation import compute_cindex
        cindex_p5 = compute_cindex(val_data['y_event'], val_data['y_time'], val_data['ds_risk'])
        assert abs(cindex_p5 - P3_DS_CINDEX) < 0.01, \
            f"DeepSurv C-index {cindex_p5:.4f} deviates from Phase 3 ({P3_DS_CINDEX})"

    def test_coxph_matches_phase2(self, val_data):
        """CoxPH C-index in Phase 5 must match Phase 2 (0.5881 ± 0.01)."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.evaluation import compute_cindex
        cindex_p5 = compute_cindex(val_data['y_event'], val_data['y_time'], val_data['cox_risk'])
        assert abs(cindex_p5 - P2_COX_CINDEX) < 0.01, \
            f"CoxPH C-index {cindex_p5:.4f} deviates from Phase 2 ({P2_COX_CINDEX})"


class TestSignConvention:
    """Test 3: Fusion sign convention — higher risk = shorter survival."""

    def test_compute_cindex_no_negation(self):
        """compute_cindex must NOT negate risk scores internally."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.evaluation import compute_cindex
        # Synthetic: higher risk should mean shorter survival
        y_event = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_time = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        risk = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # Perfect: high risk = short time
        c = compute_cindex(y_event, y_time, risk)
        assert c > 0.99, f"compute_cindex with correct sign should give C-index ~1.0, got {c:.4f}"

        # If negated, would give ~0.0 — this catches the bug
        c_neg = concordance_index_censored(y_event.astype(bool), y_time, -risk)[0]
        assert c_neg < 0.01, "Sanity check: negated risk should give C-index ~0.0"
        assert c > 0.5, "compute_cindex must not invert the sign convention"

    def test_fusion_monotonic_direction(self):
        """Strategy C fusion: increasing DeepSurv risk should increase hybrid risk
        when w_neural > 0.5 (i.e., hybrid should move in same direction as DeepSurv)."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.strategy_c import PredictionLevelFusion
        # Synthetic data
        np.random.seed(42)
        n = 100
        ds_risk = np.random.randn(n)
        ttp = np.random.exponential(300, n)
        y_time = np.random.exponential(500, n)
        y_event = np.random.binomial(1, 0.5, n)
        sc = PredictionLevelFusion(normalization_method="rank")
        sc.fit(ds_risk, ttp, y_time, y_event)
        hybrid = sc.predict(ds_risk, ttp)
        # When w_neural is high, hybrid should correlate positively with ds_risk
        rho = np.corrcoef(ds_risk, hybrid)[0, 1]
        assert rho > 0.5, \
            f"Hybrid risk should correlate positively with DeepSurv risk (rho={rho:.4f}), " \
            f"w_neural={sc.optimal_w_neural:.4f}. Sign convention may be inverted."

    def test_bootstrap_cindex_no_negation(self):
        """bootstrap_cindex must NOT negate risk scores."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.evaluation import bootstrap_cindex
        y_event = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_time = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        risk = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        mean_ci, lo, hi = bootstrap_cindex(y_event, y_time, risk, n_bootstrap=100, rng=np.random.default_rng(42))
        assert mean_ci > 0.9, \
            f"bootstrap_cindex with correct sign should give ~1.0, got {mean_ci:.4f}"


class TestFusionWeight:
    """Test 4: When w_neural ≈ 1.0, hybrid ≈ DeepSurv (no inversion)."""

    def test_w_neural_near_one_gives_deepurv(self, val_data):
        """With w_neural=0.9938, hybrid risk should be nearly identical to DeepSurv risk."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.strategy_c import PredictionLevelFusion
        mva = pd.read_csv(P5D / "mechanistic_features_val.csv")
        mtr = pd.read_csv(P5D / "mechanistic_features_train.csv")
        ttp_tr = mtr["TTP_natural"].values
        ttp_va = mva["TTP_natural"].values

        # Load training risk
        cfg = json.load(open(P3M / "final_config.json"))
        scaler = pickle.load(open(P3M / "final_scaler.pkl", "rb"))
        fn = cfg["feature_names"]; am = cfg["age_median"]
        gf = [f for f in fn if "=" not in f and f != "Age"]
        models = []
        for i in range(cfg["n_models"]):
            m = DeepSurv(len(fn), cfg["arch"]["hidden_dims"], cfg["arch"]["dropout"], cfg["arch"]["activation"])
            m.load_state_dict(torch.load(P3M / f"final_deepsurv_model_{i}.pt", map_location="cpu"))
            models.append(m)
        tr = pd.read_csv(READY / "TCGA_train.csv")
        tr = tr[~tr['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
        etr = _norm_log2cpm(pd.read_csv(READY / "TCGA_train_expression.tsv.gz", sep="\t", index_col=0))
        tr = tr[tr['Patient_ID'].astype(str).isin(etr.columns)].reset_index(drop=True)
        Xtr, ctr = _build_features(tr, etr, gf, fn, am, scaler)
        ytr_t = pd.to_numeric(ctr['Overall_Survival_Time'], errors='coerce').values
        ytr_e = pd.to_numeric(ctr['Survival_Status'], errors='coerce').fillna(0).astype(int).values
        with torch.no_grad():
            Xt = torch.FloatTensor(Xtr)
            dsr_tr = np.mean([m.eval()(Xt).cpu().numpy() for m in models], axis=0)

        sc = PredictionLevelFusion(normalization_method="rank")
        sc.fit(dsr_tr, ttp_tr, ytr_t, ytr_e)
        hybrid = sc.predict(val_data['ds_risk'], ttp_va)

        # C-index of hybrid should match DeepSurv (since w_neural ≈ 1.0)
        ci_ds = concordance_index_censored(
            val_data['y_event'].astype(bool), val_data['y_time'], val_data['ds_risk'])[0]
        ci_hy = concordance_index_censored(
            val_data['y_event'].astype(bool), val_data['y_time'], hybrid)[0]
        assert abs(ci_ds - ci_hy) < 0.005, \
            f"Hybrid C-index ({ci_hy:.4f}) should match DeepSurv ({ci_ds:.4f}) when w_neural={sc.optimal_w_neural:.4f}"


class TestFeatureChecksum:
    """Test 5: Feature matrix checksum stability."""

    def test_feature_checksum(self, val_data):
        """Feature matrix must match the known checksum from the audit."""
        cksum = hashlib.md5(val_data['X'].tobytes()).hexdigest()
        assert cksum == "85ddb1f456d04bdbe06befa59dbffdf0", \
            f"Feature checksum changed: {cksum} (expected: 85ddb1f456d04bdbe06befa59dbffdf0)"


class TestPersonalizationParity:
    """Test 6: Personalization parity — validation/GEO patients must be personalized,
    not stuck on population priors. Guards against the personalization gap reopening."""

    def test_personalized_mechanistic_features_exist(self):
        """Personalized mechanistic features CSV must exist for validation set."""
        pers_path = P5D / "mechanistic_features_val_personalized.csv"
        assert pers_path.exists(), \
            f"Personalized mechanistic features not found: {pers_path}. " \
            "Run phase5_personalization_parity.py first."

    def test_personalization_variance_increase(self):
        """Personalized TTP must have substantially more variance than population-prior TTP.
        Before personalization: 8 unique TTP values. After: should be >50."""
        mva_old = pd.read_csv(P5D / "mechanistic_features_val.csv")
        mva_pers = pd.read_csv(P5D / "mechanistic_features_val_personalized.csv")

        old_unique = mva_old["TTP_natural"].nunique()
        pers_unique = mva_pers["TTP_natural"].nunique()

        assert pers_unique > 50, \
            f"Personalized TTP has only {pers_unique} unique values (expected >50). " \
            f"Personalization may not have taken effect."
        assert pers_unique > old_unique * 2, \
            f"Personalized TTP unique values ({pers_unique}) should be >2x population-prior ({old_unique})."

    def test_personalization_variance_std(self):
        """Personalized growth_rate_alpha must have non-zero std (population priors had std=0)."""
        mva_pers = pd.read_csv(P5D / "mechanistic_features_val_personalized.csv")
        std_alpha = mva_pers["growth_rate_alpha"].std()
        assert std_alpha > 1e-6, \
            f"Personalized growth_rate_alpha std={std_alpha:.6e} (expected >0). " \
            "Personalization may not have taken effect."

    def test_pct_patients_personalized(self):
        """At least 95% of validation patients must be personalized."""
        mva_old = pd.read_csv(P5D / "mechanistic_features_val.csv")
        mva_pers = pd.read_csv(P5D / "mechanistic_features_val_personalized.csv")

        # Count patients where alpha differs from the population prior
        # Population prior alpha = 0.0008; personalized should vary
        n_total = len(mva_pers)
        n_personalized = (mva_pers["growth_rate_alpha"] != mva_old["growth_rate_alpha"].iloc[0]).sum()
        pct = n_personalized / n_total if n_total > 0 else 0

        assert pct >= 0.95, \
            f"Only {pct:.1%} ({n_personalized}/{n_total}) of validation patients personalized. " \
            f"Expected >=95%."


class TestPersonalizedHybridNoRegression:
    """Test 7: Personalized hybrid must not regress below DeepSurv.
    Even with personalization, the null result (hybrid ≈ DeepSurv) should hold."""

    def test_personalized_strategy_c_matches_deepurv(self, val_data):
        """Strategy C with personalized TTP should still match DeepSurv (null result holds)."""
        from PHASE5_NEURAL_MECHANISTIC_INTEGRATION.src.strategy_c import PredictionLevelFusion

        mtr = pd.read_csv(P5D / "mechanistic_features_train.csv")
        mva_pers = pd.read_csv(P5D / "mechanistic_features_val_personalized.csv")
        ttp_tr = mtr["TTP_natural"].values
        ttp_va_pers = mva_pers["TTP_natural"].values

        # Load training risk
        cfg = json.load(open(P3M / "final_config.json"))
        scaler = pickle.load(open(P3M / "final_scaler.pkl", "rb"))
        fn = cfg["feature_names"]; am = cfg["age_median"]
        gf = [f for f in fn if "=" not in f and f != "Age"]
        models = []
        for i in range(cfg["n_models"]):
            m = DeepSurv(len(fn), cfg["arch"]["hidden_dims"], cfg["arch"]["dropout"], cfg["arch"]["activation"])
            m.load_state_dict(torch.load(P3M / f"final_deepsurv_model_{i}.pt", map_location="cpu"))
            models.append(m)
        tr = pd.read_csv(READY / "TCGA_train.csv")
        tr = tr[~tr['Patient_ID'].isin(INVALID)].dropna(subset=["Overall_Survival_Time", "Survival_Status"]).reset_index(drop=True)
        etr = _norm_log2cpm(pd.read_csv(READY / "TCGA_train_expression.tsv.gz", sep="\t", index_col=0))
        tr = tr[tr['Patient_ID'].astype(str).isin(etr.columns)].reset_index(drop=True)
        Xtr, ctr = _build_features(tr, etr, gf, fn, am, scaler)
        ytr_t = pd.to_numeric(ctr['Overall_Survival_Time'], errors='coerce').values
        ytr_e = pd.to_numeric(ctr['Survival_Status'], errors='coerce').fillna(0).astype(int).values
        with torch.no_grad():
            Xt = torch.FloatTensor(Xtr)
            dsr_tr = np.mean([m.eval()(Xt).cpu().numpy() for m in models], axis=0)

        sc = PredictionLevelFusion(normalization_method="rank")
        sc.fit(dsr_tr, ttp_tr, ytr_t, ytr_e)
        hybrid_pers = sc.predict(val_data['ds_risk'], ttp_va_pers)

        ci_ds = concordance_index_censored(
            val_data['y_event'].astype(bool), val_data['y_time'], val_data['ds_risk'])[0]
        ci_hy = concordance_index_censored(
            val_data['y_event'].astype(bool), val_data['y_time'], hybrid_pers)[0]

        # With w_neural ≈ 1.0, hybrid should match DeepSurv (null result)
        assert abs(ci_ds - ci_hy) < 0.005, \
            f"Personalized hybrid C-index ({ci_hy:.4f}) should match DeepSurv ({ci_ds:.4f}). " \
            f"w_neural={sc.optimal_w_neural:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
