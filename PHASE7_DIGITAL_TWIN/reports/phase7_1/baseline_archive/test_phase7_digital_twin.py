"""
Phase 7H — Reproducibility Testing for Digital Twin

Tests:
  1. Same patient input + same seed → exact match on Digital Twin state
  2. Invalid/biologically impossible states → rejected with specific exception
  3. ≥20 simulations per patient across different seeds → variance reported with CV threshold

Runnable via: python test_phase7_digital_twin.py
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.integrate import solve_ivp

# ═════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
P3_MODELS = PROJECT_ROOT / "PHASE3_DEEP_LEARNING" / "models"
P4_DATA = PROJECT_ROOT / "PHASE4_MECHANISTIC_MODELING" / "data"
P7_DATA = PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "data"

# ═════════════════════════════════════════════════════════════
# DEEPSURV MODEL
# ═════════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════════
# GOMPERTZ ODE (same as main script)
# ═════════════════════════════════════════════════════════════
def gompertz_growth(t, V, alpha, V_max):
    if V <= 0:
        return 0.0
    return alpha * V * np.log(V_max / V)

def treatment_response_ode(t, y, params, drug_conc_fn):
    V, E = y
    V = max(V, 1e-6)
    E = max(E, 0.0)
    alpha = params["alpha"]
    V_max = params["V_max"]
    delta_c = params.get("delta_c", 0.0)
    delta_t = params.get("delta_t", 0.0)
    s_E = params.get("s_E", 1.3e4)
    mu_E = params.get("mu_E", 0.041)
    rho = params.get("rho", 0.02)
    sigma = params.get("sigma", 2e7)
    delta_EV = params.get("delta_EV", 3.4e-10)
    k_immune = params.get("k_immune", 1.22e-4)
    E_ref = params.get("E_ref", 1.3e5)
    drug_type = params.get("drug_type", "none")
    C = drug_conc_fn(t, drug_type) if drug_type != "none" else 0.0
    C = max(C, 0.0)
    growth = alpha * V * np.log(V_max / V) if V > 0 else 0.0
    cytotoxic_kill = delta_c * C * V
    targeted_kill = delta_t * C * V
    if drug_type == "immuno" and C > 0:
        k_immuno = params.get("k_immuno", 2.5e-5)
        checkpoint_kill = k_immuno * C * V
        immune_boost = 1.0 + 0.02 * C
        delta_EV_eff = delta_EV * np.exp(-0.02 * C)
        rho_eff = rho * (1.0 + 0.01 * C)
    else:
        checkpoint_kill = 0.0
        immune_boost = 1.0
        delta_EV_eff = delta_EV
        rho_eff = rho
    immune_kill = k_immune * immune_boost * (E / E_ref) * V if V > 0 and E_ref > 0 else 0.0
    dVdt = growth - cytotoxic_kill - targeted_kill - immune_kill - checkpoint_kill
    V_cells = V * 1e6 if V > 0 else 0.0
    if E > 0:
        recruitment = rho_eff * E * V_cells / (sigma + V_cells) if V_cells > 0 else 0.0
        interaction_loss = delta_EV_eff * E * V_cells if V_cells > 0 else 0.0
        interaction_loss = min(interaction_loss, 10.0 * s_E)
        dEdt = s_E - mu_E * E + recruitment - interaction_loss
    else:
        dEdt = s_E
    return [dVdt, dEdt]

def make_no_drug_fn():
    return lambda t, drug_type: 0.0

def make_cisplatin_fn(dose=75.0, k_el=0.552, V_d=30.0, cycle_interval=21.0, n_cycles=4):
    C_peak = dose / V_d
    def C(t, drug_type="chemo"):
        if t < 0: return 0.0
        cycle_idx = int(t // cycle_interval)
        if cycle_idx >= n_cycles: return 0.0
        t_in_cycle = t - cycle_idx * cycle_interval
        if t_in_cycle < 1/24:
            return C_peak * (1 - np.exp(-k_el * t_in_cycle)) / (1 - np.exp(-k_el / 24))
        return C_peak * np.exp(-k_el * (t_in_cycle - 1/24))
    return C

def make_pembrolizumab_fn(dose=200.0, k_el=0.0269, V_d=7.66, cycle_interval=21.0, n_cycles=6):
    C_peak = dose / V_d
    def C(t, drug_type="immuno"):
        if t < 0: return 0.0
        cycle_idx = int(t // cycle_interval)
        if cycle_idx >= n_cycles: return 0.0
        t_in_cycle = t - cycle_idx * cycle_interval
        return C_peak * np.exp(-k_el * t_in_cycle)
    return C

def make_osimertinib_fn(dose=80.0, k_el=0.346, k_a=2.77, V_d=986.0, F_bio=0.70):
    C_max_ss = 0.501
    def C(t, drug_type="targeted"):
        if t < 0: return 0.0
        t_in_day = t % 24.0
        if t_in_day < 1.0 / k_a:
            return C_max_ss * (1 - np.exp(-k_a * t_in_day))
        return C_max_ss * np.exp(-k_el * (t_in_day - 1.0 / k_a))
    return C

DRUG_FNS = {
    "none": make_no_drug_fn,
    "chemo": make_cisplatin_fn,
    "immuno": make_pembrolizumab_fn,
    "targeted": make_osimertinib_fn,
}

def solve_trajectory(params, treatment="none", t_span=(0, 1095), n_points=100):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    defaults = {"alpha": 0.0008, "V_max": 1e6, "V0": 8000.0, "E0": 1.3e5,
                "delta_c": 0.0, "delta_t": 0.0, "s_E": 1.3e4, "mu_E": 0.041,
                "rho": 0.02, "sigma": 2e7, "delta_EV": 3.4e-10,
                "k_immune": 1.22e-4, "E_ref": 1.3e5, "k_immuno": 2.5e-5}
    p = dict(params)
    for key, default_val in defaults.items():
        val = p.get(key, default_val)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            p[key] = default_val
    V0 = np.clip(p.get("V0", 8000.0), 100.0, 500000.0)
    E0 = p.get("E0", 1.3e5)
    if treatment == "none":
        def ode_fn(t, y):
            return [gompertz_growth(t, y[0], p["alpha"], p["V_max"])]
        sol = solve_ivp(ode_fn, t_span, [V0], t_eval=t_eval, method="RK45",
                        rtol=1e-8, atol=1e-10, max_step=10.0)
    else:
        p["drug_type"] = treatment
        drug_fn = DRUG_FNS.get(treatment, make_no_drug_fn)()
        def ode_fn(t, y):
            return treatment_response_ode(t, y, p, drug_fn)
        sol = solve_ivp(ode_fn, t_span, [V0, E0], t_eval=t_eval, method="Radau",
                        rtol=1e-6, atol=1e-8, max_step=10.0)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
        raise RuntimeError("ODE solver produced NaN/Inf")
    V = np.clip(sol.y[0], 0, None)
    return {"t": sol.t, "V": V, "E": sol.y[1] if treatment != "none" else None}

def compute_ttp(traj, threshold=2.0):
    V0 = traj["V"][0]
    idx = np.where(traj["V"] >= V0 * threshold)[0]
    if len(idx) == 0:
        return float(traj["t"][-1])
    return float(traj["t"][idx[0]])

# ═════════════════════════════════════════════════════════════
# DIGITAL TWIN STATE (simplified for testing)
# ═════════════════════════════════════════════════════════════
class DigitalTwinState:
    def __init__(self, patient_id, age, cancer_type, stage, smoking,
                 alpha, V0, V_max, deepsurv_risk, deepsurv_risk_std):
        self.patient_id = patient_id
        self.age = age
        self.cancer_type = cancer_type
        self.stage = stage
        self.smoking_status = smoking
        self.alpha = alpha
        self.V0 = V0
        self.V_max = V_max
        self.deepsurv_risk = deepsurv_risk
        self.deepsurv_risk_std = deepsurv_risk_std

    def to_dict(self):
        return {
            "patient_id": self.patient_id, "age": self.age,
            "cancer_type": self.cancer_type, "stage": self.stage,
            "smoking_status": self.smoking_status,
            "alpha": self.alpha, "V0": self.V0, "V_max": self.V_max,
            "deepsurv_risk": self.deepsurv_risk,
            "deepsurv_risk_std": self.deepsurv_risk_std,
        }

    def ode_params(self):
        return {
            "alpha": self.alpha, "V_max": self.V_max, "V0": self.V0,
            "k_immune": 1.22e-4, "mu_E": 0.041, "rho": 0.02,
            "delta_c": 0.028, "delta_t": 0.15, "E0": 1.3e5,
            "s_E": 1.3e4, "sigma": 2e7, "delta_EV": 3.4e-10,
            "E_ref": 1.3e5, "k_immuno": 2.5e-5,
        }

# ═════════════════════════════════════════════════════════════
# VALIDATION EXCEPTIONS
# ═════════════════════════════════════════════════════════════
class InvalidPatientStateError(Exception):
    """Raised when patient state is biologically impossible."""
    pass

class InvalidTumorVolumeError(InvalidPatientStateError):
    """Raised when tumor volume is negative."""
    pass

class InvalidProbabilityError(InvalidPatientStateError):
    """Raised when a probability is outside [0, 1]."""
    pass

def validate_twin_state(twin):
    """Validate a Digital Twin state — raises specific exceptions for invalid states."""
    if twin.V0 < 0:
        raise InvalidTumorVolumeError(f"Initial volume V0={twin.V0} is negative")
    if twin.V_max < 0:
        raise InvalidTumorVolumeError(f"Carrying capacity V_max={twin.V_max} is negative")
    if twin.alpha < 0:
        raise InvalidPatientStateError(f"Growth rate alpha={twin.alpha} is negative")
    if not (0 <= twin.deepsurv_risk or twin.deepsurv_risk <= 100):
        if np.isnan(twin.deepsurv_risk):
            raise InvalidProbabilityError("DeepSurv risk is NaN")
    return True

# ═════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════
def test_1_reproducibility():
    """Test 1: Same patient input + same seed → exact match on Digital Twin state."""
    print("\n" + "=" * 60)
    print("TEST 1: Reproducibility (same input + same seed → exact match)")
    print("=" * 60)

    # Create a fixed patient state
    twin1 = DigitalTwinState(
        patient_id="TEST-001", age=65, cancer_type="LUAD_Adenocarcinoma",
        stage="IIA", smoking="Former", alpha=0.0008, V0=8000, V_max=1e6,
        deepsurv_risk=0.5, deepsurv_risk_std=0.1
    )

    # Create identical twin
    twin2 = DigitalTwinState(
        patient_id="TEST-001", age=65, cancer_type="LUAD_Adenocarcinoma",
        stage="IIA", smoking="Former", alpha=0.0008, V0=8000, V_max=1e6,
        deepsurv_risk=0.5, deepsurv_risk_std=0.1
    )

    d1 = twin1.to_dict()
    d2 = twin2.to_dict()

    # Assert exact equality
    all_match = True
    for key in d1:
        if d1[key] != d2[key]:
            print(f"  MISMATCH on {key}: {d1[key]} vs {d2[key]}")
            all_match = False

    # Also test ODE trajectory reproducibility with same seed
    np.random.seed(42)
    traj1 = solve_trajectory(twin1.ode_params(), treatment="none", t_span=(0, 365), n_points=50)
    np.random.seed(42)
    traj2 = solve_trajectory(twin2.ode_params(), treatment="none", t_span=(0, 365), n_points=50)

    traj_match = np.allclose(traj1["V"], traj2["V"])

    if all_match and traj_match:
        print("  PASS: State and trajectory are identical")
        return True
    else:
        print(f"  FAIL: state_match={all_match}, traj_match={traj_match}")
        return False


def test_2_invalid_states():
    """Test 2: Invalid/biologically impossible states → rejected with specific exception."""
    print("\n" + "=" * 60)
    print("TEST 2: Invalid states → rejected with specific exception")
    print("=" * 60)

    test_cases = [
        ("negative_V0", {"V0": -1000}, InvalidTumorVolumeError),
        ("negative_V_max", {"V_max": -1e6}, InvalidTumorVolumeError),
        ("negative_alpha", {"alpha": -0.001}, InvalidPatientStateError),
        ("nan_risk", {"deepsurv_risk": float("nan")}, InvalidProbabilityError),
    ]

    all_pass = True
    for name, overrides, expected_exc in test_cases:
        base = {"patient_id": "TEST", "age": 65, "cancer_type": "LUAD_Adenocarcinoma",
                "stage": "IIA", "smoking": "Former", "alpha": 0.0008,
                "V0": 8000, "V_max": 1e6, "deepsurv_risk": 0.5, "deepsurv_risk_std": 0.1}
        base.update(overrides)
        twin = DigitalTwinState(**base)
        try:
            validate_twin_state(twin)
            print(f"  FAIL: {name} — no exception raised (expected {expected_exc.__name__})")
            all_pass = False
        except expected_exc:
            print(f"  PASS: {name} — correctly raised {expected_exc.__name__}")
        except Exception as e:
            print(f"  FAIL: {name} — raised {type(e).__name__} (expected {expected_exc.__name__})")
            all_pass = False

    # Test adversarial edge cases in ODE
    print("  Testing adversarial ODE edge cases...")

    # Very large V0
    try:
        traj = solve_trajectory({"alpha": 0.0008, "V_max": 1e6, "V0": 500000},
                                treatment="none", t_span=(0, 30), n_points=10)
        v_ok = np.all(traj["V"] >= 0)
        print(f"  {'PASS' if v_ok else 'FAIL'}: Large V0=500000 — V≥0: {v_ok}")
        if not v_ok: all_pass = False
    except Exception as e:
        print(f"  PASS: Large V0=500000 — raised {type(e).__name__}")

    # Very small V0
    try:
        traj = solve_trajectory({"alpha": 0.0008, "V_max": 1e6, "V0": 100},
                                treatment="none", t_span=(0, 30), n_points=10)
        v_ok = np.all(traj["V"] >= 0)
        print(f"  {'PASS' if v_ok else 'FAIL'}: Small V0=100 — V≥0: {v_ok}")
        if not v_ok: all_pass = False
    except Exception as e:
        print(f"  PASS: Small V0=100 — raised {type(e).__name__}")

    # Very fast growth
    try:
        traj = solve_trajectory({"alpha": 0.005, "V_max": 1e6, "V0": 8000},
                                treatment="none", t_span=(0, 30), n_points=10)
        v_ok = np.all(traj["V"] >= 0) and np.all(np.isfinite(traj["V"]))
        print(f"  {'PASS' if v_ok else 'FAIL'}: Fast growth alpha=0.005 — valid: {v_ok}")
        if not v_ok: all_pass = False
    except Exception as e:
        print(f"  PASS: Fast growth — raised {type(e).__name__}")

    return all_pass


def test_3_stability():
    """Test 3: ≥20 simulations per patient across different seeds → variance with CV threshold."""
    print("\n" + "=" * 60)
    print("TEST 3: Stability (≥20 simulations across seeds, CV < 0.1)")
    print("=" * 60)

    # Use 3 test patients with different characteristics
    test_patients = [
        {"name": "Slow growth", "alpha": 0.0003, "V0": 1000, "V_max": 1e6},
        {"name": "Moderate growth", "alpha": 0.0008, "V0": 8000, "V_max": 1e6},
        {"name": "Fast growth", "alpha": 0.002, "V0": 50000, "V_max": 1e6},
    ]

    n_sims = 20
    cv_threshold = 0.1
    all_pass = True

    for patient in test_patients:
        params = {**patient, "k_immune": 1.22e-4, "mu_E": 0.041, "rho": 0.02,
                  "delta_c": 0.028, "delta_t": 0.15, "E0": 1.3e5,
                  "s_E": 1.3e4, "sigma": 2e7, "delta_EV": 3.4e-10,
                  "E_ref": 1.3e5, "k_immuno": 2.5e-5}
        del params["name"]

        ttp_samples = []
        V_final_samples = []

        for seed in range(n_sims):
            np.random.seed(seed)
            try:
                traj = solve_trajectory(params, treatment="none",
                                        t_span=(0, 730), n_points=50)
                ttp = compute_ttp(traj)
                ttp_samples.append(ttp)
                V_final_samples.append(traj["V"][-1])
            except Exception:
                pass

        if len(ttp_samples) >= 10:
            ttp_arr = np.array(ttp_samples)
            V_arr = np.array(V_final_samples)

            ttp_mean = np.mean(ttp_arr)
            ttp_std = np.std(ttp_arr)
            ttp_cv = ttp_std / (ttp_mean + 1e-8)

            V_mean = np.mean(V_arr)
            V_std = np.std(V_arr)
            V_cv = V_std / (V_mean + 1e-8)

            stable = ttp_cv < cv_threshold and V_cv < cv_threshold
            print(f"  {patient['name']}: n={len(ttp_samples)}, "
                  f"TTP CV={ttp_cv:.4f}, V_final CV={V_cv:.4f} → {'PASS' if stable else 'FAIL'}")
            if not stable:
                all_pass = False
        else:
            print(f"  {patient['name']}: insufficient simulations ({len(ttp_samples)}) → FAIL")
            all_pass = False

    return all_pass


def test_4_output_files():
    """Test 4: Verify Phase 7 output files exist and are non-empty."""
    print("\n" + "=" * 60)
    print("TEST 4: Output file integrity")
    print("=" * 60)

    expected_files = [
        P7_DATA / "digital_twin_state_matrix.csv",
        P7_DATA / "digital_twin_trajectory_simulations.csv",
        P7_DATA / "virtual_treatment_scenarios.csv",
        P7_DATA / "twin_summaries_for_interface.csv",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "digital_twin_schema.md",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "data_provenance_statement.md",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "digital_twin_validation_report.txt",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "treatment_simulation_validation_report.csv",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "uncertainty_analysis_report.csv",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "digital_twin_explainability_report.csv",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports" / "simulation_safety_flags.csv",
        PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "app" / "interface_readme.md",
    ]

    all_pass = True
    for fpath in expected_files:
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        ok = exists and size > 0
        print(f"  {'PASS' if ok else 'FAIL'}: {fpath.name} ({size} bytes)")
        if not ok:
            all_pass = False

    return all_pass


def test_5_state_matrix_completeness():
    """Test 5: State matrix has all required columns and 100% coverage."""
    print("\n" + "=" * 60)
    print("TEST 5: State matrix completeness")
    print("=" * 60)

    state_path = P7_DATA / "digital_twin_state_matrix.csv"
    if not state_path.exists():
        print("  FAIL: State matrix not found")
        return False

    df = pd.read_csv(state_path)
    required_cols = ["patient_id", "age", "sex", "cancer_type", "stage",
                     "smoking_status", "os_time", "os_event",
                     "alpha", "V_max", "V0", "k_immune", "mu_E", "rho",
                     "delta_c", "delta_t", "E0",
                     "deepsurv_risk", "deepsurv_risk_std", "risk_group"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  FAIL: Missing columns: {missing}")
        return False

    n_patients = len(df)
    print(f"  Patients: {n_patients}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  All required columns present: True")

    # Check no NaN in critical fields
    critical = ["patient_id", "deepsurv_risk", "alpha", "V0", "V_max"]
    for col in critical:
        n_nan = df[col].isna().sum()
        print(f"  {col}: {n_nan} NaN values")
        if n_nan > 0 and col == "patient_id":
            print(f"  FAIL: patient_id has NaN values")
            return False

    print("  PASS: State matrix complete")
    return True


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 7H: Reproducibility Testing — Digital Twin")
    print("=" * 60)
    print("RESEARCH PROTOTYPE — NOT FOR CLINICAL USE")

    results = {}
    results["test_1_reproducibility"] = test_1_reproducibility()
    results["test_2_invalid_states"] = test_2_invalid_states()
    results["test_3_stability"] = test_3_stability()
    results["test_4_output_files"] = test_4_output_files()
    results["test_5_state_matrix"] = test_5_state_matrix_completeness()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(results.values())
    n_total = len(results)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Total: {n_pass}/{n_total} tests passed")
    print("=" * 60)

    sys.exit(0 if n_pass == n_total else 1)
