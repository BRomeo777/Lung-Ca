"""
Phase 8 CDSS — Digital Twin Simulation Module
Runs ODE tumor growth simulations for treatment scenarios.
Carries mandatory artifact warning on every treatment output.
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .config import (
    SIM_HORIZON_DAYS, SIM_N_POINTS, TREATMENT_ARTIFACT_WARNING,
    TREATMENT_LABELS, MODEL_VERSION,
)


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


def make_cisplatin_fn(dose=75.0, k_el=0.552, V_d=30.0, cycle_interval=21.0, n_cycles=4):
    C_peak = dose / V_d
    def C(t, drug_type="chemo"):
        if t < 0: return 0.0
        ci = int(t // cycle_interval)
        if ci >= n_cycles: return 0.0
        tc = t - ci * cycle_interval
        if tc < 1/24: return C_peak * (1 - np.exp(-k_el * tc)) / (1 - np.exp(-k_el / 24))
        return C_peak * np.exp(-k_el * (tc - 1/24))
    return C

def make_pembrolizumab_fn(dose=200.0, k_el=0.0269, V_d=7.66, cycle_interval=21.0, n_cycles=6):
    C_peak = dose / V_d
    def C(t, drug_type="immuno"):
        if t < 0: return 0.0
        ci = int(t // cycle_interval)
        if ci >= n_cycles: return 0.0
        tc = t - ci * cycle_interval
        return C_peak * np.exp(-k_el * tc)
    return C

def make_osimertinib_fn():
    C_max_ss = 0.501
    k_el = 0.346
    k_a = 2.77
    def C(t, drug_type="targeted"):
        if t < 0: return 0.0
        td = t % 24.0
        if td < 1.0 / k_a: return C_max_ss * (1 - np.exp(-k_a * td))
        return C_max_ss * np.exp(-k_el * (td - 1.0 / k_a))
    return C

def make_no_drug_fn():
    return lambda t, dt: 0.0

DRUG_FNS = {
    "none": make_no_drug_fn,
    "chemo": make_cisplatin_fn,
    "immuno": make_pembrolizumab_fn,
    "targeted": make_osimertinib_fn,
}


def solve_tumor_trajectory(patient_params: dict, treatment: str = "none",
                            t_span: tuple = (0, SIM_HORIZON_DAYS),
                            t_eval: np.ndarray | None = None) -> dict:
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], SIM_N_POINTS)

    defaults = {
        "alpha": 0.0008, "V_max": 1e6, "V0": 8000.0, "E0": 1.3e5,
        "delta_c": 0.0, "delta_t": 0.0, "s_E": 1.3e4, "mu_E": 0.041,
        "rho": 0.02, "sigma": 2e7, "delta_EV": 3.4e-10,
        "k_immune": 1.22e-4, "E_ref": 1.3e5, "k_immuno": 2.5e-5,
    }
    params = dict(patient_params)
    for key, dv in defaults.items():
        val = params.get(key, dv)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            params[key] = dv

    V0 = np.clip(params.get("V0", 8000.0), 100.0, 500000.0)
    E0 = params.get("E0", 1.3e5)

    if treatment == "none":
        def ode_fn(t, y):
            return [gompertz_growth(t, y[0], params["alpha"], params["V_max"])]
        sol = solve_ivp(ode_fn, t_span, [V0], t_eval=t_eval, method="RK45",
                        rtol=1e-8, atol=1e-10, max_step=10.0)
    else:
        params["drug_type"] = treatment
        drug_fn = DRUG_FNS.get(treatment, make_no_drug_fn)()
        def ode_fn(t, y):
            return treatment_response_ode(t, y, params, drug_fn)
        sol = solve_ivp(ode_fn, t_span, [V0, E0], t_eval=t_eval, method="Radau",
                        rtol=1e-6, atol=1e-8, max_step=10.0)

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
        raise RuntimeError("ODE solver produced NaN/Inf")

    V = np.clip(sol.y[0], 0, None)
    return {
        "t": sol.t,
        "V": V,
        "E": sol.y[1] if treatment != "none" else None,
        "params": params,
        "treatment": treatment,
    }


def compute_ttp(traj: dict, progression_threshold: float = 2.0) -> float:
    V0 = traj["V"][0]
    threshold = V0 * progression_threshold
    idx = np.where(traj["V"] >= threshold)[0]
    if len(idx) == 0:
        return float(SIM_HORIZON_DAYS)
    return float(traj["t"][idx[0]])


class DigitalTwinEngine:
    """Manages Digital Twin simulations for the CDSS."""

    def __init__(self):
        self._twin_df = None
        self._state_df = None

    def load(self):
        import pandas as pd
        from .config import P7_DATA
        self._twin_df = pd.read_csv(P7_DATA / "twin_summaries_for_interface.csv")
        self._state_df = pd.read_csv(P7_DATA / "digital_twin_state_matrix.csv")

    def get_existing_trajectories(self, patient_id: str) -> dict | None:
        """Get pre-computed trajectories for an existing patient."""
        if self._twin_df is None:
            self.load()
        row = self._twin_df[self._twin_df["patient_id"].astype(str) == str(patient_id)]
        if len(row) == 0:
            return None
        row = row.iloc[0]
        return {
            "patient_id": str(row["patient_id"]),
            "ttp_natural": float(row["ttp_natural"]),
            "ttp_chemo": float(row["ttp_chemo"]),
            "ttp_immuno": float(row["ttp_immuno"]),
            "ttp_targeted": float(row["ttp_targeted"]),
            "alpha": float(row["alpha"]),
            "V0": float(row["V0"]),
            "V_max": float(row["V_max"]),
            "artifact_warning": TREATMENT_ARTIFACT_WARNING,
            "model_version": MODEL_VERSION,
        }

    def simulate_treatment(self, patient_params: dict, treatment: str = "none") -> dict:
        """Run a new treatment simulation for a patient."""
        traj = solve_tumor_trajectory(patient_params, treatment=treatment)
        ttp = compute_ttp(traj)
        ttp_months = ttp / 30.44

        result = {
            "treatment": treatment,
            "treatment_label": TREATMENT_LABELS.get(treatment, treatment),
            "ttp_days": ttp,
            "ttp_months": ttp_months,
            "trajectory_t": traj["t"].tolist(),
            "trajectory_V": traj["V"].tolist(),
            "trajectory_t_months": (traj["t"] / 30.44).tolist(),
            "trajectory_diameter_mm": (
                2.0 * (3.0 * traj["V"] / (4.0 * np.pi)) ** (1.0 / 3.0)
            ).tolist(),
            "is_simulation": True,
            "is_counterfactual": treatment != "none",
            "artifact_warning": TREATMENT_ARTIFACT_WARNING,
            "model_version": MODEL_VERSION,
        }
        return result

    def simulate_all_treatments(self, patient_params: dict) -> dict:
        """Run all 4 treatment scenarios for a patient."""
        results = {}
        for tx in ["none", "chemo", "immuno", "targeted"]:
            try:
                results[tx] = self.simulate_treatment(patient_params, tx)
            except Exception as e:
                results[tx] = {
                    "treatment": tx,
                    "error": str(e),
                    "artifact_warning": TREATMENT_ARTIFACT_WARNING,
                }
        return results

    def get_population_priors(self) -> dict:
        """Get population-average ODE parameters for patients without personalized params."""
        if self._state_df is None:
            self.load()
        return {
            "alpha": float(self._state_df["alpha"].median()),
            "V_max": float(self._state_df["V_max"].median()),
            "V0": float(self._state_df["V0"].median()),
            "E0": float(self._state_df["E0"].median()),
            "k_immune": float(self._state_df["k_immune"].median()),
            "mu_E": float(self._state_df["mu_E"].median()),
            "rho": float(self._state_df["rho"].median()),
            "delta_c": float(self._state_df["delta_c"].median()),
            "delta_t": float(self._state_df["delta_t"].median()),
        }
