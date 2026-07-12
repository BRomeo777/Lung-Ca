"""
Phase 4 — Core ODE Models for NSCLC Tumor Growth

Implements:
- Gompertz growth model (natural history)
- Logistic growth model (comparison)
- Exponential growth model (baseline)
- Full treatment-response ODE system with immune effector compartment

All parameters derived from published literature (see parameter_literature_table.csv).

RESEARCH PROTOTYPE ONLY. Simulated tumor trajectories are based on population-derived
ODE parameters personalized by genomic features. They represent biologically plausible
scenarios under stated assumptions, not clinically validated predictions. This system
must not be used for clinical decision-making without prospective validation with serial
imaging data and regulatory approval.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Callable


# ═════════════════════════════════════════════════════════════
# MODEL EQUATIONS
# ═════════════════════════════════════════════════════════════

def gompertz_growth(t: float, V: float, alpha: float, V_max: float) -> float:
    """
    Gompertz tumor growth ODE.

    dV/dt = alpha * V * ln(V_max / V)

    Parameters
    ----------
    t : float — time in days
    V : float — tumor volume in mm^3
    alpha : float — Gompertz growth rate (day^-1)
    V_max : float — maximum tumor volume / carrying capacity (mm^3)

    Returns
    -------
    dVdt : float

    References
    ----------
    Benzekry S, et al. Scientific Reports 2019;9:12598.
      — Gompertz model validated for NSCLC primary tumor growth.
    Liu Z, et al. Scientific Reports 2017;7:13646.
      — Gompertz growth used for NSCLC treatment response modeling.
    """
    if V <= 0:
        return 0.0
    return alpha * V * np.log(V_max / V)


def logistic_growth(t: float, V: float, lambda_rate: float, K: float) -> float:
    """Logistic growth: dV/dt = lambda * V * (1 - V/K)."""
    if V <= 0:
        return 0.0
    return lambda_rate * V * (1.0 - V / K)


def exponential_growth(t: float, V: float, lambda_rate: float) -> float:
    """Exponential growth: dV/dt = lambda * V."""
    if V <= 0:
        return 0.0
    return lambda_rate * V


def treatment_response_ode(
    t: float,
    y: np.ndarray,
    params: Dict,
    drug_concentration_fn: Callable,
) -> np.ndarray:
    """
    Full treatment-response ODE system.

    State variables:
        y[0] = V  — tumor volume (mm^3)
        y[1] = E  — immune effector cells (count)

    Growth: Gompertz
    Cytotoxic kill: Skipper log-cell kill (fractional kill proportional to drug concentration)
    Targeted kill: First-order drug-mediated kill
    Immune kill: Fractional model (k_immune * E/E_ref * V) with checkpoint blockade
    Immune dynamics: source + Michaelis-Menten recruitment + death + interaction loss

    Note: The Norton-Simon hypothesis (kill proportional to unperturbed growth rate)
    is NOT implemented here. The simpler Skipper log-cell kill model is used instead,
    where kill = delta * C * V. This is a known simplification.

    Parameters
    ----------
    t : float — time (days)
    y : array [V, E]
    params : dict with keys:
        alpha : float — Gompertz growth rate (day^-1)
        V_max : float — carrying capacity (mm^3)
        delta_c : float — cytotoxic kill rate ((mg/L)^-1 day^-1)
        delta_i : float — immune-mediated kill rate (cell^-1 day^-1)
        delta_t : float — targeted therapy kill rate ((mg/L)^-1 day^-1)
        s_E : float — immune effector source rate (cells/day)
        mu_E : float — immune effector death rate (day^-1)
        rho : float — immune recruitment rate (day^-1)
        sigma : float — half-saturation for recruitment (cells)
        delta_EV : float — immune-tumor interaction loss rate (mm^-3 day^-1)
        drug_type : str — 'chemo', 'immuno', 'targeted', or 'none'
    drug_concentration_fn : callable — returns C(t) for given drug type

    Returns
    -------
    [dV/dt, dE/dt] : np.ndarray

    References
    ----------
    Skipper HE. Cancer Chemother Rep 1964;35:3-4 (log-cell kill hypothesis).
    Norton L, Simon R. Nature Clinical Practice Oncology 2006;3:406-407 (Norton-Simon — NOT implemented, see note above).
    Kuznetsov VA, et al. Bull Math Biol 1994;56:295-346 (immune effector dynamics).
    Wodarz D. J Theor Biol 2005;237:27-37 (checkpoint blockade modeling).
    """
    V, E = y
    V = max(V, 1e-6)   # floor to prevent log(0) and solver issues
    E = max(E, 0.0)

    alpha = params["alpha"]
    V_max = params["V_max"]
    delta_c = params.get("delta_c", 0.0)
    delta_i = params.get("delta_i", 0.0)
    delta_t = params.get("delta_t", 0.0)
    s_E = params.get("s_E", 1.3e4)
    mu_E = params.get("mu_E", 0.041)
    rho = params.get("rho", 0.02)
    sigma = params.get("sigma", 2.0e7)
    delta_EV = params.get("delta_EV", 3.4e-10)

    drug_type = params.get("drug_type", "none")
    C = drug_concentration_fn(t, drug_type) if drug_type != "none" else 0.0
    C = max(C, 0.0)

    # Gompertz growth
    growth = alpha * V * np.log(V_max / V) if V > 0 else 0.0

    # Cytotoxic kill (log-cell kill model, Norton-Simon)
    cytotoxic_kill = delta_c * C * V

    # Targeted therapy kill
    targeted_kill = delta_t * C * V

    # Immune-mediated kill
    # Baseline immune kill uses a fractional model: k_immune * (E/E_ref) * V
    # This avoids units mismatch between E (cells) and V (mm³)
    # For immunotherapy: checkpoint blockade adds direct drug-mediated kill
    # and reduces tumor-mediated immune suppression
    E_ref = params.get("E_ref", 1.3e5)  # reference immune count (= E0)
    k_immune = params.get("k_immune", 1.22e-4)  # baseline immune kill rate (day^-1)

    if drug_type == "immuno" and C > 0:
        # Checkpoint blockade effect:
        # 1. Direct immune-mediated kill proportional to drug concentration
        #    (PD-1/PD-L1 blockade enables T-cell cytotoxicity)
        k_immuno = params.get("k_immuno", 2.5e-5)  # (mg/L)^-1 day^-1
        checkpoint_kill = k_immuno * C * V
        # 2. Boost baseline immune kill
        #    At C_peak_ss ~55 mg/L: boost factor = 1 + 0.02*55 ~ 2.1 (doubles immune kill)
        immune_boost = 1.0 + 0.02 * C
        # 3. Reduce tumor-mediated immune suppression (drug blocks PD-L1 signaling)
        #    At C_peak_ss ~55 mg/L: delta_EV reduced by exp(-0.02*55) ~ 0.33 (67% reduction)
        delta_EV_eff = delta_EV * np.exp(-0.02 * C)
        # 4. Modest recruitment boost
        #    At C_peak_ss ~55 mg/L: rho_eff = rho * 1.55 (55% increase)
        rho_eff = rho * (1.0 + 0.01 * C)
    else:
        checkpoint_kill = 0.0
        immune_boost = 1.0
        delta_EV_eff = delta_EV
        rho_eff = rho

    # Baseline immune kill (fractional model, mm³/day)
    immune_kill = k_immune * immune_boost * (E / E_ref) * V if V > 0 and E_ref > 0 else 0.0

    # Tumor volume dynamics
    dVdt = growth - cytotoxic_kill - targeted_kill - immune_kill - checkpoint_kill

    # Immune effector dynamics (E in cells)
    # Convert V to cells for immune interaction terms: V_cells = V * 1e6
    V_cells = V * 1e6 if V > 0 else 0.0
    if E > 0:
        recruitment = rho_eff * E * V_cells / (sigma + V_cells) if V_cells > 0 else 0.0
        interaction_loss = delta_EV_eff * E * V_cells if V_cells > 0 else 0.0
        # Cap interaction loss to prevent extreme stiffness
        interaction_loss = min(interaction_loss, 10.0 * s_E)
        dEdt = s_E - mu_E * E + recruitment - interaction_loss
    else:
        dEdt = s_E

    return np.array([dVdt, dEdt])


# ═════════════════════════════════════════════════════════════
# ODE SOLVER
# ═════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Container for a single simulation result."""
    t: np.ndarray
    V: np.ndarray
    E: Optional[np.ndarray] = None
    params: Dict = field(default_factory=dict)
    scenario: str = "natural_history"
    patient_id: str = ""


def solve_tumor_trajectory(
    patient_params: Dict,
    treatment: str = "none",
    t_span: Tuple[float, float] = (0, 1825),
    t_eval: Optional[np.ndarray] = None,
    drug_concentration_fn: Optional[Callable] = None,
) -> SimulationResult:
    """
    Solve ODE for a single patient trajectory.

    Parameters
    ----------
    patient_params : dict
        Patient-specific ODE parameters (alpha, V_max, V0, etc.)
    treatment : str
        'none', 'chemo', 'immuno', 'targeted'
    t_span : (t_start, t_end) in days
    t_eval : array of time points for output
    drug_concentration_fn : callable for drug concentration C(t)

    Returns
    -------
    SimulationResult
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Sanitize parameters — replace NaN/inf with population defaults
    defaults = {"alpha": 0.0008, "V_max": 1e6, "V0": 8000.0, "E0": 1.3e5,
                "delta_c": 0.0, "delta_i": 0.0, "delta_t": 0.0,
                "s_E": 1.3e4, "mu_E": 0.041, "rho": 0.02, "sigma": 2e7,
                "delta_EV": 3.4e-10, "k_immune": 1.22e-4, "E_ref": 1.3e5,
                "k_immuno": 2.5e-5}
    for key, default_val in defaults.items():
        val = patient_params.get(key, default_val)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            patient_params[key] = default_val

    V0 = patient_params.get("V0", 8000.0)  # mm^3
    E0 = patient_params.get("E0", 1.3e5)   # immune effector initial count

    # Clip V0 to biologically plausible range
    V0 = np.clip(V0, 100.0, 500000.0)

    if treatment == "none":
        # Pure Gompertz growth — no immune compartment needed
        def ode_fn(t, y):
            return [gompertz_growth(t, y[0], patient_params["alpha"],
                                   patient_params["V_max"])]

        sol = solve_ivp(
            ode_fn, t_span, [V0],
            t_eval=t_eval, method="RK45",
            rtol=1e-8, atol=1e-10,
            max_step=10.0,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise RuntimeError("ODE solver produced NaN/Inf values")
        return SimulationResult(
            t=sol.t, V=sol.y[0], E=None,
            params=patient_params, scenario="natural_history"
        )
    else:
        # Full treatment-response system
        params = dict(patient_params)
        params["drug_type"] = treatment

        if drug_concentration_fn is None:
            drug_concentration_fn = make_no_drug_fn()

        def ode_fn(t, y):
            return treatment_response_ode(t, y, params, drug_concentration_fn)

        sol = solve_ivp(
            ode_fn, t_span, [V0, E0],
            t_eval=t_eval, method="Radau",  # stiff solver for immune dynamics
            rtol=1e-6, atol=1e-8,
            max_step=10.0,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise RuntimeError("ODE solver produced NaN/Inf values")
        return SimulationResult(
            t=sol.t, V=sol.y[0], E=sol.y[1],
            params=patient_params, scenario=treatment
        )


def make_no_drug_fn() -> Callable:
    """Return a drug concentration function that always returns 0."""
    return lambda t, drug_type: 0.0


def solve_trajectory_mc(
    patient_params: Dict,
    param_bounds: Dict,
    treatment: str = "none",
    t_span: Tuple[float, float] = (0, 1825),
    t_eval: Optional[np.ndarray] = None,
    n_mc: int = 200,
    rng: Optional[np.random.Generator] = None,
    drug_concentration_fn: Optional[Callable] = None,
) -> Dict:
    """
    Monte Carlo trajectory simulation with parameter uncertainty.

    Parameters
    ----------
    patient_params : dict — central parameter estimates
    param_bounds : dict — {param_name: (lower, upper)} uncertainty bounds
    n_mc : int — number of Monte Carlo samples
    rng : np.random.Generator

    Returns
    -------
    dict with keys: t, V_median, V_lower (2.5%), V_upper (97.5%), V_all
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    V_all = np.zeros((n_mc, len(t_eval)))

    for i in range(n_mc):
        # Sample parameters from bounds
        sampled = dict(patient_params)
        for key, (lo, hi) in param_bounds.items():
            if key in sampled:
                sampled[key] = rng.uniform(lo, hi)

        try:
            result = solve_tumor_trajectory(
                sampled, treatment=treatment,
                t_span=t_span, t_eval=t_eval,
                drug_concentration_fn=drug_concentration_fn,
            )
            V_all[i] = result.V
        except Exception:
            V_all[i] = np.nan

    V_median = np.nanmedian(V_all, axis=0)
    V_lower = np.nanpercentile(V_all, 2.5, axis=0)
    V_upper = np.nanpercentile(V_all, 97.5, axis=0)

    return {
        "t": t_eval,
        "V_median": V_median,
        "V_lower": V_lower,
        "V_upper": V_upper,
        "V_all": V_all,
    }


# ═════════════════════════════════════════════════════════════
# VOLUME-TO-DIAMETER CONVERSION
# ═════════════════════════════════════════════════════════════

def volume_to_diameter(V_mm3: float) -> float:
    """Convert tumor volume (mm^3) to diameter (mm) assuming sphere."""
    return 2.0 * (3.0 * V_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)


def diameter_to_volume(d_mm: float) -> float:
    """Convert tumor diameter (mm) to volume (mm^3) assuming sphere."""
    r = d_mm / 2.0
    return (4.0 / 3.0) * np.pi * r ** 3


# ═════════════════════════════════════════════════════════════
# TIME-TO-PROGRESSION CALCULATION
# ═════════════════════════════════════════════════════════════

def time_to_progression(result: SimulationResult,
                        progression_threshold: float = 2.0) -> float:
    """
    Calculate time to progression (tumor doubling from baseline).

    Parameters
    ----------
    result : SimulationResult
    progression_threshold : float — fold increase from V0 defining progression

    Returns
    -------
    float — time to progression in days (np.inf if never reached)
    """
    V0 = result.V[0]
    threshold = V0 * progression_threshold
    idx = np.where(result.V >= threshold)[0]
    if len(idx) == 0:
        return np.inf
    return result.t[idx[0]]


def time_to_clinical_threshold(result: SimulationResult,
                               threshold_volume: float = 500.0) -> float:
    """
    Time to reach clinical threshold volume (default 500 mm^3 ~ 1 cm diameter).
    """
    idx = np.where(result.V >= threshold_volume)[0]
    if len(idx) == 0:
        return np.inf
    return result.t[idx[0]]


if __name__ == "__main__":
    # Quick test: simulate Gompertz growth with population priors
    params = {
        "alpha": 5e-4,      # day^-1
        "V_max": 1e6,       # mm^3 (10^12 cells)
        "V0": 8000,         # mm^3 (Stage I)
    }
    result = solve_tumor_trajectory(params, treatment="none", t_span=(0, 1825))
    print(f"V(0) = {result.V[0]:.1f} mm^3")
    print(f"V(1yr) = {result.V[len(result.t)//5]:.1f} mm^3")
    print(f"V(5yr) = {result.V[-1]:.1f} mm^3")
    print(f"Diameter(5yr) = {volume_to_diameter(result.V[-1]):.1f} mm")
    ttp = time_to_progression(result)
    print(f"Time to progression (doubling): {ttp:.0f} days ({ttp/30.4:.1f} months)")
