"""
Phase 4 — Treatment Response & Pharmacokinetic Models

Implements:
- One-compartment PK models for cisplatin, pembrolizumab, osimertinib
- Drug concentration functions C(t) for each treatment scenario
- Treatment schedule definitions (cycling, daily, Q3W)

RESEARCH PROTOTYPE ONLY. Simulated tumor trajectories are based on population-derived
ODE parameters personalized by genomic features. They represent biologically plausible
scenarios under stated assumptions, not clinically validated predictions. This system
must not be used for clinical decision-making without prospective validation with serial
imaging data and regulatory approval.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Callable, List, Tuple


# ═════════════════════════════════════════════════════════════
# PHARMACOKINETIC PARAMETERS (from published literature)
# ═════════════════════════════════════════════════════════════

PK_PARAMS = {
    "cisplatin": {
        "k_el": 0.552,       # day^-1 (t_half = 30 hours; FDA label, Hanada 2024)
        "k_a": 0.0,          # IV bolus — no absorption
        "V_d": 30.0,         # L (population PK, de Jongh 2003)
        "dose": 75.0,        # mg/m^2 (standard NSCLC regimen)
        "route": "IV",
        "schedule": "Q21D",  # every 21 days
        "n_cycles": 4,
        "C_max": 2.5,        # mg/L (peak plasma concentration, 75 mg/m^2)
        "citation": "Hanada et al., PMC11585315, 2024; de Jongh et al., Clin Cancer Res 2003",
    },
    "pembrolizumab": {
        "k_el": 0.0269,      # day^-1 (t_half = 25.8 days; FDA label 2014)
        "k_a": 0.0,          # IV infusion
        "V_d": 7.66,         # L (FDA population PK)
        "dose": 200.0,       # mg fixed dose
        "route": "IV",
        "schedule": "Q21D",  # every 21 days
        "n_cycles": 6,
        "C_max": 26.1,       # mg/L (steady-state Cmax, 200mg Q3W)
        "citation": "FDA Pembrolizumab Clinical Pharmacology Review, 2014; Centanni et al., PMC5270291, 2017",
    },
    "nivolumab": {
        "k_el": 0.0277,      # day^-1 (t_half = 25 days; Bajaj 2017)
        "k_a": 0.0,
        "V_d": 8.0,          # L (population PK)
        "dose": 240.0,       # mg fixed dose
        "route": "IV",
        "schedule": "Q14D",  # every 14 days
        "n_cycles": 8,
        "C_max": 28.0,       # mg/L
        "citation": "Bajaj et al., CPT Pharmacometrics Syst Pharmacol 2017;6(12):1235-1245",
    },
    "osimertinib": {
        "k_el": 0.346,       # day^-1 (t_half = 48 hours; Vishwanathan 2020)
        "k_a": 2.77,         # day^-1 (first-order absorption, Tmax ~6h)
        "V_d": 986.0,        # L (apparent Vd/F)
        "dose": 80.0,        # mg daily oral
        "route": "oral",
        "schedule": "QD",    # once daily
        "n_cycles": 365,     # continuous daily dosing
        "F_bio": 0.70,       # bioavailability ~70% (estimated)
        "C_max": 0.501,      # umol/L at steady state (80mg daily)
        "citation": "Vishwanathan et al., Br J Clin Pharmacol 2020;86(3):547-557; Pilla Reddy et al., PRP 2024",
    },
}


# ═════════════════════════════════════════════════════════════
# DRUG CONCENTRATION FUNCTIONS
# ═════════════════════════════════════════════════════════════

def make_cisplatin_concentration_fn(
    dose: float = 75.0,
    k_el: float = 0.552,
    V_d: float = 30.0,
    cycle_interval: float = 21.0,
    n_cycles: int = 4,
    infusion_duration: float = 1/24,  # 1 hour infusion
) -> Callable:
    """
    Cisplatin plasma concentration C(t).

    One-compartment IV infusion model:
    During infusion: C(t) = (Dose/T_inf/V_d) * (1 - exp(-k_el*t)) / k_el
    After infusion: C(t) = C_peak * exp(-k_el * (t - T_inf))

    References
    ----------
    Hanada et al., Understanding Cisplatin Pharmacokinetics, PMC11585315, 2024.
    de Jongh et al., Clin Cancer Res 2003;9:4050-4056.
    """
    C_peak = dose / V_d  # simplified peak concentration (mg/L)

    def C(t: float, drug_type: str = "chemo") -> float:
        if t < 0 or np.isnan(t) or np.isinf(t):
            return 0.0
        # Determine which cycle we're in
        cycle_num = int(t / cycle_interval)
        if cycle_num >= n_cycles:
            return 0.0  # treatment completed

        # Time since last dose
        t_since_dose = t - cycle_num * cycle_interval

        if t_since_dose < infusion_duration:
            # During infusion — linear rise
            return C_peak * (t_since_dose / infusion_duration)
        else:
            # After infusion — exponential decay
            t_post = t_since_dose - infusion_duration
            return C_peak * np.exp(-k_el * t_post)

    return C


def make_pembrolizumab_concentration_fn(
    dose: float = 200.0,
    k_el: float = 0.0269,
    V_d: float = 7.66,
    cycle_interval: float = 21.0,
    n_cycles: int = 6,
    infusion_duration: float = 0.5/24,  # 30 min infusion
) -> Callable:
    """
    Pembrolizumab plasma concentration C(t).

    One-compartment IV infusion model with long half-life (25.8 days).
    Accumulates at steady state (~2.1-fold with Q3W dosing).

    References
    ----------
    FDA Pembrolizumab Clinical Pharmacology Review, BLA 125514, 2014.
    Centanni et al., Clin Pharmacol Ther 2017;101(5):632-640.
    """
    # Use steady-state accumulation factor
    accumulation = 2.1  # Q3W dosing
    C_peak_ss = (dose / V_d) * accumulation  # mg/L at steady state

    def C(t: float, drug_type: str = "immuno") -> float:
        if t < 0 or np.isnan(t) or np.isinf(t):
            return 0.0
        cycle_num = int(t / cycle_interval)
        if cycle_num >= n_cycles:
            # After treatment — slow decay from last dose
            t_after = t - n_cycles * cycle_interval
            return C_peak_ss * np.exp(-k_el * t_after)

        # Time since last dose
        t_since_dose = t - cycle_num * cycle_interval

        if t_since_dose < infusion_duration:
            return C_peak_ss * (t_since_dose / infusion_duration)
        else:
            t_post = t_since_dose - infusion_duration
            # Include residual from previous cycles
            residual = 0.0
            for prev_cycle in range(cycle_num):
                t_prev = t - prev_cycle * cycle_interval - infusion_duration
                residual += C_peak_ss * np.exp(-k_el * t_prev)
            return C_peak_ss * np.exp(-k_el * t_post) + residual

    return C


def make_osimertinib_concentration_fn(
    dose: float = 80.0,
    k_el: float = 0.346,
    k_a: float = 2.77,
    V_d: float = 986.0,
    F_bio: float = 0.70,
    daily: bool = True,
    duration_days: float = 365.0,
) -> Callable:
    """
    Osimertinib plasma concentration C(t).

    One-compartment oral model with first-order absorption:
    C(t) = (F * Dose * k_a) / (V_d * (k_a - k_el)) * (exp(-k_el*t) - exp(-k_a*t))

    At steady state (~15 days), Css,max ~501 nmol/L, Css,min ~417 nmol/L.
    Convert nmol/L to mg/L: MW osimertinib = 489.3 g/mol
    501 nmol/L = 0.501 * 489.3 / 1000 = 0.245 mg/L

    References
    ----------
    Vishwanathan et al., Br J Clin Pharmacol 2020;86(3):547-557.
    Pilla Reddy et al., CPT Pharmacometrics Syst Pharmacol 2024.
    """
    # Steady-state concentration (simplified — use average Css)
    Css_avg = 0.245  # mg/L (average steady-state, 80mg daily)
    tmax = 0.25  # ~6 hours in days

    def C(t: float, drug_type: str = "targeted") -> float:
        if t < 0 or t > duration_days or np.isnan(t) or np.isinf(t):
            return 0.0

        # Daily dosing — compute concentration from each daily dose
        # Simplified: use steady-state approximation after 15 days
        if t < 15:
            # Build-up phase
            day = int(t)
            t_since_dose = t - day
            conc = 0.0
            for d in range(day + 1):
                td = t - d
                if td >= 0:
                    conc += (F_bio * dose * k_a) / (V_d * (k_a - k_el)) * \
                            (np.exp(-k_el * td) - np.exp(-k_a * td))
            return max(conc, 0.0)
        else:
            # Steady state — oscillates between Cmin and Cmax
            day = int(t)
            t_since_dose = t - day
            # Bateman function for single dose at steady state
            c_single = (F_bio * dose * k_a) / (V_d * (k_a - k_el)) * \
                       (np.exp(-k_el * t_since_dose) - np.exp(-k_a * t_since_dose))
            # Add residual from previous doses (geometric series at steady state)
            residual_factor = 1.0 / (1.0 - np.exp(-k_el))  # accumulation factor
            return max(c_single * residual_factor, 0.0)

    return C


# ═════════════════════════════════════════════════════════════
# TREATMENT SCHEDULE FACTORY
# ═════════════════════════════════════════════════════════════

def get_drug_concentration_fn(treatment: str) -> Callable:
    """
    Get the drug concentration function for a given treatment type.

    Parameters
    ----------
    treatment : str — 'chemo', 'immuno', 'targeted', 'none'

    Returns
    -------
    callable C(t, drug_type) -> float
    """
    if treatment == "chemo":
        return make_cisplatin_concentration_fn()
    elif treatment == "immuno":
        return make_pembrolizumab_concentration_fn()
    elif treatment == "targeted":
        return make_osimertinib_concentration_fn()
    else:
        return lambda t, drug_type="none": 0.0


# ═════════════════════════════════════════════════════════════
# TREATMENT-SPECIFIC ODE PARAMETERS
# ═════════════════════════════════════════════════════════════

TREATMENT_PARAMS = {
    "chemo": {
        "delta_c": 0.028,    # (mg/L)^-1 day^-1 — cisplatin cell kill rate
        "delta_i": 0.0,      # no immune activation
        "delta_t": 0.0,      # no targeted therapy
        "citation": "Liu et al., Sci Rep 2017;7:13646 (NSCLC-specific, beta_c=0.028)",
    },
    "immuno": {
        "delta_c": 0.0,      # no cytotoxic
        "delta_i": 0.0,      # baseline immune kill handled by k_immune in ODE
        "delta_t": 0.0,
        "k_immuno": 2.5e-5,  # checkpoint blockade kill rate (mg/L)^-1 day^-1
        "citation": "Wodarz D, J Theor Biol 2005;237:27-37; Kuznetsov VA, Bull Math Biol 1994; "
                    "checkpoint blockade effect calibrated to KEYNOTE-024 PFS (Reck M et al. NEJM 2016)",
    },
    "targeted": {
        "delta_c": 0.0,
        "delta_i": 0.0,
        "delta_t": 0.15,     # (mg/L)^-1 day^-1 — osimertinib kill rate (EGFR mut)
        "citation": "Inferred from FLAURA trial response rates; Pilla Reddy et al., 2024",
    },
    "none": {
        "delta_c": 0.0,
        "delta_i": 0.0,
        "delta_t": 0.0,
    },
}


if __name__ == "__main__":
    # Quick test: plot drug concentration profiles
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 126, 500)  # 6 months

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, drug in zip(axes, ["chemo", "immuno", "targeted"]):
        C_fn = get_drug_concentration_fn(drug)
        conc = [C_fn(ti, drug) for ti in t]
        ax.plot(t, conc)
        ax.set_title(f"{drug} concentration")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("C(t) (mg/L)")

    plt.tight_layout()
    plt.savefig("drug_concentration_test.png", dpi=100)
    print("Saved drug_concentration_test.png")
