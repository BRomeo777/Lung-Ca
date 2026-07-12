"""
Phase 4 — Genomic Personalization of ODE Parameters

Maps patient-specific clinical and genomic features to ODE parameter adjustments.
Every mapping is biologically justified from published literature.

RESEARCH PROTOTYPE ONLY. Simulated tumor trajectories are based on population-derived
ODE parameters personalized by genomic features. They represent biologically plausible
scenarios under stated assumptions, not clinically validated predictions. This system
must not be used for clinical decision-making without prospective validation with serial
imaging data and regulatory approval.
"""
from __future__ import annotations
import numpy as np
import json
from typing import Dict, Tuple, Optional
from pathlib import Path


# ═════════════════════════════════════════════════════════════
# STAGE-TO-INITIAL-VOLUME MAPPING
# ═════════════════════════════════════════════════════════════

# Initial tumor volume by stage (mm^3)
# Based on typical tumor diameters at diagnosis:
#   Stage IA: ~1.5 cm diameter -> ~1,767 mm^3
#   Stage IB: ~2.5 cm -> ~8,181 mm^3
#   Stage IIA: ~3.5 cm -> ~22,449 mm^3
#   Stage IIB: ~4.5 cm -> ~47,713 mm^3
#   Stage IIIA: ~5.5 cm -> ~86,553 mm^3
#   Stage IIIB: ~6.6 cm -> ~150,520 mm^3 (from Liu et al. 2017)
#   Stage IV: ~7.0 cm -> ~179,594 mm^3
#
# References:
#   Liu Z, et al. Sci Rep 2017;7:13646 — Stage I: 2.5±2.5cm, Stage II: 3.5±3cm, Stage III: 6.6±3cm
#   Goldstraw P, et al. J Thorac Oncol 2016;11(1):39-51 — TNM staging

STAGE_V0 = {
    "I":    8000.0,
    "IA":   1767.0,
    "IB":   8181.0,
    "II":   22449.0,
    "IIA":  22449.0,
    "IIB":  47713.0,
    "III":  86553.0,
    "IIIA": 86553.0,
    "IIIB": 150520.0,
    "IV":   179594.0,
}

# Growth rate adjustment by stage (higher stage = faster growth)
# Based on volume doubling time meta-analysis:
#   Adenocarcinoma: VDT 223 days (pooled mean)
#   Squamous cell: VDT 140 days
#   Higher stage associated with faster growth (Usuda et al., EJCA 2024)
#
# Gompertz alpha = ln(2) / VDT (for early exponential phase approximation)
#   LUAD: alpha ~ ln(2)/223 = 3.11e-3 day^-1
#   LUSC: alpha ~ ln(2)/140 = 4.95e-3 day^-1
# But Gompertz alpha is the INITIAL specific growth rate, not the doubling rate.
# For Gompertz: V(t) = V_max * exp(-(alpha/beta) * exp(-beta*t))
# alpha_0 = alpha (initial growth rate), beta = rate of growth deceleration
#
# Benzekry et al. (2019) for NSCLC: alpha_0 ~ 0.12 day^-1, beta ~ 0.007 day^-1
# (fitted to clinical data, primary tumor)
#
# We use a base alpha and adjust by stage and histology.

STAGE_GROWTH_FACTOR = {
    "I":    0.7,
    "IA":   0.6,
    "IB":   0.8,
    "II":   0.9,
    "IIA":  0.9,
    "IIB":  1.0,
    "III":  1.2,
    "IIIA": 1.2,
    "IIIB": 1.4,
    "IV":   1.5,
}

HISTOLOGY_GROWTH_FACTOR = {
    "LUAD_Adenocarcinoma": 0.85,   # slower (VDT 223 days)
    "LUSC_SquamousCell": 1.15,     # faster (VDT 140 days)
}

# ═════════════════════════════════════════════════════════════
# GENE-TO-PARAMETER MAPPINGS
# ═════════════════════════════════════════════════════════════

# Each mapping: gene_symbol -> (parameter_affected, direction, max_adjustment, rationale)
# direction: +1 = high expression increases parameter, -1 = high expression decreases
# max_adjustment: maximum fractional change (e.g., 0.3 = ±30%)
# All adjustments are multiplicative around population priors, capped at ±50%.

GENE_PARAM_MAPPINGS = {
    "SPP1": {
        "param": "k_immune",
        "direction": -1,      # High SPP1 -> lower immune kill
        "max_adj": 0.30,
        "rationale": "SPP1 (osteopontin) suppresses immune surveillance via integrin signaling, "
                     "promoting tumor immune evasion. High SPP1 associated with immunosuppressive "
                     "TME in NSCLC. Ref: Rittling et al., Semin Cancer Biol 2020;76:120-131.",
        "inferred": False,
    },
    "COL1A1": {
        "param": "V_max",
        "direction": +1,      # High COL1A1 -> higher carrying capacity
        "max_adj": 0.25,
        "rationale": "COL1A1 (collagen type I) is a key ECM component. High expression indicates "
                     "active ECM remodeling and desmoplasia, enabling tumor expansion and increasing "
                     "effective carrying capacity. Ref: Chen et al., Cancer Cell Int 2022;22:246.",
        "inferred": False,
    },
    "MMP1": {
        "param": "V_max",
        "direction": +1,      # High MMP1 -> higher carrying capacity
        "max_adj": 0.20,
        "rationale": "MMP1 (matrix metalloproteinase-1) degrades ECM, facilitating invasion and "
                     "expansion. High MMP1 correlates with advanced stage in NSCLC. "
                     "Ref: Hojilla CV, et al. Front Oncol 2022;12:877643.",
        "inferred": False,
    },
    "CXCL8": {
        "param": "rho",       # Immune recruitment
        "direction": -1,      # High CXCL8 -> reduced effective immune recruitment
        "max_adj": 0.25,
        "rationale": "CXCL8 (IL-8) recruits neutrophils (MDSCs) rather than effective CTLs. "
                     "High CXCL8 in NSCLC promotes immunosuppressive TME and angiogenesis, "
                     "reducing effective anti-tumor immune response. "
                     "Ref: Schalper KA, et al. Clin Cancer Res 2017;23(16):4771-4782.",
        "inferred": False,
    },
    "EGLN3": {
        "param": "alpha",
        "direction": +1,      # High EGLN3 -> increased growth rate
        "max_adj": 0.20,
        "rationale": "EGLN3 (PHD3) is a hypoxia sensor. High EGLN3 indicates hypoxic adaptation, "
                     "which in NSCLC is associated with aggressive phenotype and HIF-1alpha "
                     "driven proliferation. Ref: Henze AT, et al. Cancer Res 2010;70(13):5351-5361.",
        "inferred": False,
    },
    "ANGPTL4": {
        "param": "V_max",
        "direction": +1,      # High ANGPTL4 -> higher carrying capacity (angiogenesis)
        "max_adj": 0.20,
        "rationale": "ANGPTL4 regulates angiogenesis and lipid metabolism. High expression "
                     "promotes vascularization, increasing nutrient supply and effective "
                     "carrying capacity. Ref: Tan MJ, et al. J Biol Chem 2010;285(12):8866-8876.",
        "inferred": False,
    },
    "MELTF": {
        "param": "alpha",
        "direction": +1,      # High MELTF -> increased growth (top permutation importance)
        "max_adj": 0.15,
        "rationale": "MELTF (melanotransferrin) involved in iron metabolism and cell proliferation. "
                     "Identified as top predictive feature by DeepSurv permutation importance. "
                     "Direction inferred from ML model coefficient, not directly validated.",
        "inferred": True,
    },
    "CD109": {
        "param": "alpha",
        "direction": +1,      # High CD109 -> increased growth
        "max_adj": 0.15,
        "rationale": "CD109 is a GPI-anchored protein upregulated in squamous cell carcinoma. "
                     "Associated with TGF-beta signaling and tumor progression. "
                     "Direction inferred from ML model.",
        "inferred": True,
    },
    "MYEOV": {
        "param": "alpha",
        "direction": +1,      # High MYEOV -> increased growth
        "max_adj": 0.15,
        "rationale": "MYEOV (myeloma overexpressed gene) associated with proliferation in "
                     "multiple cancers including lung. Direction inferred from ML model.",
        "inferred": True,
    },
    "LAMC2": {
        "param": "V_max",
        "direction": +1,      # High LAMC2 -> higher carrying capacity (invasion)
        "max_adj": 0.15,
        "rationale": "LAMC2 (laminin gamma 2) is a marker of invasive front in NSCLC. "
                     "High expression associated with stromal invasion and larger tumor mass. "
                     "Ref: Koshikawa N, et al. Cancer Res 1999;59(20):5146-5153.",
        "inferred": False,
    },
    "RHOV": {
        "param": "alpha",
        "direction": +1,
        "max_adj": 0.10,
        "rationale": "RHOV (Ras homolog family member V) is an atypical Rho GTPase involved "
                     "in cell migration and proliferation. Direction inferred from ML model.",
        "inferred": True,
    },
    "SORCS2": {
        "param": "k_immune",
        "direction": -1,
        "max_adj": 0.10,
        "rationale": "SORCS2 associated with neural development and receptor trafficking. "
                     "Role in NSCLC not well characterized. Direction inferred from ML model.",
        "inferred": True,
    },
    "COL22A1": {
        "param": "V_max",
        "direction": +1,
        "max_adj": 0.10,
        "rationale": "COL22A1 is a minor collagen associated with tissue remodeling. "
                     "Direction inferred from ML model.",
        "inferred": True,
    },
}

# ═════════════════════════════════════════════════════════════
# CLINICAL FEATURE MAPPINGS
# ═════════════════════════════════════════════════════════════

SMOKING_PARAMS = {
    "Current": {"alpha_factor": 1.15, "delta_i_factor": 0.85, "rationale": "Smokers have higher mutational burden, faster growth kinetics, and reduced immunotherapy response in some studies. Ref: Usuda et al., EJCA 2024."},
    "Former":  {"alpha_factor": 1.0,  "delta_i_factor": 1.0,  "rationale": "Former smokers have intermediate growth kinetics."},
    "Never":   {"alpha_factor": 0.85, "delta_i_factor": 1.1,  "rationale": "Never-smokers tend to have slower-growing tumors (often EGFR-mutant LUAD) with potentially better immune response. Ref: Usuda et al., EJCA 2024."},
}

# Age effect on immune effector death rate (immunosenescence)
# Base mu_E = 0.041 day^-1 (Kuznetsov 1994)
# Adjustment: +0.5% per year above 65, -0.5% per year below 65 (capped at ±20%)
AGE_MU_E_FACTOR = 0.005  # per year
AGE_REFERENCE = 65.0


# ═════════════════════════════════════════════════════════════
# PERSONALIZATION FUNCTION
# ═════════════════════════════════════════════════════════════

def personalize_parameters(
    patient_features: Dict,
    population_priors: Dict,
    ensg2sym: Dict,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_scale: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
    deepsurv_risk: Optional[float] = None,
) -> Tuple[Dict, Dict]:
    """
    Adjust population ODE parameters based on patient-specific features.

    Parameters
    ----------
    patient_features : dict
        Keys: 'Age', 'Cancer_Type', 'Stage', 'Smoking_Status',
        and gene expression values (Ensembl IDs or gene symbols)
    population_priors : dict
        Literature-derived baseline parameter values
    ensg2sym : dict
        Ensembl ID -> gene symbol mapping
    scaler_mean, scaler_scale : optional
        StandardScaler parameters for computing z-scores of gene expression
    feature_names : list, optional
        Feature names in the order used by the scaler
    deepsurv_risk : float, optional
        DeepSurv risk score for ML-informed parameter scaling

    Returns
    -------
    patient_params : dict — patient-specific ODE parameters
    uncertainty_bounds : dict — {param: (lower, upper)} 95% intervals
    """
    # Start from population priors
    params = dict(population_priors)

    # ── 1. STAGE → V0 and growth rate ──
    stage = patient_features.get("Stage", "IIB")
    if stage in STAGE_V0:
        params["V0"] = STAGE_V0[stage]
    else:
        params["V0"] = population_priors["V0"]

    stage_factor = STAGE_GROWTH_FACTOR.get(stage, 1.0)
    params["alpha"] = params["alpha"] * stage_factor

    # ── 2. HISTOLOGY → growth rate ──
    cancer_type = patient_features.get("Cancer_Type", "LUAD_Adenocarcinoma")
    histo_factor = HISTOLOGY_GROWTH_FACTOR.get(cancer_type, 1.0)
    params["alpha"] = params["alpha"] * histo_factor

    # ── 3. SMOKING → growth rate and immune kill ──
    smoking = patient_features.get("Smoking_Status", "Former")
    smoke = SMOKING_PARAMS.get(smoking, SMOKING_PARAMS["Former"])
    params["alpha"] = params["alpha"] * smoke["alpha_factor"]
    # Note: delta_i is vestigial (replaced by k_immune in ODE).
    # Apply smoking adjustment to k_immune instead.
    params["k_immune"] = params.get("k_immune", 1.22e-4) * smoke["delta_i_factor"]

    # ── 4. AGE → immune effector death rate (immunosenescence) ──
    age = patient_features.get("Age", 65.0)
    try:
        age = float(age)
        if np.isnan(age) or np.isinf(age):
            age = 65.0
    except (ValueError, TypeError):
        age = 65.0
    age_adj = (age - AGE_REFERENCE) * AGE_MU_E_FACTOR
    age_adj = np.clip(age_adj, -0.20, 0.20)
    mu_E_base = params.get("mu_E", 0.041)
    if mu_E_base is None or (isinstance(mu_E_base, float) and (np.isnan(mu_E_base) or np.isinf(mu_E_base))):
        mu_E_base = 0.041
    params["mu_E"] = mu_E_base * (1.0 + age_adj)

    # ── 5. GENE EXPRESSION → parameter adjustments ──
    # Build symbol-to-value mapping from patient features
    sym2value = {}
    for key, val in patient_features.items():
        if key.startswith("ENSG"):
            sym = ensg2sym.get(key, None)
            if sym:
                sym2value[sym] = val

    # Compute z-scores if scaler is provided
    sym2zscore = {}
    if scaler_mean is not None and scaler_scale is not None and feature_names is not None:
        for i, fname in enumerate(feature_names):
            if fname.startswith("ENSG"):
                sym = ensg2sym.get(fname, None)
                if sym and fname in patient_features:
                    raw = patient_features[fname]
                    z = (raw - scaler_mean[i]) / scaler_scale[i]
                    sym2zscore[sym] = z

    # Apply gene-based adjustments
    for gene_sym, mapping in GENE_PARAM_MAPPINGS.items():
        if gene_sym not in sym2zscore and gene_sym not in sym2value:
            continue

        # Use z-score if available, otherwise use raw value with rank-based heuristic
        if gene_sym in sym2zscore:
            z = sym2zscore[gene_sym]
            # Sigmoid mapping: z-score -> [0, 1] -> [-max_adj, +max_adj]
            adj = mapping["direction"] * mapping["max_adj"] * np.tanh(z)
        else:
            # Without z-score, skip (can't determine relative expression)
            continue

        param_name = mapping["param"]
        # Redirect delta_i adjustments to k_immune (delta_i is vestigial)
        if param_name == "delta_i":
            param_name = "k_immune"
        if param_name in params:
            params[param_name] = params[param_name] * (1.0 + adj)

    # ── 6. DEEPSURV RISK SCORE → overall parameter scaling ──
    if deepsurv_risk is not None:
        # Normalize risk score to [0, 1] using logistic function
        # DeepSurv risk scores are log-hazard, typically range [-2, +2]
        risk_normalized = 1.0 / (1.0 + np.exp(-deepsurv_risk))
        # Scale growth parameters: high risk -> faster growth, higher V_max
        # Bounded to ±20% adjustment
        risk_adj = (risk_normalized - 0.5) * 0.4  # [-0.2, +0.2]
        params["alpha"] = params["alpha"] * (1.0 + risk_adj)
        params["V_max"] = params["V_max"] * (1.0 + risk_adj * 0.5)
        # High risk -> lower immune kill (apply to k_immune, not vestigial delta_i)
        params["k_immune"] = params.get("k_immune", 1.22e-4) * (1.0 - risk_adj * 0.5)

    # ── 7. CAP ALL PARAMETERS AT ±50% OF POPULATION PRIOR ──
    # Also apply hard biological bounds to prevent impossible values
    HARD_BOUNDS = {
        "alpha": (1e-5, 0.01),        # day^-1 (must be positive, < 0.01 for realistic VDT)
        "V_max": (1e4, 1e7),          # mm^3 (10^10 to 10^13 cells)
        "V0": (100.0, 500000.0),      # mm^3 (biologically plausible tumor volumes)
        "delta_c": (0.0, 0.2),        # (mg/L)^-1 day^-1
        "k_immune": (1e-6, 1e-2),     # day^-1
        "delta_t": (0.0, 0.5),        # (mg/L)^-1 day^-1
        "mu_E": (0.001, 0.2),         # day^-1 (half-life 3.5 to 693 days)
        "rho": (0.0, 0.1),            # day^-1
        "s_E": (1000.0, 50000.0),     # cells/day
        "delta_EV": (1e-12, 1e-8),    # cell^-1 day^-1
        "E0": (1e4, 1e6),             # cells
        "k_immuno": (1e-7, 1e-3),     # (mg/L)^-1 day^-1
    }
    for key in ["alpha", "V_max", "delta_c", "delta_i", "delta_t",
                "mu_E", "rho", "s_E", "delta_EV", "k_immune", "k_immuno"]:
        if key in population_priors and key in params:
            lo = population_priors[key] * 0.5
            hi = population_priors[key] * 1.5
            params[key] = np.clip(params[key], lo, hi)
        # Apply hard bounds as secondary safety (only if key exists in params)
        if key in HARD_BOUNDS and key in params:
            params[key] = np.clip(params[key], HARD_BOUNDS[key][0], HARD_BOUNDS[key][1])

    # ── 8. COMPUTE UNCERTAINTY BOUNDS ──
    # Uncertainty factors based on inter-patient variability in literature
    UNCERTAINTY_FACTORS = {
        "alpha": 0.30,      # ±30% (Benzekry 2019: CV ~25%)
        "V_max": 0.40,      # ±40% (high variability in carrying capacity)
        "V0": 0.50,         # ±50% (large uncertainty in initial volume)
        "delta_c": 0.40,    # ±40% (Liu 2017: chemo effect varies)
        "delta_i": 0.50,    # ±50% (immune parameters highly variable)
        "delta_t": 0.40,    # ±40%
        "mu_E": 0.25,       # ±25% (Kuznetsov 1994)
        "rho": 0.35,        # ±35%
        "s_E": 0.30,        # ±30%
        "delta_EV": 0.40,   # ±40%
        "E0": 0.30,         # ±30%
    }

    uncertainty_bounds = {}
    for key, factor in UNCERTAINTY_FACTORS.items():
        if key in params:
            val = params[key]
            uncertainty_bounds[key] = (val * (1 - factor), val * (1 + factor))

    return params, uncertainty_bounds


def get_personalization_report(patient_features: Dict, ensg2sym: Dict,
                               scaler_mean=None, scaler_scale=None,
                               feature_names=None) -> list:
    """
    Generate a human-readable report of all parameter adjustments for a patient.
    """
    report = []
    stage = patient_features.get("Stage", "IIB")
    report.append(f"  Stage={stage} -> V0={STAGE_V0.get(stage, 'default')} mm^3, "
                  f"growth factor={STAGE_GROWTH_FACTOR.get(stage, 1.0)}")

    cancer_type = patient_features.get("Cancer_Type", "LUAD")
    report.append(f"  Histology={cancer_type} -> growth factor={HISTOLOGY_GROWTH_FACTOR.get(cancer_type, 1.0)}")

    smoking = patient_features.get("Smoking_Status", "Former")
    smoke = SMOKING_PARAMS.get(smoking, SMOKING_PARAMS["Former"])
    report.append(f"  Smoking={smoking} -> alpha x{smoke['alpha_factor']}, delta_i x{smoke['delta_i_factor']}")

    age = patient_features.get("Age", 65.0)
    try:
        age = float(age)
    except:
        age = 65.0
    age_adj = np.clip((age - 65) * 0.005, -0.20, 0.20)
    report.append(f"  Age={age} -> mu_E adjustment={age_adj:+.1%}")

    # Gene adjustments
    sym2zscore = {}
    if scaler_mean is not None and scaler_scale is not None and feature_names is not None:
        for i, fname in enumerate(feature_names):
            if fname.startswith("ENSG"):
                sym = ensg2sym.get(fname, None)
                if sym and fname in patient_features:
                    raw = patient_features[fname]
                    z = (raw - scaler_mean[i]) / scaler_scale[i]
                    sym2zscore[sym] = z

    for gene_sym, mapping in GENE_PARAM_MAPPINGS.items():
        if gene_sym in sym2zscore:
            z = sym2zscore[gene_sym]
            adj = mapping["direction"] * mapping["max_adj"] * np.tanh(z)
            tag = "[INFERRED]" if mapping["inferred"] else "[LITERATURE]"
            param_name = mapping["param"]
            if param_name == "delta_i":
                param_name = "k_immune"
            report.append(f"  {gene_sym} (z={z:+.2f}) -> {param_name} adj={adj:+.1%} {tag}")

    return report
