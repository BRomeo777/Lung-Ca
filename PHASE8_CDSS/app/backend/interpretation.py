"""
Phase 8 CDSS — Clinical Interpretation Layer
Extends Phase 7.1's clinical_interpretation_guidance.md with 20+ permitted and 20+ forbidden statements.
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
from .config import CLINICAL_DISCLAIMER, TREATMENT_ARTIFACT_WARNING

# ── PERMITTED STATEMENTS (21) ──
PERMITTED_STATEMENTS = [
    # From Phase 7.1 (5 seed statements)
    "The Digital Twin estimates disease trajectories under explicit modeling assumptions.",
    "The Digital Twin ranks patients by predicted risk using a validated neural survival model.",
    "The Digital Twin quantifies uncertainty in simulated outcomes via Monte Carlo sampling.",
    "The Digital Twin identifies mechanistic drivers of tumor progression per patient.",
    "The Digital Twin provides a reproducible computational framework for hypothesis generation.",
    # Phase 8 extensions (16 additional)
    "This patient's risk score places them in the {risk_group} risk category relative to the validation cohort.",
    "The model's discrimination ability is C-index 0.6333 (95% CI: 0.5748-0.7021) on internal validation.",
    "The survival probability shown is UNCALIBRATED and should be interpreted with caution — use for relative ranking only.",
    "The Digital Twin simulation suggests a projected time-to-progression of {ttp_months:.1f} months under the {treatment_label} scenario.",
    "This is a computational hypothesis, subject to the treatment-simulation artifact warning, and requires clinician interpretation.",
    "The uncertainty in this prediction is classified as {uncertainty_level}, with a 95% CI of [{ci_lo:.1f}-{ci_hi:.1f}] months.",
    "The top contributing features to this patient's risk score are: {top_features}.",
    "The mechanistic model projects tumor volume evolution based on Gompertz growth kinetics with patient-specific parameters.",
    "All simulated treatment outcomes are counterfactual projections under idealized conditions and are labeled as such.",
    "The model was trained on TCGA data (n=737) and validated on an internal validation set (n=182).",
    "External validation was performed on 5 independent cohorts with a mean C-index of 0.6588.",
    "This patient's ODE parameters are {param_origin} — {'personalized from gene expression' if 'PERSONALIZED' else 'population-average priors'}.",
    "The growth-survival correlation (rho=-0.1786, p=0.0004) confirms that faster-growing tumors are associated with shorter survival.",
    "No prediction is provided without an accompanying explanation and uncertainty estimate.",
    "The system enforces hard blocks for Stage IIIB/IV patients based on Phase 7.1 insufficient sample findings.",
    "This CDSS is a research prototype for hypothesis generation and decision support research, not for autonomous clinical decision-making.",
]

# ── FORBIDDEN STATEMENTS (21) ──
FORBIDDEN_STATEMENTS = [
    # From Phase 7.1 (5 seed statements)
    "The Digital Twin determines the best treatment.",
    "The Digital Twin predicts individual patient survival.",
    "The Digital Twin can replace oncologist judgment.",
    "Simulated treatment outcomes represent what would actually happen.",
    "The Digital Twin is validated for clinical deployment.",
    # Phase 8 extensions (16 additional)
    "This patient should receive {treatment}.",
    "Treatment X is superior to Treatment Y for this patient.",
    "The model recommends {treatment} for this patient.",
    "The survival probability of {value}% is clinically accurate.",
    "This patient's predicted survival time is {months} months.",
    "The Digital Twin can identify non-responders to {treatment}.",
    "Treatment differences shown here are clinically meaningful.",
    "The model accounts for treatment resistance, toxicity, or discontinuation.",
    "This prediction can be used for treatment planning without clinician oversight.",
    "The calibration of this model has been validated for clinical use.",
    "The Digital Twin is FDA-approved or regulatory-ready.",
    "This system can be used to determine patient eligibility for clinical trials.",
    "The simulation results should replace pathology or imaging findings.",
    "The model can predict response to combination therapy.",
    "The uncertainty interval guarantees that the true outcome falls within the stated range.",
    "This system has been validated in a prospective clinical trial.",
    "The Digital Twin can be used without the treatment-simulation artifact warning.",
]


def get_permitted_statement(template: str, **kwargs) -> str:
    """Format a permitted statement with patient-specific values."""
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return template


def get_forbidden_statement(template: str, **kwargs) -> str:
    """Format a forbidden statement (for documentation/education)."""
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return template


def get_treatment_interpretation(treatment_label: str) -> str:
    """Return the mandatory clinical interpretation for treatment simulation results."""
    return (
        f"This Digital Twin simulation suggests improved predicted outcomes under "
        f"the simulated {treatment_label} scenario. "
        f"This is a computational hypothesis, subject to the treatment-simulation "
        f"artifact warning, and requires clinician interpretation.\n\n"
        f"⚠️ ARTIFACT WARNING: {TREATMENT_ARTIFACT_WARNING}"
    )


def get_risk_interpretation(risk_group: str) -> str:
    """Return permitted interpretation for risk group."""
    return (
        f"This patient's risk score places them in the {risk_group} risk category "
        f"relative to the validation cohort. This ranking is based on a validated "
        f"neural survival model (C-index 0.6333). The absolute survival probability "
        f"shown is UNCALIBRATED and should be interpreted with caution."
    )


def get_all_statements() -> dict:
    """Return all permitted and forbidden statements for the Evidence Panel."""
    return {
        "permitted": PERMITTED_STATEMENTS,
        "forbidden": FORBIDDEN_STATEMENTS,
        "n_permitted": len(PERMITTED_STATEMENTS),
        "n_forbidden": len(FORBIDDEN_STATEMENTS),
        "disclaimer": CLINICAL_DISCLAIMER,
    }
