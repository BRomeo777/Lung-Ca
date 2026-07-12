"""
HybridPatientState — Unified patient representation combining neural and mechanistic components.

This is the core data structure of the Phase 5 integration engine
and the input to the Phase 6 Digital Twin.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


CLINICAL_USE_WARNING = (
    "RESEARCH PROTOTYPE ONLY. This output must not be used for "
    "clinical decision-making without prospective validation, "
    "independent external validation, and regulatory approval."
)


@dataclass
class HybridPatientState:
    """Unified patient representation combining neural and mechanistic components."""

    # Identity
    patient_id: str = ""
    cohort_source: str = ""
    analysis_role: str = ""

    # Neural component (Phase 3)
    deepsurv_risk_score: float = float("nan")
    deepsurv_risk_std: float = float("nan")
    deepsurv_risk_ci_lower: float = float("nan")
    deepsurv_risk_ci_upper: float = float("nan")
    deepsurv_risk_percentile: float = float("nan")
    coxph_risk_score: float = float("nan")
    risk_group: str = ""

    # Mechanistic component (Phase 4)
    ode_alpha: float = float("nan")
    ode_V_max: float = float("nan")
    ode_V0: float = float("nan")
    ode_delta_c: float = float("nan")
    ode_k_immune: float = float("nan")
    ode_mu_E: float = float("nan")
    ode_rho: float = float("nan")
    ode_parameter_uncertainty: Dict = field(default_factory=dict)

    ttp_natural_median: float = float("nan")
    ttp_natural_ci: Tuple = (float("nan"), float("nan"))
    ttp_chemo_median: float = float("nan")
    ttp_chemo_ci: Tuple = (float("nan"), float("nan"))
    ttp_immuno_median: float = float("nan")
    ttp_immuno_ci: Tuple = (float("nan"), float("nan"))
    ttp_targeted_median: float = float("nan")
    ttp_targeted_ci: Tuple = (float("nan"), float("nan"))

    benefit_chemo_days: float = float("nan")
    benefit_immuno_days: float = float("nan")
    benefit_targeted_days: float = float("nan")

    immune_activity_score: float = float("nan")
    simulation_confidence: str = ""

    # Integration component (Phase 5)
    hybrid_risk_score: float = float("nan")
    hybrid_risk_ci: Tuple = (float("nan"), float("nan"))
    integration_strategy: str = ""
    integration_weights: Dict = field(default_factory=dict)
    neural_mechanistic_agreement: float = float("nan")
    consistency_violation: bool = False
    consistency_violation_reason: str = ""

    # Explainability
    top_neural_features: List[Tuple] = field(default_factory=list)
    top_mechanistic_parameters: List[Tuple] = field(default_factory=list)
    dominant_hallmarks: List[str] = field(default_factory=list)
    causal_chain_summary: str = ""
    evidence_tags_applied: List[str] = field(default_factory=list)

    # Clinical output
    recommended_monitoring_intensity: str = ""
    treatment_comparison_summary: str = ""
    clinical_confidence_statement: str = ""

    # Metadata
    phase3_model_version: str = "Phase3-DeepSurv-Top2"
    phase4_model_version: str = "Phase4-Gompertz-Hardened"
    phase5_integration_version: str = "Phase5-v1.0"
    generation_timestamp: str = ""

    clinical_use_warning: str = CLINICAL_USE_WARNING

    def to_dict(self) -> Dict:
        """Convert to flat dictionary for CSV serialization."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = "; ".join(str(x) for x in v) if v else ""
            elif isinstance(v, dict):
                d[k] = str(v)
            else:
                d[k] = v
        return d
