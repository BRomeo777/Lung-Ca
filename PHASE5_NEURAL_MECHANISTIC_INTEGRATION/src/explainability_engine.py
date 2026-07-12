"""
Hybrid Explainability Engine — Multi-level explanation for hybrid model predictions.

Level 1: Feature-level (SHAP-like permutation importance for ML researchers)
Level 2: Hallmark-level (Cancer Hallmarks attribution for biologists)
Level 3: Clinical-level (Natural language summary for clinicians)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# Hallmark mapping from gene symbols to Cancer Hallmarks
HALLMARK_MAP = {
    "SPP1": ["Immune Evasion", "Invasion & Metastasis"],
    "COL1A1": ["Invasion & Metastasis", "Angiogenesis"],
    "MMP1": ["Invasion & Metastasis"],
    "CXCL8": ["Immune Evasion", "Angiogenesis", "Sustained Proliferation"],
    "EGLN3": ["Angiogenesis"],
    "ANGPTL4": ["Angiogenesis", "Invasion & Metastasis"],
    "LAMC2": ["Invasion & Metastasis"],
    "MELTF": ["Sustained Proliferation"],
    "CD109": ["Immune Evasion"],
    "MYEOV": ["Sustained Proliferation"],
    "RHOV": ["Invasion & Metastasis"],
    "SORCS2": ["Immune Evasion"],
    "COL22A1": ["Invasion & Metastasis"],
}


class HybridExplainabilityEngine:
    """Multi-level explanation engine for the hybrid model."""

    def __init__(self, ensg2sym: Dict, feature_names: List[str],
                 permutation_importance: Optional[pd.DataFrame] = None):
        self.ensg2sym = ensg2sym
        self.feature_names = feature_names
        self.permutation_importance = permutation_importance

    def explain_patient(
        self,
        patient_id: str,
        deepsurv_risk: float,
        hybrid_risk: float,
        ttp_natural: float,
        ttp_chemo: float,
        ttp_immuno: float,
        ttp_targeted: float,
        patient_features: Optional[pd.Series] = None,
        ode_params: Optional[Dict] = None,
        simulation_confidence: str = "MODERATE",
    ) -> Dict:
        """Generate complete explanation for one patient."""

        # Level 1: Feature-level
        top_features = self._get_top_features(patient_features)

        # Level 2: Hallmark-level
        hallmarks = self._attribute_hallmarks(top_features, ode_params)

        # Level 3: Clinical-level
        clinical_summary = self._generate_clinical_narrative(
            patient_id, deepsurv_risk, hybrid_risk,
            ttp_natural, ttp_chemo, ttp_immuno, ttp_targeted,
            hallmarks, simulation_confidence)

        return {
            "level1_features": top_features,
            "level2_hallmarks": hallmarks,
            "level3_clinical": clinical_summary,
        }

    def _get_top_features(self, patient_features: Optional[pd.Series]) -> List[Tuple]:
        """Get top contributing features for this patient."""
        if self.permutation_importance is not None:
            top = self.permutation_importance.head(10)
            result = []
            for _, row in top.iterrows():
                feat = row["feature"]
                sym = self.ensg2sym.get(feat, feat)
                result.append((sym, row["importance_mean"]))
            return result
        return []

    def _attribute_hallmarks(
        self,
        top_features: List[Tuple],
        ode_params: Optional[Dict],
    ) -> Dict:
        """Attribute features to Cancer Hallmarks."""
        hallmark_scores = {}
        for feat_name, importance in top_features:
            if feat_name in HALLMARK_MAP:
                for hk in HALLMARK_MAP[feat_name]:
                    hallmark_scores[hk] = hallmark_scores.get(hk, 0) + importance

        # Add mechanistic contributions
        if ode_params:
            alpha = ode_params.get("alpha", 0.0008)
            if alpha > 0.001:
                hallmark_scores["Sustained Proliferation"] = hallmark_scores.get(
                    "Sustained Proliferation", 0) + 0.02
            k_immune = ode_params.get("k_immune", 1.22e-4)
            if k_immune < 8e-5:
                hallmark_scores["Immune Evasion"] = hallmark_scores.get(
                    "Immune Evasion", 0) + 0.015

        # Sort by score
        sorted_hk = sorted(hallmark_scores.items(), key=lambda x: -x[1])
        return {
            "dominant_hallmarks": [hk for hk, _ in sorted_hk[:3]],
            "hallmark_scores": dict(sorted_hk),
        }

    def _generate_clinical_narrative(
        self,
        patient_id: str,
        deepsurv_risk: float,
        hybrid_risk: float,
        ttp_natural: float,
        ttp_chemo: float,
        ttp_immuno: float,
        ttp_targeted: float,
        hallmarks: Dict,
        simulation_confidence: str,
    ) -> str:
        """Generate natural language clinical summary."""
        ttp_nat_mo = ttp_natural / 30.44 if np.isfinite(ttp_natural) else 0
        ttp_chemo_mo = ttp_chemo / 30.44 if np.isfinite(ttp_chemo) else 0
        ttp_immuno_mo = ttp_immuno / 30.44 if np.isfinite(ttp_immuno) else 0
        ttp_targeted_mo = ttp_targeted / 30.44 if np.isfinite(ttp_targeted) else 0

        benefit_chemo = (ttp_chemo - ttp_natural) / 30.44
        benefit_immuno = (ttp_immuno - ttp_natural) / 30.44
        benefit_targeted = (ttp_targeted - ttp_natural) / 30.44

        risk_label = "high-risk" if hybrid_risk > 0.5 else "low-risk"
        dominant = ", ".join(hallmarks.get("dominant_hallmarks", ["Unknown"]))

        narrative = (
            f"Patient {patient_id}: {risk_label} molecular profile dominated by {dominant}. "
            f"Mechanistic simulation predicts progression at {ttp_nat_mo:.1f} months under natural history. "
            f"Chemotherapy shows {'moderate' if benefit_chemo > 60 else 'limited'} predicted benefit "
            f"(+{benefit_chemo:.1f} months). "
            f"Immunotherapy benefit predicted {'limited' if benefit_immuno < 60 else 'moderate'} "
            f"(+{benefit_immuno:.1f} months). "
            f"Targeted therapy benefit: +{benefit_targeted:.1f} months. "
            f"Simulation confidence: {simulation_confidence}. "
            f"[RESEARCH PROTOTYPE — NOT FOR CLINICAL USE]"
        )
        return narrative

    def generate_treatment_comparison_narrative(
        self,
        ttp_natural: float,
        ttp_chemo: float,
        ttp_immuno: float,
        ttp_targeted: float,
        simulation_confidence: str = "MODERATE",
    ) -> str:
        """Generate natural language treatment comparison."""
        ttp_nat_mo = ttp_natural / 30.44 if np.isfinite(ttp_natural) else 0
        benefits = {
            "chemotherapy": (ttp_chemo - ttp_natural) / 30.44,
            "immunotherapy": (ttp_immuno - ttp_natural) / 30.44,
            "targeted therapy": (ttp_targeted - ttp_natural) / 30.44,
        }

        best_treatment = max(benefits, key=benefits.get)
        best_benefit = benefits[best_treatment]

        lines = [
            f"Natural history: projected progression at {ttp_nat_mo:.1f} months.",
        ]
        for tx, benefit in benefits.items():
            if np.isfinite(benefit):
                lines.append(f"{tx.capitalize()}: +{benefit:.1f} months benefit.")

        lines.append(f"Recommended priority scenario: {best_treatment} (+{best_benefit:.1f} months).")
        lines.append(f"Confidence: {simulation_confidence}.")
        lines.append("[RESEARCH PROTOTYPE — NOT FOR CLINICAL USE]")

        return " ".join(lines)
