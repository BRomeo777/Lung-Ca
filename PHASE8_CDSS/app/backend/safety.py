"""
Phase 8 CDSS — Safety Layer
Enforces Phase 7.1 constraint contract: hard blocks, validation, OOD detection.
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .config import (
    BLOCKED_STAGES, SUPPORTED_STAGES, SUPPORTED_CANCER_TYPES,
    SUPPORTED_SMOKING, STAGE_BLOCK_MESSAGE, CLINICAL_DISCLAIMER,
)


@dataclass
class SafetyResult:
    allowed: bool
    blocked: bool = False
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    block_reason: str = ""
    block_source: str = ""  # "phase7_1_constraint" or "validation" or "ood"


class SafetyLayer:
    """Enforces Phase 7.1 constraints as hard blocks, not soft warnings."""

    def __init__(self):
        self.blocked_stages = set(BLOCKED_STAGES)
        self.supported_stages = set(SUPPORTED_STAGES)
        self.supported_cancer_types = set(SUPPORTED_CANCER_TYPES)
        self.supported_smoking = set(SUPPORTED_SMOKING)

    def check(self, age: float, cancer_type: str, stage: str,
              smoking_status: str, **kwargs) -> SafetyResult:
        result = SafetyResult(allowed=True)

        # ── 1. MISSING VARIABLES ──
        if age is None:
            result.errors.append("Age is required")
            result.allowed = False
        elif not (0 < age < 120):
            result.errors.append(f"Age {age} is outside biological range (0-120)")
            result.allowed = False

        if cancer_type is None or str(cancer_type).strip() == "":
            result.errors.append("Cancer type is required")
            result.allowed = False

        if stage is None or str(stage).strip() == "":
            result.errors.append("Stage is required")
            result.allowed = False

        if smoking_status is None or str(smoking_status).strip() == "":
            result.warnings.append("Smoking status not provided — defaulting to 'Never'")

        if not result.allowed:
            result.blocked = True
            result.block_reason = "; ".join(result.errors)
            result.block_source = "validation"
            return result

        # ── 2. IMPOSSIBLE BIOLOGICAL VALUES ──
        if isinstance(age, (int, float)) and (age < 18 or age > 100):
            result.warnings.append(f"Age {age} is outside typical lung cancer range (18-100)")

        # ── 3. STAGE CHECK — HARD BLOCK for IIIB/IV (checked before subtype) ──
        stage_mapped = self._map_stage(stage)
        if stage_mapped is None:
            result.errors.append(f"Unsupported stage: {stage}")
            result.blocked = True
            result.block_reason = f"Unsupported stage: {stage}. Supported: {SUPPORTED_STAGES}"
            result.block_source = "validation"
            result.allowed = False
            return result

        if stage_mapped in self.blocked_stages:
            result.blocked = True
            result.block_reason = STAGE_BLOCK_MESSAGE
            result.block_source = "phase7_1_constraint"
            result.allowed = False
            return result

        # ── 4. UNSUPPORTED SUBTYPE ──
        ct_mapped = self._map_cancer_type(cancer_type)
        if ct_mapped is None:
            result.errors.append(f"Unsupported cancer type: {cancer_type}")
            result.blocked = True
            result.block_reason = f"Unsupported cancer type: {cancer_type}. Supported: {SUPPORTED_CANCER_TYPES}"
            result.block_source = "validation"
            result.allowed = False
            return result

        # ── 5. OOD DETECTION (generic, for cases not already blocked) ──
        # Unusual age-stage combinations
        if isinstance(age, (int, float)):
            if age < 30 and stage_mapped in ["III", "IIIA"]:
                result.warnings.append(f"Unusual age-stage combination: age {age} with Stage {stage_mapped}")

        # Check for unusual biomarker combinations (if provided)
        gene_expr = kwargs.get("gene_expression")
        if gene_expr is not None:
            # Simple OOD: check if all gene values are zero (no expression data)
            if all(v == 0 for v in gene_expr.values() if isinstance(v, (int, float))):
                result.warnings.append(
                    "No gene expression data provided — prediction will use clinical features only. "
                    "This may reduce accuracy."
                )

        # ── 6. Smoking status mapping ──
        sm_mapped = self._map_smoking(smoking_status)
        if sm_mapped is None and smoking_status:
            result.warnings.append(f"Unrecognized smoking status '{smoking_status}' — defaulting to 'Never'")

        return result

    @staticmethod
    def _map_cancer_type(v):
        s = str(v).lower()
        if "luad" in s or "adc" in s or "adenocarcinoma" in s:
            return "LUAD_Adenocarcinoma"
        if "lusc" in s or "sqc" in s or "squamous" in s or "scc" in s:
            return "LUSC_SquamousCell"
        return None

    @staticmethod
    def _map_stage(v):
        s = str(v).strip()
        if s in SUPPORTED_STAGES + BLOCKED_STAGES:
            return s
        return None

    @staticmethod
    def _map_smoking(v):
        s = str(v).lower()
        if "former" in s:
            return "Former"
        if "current" in s:
            return "Current"
        if "never" in s:
            return "Never"
        return None
