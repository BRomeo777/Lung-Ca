"""
Phase 8 CDSS — FastAPI Backend
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd

from .config import (
    CLINICAL_DISCLAIMER, TREATMENT_ARTIFACT_WARNING, CALIBRATION_WARNING,
    STAGE_BLOCK_MESSAGE, MODEL_VERSION, PHASE8_VERSION, P7_1_DIR,
)
from .safety import SafetyLayer, SafetyResult
from .prediction import PredictionEngine
from .digital_twin import DigitalTwinEngine
from .explainability import ExplainabilityEngine
from .interpretation import (
    get_treatment_interpretation, get_risk_interpretation,
    get_all_statements, PERMITTED_STATEMENTS, FORBIDDEN_STATEMENTS,
)

app = FastAPI(
    title="Phase 8 CDSS — Lung Cancer Digital Twin",
    description=CLINICAL_DISCLAIMER,
    version=PHASE8_VERSION,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Initialize engines ──
safety = SafetyLayer()
prediction_engine = PredictionEngine()
dt_engine = DigitalTwinEngine()
expl_engine = ExplainabilityEngine()


# ── Data Models ──
class PatientInput(BaseModel):
    patient_id: Optional[str] = None
    age: float = Field(..., description="Patient age")
    cancer_type: str = Field(..., description="LUAD, LUSC, adenocarcinoma, squamous")
    stage: str = Field(..., description="I, IA, IB, II, IIA, IIB, III, IIIA, IIIB, IV")
    smoking_status: str = Field("Never", description="Current, Former, Never")
    gene_expression: Optional[dict] = Field(None, description="Dict of Ensembl ID -> expression value")


class SimulationInput(BaseModel):
    patient_id: Optional[str] = None
    age: float
    cancer_type: str
    stage: str
    smoking_status: str = "Never"
    treatment: str = Field("none", description="none, chemo, immuno, targeted")
    alpha: Optional[float] = None
    V0: Optional[float] = None
    V_max: Optional[float] = None
    k_immune: Optional[float] = None


# ── Endpoints ──
@app.get("/")
async def root():
    return {
        "system": "Phase 8 CDSS — Lung Cancer Digital Twin",
        "version": PHASE8_VERSION,
        "model_version": MODEL_VERSION,
        "disclaimer": CLINICAL_DISCLAIMER,
        "endpoints": ["/health", "/predict", "/simulate", "/explain", "/evidence", "/statements"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": PHASE8_VERSION}


@app.post("/predict")
async def predict(patient: PatientInput):
    """Get risk prediction for a patient. Enforces Phase 7.1 constraints."""
    # Step 1: Safety check
    result = safety.check(
        age=patient.age, cancer_type=patient.cancer_type,
        stage=patient.stage, smoking_status=patient.smoking_status,
        gene_expression=patient.gene_expression,
    )

    if result.blocked:
        return {
            "allowed": False,
            "blocked": True,
            "block_reason": result.block_reason,
            "block_source": result.block_source,
            "disclaimer": CLINICAL_DISCLAIMER,
            "warnings": result.warnings,
        }

    # Step 2: Check if existing patient
    if patient.patient_id:
        existing = prediction_engine.predict_existing(patient.patient_id)
        if existing:
            # Get survival curve
            t_eval = np.linspace(1, 1095, 100)
            S = prediction_engine.risk_to_survival_curve(existing["deepsurv_risk"], t_eval)
            surv_at = prediction_engine.get_survival_at_timepoints(existing["deepsurv_risk"])
            return {
                "allowed": True,
                "blocked": False,
                "patient": existing,
                "survival_curve": {
                    "t_days": t_eval.tolist(),
                    "t_months": (t_eval / 30.44).tolist(),
                    "S": S.tolist(),
                },
                "survival_at_timepoints": surv_at,
                "calibration_warning": CALIBRATION_WARNING,
                "risk_interpretation": get_risk_interpretation(existing["risk_group"]),
                "warnings": result.warnings,
                "disclaimer": CLINICAL_DISCLAIMER,
                "model_version": MODEL_VERSION,
            }

    # Step 3: New patient prediction
    pred = prediction_engine.predict_new(
        age=patient.age, cancer_type=patient.cancer_type,
        stage=patient.stage, smoking_status=patient.smoking_status,
        gene_expression=patient.gene_expression,
    )

    t_eval = np.linspace(1, 1095, 100)
    S = prediction_engine.risk_to_survival_curve(pred["deepsurv_risk"], t_eval)
    surv_at = prediction_engine.get_survival_at_timepoints(pred["deepsurv_risk"])

    return {
        "allowed": True,
        "blocked": False,
        "patient": pred,
        "survival_curve": {
            "t_days": t_eval.tolist(),
            "t_months": (t_eval / 30.44).tolist(),
            "S": S.tolist(),
        },
        "survival_at_timepoints": surv_at,
        "calibration_warning": CALIBRATION_WARNING,
        "risk_interpretation": get_risk_interpretation(pred["risk_group"]),
        "warnings": result.warnings,
        "disclaimer": CLINICAL_DISCLAIMER,
        "model_version": MODEL_VERSION,
    }


@app.post("/simulate")
async def simulate(sim: SimulationInput):
    """Run Digital Twin treatment simulation. Always carries artifact warning."""
    # Safety check first
    result = safety.check(
        age=sim.age, cancer_type=sim.cancer_type,
        stage=sim.stage, smoking_status=sim.smoking_status,
    )
    if result.blocked:
        return {
            "allowed": False,
            "blocked": True,
            "block_reason": result.block_reason,
            "block_source": result.block_source,
            "disclaimer": CLINICAL_DISCLAIMER,
        }

    # Get patient params
    if sim.patient_id:
        existing = dt_engine.get_existing_trajectories(sim.patient_id)
        if existing and sim.treatment == "none":
            return {
                "allowed": True,
                "blocked": False,
                "simulation": existing,
                "artifact_warning": TREATMENT_ARTIFACT_WARNING,
                "interpretation": get_treatment_interpretation(
                    "Natural History (no treatment)" if sim.treatment == "none" else
                    {"chemo": "Chemotherapy", "immuno": "Immunotherapy", "targeted": "Targeted Therapy"}.get(sim.treatment, sim.treatment)
                ),
                "disclaimer": CLINICAL_DISCLAIMER,
            }

    # Build params
    priors = dt_engine.get_population_priors()
    patient_params = {
        "alpha": sim.alpha or priors["alpha"],
        "V_max": sim.V_max or priors["V_max"],
        "V0": sim.V0 or priors["V0"],
        "k_immune": sim.k_immune or priors["k_immune"],
        "E0": priors["E0"],
        "mu_E": priors["mu_E"],
        "rho": priors["rho"],
        "delta_c": priors["delta_c"],
        "delta_t": priors["delta_t"],
    }

    try:
        sim_result = dt_engine.simulate_treatment(patient_params, sim.treatment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

    tx_label = sim_result.get("treatment_label", sim.treatment)
    return {
        "allowed": True,
        "blocked": False,
        "simulation": sim_result,
        "artifact_warning": TREATMENT_ARTIFACT_WARNING,
        "interpretation": get_treatment_interpretation(tx_label),
        "disclaimer": CLINICAL_DISCLAIMER,
    }


@app.post("/explain")
async def explain(patient: PatientInput):
    """Get explainability outputs for a patient."""
    result = safety.check(
        age=patient.age, cancer_type=patient.cancer_type,
        stage=patient.stage, smoking_status=patient.smoking_status,
    )
    if result.blocked:
        return {
            "allowed": False,
            "blocked": True,
            "block_reason": result.block_reason,
        }

    # Get feature contributions
    contrib = prediction_engine.get_feature_contributions(
        age=patient.age, cancer_type=patient.cancer_type,
        stage=patient.stage, smoking_status=patient.smoking_status,
        gene_expression=patient.gene_expression,
    )

    # Get patient params for mechanistic explanation
    priors = dt_engine.get_population_priors()
    explanation = expl_engine.get_patient_explanation(priors, contrib)

    return {
        "allowed": True,
        "blocked": False,
        "explanation": explanation,
        "disclaimer": CLINICAL_DISCLAIMER,
    }


@app.get("/evidence")
async def evidence():
    """Get evidence panel data from Phase 7.1 deliverables."""
    import json
    evidence_data = {
        "model_version": MODEL_VERSION,
        "training_cohort": "TCGA (n=737 train, n=182 validation)",
        "external_validation": "5 cohorts: GSE30219, GSE50081, GSE72094, GSE31210, GSE31210",
        "external_validation_cindex": 0.6588,
        "internal_cindex": 0.6333,
        "internal_cindex_ci": [0.5748, 0.7021],
        "calibration_slope": 0.6189,
        "calibration_target": "0.8-1.2",
        "calibration_status": "UNCALIBRATED — Platt scaling evaluated but not deployed",
        "ibs": 0.7196,
        "n_sims": 3676,
        "safety_flags": 0,
        "reproducibility": "5/5 tests passed",
        "biological_plausibility": "rho=-0.1786, p=0.0004 (growth-survival correlation)",
        "blocked_stages": ["IIIB", "IV"],
        "blocked_reason": "Insufficient sample size (n=7, n=10) in Phase 7.1 validation",
        "treatment_artifact": "100% treatment benefit — structural property of ODE model",
        "treatment_assumptions": "No resistance, no toxicity, no discontinuation, no dose modification",
        "disclaimer": CLINICAL_DISCLAIMER,
    }

    # Load Phase 7.1 position statement if available
    position_path = P7_1_DIR / "phase7_scientific_position_statement.md"
    if position_path.exists():
        evidence_data["phase7_1_position_statement"] = position_path.read_text(encoding="utf-8")

    return evidence_data


@app.get("/statements")
async def statements():
    """Get all permitted and forbidden clinical interpretation statements."""
    return get_all_statements()


@app.get("/patients")
async def list_patients(limit: int = 20):
    """List patients from the state matrix for browsing."""
    if prediction_engine._state_df is None:
        prediction_engine.load()
    df = prediction_engine._state_df.head(limit)
    return {
        "patients": df[["patient_id", "age", "cancer_type", "stage", "risk_group"]].to_dict(orient="records"),
        "total": len(prediction_engine._state_df),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
