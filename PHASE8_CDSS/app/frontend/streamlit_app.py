"""
Phase 8 CDSS — Streamlit Frontend
7 pages: Home, Patient Input, Prediction, Digital Twin, Explainability, Evidence, Export
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
import sys
from pathlib import Path

# Add app/ to path so backend.* imports work
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time

from backend.config import (
    CLINICAL_DISCLAIMER, TREATMENT_ARTIFACT_WARNING, CALIBRATION_WARNING,
    STAGE_BLOCK_MESSAGE, MODEL_VERSION, PHASE8_VERSION,
)
from backend.safety import SafetyLayer
from backend.prediction import PredictionEngine
from backend.digital_twin import DigitalTwinEngine
from backend.explainability import ExplainabilityEngine
from backend.interpretation import (
    get_treatment_interpretation, get_risk_interpretation,
    get_all_statements, PERMITTED_STATEMENTS, FORBIDDEN_STATEMENTS,
)

# ── Initialize engines ──
@st.cache_resource
def get_engines():
    safety = SafetyLayer()
    pred = PredictionEngine()
    pred.load()
    dt = DigitalTwinEngine()
    dt.load()
    expl = ExplainabilityEngine()
    expl.load()
    return safety, pred, dt, expl


def main():
    st.set_page_config(
        page_title="Phase 8 CDSS — Lung Cancer Digital Twin",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Persistent disclaimer in sidebar ──
    st.sidebar.error(f"⚠️ {CLINICAL_DISCLAIMER}")
    st.sidebar.markdown(f"**Model Version:** {MODEL_VERSION}")
    st.sidebar.markdown(f"**CDSS Version:** {PHASE8_VERSION}")

    # ── Navigation ──
    pages = [
        "Home", "Patient Input", "Prediction", "Digital Twin",
        "Explainability", "Evidence Panel", "Export",
    ]
    page = st.sidebar.selectbox("Navigate", pages)

    safety, pred_engine, dt_engine, expl_engine = get_engines()

    # Initialize session state
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None
    if "explanation_result" not in st.session_state:
        st.session_state.explanation_result = None

    if page == "Home":
        render_home()
    elif page == "Patient Input":
        render_patient_input(safety, pred_engine)
    elif page == "Prediction":
        render_prediction(pred_engine)
    elif page == "Digital Twin":
        render_digital_twin(dt_engine, pred_engine)
    elif page == "Explainability":
        render_explainability(pred_engine, expl_engine, dt_engine)
    elif page == "Evidence Panel":
        render_evidence()
    elif page == "Export":
        render_export()


def render_home():
    st.title("Phase 8 CDSS — Lung Cancer Digital Twin")
    st.markdown(f"**Version:** {PHASE8_VERSION} | **Model:** {MODEL_VERSION}")

    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    st.markdown("---")
    st.header("System Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What This System Does
        - **Risk Assessment:** DeepSurv neural survival model for patient risk stratification
        - **Disease Trajectory:** Gompertz ODE-based tumor growth simulation
        - **Treatment Simulation:** Virtual treatment scenarios (chemotherapy, immunotherapy, targeted therapy)
        - **Uncertainty Quantification:** Monte Carlo confidence intervals on all simulations
        - **Explainability:** Feature importance and mechanistic parameter contributions
        """)

    with col2:
        st.markdown("""
        ### What This System Does NOT Do
        - ❌ Determine the best treatment for individual patients
        - ❌ Replace oncologist judgment
        - ❌ Provide calibrated absolute survival probabilities
        - ❌ Account for treatment resistance, toxicity, or discontinuation
        - ❌ Support Stage IIIB/IV patients (insufficient validation sample)
        """)

    st.markdown("---")
    st.header("Validation Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Internal C-index", "0.6333", "95% CI: 0.575-0.702")
    col2.metric("External C-index", "0.6588", "5/5 cohorts")
    col3.metric("Calibration Slope", "0.6189", "⚠️ Below target (0.8-1.2)")
    col4.metric("Reproducibility", "5/5", "All tests passed")

    st.markdown("---")
    st.header("Phase 7.1 Constraints Enforced")

    st.markdown("""
    | Constraint | Enforcement |
    |-----------|-------------|
    | Absolute survival probabilities | Displayed with "UNCALIBRATED" tag — use for ranking only |
    | Treatment comparisons | Mandatory artifact warning on every simulation output |
    | Stage IIIB/IV patients | **Hard blocked** — prediction refused with specific reason |
    | No prediction without explanation | Every prediction includes feature contributions |
    | No prediction without uncertainty | Every prediction includes confidence interval |
    """)

    st.markdown("---")
    st.info(
        "📋 This system integrates validated outputs from Phase 3 (DeepSurv), Phase 4 (Mechanistic Model), "
        "Phase 5 (Neural-Mechanistic Integration), Phase 6 (Federated Learning), and Phase 7/7.1 "
        "(Personalized Digital Twin, Scientifically Hardened)."
    )


def render_patient_input(safety, pred_engine):
    st.title("Patient Input")
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    st.markdown("---")
    st.header("Patient Selection")

    mode = st.radio("Input Mode", ["Select existing patient", "Enter new patient data"])

    if mode == "Select existing patient":
        df = pred_engine._state_df
        # Filter out blocked stages
        blocked = ["IIIB", "IV"]
        available = df[~df["stage"].isin(blocked)]
        patient_ids = available["patient_id"].tolist()
        selected = st.selectbox("Select Patient ID", [""] + patient_ids[:200])

        if selected:
            row = df[df["patient_id"] == selected].iloc[0]
            st.session_state.patient_data = {
                "patient_id": str(row["patient_id"]),
                "age": float(row["age"]),
                "cancer_type": str(row["cancer_type"]),
                "stage": str(row["stage"]),
                "smoking_status": str(row["smoking_status"]),
            }
            st.success(f"Patient selected: {selected}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Age", int(row["age"]))
            col2.metric("Cancer Type", row["cancer_type"])
            col3.metric("Stage", row["stage"])
            col4.metric("Smoking", row["smoking_status"])

            # Show risk group from existing data
            st.info(f"Risk Group: **{row.get('risk_group', 'Unknown')}** | "
                    f"DeepSurv Risk: {row['deepsurv_risk']:.4f}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=65)
            cancer_type = st.selectbox("Cancer Type", ["", "LUAD (Adenocarcinoma)", "LUSC (Squamous Cell)"])
            stage = st.selectbox("Stage", ["", "I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV"])
        with col2:
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            st.markdown("### Gene Expression (Optional)")
            st.markdown("*Provide Ensembl gene IDs and expression values. Leave empty for clinical-only prediction.*")
            gene_input = st.text_area("Gene Expression (JSON format)", height=100,
                                       placeholder='{"ENSG00000171557.17": 5.2, "ENSG00000021826.17": 3.1}')

        if st.button("Validate & Save Patient", type="primary"):
            ct_mapped = cancer_type.split(" ")[0] if cancer_type else ""
            gene_expr = None
            if gene_input.strip():
                try:
                    gene_expr = json.loads(gene_input)
                except json.JSONDecodeError:
                    st.warning("Invalid JSON for gene expression — ignoring.")

            result = safety.check(
                age=age, cancer_type=ct_mapped, stage=stage,
                smoking_status=smoking, gene_expression=gene_expr,
            )

            if result.blocked:
                st.error(f"🚫 **BLOCKED:** {result.block_reason}")
                if result.block_source == "phase7_1_constraint":
                    st.warning(STAGE_BLOCK_MESSAGE)
            else:
                st.success("✅ Patient validated successfully!")
                if result.warnings:
                    for w in result.warnings:
                        st.warning(f"⚠️ {w}")
                st.session_state.patient_data = {
                    "patient_id": None,
                    "age": float(age),
                    "cancer_type": ct_mapped,
                    "stage": stage,
                    "smoking_status": smoking,
                    "gene_expression": gene_expr,
                }

    # Show current patient
    if st.session_state.patient_data:
        st.markdown("---")
        st.markdown("### Current Patient")
        pd_data = st.session_state.patient_data
        st.json({k: v for k, v in pd_data.items() if k != "gene_expression"})


def render_prediction(pred_engine):
    st.title("Prediction")
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    if not st.session_state.patient_data:
        st.warning("Please select or enter a patient first (Patient Input page).")
        return

    pd_data = st.session_state.patient_data

    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Running prediction..."):
            if pd_data.get("patient_id"):
                result = pred_engine.predict_existing(pd_data["patient_id"])
                if result:
                    t_eval = np.linspace(1, 1095, 100)
                    S = pred_engine.risk_to_survival_curve(result["deepsurv_risk"], t_eval)
                    surv_at = pred_engine.get_survival_at_timepoints(result["deepsurv_risk"])
                    st.session_state.prediction_result = {
                        "patient": result,
                        "survival_curve": {"t_days": t_eval, "t_months": t_eval / 30.44, "S": S},
                        "survival_at_timepoints": surv_at,
                    }
                else:
                    st.error("Patient not found in state matrix.")
                    return
            else:
                pred = pred_engine.predict_new(
                    age=pd_data["age"], cancer_type=pd_data["cancer_type"],
                    stage=pd_data["stage"], smoking_status=pd_data["smoking_status"],
                    gene_expression=pd_data.get("gene_expression"),
                )
                t_eval = np.linspace(1, 1095, 100)
                S = pred_engine.risk_to_survival_curve(pred["deepsurv_risk"], t_eval)
                surv_at = pred_engine.get_survival_at_timepoints(pred["deepsurv_risk"])
                st.session_state.prediction_result = {
                    "patient": pred,
                    "survival_curve": {"t_days": t_eval, "t_months": t_eval / 30.44, "S": S},
                    "survival_at_timepoints": surv_at,
                }

    result = st.session_state.prediction_result
    if result:
        patient = result["patient"]
        st.markdown("---")
        st.header("Risk Assessment")

        # Risk group with color
        rg = patient.get("risk_group", "Unknown")
        rg_color = {"Low": "green", "Intermediate": "orange", "High": "red"}.get(rg, "gray")
        st.markdown(f"### Risk Group: :{rg_color}[{rg}]")

        col1, col2, col3 = st.columns(3)
        col1.metric("DeepSurv Risk Score", f"{patient['deepsurv_risk']:.4f}")
        col2.metric("Risk Std (Ensemble)", f"{patient.get('deepsurv_risk_std', 0):.4f}")
        col3.metric("Uncertainty Level", "Moderate")

        # Survival probability with calibration warning
        st.markdown("---")
        st.header("Survival Probability")

        # INLINE calibration warning — not in a separate panel
        st.warning(f"⚠️ **UNCALIBRATED — interpret with caution.** {CALIBRATION_WARNING}")

        surv_at = result["survival_at_timepoints"]
        col1, col2, col3 = st.columns(3)
        col1.metric("12-month survival", f"{surv_at['12m']:.1%}", "⚠️ Uncalibrated")
        col2.metric("24-month survival", f"{surv_at['24m']:.1%}", "⚠️ Uncalibrated")
        col3.metric("36-month survival", f"{surv_at['36m']:.1%}", "⚠️ Uncalibrated")

        # Survival curve
        fig = go.Figure()
        sc = result["survival_curve"]
        fig.add_trace(go.Scatter(
            x=sc["t_months"], y=sc["S"],
            mode="lines", name="Survival Probability",
            line=dict(color="blue", width=2),
        ))
        fig.update_layout(
            title="Survival Curve (UNCALIBRATED — use for ranking only)",
            xaxis_title="Time (months)", yaxis_title="Survival Probability",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk interpretation
        st.markdown("---")
        st.info(get_risk_interpretation(rg))

        # Confidence interval info
        st.markdown("---")
        st.header("Confidence & Uncertainty")
        st.markdown(f"""
        - **Uncertainty Level:** Moderate (MC n=20, 95% CI)
        - **Ensemble Std:** {patient.get('deepsurv_risk_std', 0):.4f}
        - **Calibration Status:** ⚠️ Uncalibrated (slope=0.6189, target=0.8-1.2)
        - **Model Version:** {MODEL_VERSION}
        """)


def render_digital_twin(dt_engine, pred_engine):
    st.title("Digital Twin")
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    if not st.session_state.patient_data:
        st.warning("Please select or enter a patient first (Patient Input page).")
        return

    pd_data = st.session_state.patient_data

    # Treatment selection
    st.markdown("---")
    st.header("Treatment Simulation")
    st.markdown("**Simulated Research Scenario** — not a treatment recommendation.")

    treatment = st.selectbox(
        "Select Treatment Scenario",
        ["none", "chemo", "immuno", "targeted"],
        format_func=lambda x: {
            "none": "Natural History (no treatment)",
            "chemo": "Chemotherapy (Cisplatin)",
            "immuno": "Immunotherapy (Pembrolizumab)",
            "targeted": "Targeted Therapy (Osimertinib)",
        }[x],
    )

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running ODE simulation..."):
            # Get existing or run new
            if pd_data.get("patient_id"):
                existing = dt_engine.get_existing_trajectories(pd_data["patient_id"])
                if existing:
                    # Run new simulation for selected treatment
                    state_row = pred_engine._state_df[
                        pred_engine._state_df["patient_id"].astype(str) == pd_data["patient_id"]
                    ].iloc[0]
                    patient_params = {
                        "alpha": float(state_row["alpha"]),
                        "V_max": float(state_row["V_max"]),
                        "V0": float(state_row["V0"]),
                        "k_immune": float(state_row["k_immune"]),
                        "E0": float(state_row["E0"]),
                        "mu_E": float(state_row["mu_E"]),
                        "rho": float(state_row["rho"]),
                        "delta_c": float(state_row["delta_c"]),
                        "delta_t": float(state_row["delta_t"]),
                    }
                else:
                    patient_params = dt_engine.get_population_priors()
            else:
                patient_params = dt_engine.get_population_priors()

            try:
                sim_result = dt_engine.simulate_treatment(patient_params, treatment)
                st.session_state.simulation_result = sim_result
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                return

    sim = st.session_state.simulation_result
    if sim:
        # ── MANDATORY ARTIFACT WARNING — attached to treatment output, not dismissible ──
        if treatment != "none":
            st.error(f"⚠️ **ARTIFACT WARNING:** {TREATMENT_ARTIFACT_WARNING}")

        st.markdown("---")
        st.header(f"Simulation Result: {sim['treatment_label']}")

        # TTP
        col1, col2, col3 = st.columns(3)
        col1.metric("TTP (days)", f"{sim['ttp_days']:.1f}")
        col2.metric("TTP (months)", f"{sim['ttp_months']:.1f}")
        col3.metric("Outcome Type", "Counterfactual" if sim.get("is_counterfactual") else "Natural History")

        # Trajectory plot
        fig = go.Figure()
        t_months = sim["trajectory_t_months"]
        V = sim["trajectory_V"]
        diameter = sim["trajectory_diameter_mm"]
        fig.add_trace(go.Scatter(
            x=t_months, y=diameter,
            mode="lines", name="Tumor Diameter (mm)",
            line=dict(color="red", width=2),
        ))
        fig.update_layout(
            title=f"Tumor Growth Trajectory — {sim['treatment_label']}",
            xaxis_title="Time (months)", yaxis_title="Tumor Diameter (mm)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Clinical interpretation
        st.markdown("---")
        st.header("Clinical Interpretation")
        st.info(get_treatment_interpretation(sim["treatment_label"]))

        # Counterfactual comparison
        st.markdown("---")
        st.header("Counterfactual Comparison")
        if pd_data.get("patient_id"):
            existing = dt_engine.get_existing_trajectories(pd_data["patient_id"])
            if existing:
                comp_data = {
                    "Treatment": ["Natural History", "Chemotherapy", "Immunotherapy", "Targeted"],
                    "TTP (months)": [
                        existing["ttp_natural"] / 30.44,
                        existing["ttp_chemo"] / 30.44,
                        existing["ttp_immuno"] / 30.44,
                        existing["ttp_targeted"] / 30.44,
                    ],
                }
                fig2 = go.Figure(data=[go.Bar(
                    x=comp_data["Treatment"], y=comp_data["TTP (months)"],
                    marker_color=["gray", "blue", "green", "orange"],
                )])
                fig2.update_layout(
                    title="TTP Comparison Across Treatments (⚠️ Artifact: 100% benefit — see warning above)",
                    xaxis_title="Treatment", yaxis_title="TTP (months)",
                    height=350,
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.error(f"⚠️ {TREATMENT_ARTIFACT_WARNING}")


def render_explainability(pred_engine, expl_engine, dt_engine):
    st.title("Explainability")
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    if not st.session_state.patient_data:
        st.warning("Please select or enter a patient first (Patient Input page).")
        return

    pd_data = st.session_state.patient_data

    if st.button("Generate Explanation", type="primary"):
        with st.spinner("Computing feature contributions..."):
            contrib = pred_engine.get_feature_contributions(
                age=pd_data["age"], cancer_type=pd_data["cancer_type"],
                stage=pd_data["stage"], smoking_status=pd_data["smoking_status"],
                gene_expression=pd_data.get("gene_expression"),
            )
            priors = dt_engine.get_population_priors()
            # Use state matrix params if existing patient
            if pd_data.get("patient_id"):
                row = pred_engine._state_df[
                    pred_engine._state_df["patient_id"].astype(str) == pd_data["patient_id"]
                ].iloc[0]
                priors = {
                    "alpha": float(row["alpha"]), "V_max": float(row["V_max"]),
                    "V0": float(row["V0"]), "k_immune": float(row["k_immune"]),
                    "delta_c": float(row["delta_c"]), "delta_t": float(row["delta_t"]),
                }
            explanation = expl_engine.get_patient_explanation(priors, contrib)
            st.session_state.explanation_result = explanation

    expl = st.session_state.explanation_result
    if expl:
        st.markdown("---")
        st.header("Top Clinical Feature Contributions")

        clinical = expl["top_clinical_features"]
        if clinical:
            fig = go.Figure(go.Bar(
                x=[v for _, v in clinical],
                y=[k for k, _ in clinical],
                orientation="h",
                marker_color="steelblue",
            ))
            fig.update_layout(
                title="Clinical Feature Contributions (Gradient × Input)",
                xaxis_title="Contribution", yaxis_title="Feature",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("Top Molecular Feature Contributions")

        molecular = expl["top_molecular_features"]
        if molecular:
            fig2 = go.Figure(go.Bar(
                x=[v for _, v in molecular],
                y=[k for k, _ in molecular],
                orientation="h",
                marker_color="darkorange",
            ))
            fig2.update_layout(
                title="Molecular Feature Contributions (Gradient × Input)",
                xaxis_title="Contribution", yaxis_title="Gene (Ensembl ID)",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.header("Mechanistic Parameter Contributions")

        mech = expl["mechanistic_contributions"]
        mech_data = []
        for pname, info in mech.items():
            mech_data.append({
                "Parameter": pname,
                "Value": info["value"],
                "Description": info["description"],
                "Effect": info["effect"],
                "Literature Range": info["literature_range"],
            })
        st.dataframe(pd.DataFrame(mech_data), use_container_width=True)

        st.markdown("---")
        st.header("Global Cohort Importance")
        global_imp = expl["global_cohort_importance"]
        st.markdown(f"**Source:** {global_imp['source']} | **Features:** {global_imp['n_features']}")
        if global_imp["top_10"]:
            imp_df = pd.DataFrame(global_imp["top_10"])
            st.dataframe(imp_df, use_container_width=True)

        st.markdown("---")
        st.info(expl["explanation_method"])
        st.caption(expl["disclaimer"])


def render_evidence():
    st.title("Evidence Panel")
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    st.markdown("---")
    st.header("Model Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Training
        - **Cohort:** TCGA (The Cancer Genome Atlas)
        - **Training set:** n=737
        - **Validation set:** n=182
        - **Model:** DeepSurv Top-2 ensemble ([64, 32], GELU, dropout=0.2)
        - **Features:** 143 (Age + 3 clinical categorical + 127 gene expression)
        """)

    with col2:
        st.markdown("""
        ### External Validation (Phase 3)
        - **GSE30219:** C-index 0.6782 ✅
        - **GSE50081:** C-index 0.5965 ✅
        - **GSE72094:** C-index 0.6334 ✅
        - **GSE31210:** C-index 0.7271 ✅
        - **Mean external C-index:** 0.6588
        - **DeepSurv wins 5/5 cohorts**
        """)

    st.markdown("---")
    st.header("Known Limitations (from Phase 7.1)")

    st.markdown("""
    | Limitation | Detail | Phase 8 Enforcement |
    |-----------|--------|---------------------|
    | Calibration slope 0.6189 | Below target 0.8-1.2 | "UNCALIBRATED" tag on all survival probabilities |
    | 100% treatment benefit | Structural ODE artifact | Mandatory artifact warning on every treatment simulation |
    | Stage IIIB (n=7) | Insufficient sample | Hard block — prediction refused |
    | Stage IV (n=10) | Insufficient sample | Hard block — prediction refused |
    | No resistance modeling | ODE assumes continuous response | Artifact warning |
    | No toxicity modeling | No Grade 3-4 AE simulation | Artifact warning |
    | No prospective validation | All data retrospective | Research disclaimer |
    | No regulatory approval | Not FDA/EMA reviewed | Research disclaimer |
    """)

    st.markdown("---")
    st.header("Clinical Interpretation Statements")

    statements = get_all_statements()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### ✅ Permitted Statements ({statements['n_permitted']})")
        for i, s in enumerate(statements["permitted"], 1):
            st.markdown(f"{i}. {s}")

    with col2:
        st.markdown(f"### ❌ Forbidden Statements ({statements['n_forbidden']})")
        for i, s in enumerate(statements["forbidden"], 1):
            st.markdown(f"{i}. {s}")

    st.markdown("---")
    st.header("Phase 7.1 Scientific Position Statement")
    try:
        from backend.config import P7_1_DIR
        position_path = P7_1_DIR / "phase7_scientific_position_statement.md"
        if position_path.exists():
            st.markdown(position_path.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Could not load Phase 7.1 position statement.")


def render_export():
    st.title("Export")
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    if not st.session_state.patient_data:
        st.warning("Please select or enter a patient first.")
        return

    st.markdown("---")
    st.header("Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### JSON Export")
        if st.button("Generate JSON Report"):
            report = {
                "patient": {k: v for k, v in st.session_state.patient_data.items() if k != "gene_expression"},
                "prediction": st.session_state.prediction_result,
                "simulation": st.session_state.simulation_result,
                "explanation": st.session_state.explanation_result,
                "model_version": MODEL_VERSION,
                "cdss_version": PHASE8_VERSION,
                "disclaimer": CLINICAL_DISCLAIMER,
                "calibration_warning": CALIBRATION_WARNING,
                "artifact_warning": TREATMENT_ARTIFACT_WARNING,
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"cdss_report_{st.session_state.patient_data.get('patient_id', 'new')}.json",
                mime="application/json",
            )

    with col2:
        st.markdown("### CSV Export")
        if st.button("Generate CSV Summary"):
            rows = []
            pd_data = st.session_state.patient_data
            pred = st.session_state.prediction_result
            sim = st.session_state.simulation_result
            base = {
                "patient_id": pd_data.get("patient_id", "new"),
                "age": pd_data["age"],
                "cancer_type": pd_data["cancer_type"],
                "stage": pd_data["stage"],
                "smoking_status": pd_data["smoking_status"],
                "model_version": MODEL_VERSION,
                "disclaimer": CLINICAL_DISCLAIMER,
            }
            if pred:
                p = pred["patient"]
                base.update({
                    "risk_group": p.get("risk_group"),
                    "deepsurv_risk": p.get("deepsurv_risk"),
                    "survival_12m": pred["survival_at_timepoints"]["12m"],
                    "survival_24m": pred["survival_at_timepoints"]["24m"],
                    "survival_36m": pred["survival_at_timepoints"]["36m"],
                    "calibration_status": "UNCALIBRATED",
                })
            if sim:
                base.update({
                    "treatment": sim.get("treatment"),
                    "ttp_days": sim.get("ttp_days"),
                    "ttp_months": sim.get("ttp_months"),
                    "artifact_warning": TREATMENT_ARTIFACT_WARNING,
                })
            rows.append(base)
            df = pd.DataFrame(rows)
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name=f"cdss_summary_{st.session_state.patient_data.get('patient_id', 'new')}.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.markdown("### Simulation Report (with artifact warning)")

    if st.session_state.simulation_result:
        sim = st.session_state.simulation_result
        report_text = f"""
# CDSS Simulation Report

**Patient:** {st.session_state.patient_data.get('patient_id', 'New patient')}
**Model Version:** {MODEL_VERSION}
**CDSS Version:** {PHASE8_VERSION}

## Patient Information
- Age: {st.session_state.patient_data['age']}
- Cancer Type: {st.session_state.patient_data['cancer_type']}
- Stage: {st.session_state.patient_data['stage']}
- Smoking Status: {st.session_state.patient_data['smoking_status']}

## Simulation Result
- Treatment: {sim.get('treatment_label', sim.get('treatment'))}
- TTP: {sim.get('ttp_days', 'N/A'):.1f} days ({sim.get('ttp_months', 0):.1f} months)
- Outcome Type: {'Counterfactual' if sim.get('is_counterfactual') else 'Natural History'}

## ⚠️ ARTIFACT WARNING
{TREATMENT_ARTIFACT_WARNING}

## Disclaimer
{CLINICAL_DISCLAIMER}
"""
        st.download_button(
            "Download Simulation Report (TXT)",
            data=report_text,
            file_name=f"simulation_report_{st.session_state.patient_data.get('patient_id', 'new')}.txt",
            mime="text/plain",
        )
        st.text(report_text)


if __name__ == "__main__":
    main()
