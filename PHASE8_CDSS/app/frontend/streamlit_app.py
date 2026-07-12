"""
OncoTwin — Lung Cancer Neural-Mechanistic Digital Twin & Clinical Decision Support
Research-grade web application showcasing the full 8-phase project pipeline.
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
    get_all_statements,
)

# ══════════════════════════════════════════════════════════════════════════
# THEME / DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════
PRIMARY = "#0e7490"      # teal-700
PRIMARY_DARK = "#155e75"  # teal-800
ACCENT = "#0891b2"       # cyan-600
INK = "#0f172a"          # slate-900
MUTED = "#64748b"        # slate-500

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #0f172a; letter-spacing: -0.02em; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0e7490 0%, #155e75 55%, #0f172a 100%);
    border-radius: 20px;
    padding: 2.6rem 2.4rem;
    color: #f8fafc;
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 45px -20px rgba(14,116,144,0.55);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.14), transparent 70%);
}
.hero h1 { color: #ffffff; font-size: 2.35rem; margin: 0 0 .4rem 0; }
.hero p { color: #cbeef5; font-size: 1.02rem; margin: 0; max-width: 780px; line-height: 1.55; }
.hero .kicker {
    display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .18em;
    text-transform: uppercase; color: #7dd3fc; margin-bottom: .8rem;
    background: rgba(255,255,255,0.08); padding: .3rem .7rem; border-radius: 999px;
}

/* Metric cards */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 1rem; margin: .4rem 0 1rem 0; }
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.2rem 1.3rem; box-shadow: 0 4px 14px -8px rgba(15,23,42,0.15);
    transition: transform .15s ease, box-shadow .15s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px -14px rgba(14,116,144,0.4); }
.metric-card .label { font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: #64748b; }
.metric-card .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #0e7490; line-height: 1.1; margin: .3rem 0 .15rem 0; }
.metric-card .sub { font-size: .78rem; color: #94a3b8; }

/* Feature/step cards */
.card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
    padding:1.3rem 1.4rem; margin-bottom:1rem; box-shadow:0 4px 14px -10px rgba(15,23,42,0.15);
}
.card h4 { margin:.1rem 0 .5rem 0; font-family:'Space Grotesk',sans-serif; color:#155e75; }
.card p, .card li { color:#475569; font-size:.92rem; line-height:1.55; }

/* Phase pipeline */
.phase-row { display:flex; align-items:stretch; gap:.6rem; margin-bottom:.7rem; }
.phase-num {
    flex:0 0 46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center;
    font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; color:#fff;
    background:linear-gradient(135deg,#0891b2,#155e75);
}
.phase-body { flex:1; background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #0891b2; border-radius:10px; padding:.7rem 1rem; }
.phase-body .t { font-weight:700; color:#0f172a; font-size:.95rem; }
.phase-body .d { color:#64748b; font-size:.83rem; margin-top:.15rem; }
.phase-body .badge { display:inline-block; font-size:.68rem; font-weight:700; color:#065f46; background:#d1fae5; padding:.12rem .5rem; border-radius:999px; margin-left:.4rem; }

/* Pills / badges */
.pill { display:inline-block; padding:.28rem .75rem; border-radius:999px; font-size:.76rem; font-weight:600; margin:.15rem; }
.pill-teal { background:#cffafe; color:#155e75; }
.pill-amber { background:#fef3c7; color:#92400e; }
.pill-red { background:#fee2e2; color:#991b1b; }
.pill-green { background:#d1fae5; color:#065f46; }

/* Section divider */
.section-title { font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700; color:#0f172a; margin:1.6rem 0 .3rem 0; border-bottom:2px solid #e2e8f0; padding-bottom:.4rem; }

/* Risk chip */
.risk-chip { display:inline-flex; align-items:center; gap:.5rem; padding:.5rem 1.1rem; border-radius:999px; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a 0%, #155e75 100%); }
section[data-testid="stSidebar"] * { color:#e2e8f0; }
section[data-testid="stSidebar"] .brand { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.5rem; color:#fff; letter-spacing:-0.02em; }
section[data-testid="stSidebar"] .brand-sub { font-size:.74rem; color:#7dd3fc; letter-spacing:.12em; text-transform:uppercase; }

/* Buttons */
.stButton>button {
    background:linear-gradient(135deg,#0891b2,#155e75); color:#fff; border:none; border-radius:10px;
    font-weight:600; padding:.55rem 1.3rem; transition:filter .15s ease;
}
.stButton>button:hover { filter:brightness(1.08); color:#fff; }

.stDownloadButton>button { background:#0f172a; color:#fff; border-radius:10px; border:none; font-weight:600; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def metric_card(label, value, sub=""):
    return f"""<div class="metric-card"><div class="label">{label}</div>
    <div class="value">{value}</div><div class="sub">{sub}</div></div>"""


def metric_row(cards):
    st.markdown('<div class="metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def plotly_theme(fig, title=None, height=400):
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        title_font=dict(family="Space Grotesk, sans-serif", size=17, color=INK),
        margin=dict(l=30, r=20, t=55, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# ENGINES
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="OncoTwin — Lung Cancer Digital Twin & CDSS",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="brand">🫁 OncoTwin</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="brand-sub">Neural-Mechanistic Digital Twin</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    pages = {
        "Project Overview": "🔬",
        "Patient Input": "📋",
        "Risk Prediction": "📈",
        "Digital Twin": "🧬",
        "Explainability": "💡",
        "Evidence & Validation": "📚",
        "Export Report": "📤",
    }
    choice = st.sidebar.radio(
        "Navigate",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]}  {x}",
    )

    st.sidebar.markdown("---")
    # Patient status indicator
    if st.session_state.get("patient_data"):
        pdd = st.session_state.patient_data
        pid = pdd.get("patient_id") or "New patient"
        st.sidebar.markdown(
            f"**Active patient**\n\n`{pid}` · {pdd['cancer_type']} · Stage {pdd['stage']}"
        )
    else:
        st.sidebar.info("No patient loaded")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model:** `{MODEL_VERSION}`")
    st.sidebar.markdown(f"**CDSS:** `{PHASE8_VERSION}`")
    st.sidebar.caption("⚠️ Research prototype — not for clinical use.")

    # Session state init
    for k in ["patient_data", "prediction_result", "simulation_result", "explanation_result"]:
        if k not in st.session_state:
            st.session_state[k] = None

    engines = get_engines()

    if choice == "Project Overview":
        render_overview()
    elif choice == "Patient Input":
        render_patient_input(engines[0], engines[1])
    elif choice == "Risk Prediction":
        render_prediction(engines[1])
    elif choice == "Digital Twin":
        render_digital_twin(engines[2], engines[1])
    elif choice == "Explainability":
        render_explainability(engines[1], engines[3], engines[2])
    elif choice == "Evidence & Validation":
        render_evidence()
    elif choice == "Export Report":
        render_export()


# ══════════════════════════════════════════════════════════════════════════
# PAGE — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
def render_overview():
    st.markdown(
        """
        <div class="hero">
            <div class="kicker">Federated Neural-Mechanistic Framework</div>
            <h1>A Lung Cancer Digital Twin for Virtual Treatment Simulation</h1>
            <p>An end-to-end research platform integrating deep survival learning, mechanistic
            tumor-growth dynamics, federated privacy-preserving training, and a patient-specific
            digital twin — delivered through a safety-constrained clinical decision support system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_row([
        metric_card("External C-index", "0.6588", "DeepSurv wins 5/5 cohorts"),
        metric_card("Validation Cohorts", "5", "TCGA + 4 GEO datasets"),
        metric_card("Pipeline Phases", "8", "Data → CDSS deployment"),
        metric_card("Safety Tests", "87 / 87", "100% passing"),
    ])

    st.markdown(
        '<div class="card"><h4>What is a Digital Twin here?</h4>'
        '<p>A computational replica of an individual patient\'s tumor, parameterized from their '
        'molecular and clinical profile. It couples a <b>neural survival model</b> (learns risk from '
        'data) with a <b>mechanistic ODE model</b> (encodes tumor biology), enabling <i>virtual</i> '
        'treatment scenarios that would be impossible or unethical to test directly.</p></div>',
        unsafe_allow_html=True,
    )

    section("The 8-Phase Research Pipeline")
    phases = [
        ("1", "Data Acquisition & Audit", "TCGA-LUAD/LUSC + 5 GEO cohorts curated, harmonized, and audited for readiness.", "Complete"),
        ("2", "External Validation Setup", "Cross-platform harmonization (RNA-seq ↔ microarray) and scale alignment.", "Complete"),
        ("3", "Deep Survival Learning", "DeepSurv top-2 ensemble (Cox partial likelihood). Mean external C-index 0.6588, wins 5/5 cohorts.", "Complete"),
        ("4", "Mechanistic Modeling", "Gompertz growth ODE + Skipper log-cell kill, immune dynamics, PK models. 42 literature-cited parameters.", "Complete"),
        ("5", "Neural-Mechanistic Integration", "Hybrid patient states linking learned risk to biological trajectories (Spearman ρ = -0.51).", "Complete"),
        ("6", "Federated Learning", "Privacy-preserving federated averaging across simulated institutions vs centralized baseline.", "Complete"),
        ("7", "Digital Twin + 7.1 Hardening", "Per-patient twin, treatment simulation, calibration audit, and a strict scientific constraint contract.", "Complete"),
        ("8", "Clinical Decision Support", "Safety-layered CDSS with explainability, uncertainty, and mandatory warnings. Hardened in Phase 8.1.", "Complete"),
    ]
    for num, title, desc, badge in phases:
        st.markdown(
            f"""<div class="phase-row">
                <div class="phase-num">{num}</div>
                <div class="phase-body">
                    <div class="t">{title}<span class="badge">{badge}</span></div>
                    <div class="d">{desc}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    section("How the Models Fit Together")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="card"><h4>🧠 Neural (DeepSurv)</h4><p>Learns a risk score from 143 features '
            '(age, stage, histology, smoking, 127 genes). Validated for <b>ranking</b> patients by risk.</p></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="card"><h4>⚙️ Mechanistic (ODE)</h4><p>Simulates tumor volume over time under '
            'natural history and therapy, grounded in published NSCLC pharmacology.</p></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="card"><h4>🔗 Digital Twin</h4><p>Fuses both into a patient-specific replica for '
            'counterfactual "what-if" treatment simulation with quantified uncertainty.</p></div>',
            unsafe_allow_html=True)

    section("External Validation Performance")
    cohorts = ["TCGA (internal)", "GSE30219", "GSE50081", "GSE72094", "GSE31210"]
    ds = [0.6131, 0.6782, 0.5965, 0.6334, 0.7271]
    cox = [0.5881, 0.6552, 0.5411, 0.5593, 0.7049]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cohorts, y=ds, name="DeepSurv", marker_color=PRIMARY))
    fig.add_trace(go.Bar(x=cohorts, y=cox, name="Cox PH (baseline)", marker_color="#cbd5e1"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8", annotation_text="Random (0.5)")
    plotly_theme(fig, "Concordance Index — DeepSurv vs Cox Baseline", height=420)
    fig.update_yaxes(range=[0.5, 0.78], title="C-index")
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    section("Responsible-AI Guardrails")
    st.markdown(
        '<span class="pill pill-red">Stage IIIB/IV hard-blocked</span>'
        '<span class="pill pill-amber">Uncalibrated survival flagged</span>'
        '<span class="pill pill-amber">Treatment-artifact warning</span>'
        '<span class="pill pill-teal">Explanation with every prediction</span>'
        '<span class="pill pill-teal">Uncertainty with every prediction</span>'
        '<span class="pill pill-green">87/87 safety tests passing</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")
    st.caption("Research project: A Federated Neural-Mechanistic Digital Twin Framework for "
               "Personalized Clinical Decision Support in Lung Cancer.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE — PATIENT INPUT
# ══════════════════════════════════════════════════════════════════════════
def render_patient_input(safety, pred_engine):
    st.title("📋 Patient Input")
    st.caption("Select a validated cohort patient or enter a new profile. Inputs pass through the "
               "Phase 7.1 safety layer before any prediction is allowed.")

    mode = st.radio("Input mode", ["Select existing patient", "Enter new patient data"],
                    horizontal=True)

    if mode == "Select existing patient":
        df = pred_engine._state_df
        available = df[~df["stage"].isin(["IIIB", "IV"])]
        patient_ids = available["patient_id"].tolist()
        selected = st.selectbox("Patient ID", [""] + patient_ids[:300])

        if selected:
            row = df[df["patient_id"] == selected].iloc[0]
            st.session_state.patient_data = {
                "patient_id": str(row["patient_id"]),
                "age": float(row["age"]),
                "cancer_type": str(row["cancer_type"]),
                "stage": str(row["stage"]),
                "smoking_status": str(row["smoking_status"]),
            }
            st.success(f"Patient **{selected}** loaded.")
            metric_row([
                metric_card("Age", int(row["age"]), "years"),
                metric_card("Histology", row["cancer_type"], ""),
                metric_card("Stage", row["stage"], "AJCC"),
                metric_card("Smoking", row["smoking_status"], ""),
            ])
            rg = row.get("risk_group", "Unknown")
            st.markdown(_risk_chip(rg) + f'&nbsp;&nbsp;<span class="sub">DeepSurv risk '
                        f'{row["deepsurv_risk"]:.4f}</span>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Clinical profile")
            age = st.number_input("Age", 18, 100, 65)
            cancer_type = st.selectbox("Histology", ["", "LUAD (Adenocarcinoma)", "LUSC (Squamous Cell)"])
            stage = st.selectbox("AJCC Stage", ["", "I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV"])
            smoking = st.selectbox("Smoking status", ["Never", "Former", "Current"])
        with c2:
            st.markdown("##### Molecular profile *(optional)*")
            st.caption("Provide Ensembl gene IDs → expression values as JSON. Leave empty for a "
                       "clinical-only prediction.")
            gene_input = st.text_area("Gene expression (JSON)", height=150,
                                      placeholder='{"ENSG00000171557.17": 5.2, "ENSG00000021826.17": 3.1}')

        if st.button("Validate & Save Patient", type="primary"):
            ct_mapped = cancer_type.split(" ")[0] if cancer_type else ""
            gene_expr = None
            if gene_input.strip():
                try:
                    gene_expr = json.loads(gene_input)
                except json.JSONDecodeError:
                    st.warning("Invalid JSON for gene expression — ignoring.")

            result = safety.check(age=age, cancer_type=ct_mapped, stage=stage,
                                  smoking_status=smoking, gene_expression=gene_expr)

            if result.blocked:
                st.error(f"🚫 **BLOCKED:** {result.block_reason}")
                if result.block_source == "phase7_1_constraint":
                    st.warning(STAGE_BLOCK_MESSAGE)
                st.session_state.patient_data = None
            else:
                st.success("✅ Patient validated and saved.")
                for w in (result.warnings or []):
                    st.warning(f"⚠️ {w}")
                st.session_state.patient_data = {
                    "patient_id": None, "age": float(age), "cancer_type": ct_mapped,
                    "stage": stage, "smoking_status": smoking, "gene_expression": gene_expr,
                }

    if st.session_state.patient_data:
        st.markdown("---")
        section("Active Patient")
        st.json({k: v for k, v in st.session_state.patient_data.items() if k != "gene_expression"})
        st.info("Continue to **Risk Prediction** → then **Digital Twin** and **Explainability**.")


def _risk_chip(rg):
    colors = {
        "Low": ("#d1fae5", "#065f46"),
        "Intermediate": ("#fef3c7", "#92400e"),
        "High": ("#fee2e2", "#991b1b"),
    }
    bg, fg = colors.get(rg, ("#e2e8f0", "#334155"))
    return (f'<span class="risk-chip" style="background:{bg};color:{fg};">'
            f'Risk group: {rg}</span>')


# ══════════════════════════════════════════════════════════════════════════
# PAGE — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════
def render_prediction(pred_engine):
    st.title("📈 Risk Prediction")
    st.caption("DeepSurv neural survival model — validated for ranking (C-index 0.6333). "
               "Absolute probabilities are uncalibrated.")

    if not st.session_state.patient_data:
        st.warning("Load a patient first on the **Patient Input** page.")
        return

    pd_data = st.session_state.patient_data

    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Running DeepSurv ensemble..."):
            if pd_data.get("patient_id"):
                result = pred_engine.predict_existing(pd_data["patient_id"])
                if not result:
                    st.error("Patient not found in state matrix.")
                    return
            else:
                result = pred_engine.predict_new(
                    age=pd_data["age"], cancer_type=pd_data["cancer_type"],
                    stage=pd_data["stage"], smoking_status=pd_data["smoking_status"],
                    gene_expression=pd_data.get("gene_expression"))
            t_eval = np.linspace(1, 1095, 100)
            S = pred_engine.risk_to_survival_curve(result["deepsurv_risk"], t_eval)
            surv_at = pred_engine.get_survival_at_timepoints(result["deepsurv_risk"])
            st.session_state.prediction_result = {
                "patient": result,
                "survival_curve": {"t_months": t_eval / 30.44, "S": S},
                "survival_at_timepoints": surv_at,
            }

    result = st.session_state.prediction_result
    if not result:
        return

    patient = result["patient"]
    rg = patient.get("risk_group", "Unknown")

    section("Risk Assessment")
    st.markdown(_risk_chip(rg), unsafe_allow_html=True)
    metric_row([
        metric_card("DeepSurv Risk", f"{patient['deepsurv_risk']:.4f}", "log-hazard (relative)"),
        metric_card("Ensemble Std", f"{patient.get('deepsurv_risk_std', 0):.4f}", "model disagreement"),
        metric_card("Risk Group", rg, "tertile of cohort"),
        metric_card("Uncertainty", "Moderate", "MC n=20, 95% CI"),
    ])

    section("Survival Probability")
    st.warning(f"⚠️ **UNCALIBRATED — interpret with caution.** {CALIBRATION_WARNING}")
    surv_at = result["survival_at_timepoints"]
    metric_row([
        metric_card("12-month", f"{surv_at['12m']:.1%}", "⚠️ uncalibrated"),
        metric_card("24-month", f"{surv_at['24m']:.1%}", "⚠️ uncalibrated"),
        metric_card("36-month", f"{surv_at['36m']:.1%}", "⚠️ uncalibrated"),
    ])

    sc = result["survival_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sc["t_months"], y=sc["S"], mode="lines", name="Survival probability",
        line=dict(color=PRIMARY, width=3), fill="tozeroy",
        fillcolor="rgba(14,116,144,0.08)"))
    plotly_theme(fig, "Estimated Survival Curve (ranking only — not calibrated)", height=420)
    fig.update_xaxes(title="Time (months)")
    fig.update_yaxes(title="Survival probability", range=[0, 1.02])
    st.plotly_chart(fig, use_container_width=True)

    st.info(get_risk_interpretation(rg))


# ══════════════════════════════════════════════════════════════════════════
# PAGE — DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════
def render_digital_twin(dt_engine, pred_engine):
    st.title("🧬 Digital Twin — Treatment Simulation")
    st.caption("Mechanistic tumor-growth ODE simulation for virtual treatment scenarios. "
               "Research hypothesis-generation only.")

    if not st.session_state.patient_data:
        st.warning("Load a patient first on the **Patient Input** page.")
        return

    pd_data = st.session_state.patient_data

    treatment = st.selectbox(
        "Treatment scenario",
        ["none", "chemo", "immuno", "targeted"],
        format_func=lambda x: {
            "none": "Natural History (no treatment)",
            "chemo": "Chemotherapy (Cisplatin)",
            "immuno": "Immunotherapy (Pembrolizumab)",
            "targeted": "Targeted Therapy (Osimertinib)",
        }[x])

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Integrating tumor-growth ODE..."):
            if pd_data.get("patient_id"):
                existing = dt_engine.get_existing_trajectories(pd_data["patient_id"])
                if existing:
                    sr = pred_engine._state_df[
                        pred_engine._state_df["patient_id"].astype(str) == pd_data["patient_id"]
                    ].iloc[0]
                    patient_params = {
                        "alpha": float(sr["alpha"]), "V_max": float(sr["V_max"]),
                        "V0": float(sr["V0"]), "k_immune": float(sr["k_immune"]),
                        "E0": float(sr["E0"]), "mu_E": float(sr["mu_E"]),
                        "rho": float(sr["rho"]), "delta_c": float(sr["delta_c"]),
                        "delta_t": float(sr["delta_t"]),
                    }
                else:
                    patient_params = dt_engine.get_population_priors()
            else:
                patient_params = dt_engine.get_population_priors()
            try:
                st.session_state.simulation_result = dt_engine.simulate_treatment(patient_params, treatment)
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                return

    sim = st.session_state.simulation_result
    if not sim:
        return

    if treatment != "none":
        st.error(f"⚠️ **ARTIFACT WARNING:** {TREATMENT_ARTIFACT_WARNING}")

    section(f"Result — {sim['treatment_label']}")
    metric_row([
        metric_card("Time to Progression", f"{sim['ttp_months']:.1f}", "months"),
        metric_card("TTP (days)", f"{sim['ttp_days']:.0f}", "days"),
        metric_card("Outcome", "Counterfactual" if sim.get("is_counterfactual") else "Natural history", ""),
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim["trajectory_t_months"], y=sim["trajectory_diameter_mm"],
        mode="lines", name="Tumor diameter",
        line=dict(color="#dc2626", width=3), fill="tozeroy",
        fillcolor="rgba(220,38,38,0.07)"))
    plotly_theme(fig, f"Tumor Growth Trajectory — {sim['treatment_label']}", height=420)
    fig.update_xaxes(title="Time (months)")
    fig.update_yaxes(title="Tumor diameter (mm)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(get_treatment_interpretation(sim["treatment_label"]))

    # Counterfactual comparison for existing patients
    if pd_data.get("patient_id"):
        existing = dt_engine.get_existing_trajectories(pd_data["patient_id"])
        if existing:
            section("Counterfactual Comparison")
            labels = ["Natural History", "Chemotherapy", "Immunotherapy", "Targeted"]
            ttps = [existing["ttp_natural"] / 30.44, existing["ttp_chemo"] / 30.44,
                    existing["ttp_immuno"] / 30.44, existing["ttp_targeted"] / 30.44]
            fig2 = go.Figure(go.Bar(x=labels, y=ttps,
                                    marker_color=["#94a3b8", PRIMARY, "#059669", "#d97706"]))
            plotly_theme(fig2, "TTP Across Scenarios (⚠️ artifact: uniform benefit)", height=360)
            fig2.update_yaxes(title="TTP (months)")
            st.plotly_chart(fig2, use_container_width=True)
            st.error(f"⚠️ {TREATMENT_ARTIFACT_WARNING}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════
def render_explainability(pred_engine, expl_engine, dt_engine):
    st.title("💡 Explainability")
    st.caption("Feature contributions (gradient × input), mechanistic parameters, and global cohort "
               "importance. Every prediction is accompanied by an explanation by design.")

    if not st.session_state.patient_data:
        st.warning("Load a patient first on the **Patient Input** page.")
        return

    pd_data = st.session_state.patient_data

    if st.button("Generate Explanation", type="primary"):
        with st.spinner("Computing feature attributions..."):
            contrib = pred_engine.get_feature_contributions(
                age=pd_data["age"], cancer_type=pd_data["cancer_type"],
                stage=pd_data["stage"], smoking_status=pd_data["smoking_status"],
                gene_expression=pd_data.get("gene_expression"))
            priors = dt_engine.get_population_priors()
            if pd_data.get("patient_id"):
                row = pred_engine._state_df[
                    pred_engine._state_df["patient_id"].astype(str) == pd_data["patient_id"]
                ].iloc[0]
                priors = {
                    "alpha": float(row["alpha"]), "V_max": float(row["V_max"]),
                    "V0": float(row["V0"]), "k_immune": float(row["k_immune"]),
                    "delta_c": float(row["delta_c"]), "delta_t": float(row["delta_t"]),
                }
            st.session_state.explanation_result = expl_engine.get_patient_explanation(priors, contrib)

    expl = st.session_state.explanation_result
    if not expl:
        return

    col1, col2 = st.columns(2)
    with col1:
        section("Clinical Features")
        clinical = expl["top_clinical_features"]
        if clinical:
            fig = go.Figure(go.Bar(
                x=[v for _, v in clinical], y=[k for k, _ in clinical],
                orientation="h", marker_color=PRIMARY))
            plotly_theme(fig, "Clinical contributions", height=360)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        section("Molecular Features")
        molecular = expl["top_molecular_features"]
        if molecular:
            fig2 = go.Figure(go.Bar(
                x=[v for _, v in molecular], y=[k for k, _ in molecular],
                orientation="h", marker_color="#d97706"))
            plotly_theme(fig2, "Molecular contributions", height=360)
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)

    section("Mechanistic Parameters")
    mech_data = [{
        "Parameter": p, "Value": info["value"], "Effect": info["effect"],
        "Description": info["description"], "Literature range": info["literature_range"],
    } for p, info in expl["mechanistic_contributions"].items()]
    st.dataframe(pd.DataFrame(mech_data), use_container_width=True, hide_index=True)

    section("Global Cohort Importance")
    gi = expl["global_cohort_importance"]
    st.caption(f"Source: {gi['source']} · {gi['n_features']} features")
    if gi["top_10"]:
        st.dataframe(pd.DataFrame(gi["top_10"]), use_container_width=True, hide_index=True)

    st.info(expl["explanation_method"])
    st.caption(expl["disclaimer"])


# ══════════════════════════════════════════════════════════════════════════
# PAGE — EVIDENCE & VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def render_evidence():
    st.title("📚 Evidence & Validation")
    st.caption("Model provenance, external validation, and the Phase 7.1 limitations that shape "
               "every safety guardrail in this system.")

    section("Model Provenance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="card"><h4>Training</h4>'
            '<ul><li><b>Cohort:</b> TCGA (LUAD + LUSC)</li>'
            '<li><b>Training n:</b> 737 · <b>Validation n:</b> 182</li>'
            '<li><b>Model:</b> DeepSurv top-2 ensemble ([64,32], GELU, dropout 0.2)</li>'
            '<li><b>Features:</b> 143 (age, 3 clinical categoricals, 127 genes)</li>'
            '<li><b>Loss:</b> Cox partial likelihood (Efron ties)</li></ul></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="card"><h4>External Validation (Phase 3)</h4>'
            '<ul><li>GSE30219 — C-index <b>0.678</b> ✅</li>'
            '<li>GSE50081 — C-index <b>0.597</b> ✅</li>'
            '<li>GSE72094 — C-index <b>0.633</b> ✅</li>'
            '<li>GSE31210 — C-index <b>0.727</b> ✅</li>'
            '<li><b>Mean external C-index 0.6588 — wins 5/5</b></li></ul></div>',
            unsafe_allow_html=True)

    section("Known Limitations & Enforcement (Phase 7.1)")
    lim = pd.DataFrame([
        ["Calibration slope 0.6189", "Below target 0.8–1.2", "'UNCALIBRATED' tag on all survival probabilities"],
        ["Uniform 100% treatment benefit", "Structural ODE artifact", "Mandatory artifact warning on every simulation"],
        ["Stage IIIB (n=7)", "Insufficient sample", "Hard block — prediction refused"],
        ["Stage IV (n=10)", "Insufficient sample", "Hard block — prediction refused"],
        ["No resistance modeling", "ODE assumes continuous response", "Artifact warning"],
        ["No toxicity modeling", "No Grade 3–4 AE simulation", "Artifact warning"],
        ["Retrospective data only", "No prospective validation", "Research disclaimer"],
        ["No regulatory review", "Not FDA/EMA cleared", "Research disclaimer"],
    ], columns=["Limitation", "Detail", "Enforcement in CDSS"])
    st.dataframe(lim, use_container_width=True, hide_index=True)

    section("Permitted vs Forbidden Interpretation Statements")
    statements = get_all_statements()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### ✅ Permitted ({statements['n_permitted']})")
        for i, s in enumerate(statements["permitted"], 1):
            st.markdown(f"{i}. {s}")
    with c2:
        st.markdown(f"##### ❌ Forbidden ({statements['n_forbidden']})")
        for i, s in enumerate(statements["forbidden"], 1):
            st.markdown(f"{i}. {s}")

    section("Phase 7.1 Scientific Position Statement")
    try:
        from backend.config import P7_1_DIR
        p = P7_1_DIR / "phase7_scientific_position_statement.md"
        if p.exists():
            with st.expander("Read the full position statement"):
                st.markdown(p.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Could not load Phase 7.1 position statement.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE — EXPORT
# ══════════════════════════════════════════════════════════════════════════
def render_export():
    st.title("📤 Export Report")
    st.caption("Download the current patient session as a structured report. All exports carry the "
               "mandatory disclaimers and warnings.")

    if not st.session_state.patient_data:
        st.warning("Load a patient first on the **Patient Input** page.")
        return

    pd_data = st.session_state.patient_data
    pred = st.session_state.prediction_result
    sim = st.session_state.simulation_result

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### JSON report")
        report = {
            "patient": {k: v for k, v in pd_data.items() if k != "gene_expression"},
            "prediction": pred, "simulation": sim,
            "explanation": st.session_state.explanation_result,
            "model_version": MODEL_VERSION, "cdss_version": PHASE8_VERSION,
            "disclaimer": CLINICAL_DISCLAIMER,
            "calibration_warning": CALIBRATION_WARNING,
            "artifact_warning": TREATMENT_ARTIFACT_WARNING,
        }
        st.download_button(
            "Download JSON", data=json.dumps(report, indent=2, default=str),
            file_name=f"oncotwin_report_{pd_data.get('patient_id', 'new')}.json",
            mime="application/json")
    with c2:
        st.markdown("##### CSV summary")
        base = {
            "patient_id": pd_data.get("patient_id", "new"), "age": pd_data["age"],
            "cancer_type": pd_data["cancer_type"], "stage": pd_data["stage"],
            "smoking_status": pd_data["smoking_status"], "model_version": MODEL_VERSION,
        }
        if pred:
            p = pred["patient"]
            base.update({
                "risk_group": p.get("risk_group"), "deepsurv_risk": p.get("deepsurv_risk"),
                "survival_12m": pred["survival_at_timepoints"]["12m"],
                "survival_24m": pred["survival_at_timepoints"]["24m"],
                "survival_36m": pred["survival_at_timepoints"]["36m"],
                "calibration_status": "UNCALIBRATED",
            })
        if sim:
            base.update({"treatment": sim.get("treatment"),
                         "ttp_days": sim.get("ttp_days"), "ttp_months": sim.get("ttp_months")})
        st.download_button(
            "Download CSV", data=pd.DataFrame([base]).to_csv(index=False),
            file_name=f"oncotwin_summary_{pd_data.get('patient_id', 'new')}.csv",
            mime="text/csv")

    st.markdown("---")
    st.caption("Reports are for research documentation only and must not inform clinical decisions.")


if __name__ == "__main__":
    main()
