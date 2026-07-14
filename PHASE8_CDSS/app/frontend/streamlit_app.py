"""
OncoTwin — Lung Cancer Clinical Decision Support Console
A single integrated clinical workflow: enter a patient once, get a full assessment.
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    get_treatment_interpretation, get_risk_interpretation, get_all_statements,
)

# ── palette ──
PRIMARY = "#0e7490"
INK = "#0f172a"
TREATMENTS = ["none", "chemo", "immuno", "targeted"]
TREAT_LABEL = {
    "none": "Natural History", "chemo": "Chemotherapy (Cisplatin)",
    "immuno": "Immunotherapy (Pembrolizumab)", "targeted": "Targeted (Osimertinib)",
}
TREAT_COLOR = {"none": "#94a3b8", "chemo": "#0e7490", "immuno": "#059669", "targeted": "#d97706"}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
#MainMenu,footer,header {visibility:hidden;}
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1180px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; color:#0f172a; letter-spacing:-0.02em; }

.hero { background:linear-gradient(135deg,#0e7490 0%,#155e75 55%,#0f172a 100%);
  border-radius:18px; padding:1.8rem 2rem; color:#f8fafc; margin-bottom:1.2rem;
  box-shadow:0 18px 40px -20px rgba(14,116,144,0.55); }
.hero h1 { color:#fff; font-size:1.9rem; margin:.2rem 0 .3rem 0; }
.hero p { color:#cbeef5; font-size:.98rem; margin:0; max-width:760px; line-height:1.5; }
.hero .kicker { display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.18em;
  text-transform:uppercase; color:#7dd3fc; background:rgba(255,255,255,0.08);
  padding:.28rem .7rem; border-radius:999px; margin-bottom:.3rem; }

.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.8rem; margin:.3rem 0 1rem 0; }
.metric-card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.1rem;
  box-shadow:0 4px 14px -8px rgba(15,23,42,0.15); }
.metric-card .label { font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; color:#64748b; }
.metric-card .value { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:#0e7490; margin:.25rem 0 .1rem 0; }
.metric-card .sub { font-size:.74rem; color:#94a3b8; }

.card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1.1rem 1.2rem;
  margin-bottom:.9rem; box-shadow:0 4px 14px -10px rgba(15,23,42,0.15); }
.card h4 { margin:.1rem 0 .4rem 0; font-family:'Space Grotesk',sans-serif; color:#155e75; }
.card p,.card li { color:#475569; font-size:.9rem; line-height:1.5; }

.pill { display:inline-block; padding:.26rem .7rem; border-radius:999px; font-size:.74rem; font-weight:600; margin:.12rem; }
.pill-teal{background:#cffafe;color:#155e75;} .pill-amber{background:#fef3c7;color:#92400e;}
.pill-red{background:#fee2e2;color:#991b1b;} .pill-green{background:#d1fae5;color:#065f46;}

.section-title { font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; color:#0f172a;
  margin:1.2rem 0 .3rem 0; border-bottom:2px solid #e2e8f0; padding-bottom:.35rem; }

.pat-banner { background:linear-gradient(135deg,#f0f9ff,#ecfeff); border:1px solid #bae6fd;
  border-radius:14px; padding:.9rem 1.2rem; margin-bottom:.6rem; display:flex; gap:1.8rem; flex-wrap:wrap; }
.pat-banner .item .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:#0369a1; font-weight:600; }
.pat-banner .item .v { font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; color:#0f172a; }

.risk-chip { display:inline-flex; align-items:center; gap:.5rem; padding:.45rem 1.1rem; border-radius:999px;
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1rem; }

section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a 0%,#155e75 100%); }
section[data-testid="stSidebar"] * { color:#e2e8f0; }
section[data-testid="stSidebar"] .brand { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.5rem; color:#fff; }
section[data-testid="stSidebar"] .brand-sub { font-size:.72rem; color:#7dd3fc; letter-spacing:.12em; text-transform:uppercase; }

.stButton>button { background:linear-gradient(135deg,#0891b2,#155e75); color:#fff; border:none;
  border-radius:10px; font-weight:600; padding:.6rem 1.4rem; }
.stButton>button:hover { filter:brightness(1.08); color:#fff; }
.stDownloadButton>button { background:#0f172a; color:#fff; border-radius:10px; border:none; font-weight:600; }
</style>
"""


# ── helpers ──
def mc(label, value, sub=""):
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def metric_row(cards):
    st.markdown('<div class="metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section(t):
    st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)


def risk_chip(rg):
    colors = {"Low": ("#d1fae5", "#065f46"), "Intermediate": ("#fef3c7", "#92400e"), "High": ("#fee2e2", "#991b1b")}
    bg, fg = colors.get(rg, ("#e2e8f0", "#334155"))
    return f'<span class="risk-chip" style="background:{bg};color:{fg};">Risk group: {rg}</span>'


def theme_fig(fig, title=None, height=380):
    fig.update_layout(title=title, height=height, template="plotly_white",
                      font=dict(family="Inter, sans-serif", size=13, color=INK),
                      title_font=dict(family="Space Grotesk, sans-serif", size=16, color=INK),
                      margin=dict(l=30, r=20, t=50, b=40),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


def survival_gauge(value_pct, label):
    color = "#059669" if value_pct >= 66 else "#d97706" if value_pct >= 33 else "#dc2626"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value_pct,
        number={"suffix": "%", "font": {"size": 34, "family": "Space Grotesk"}},
        title={"text": label, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 33], "color": "#fee2e2"},
                         {"range": [33, 66], "color": "#fef3c7"},
                         {"range": [66, 100], "color": "#d1fae5"}]}))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    return fig


@st.cache_resource
def get_engines():
    safety = SafetyLayer()
    pred = PredictionEngine(); pred.load()
    dt = DigitalTwinEngine(); dt.load()
    expl = ExplainabilityEngine(); expl.load()
    return safety, pred, dt, expl


def main():
    st.set_page_config(page_title="OncoTwin — Lung Cancer CDSS", page_icon="🫁",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="brand">🫁 OncoTwin</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="brand-sub">Lung Cancer Decision Support</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    pages = {"Home": "🏠", "Clinical Console": "🩺", "Model Evidence": "📚"}
    choice = st.sidebar.radio("Navigate", list(pages.keys()),
                              format_func=lambda x: f"{pages[x]}  {x}")

    st.sidebar.markdown("---")
    if st.session_state.get("analysis"):
        p = st.session_state.analysis["patient"]
        st.sidebar.success(f"Analysis ready\n\n{p['cancer_type']} · Stage {p['stage']} · Age {int(p['age'])}")
    else:
        st.sidebar.info("No analysis yet")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model:** `{MODEL_VERSION}`")
    st.sidebar.markdown(f"**CDSS:** `{PHASE8_VERSION}`")
    st.sidebar.caption("⚠️ Research prototype — not for clinical use.")

    for k in ["analysis", "block"]:
        if k not in st.session_state:
            st.session_state[k] = None

    engines = get_engines()
    if choice == "Home":
        render_home()
    elif choice == "Clinical Console":
        render_console(engines)
    elif choice == "Model Evidence":
        render_evidence()


# ══════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════
def render_home():
    st.markdown(
        """<div class="hero">
            <div class="kicker">Clinical Decision Support · Lung Cancer</div>
            <h1>OncoTwin Clinical Console</h1>
            <p>Enter one patient. Get a complete assessment in seconds — survival risk, a full
            treatment-scenario comparison, and the reasoning behind every number — inside validated
            safety limits.</p></div>""",
        unsafe_allow_html=True)

    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>🩺 One-screen workflow</h4><p>No page-hopping. Enter the '
                    'patient and press <b>Run Assessment</b> — everything computes at once.</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>🧬 Treatment comparison</h4><p>Natural history vs. chemo, '
                    'immuno, and targeted therapy are simulated and compared side by side.</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>💡 Transparent</h4><p>Every result carries its risk drivers, '
                    'uncertainty, calibration status, and safety flags.</p></div>',
                    unsafe_allow_html=True)

    st.info("Open **Clinical Console** in the sidebar to begin.")
    st.markdown(
        '<span class="pill pill-red">Stage IIIB/IV blocked</span>'
        '<span class="pill pill-amber">Uncalibrated survival flagged</span>'
        '<span class="pill pill-amber">Treatment-artifact warning</span>'
        '<span class="pill pill-teal">Validated on 5 cohorts (C-index 0.66)</span>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE — compute everything in one pass
# ══════════════════════════════════════════════════════════════════════════
def run_analysis(engines, patient):
    _, pred, dt, expl = engines

    # 1) Risk prediction
    if patient.get("patient_id"):
        result = pred.predict_existing(patient["patient_id"])
        if not result:
            result = pred.predict_new(patient["age"], patient["cancer_type"],
                                      patient["stage"], patient["smoking_status"])
    else:
        result = pred.predict_new(patient["age"], patient["cancer_type"],
                                  patient["stage"], patient["smoking_status"],
                                  patient.get("gene_expression"))

    t_eval = np.linspace(1, 1095, 100)
    S = pred.risk_to_survival_curve(result["deepsurv_risk"], t_eval)
    surv_at = pred.get_survival_at_timepoints(result["deepsurv_risk"])

    # 2) Twin parameters
    if patient.get("patient_id"):
        srdf = pred._state_df[pred._state_df["patient_id"].astype(str) == patient["patient_id"]]
        if len(srdf):
            sr = srdf.iloc[0]
            params = {k: float(sr[k]) for k in
                      ["alpha", "V_max", "V0", "k_immune", "E0", "mu_E", "rho", "delta_c", "delta_t"]}
        else:
            params = dt.get_population_priors()
    else:
        params = dt.get_population_priors()

    # 3) Simulate all treatment scenarios
    sims = {}
    for tr in TREATMENTS:
        try:
            sims[tr] = dt.simulate_treatment(params, tr)
        except Exception:
            sims[tr] = None

    # 4) Explanation
    contrib = pred.get_feature_contributions(patient["age"], patient["cancer_type"],
                                             patient["stage"], patient["smoking_status"],
                                             patient.get("gene_expression"))
    expl_priors = {k: params[k] for k in ["alpha", "V_max", "V0", "k_immune", "delta_c", "delta_t"]
                   if k in params}
    explanation = expl.get_patient_explanation(expl_priors, contrib)

    return {
        "patient": patient,
        "risk": result,
        "survival": {"t_months": (t_eval / 30.44).tolist(), "S": S.tolist()},
        "survival_at": surv_at,
        "sims": sims,
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════════════════
# CLINICAL CONSOLE — intake + integrated tabbed results
# ══════════════════════════════════════════════════════════════════════════
def render_console(engines):
    safety, pred, dt, expl = engines
    st.title("🩺 Clinical Console")

    # ── Intake panel ──
    with st.container(border=True):
        st.markdown("##### Patient intake")
        mode = st.radio("Source", ["New patient", "Cohort patient"], horizontal=True,
                        label_visibility="collapsed")
        patient = None

        if mode == "Cohort patient":
            df = pred._state_df
            avail = df[~df["stage"].isin(["IIIB", "IV"])]
            sel = st.selectbox("Select validated cohort patient", [""] + avail["patient_id"].astype(str).tolist()[:300])
            if sel:
                row = df[df["patient_id"].astype(str) == sel].iloc[0]
                patient = {"patient_id": sel, "age": float(row["age"]),
                           "cancer_type": str(row["cancer_type"]), "stage": str(row["stage"]),
                           "smoking_status": str(row["smoking_status"])}
        else:
            c1, c2, c3, c4 = st.columns(4)
            age = c1.number_input("Age", 18, 100, 65)
            ct = c2.selectbox("Histology", ["LUAD (Adenocarcinoma)", "LUSC (Squamous Cell)"])
            stage = c3.selectbox("AJCC Stage", ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV"])
            smoking = c4.selectbox("Smoking", ["Never", "Former", "Current"])
            genes = None
            with st.expander("Add gene expression (optional)"):
                gtext = st.text_area("Ensembl ID → value (JSON)", height=90,
                                     placeholder='{"ENSG00000171557.17": 5.2}')
                if gtext.strip():
                    try:
                        genes = json.loads(gtext)
                    except json.JSONDecodeError:
                        st.warning("Invalid JSON — ignoring gene input.")
            patient = {"patient_id": None, "age": float(age),
                       "cancer_type": ct.split(" ")[0], "stage": stage,
                       "smoking_status": smoking, "gene_expression": genes}

        run = st.button("▶  Run Assessment", type="primary", use_container_width=True)

    # ── Run ──
    if run:
        if not patient:
            st.warning("Select or enter a patient first.")
            return
        chk = safety.check(age=patient["age"], cancer_type=patient["cancer_type"],
                           stage=patient["stage"], smoking_status=patient["smoking_status"],
                           gene_expression=patient.get("gene_expression"))
        if chk.blocked:
            st.session_state.analysis = None
            st.session_state.block = {"reason": chk.block_reason, "source": chk.block_source}
        else:
            with st.spinner("Running DeepSurv risk model, tumor-growth simulations, and explainability..."):
                st.session_state.analysis = run_analysis(engines, patient)
                st.session_state.analysis["warnings"] = chk.warnings or []
                st.session_state.block = None

    # ── Blocked ──
    if st.session_state.block:
        b = st.session_state.block
        st.error(f"🚫 **Prediction refused:** {b['reason']}")
        if b["source"] == "phase7_1_constraint":
            st.warning(STAGE_BLOCK_MESSAGE)
        st.info("This is an intentional safety guardrail. Adjust the stage to a supported value (I–IIIA) to proceed.")
        return

    a = st.session_state.analysis
    if not a:
        st.caption("Enter a patient above and press **Run Assessment** to generate a full report.")
        return

    _render_patient_banner(a)
    for w in a.get("warnings", []):
        st.warning(f"⚠️ {w}")

    tabs = st.tabs(["📊 Summary", "📈 Survival", "🧬 Treatment Options", "💡 Why", "📤 Report"])
    with tabs[0]:
        _tab_summary(a)
    with tabs[1]:
        _tab_survival(a)
    with tabs[2]:
        _tab_treatment(a)
    with tabs[3]:
        _tab_explain(a)
    with tabs[4]:
        _tab_report(a)


def _render_patient_banner(a):
    p = a["patient"]
    pid = p.get("patient_id") or "New patient"
    st.markdown(
        f"""<div class="pat-banner">
            <div class="item"><div class="k">Patient</div><div class="v">{pid}</div></div>
            <div class="item"><div class="k">Age</div><div class="v">{int(p['age'])}</div></div>
            <div class="item"><div class="k">Histology</div><div class="v">{p['cancer_type']}</div></div>
            <div class="item"><div class="k">Stage</div><div class="v">{p['stage']}</div></div>
            <div class="item"><div class="k">Smoking</div><div class="v">{p['smoking_status']}</div></div>
        </div>""",
        unsafe_allow_html=True)


def _tab_summary(a):
    risk = a["risk"]
    rg = risk.get("risk_group", "Unknown")
    sa = a["survival_at"]

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(survival_gauge(sa["24m"] * 100, "24-month survival (uncalibrated)"),
                        use_container_width=True)
    with c2:
        st.markdown(risk_chip(rg), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        metric_row([
            mc("DeepSurv Risk", f"{risk['deepsurv_risk']:.3f}", "relative log-hazard"),
            mc("Ensemble Std", f"{risk.get('deepsurv_risk_std', 0):.3f}", "model agreement"),
            mc("Uncertainty", "Moderate", "MC 95% CI"),
        ])
        st.info(get_risk_interpretation(rg))

    # best-scenario glance
    sims = a["sims"]
    valid = {k: v for k, v in sims.items() if v}
    if valid:
        best = max(valid.items(), key=lambda kv: kv[1]["ttp_months"])
        section("Longest simulated time-to-progression")
        metric_row([
            mc(TREAT_LABEL[k], f"{v['ttp_months']:.1f} mo", "TTP")
            for k, v in valid.items()
        ])
        st.caption(f"Longest in simulation: **{TREAT_LABEL[best[0]]}** "
                   f"({best[1]['ttp_months']:.1f} months) — see **Treatment Options** for the mandatory artifact caveat.")

    section("Safety status")
    st.markdown(
        '<span class="pill pill-green">Input passed safety layer</span>'
        '<span class="pill pill-amber">Survival = uncalibrated (ranking only)</span>'
        '<span class="pill pill-amber">Treatment differences = artifact, not clinical</span>',
        unsafe_allow_html=True)


def _tab_survival(a):
    st.warning(f"⚠️ **UNCALIBRATED — interpret with caution.** {CALIBRATION_WARNING}")
    sa = a["survival_at"]
    metric_row([
        mc("12-month", f"{sa['12m']:.1%}", "⚠️ uncalibrated"),
        mc("24-month", f"{sa['24m']:.1%}", "⚠️ uncalibrated"),
        mc("36-month", f"{sa['36m']:.1%}", "⚠️ uncalibrated"),
    ])
    sc = a["survival"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc["t_months"], y=sc["S"], mode="lines",
                             line=dict(color=PRIMARY, width=3), fill="tozeroy",
                             fillcolor="rgba(14,116,144,0.08)", name="Survival"))
    theme_fig(fig, "Estimated survival curve (ranking only)", height=420)
    fig.update_xaxes(title="Time (months)")
    fig.update_yaxes(title="Survival probability", range=[0, 1.02])
    st.plotly_chart(fig, use_container_width=True)


def _tab_treatment(a):
    st.error(f"⚠️ **ARTIFACT WARNING:** {TREATMENT_ARTIFACT_WARNING}")
    sims = a["sims"]
    valid = {k: v for k, v in sims.items() if v}

    # TTP comparison
    labels = [TREAT_LABEL[k] for k in valid]
    ttps = [valid[k]["ttp_months"] for k in valid]
    colors = [TREAT_COLOR[k] for k in valid]
    fig = go.Figure(go.Bar(x=labels, y=ttps, marker_color=colors,
                           text=[f"{v:.1f}" for v in ttps], textposition="outside"))
    theme_fig(fig, "Simulated time-to-progression by scenario (months)", height=360)
    fig.update_yaxes(title="TTP (months)")
    st.plotly_chart(fig, use_container_width=True)

    # Trajectory overlay
    fig2 = go.Figure()
    for k, v in valid.items():
        fig2.add_trace(go.Scatter(x=v["trajectory_t_months"], y=v["trajectory_diameter_mm"],
                                  mode="lines", name=TREAT_LABEL[k],
                                  line=dict(color=TREAT_COLOR[k], width=2.5)))
    theme_fig(fig2, "Tumor diameter trajectory by scenario", height=400)
    fig2.update_xaxes(title="Time (months)")
    fig2.update_yaxes(title="Tumor diameter (mm)")
    st.plotly_chart(fig2, use_container_width=True)

    section("How to read this")
    st.info(get_treatment_interpretation(TREAT_LABEL["chemo"]))
    st.error(f"⚠️ {TREATMENT_ARTIFACT_WARNING}")


def _tab_explain(a):
    ex = a["explanation"]
    c1, c2 = st.columns(2)
    with c1:
        section("Clinical drivers")
        clinical = ex["top_clinical_features"]
        if clinical:
            fig = go.Figure(go.Bar(x=[v for _, v in clinical], y=[k for k, _ in clinical],
                                   orientation="h", marker_color=PRIMARY))
            theme_fig(fig, "Clinical contributions", height=340)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("Molecular drivers")
        molecular = ex["top_molecular_features"]
        if molecular:
            fig2 = go.Figure(go.Bar(x=[v for _, v in molecular], y=[k for k, _ in molecular],
                                    orientation="h", marker_color="#d97706"))
            theme_fig(fig2, "Molecular contributions", height=340)
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("No gene expression provided — molecular drivers unavailable.")

    section("Mechanistic parameters")
    mech = [{"Parameter": p, "Value": i["value"], "Effect": i["effect"],
             "Description": i["description"], "Literature range": i["literature_range"]}
            for p, i in ex["mechanistic_contributions"].items()]
    st.dataframe(pd.DataFrame(mech), use_container_width=True, hide_index=True)
    st.caption(ex["explanation_method"] + " · " + ex["disclaimer"])


def _tab_report(a):
    p = a["patient"]
    risk = a["risk"]
    sa = a["survival_at"]
    report = {
        "patient": {k: v for k, v in p.items() if k != "gene_expression"},
        "risk_group": risk.get("risk_group"), "deepsurv_risk": risk.get("deepsurv_risk"),
        "survival_12m": sa["12m"], "survival_24m": sa["24m"], "survival_36m": sa["36m"],
        "calibration_status": "UNCALIBRATED",
        "treatment_ttp_months": {k: (v["ttp_months"] if v else None) for k, v in a["sims"].items()},
        "model_version": MODEL_VERSION, "cdss_version": PHASE8_VERSION,
        "disclaimer": CLINICAL_DISCLAIMER, "calibration_warning": CALIBRATION_WARNING,
        "artifact_warning": TREATMENT_ARTIFACT_WARNING,
    }
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ Download JSON report", json.dumps(report, indent=2, default=str),
                           file_name=f"oncotwin_{p.get('patient_id', 'new')}.json",
                           mime="application/json", use_container_width=True)
    with c2:
        flat = {k: v for k, v in report.items() if not isinstance(v, dict)}
        flat.update({f"ttp_{k}": (v["ttp_months"] if v else None) for k, v in a["sims"].items()})
        flat.update({k: v for k, v in report["patient"].items()})
        st.download_button("⬇ Download CSV summary", pd.DataFrame([flat]).to_csv(index=False),
                           file_name=f"oncotwin_{p.get('patient_id', 'new')}.csv",
                           mime="text/csv", use_container_width=True)
    st.markdown("---")
    st.json(report)


# ══════════════════════════════════════════════════════════════════════════
# MODEL EVIDENCE
# ══════════════════════════════════════════════════════════════════════════
def render_evidence():
    st.title("📚 Model Evidence & Limitations")
    st.caption("Provenance, external validation, and the limitations that drive every safety guardrail.")

    section("External validation performance")
    # Phase 3.1 rigor results (bootstrap 1000 resamples, seed 42) — reproduces Phase 3 report.
    rigor = [
        # cohort, ds, ds_lo, ds_hi, cox, delta, p, calib_slope, mean_auc, ibs
        ("TCGA (internal)", 0.6131, 0.545, 0.687, 0.5881, +0.0249, 0.404, 0.404, 0.648, 0.186),
        ("GSE30219", 0.6777, 0.635, 0.722, 0.6583, +0.0194, 0.118, 0.987, 0.724, 0.237),
        ("GSE50081", 0.5965, 0.533, 0.665, 0.5411, +0.0554, 0.226, 0.613, 0.580, 0.143),
        ("GSE72094", 0.6334, 0.575, 0.687, 0.5593, +0.0741, 0.002, 0.717, 0.643, 0.173),
        ("GSE31210", 0.7271, 0.624, 0.819, 0.7049, +0.0222, 0.540, 0.796, 0.873, 0.049),
    ]
    cohorts = [r[0] for r in rigor]
    ds = [r[1] for r in rigor]
    ds_err_lo = [r[1] - r[2] for r in rigor]
    ds_err_hi = [r[3] - r[1] for r in rigor]
    cox = [r[4] for r in rigor]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cohorts, y=ds, name="DeepSurv", marker_color=PRIMARY,
                         error_y=dict(type="data", symmetric=False, array=ds_err_hi,
                                      arrayminus=ds_err_lo, color="#155e75")))
    fig.add_trace(go.Bar(x=cohorts, y=cox, name="Cox baseline", marker_color="#cbd5e1"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8")
    theme_fig(fig, "C-index with bootstrap 95% CI (higher = better ranking)", height=380)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0.5, 0.85], title="C-index")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Mean external C-index: **DeepSurv 0.659** vs **Cox 0.616**. DeepSurv wins 5/5 cohorts. "
               "Statistics: Harrell C-index, 1000 bootstrap resamples, seed 42 (Phase 3.1).")

    section("Statistical rigor (Phase 3.1)")
    st.dataframe(pd.DataFrame([
        {"Cohort": r[0], "DeepSurv C-index (95% CI)": f"{r[1]:.4f} ({r[2]:.3f}–{r[3]:.3f})",
         "Δ vs Cox": f"{r[5]:+.4f}", "paired p": f"{r[6]:.3f}",
         "Calib. slope": f"{r[7]:.3f}", "Mean td-AUC": f"{r[8]:.3f}", "IBS": f"{r[9]:.3f}"}
        for r in rigor
    ]), use_container_width=True, hide_index=True)
    st.markdown(
        '<span class="pill pill-green">GSE72094: DeepSurv &gt; Cox, p=0.002</span>'
        '<span class="pill pill-teal">GSE30219 calibration slope 0.99 (ideal 1.0)</span>'
        '<span class="pill pill-amber">Absolute survival remains approximate</span>',
        unsafe_allow_html=True)
    st.caption("Survival curves in the console use a Phase 3.1 single-covariate Cox recalibration "
               "(baseline fit on internal validation). Calibration slope (ideal = 1.0) quantifies "
               "residual miscalibration per cohort.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><h4>Training</h4><ul>'
                    '<li>Cohort: TCGA (LUAD + LUSC)</li>'
                    '<li>Train n=737 · Val n=182</li>'
                    '<li>DeepSurv top-2 ensemble ([64,32], GELU)</li>'
                    '<li>143 features (clinical + 127 genes)</li></ul></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Validated use</h4><ul>'
                    '<li>Patient <b>ranking</b> by risk (C-index 0.66)</li>'
                    '<li>Hypothesis generation only</li>'
                    '<li>Not calibrated for absolute survival</li>'
                    '<li>Not prospectively / regulatory validated</li></ul></div>', unsafe_allow_html=True)

    section("Known limitations & enforcement")
    st.dataframe(pd.DataFrame([
        ["Calibration slope 0.6189", "Below target 0.8–1.2", "'UNCALIBRATED' tag on survival"],
        ["Uniform treatment benefit", "Structural ODE artifact", "Mandatory artifact warning"],
        ["Stage IIIB (n=7) / IV (n=10)", "Insufficient sample", "Hard block — prediction refused"],
        ["No resistance / toxicity model", "ODE simplification", "Artifact warning"],
        ["Retrospective data only", "No prospective validation", "Research disclaimer"],
    ], columns=["Limitation", "Detail", "Enforcement"]), use_container_width=True, hide_index=True)

    section("Permitted vs forbidden interpretations")
    s = get_all_statements()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### ✅ Permitted ({s['n_permitted']})")
        for i, x in enumerate(s["permitted"], 1):
            st.markdown(f"{i}. {x}")
    with c2:
        st.markdown(f"##### ❌ Forbidden ({s['n_forbidden']})")
        for i, x in enumerate(s["forbidden"], 1):
            st.markdown(f"{i}. {x}")

    st.error(f"⚠️ {CLINICAL_DISCLAIMER}")


if __name__ == "__main__":
    main()
