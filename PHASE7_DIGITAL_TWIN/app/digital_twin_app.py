"""
Digital Twin Interface — Streamlit Research Prototype

Framework: Streamlit
Fidelity: Live backend calling the Digital Twin simulation engine
Out of scope: Real-time patient monitoring, clinical decision support,
              regulatory-approved medical device functionality.

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
This system generates patient-specific computational simulations that support
hypothesis generation and future clinical decision-support research. It does
NOT determine the best treatment for individual patients.

Usage:
    streamlit run digital_twin_app.py
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

import streamlit as st

# ═════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
P3_MODELS = PROJECT_ROOT / "PHASE3_DEEP_LEARNING" / "models"
P4_DATA = PROJECT_ROOT / "PHASE4_MECHANISTIC_MODELING" / "data"
P7_DATA = PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "data"

CLINICAL_USE_WARNING = (
    "⚠️ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. "
    "This system generates computational simulations for hypothesis generation. "
    "It does NOT determine the best treatment for individual patients."
)

# ═════════════════════════════════════════════════════════════
# DEEPSURV MODEL
# ═════════════════════════════════════════════════════════════
ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}

class DeepSurv(nn.Module):
    def __init__(self, n_features, hidden_dims=[64, 32], dropout=0.2, activation="gelu"):
        super().__init__()
        act_fn = ACTIVATIONS[activation]
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(in_dim))
            linear = nn.Linear(in_dim, h_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        out_layer = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

@st.cache_resource
def load_models():
    with open(P3_MODELS / "final_config.json", "r") as f:
        config = json.load(f)
    with open(P3_MODELS / "final_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    models = []
    for i in range(config["n_models"]):
        m = DeepSurv(len(config["feature_names"]),
                      hidden_dims=config["arch"]["hidden_dims"],
                      dropout=config["arch"]["dropout"],
                      activation=config["arch"]["activation"])
        m.load_state_dict(torch.load(P3_MODELS / f"final_deepsurv_model_{i}.pt",
                                      map_location="cpu", weights_only=True))
        m.eval()
        models.append(m)
    return config, scaler, models

@st.cache_data
def load_twin_summaries():
    path = P7_DATA / "twin_summaries_for_interface.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_population_priors():
    with open(P4_DATA / "population_priors.json", "r") as f:
        return json.load(f)

# ═════════════════════════════════════════════════════════════
# GOMPERTZ ODE
# ═════════════════════════════════════════════════════════════
def gompertz_growth(t, V, alpha, V_max):
    if V <= 0:
        return 0.0
    return alpha * V * np.log(V_max / V)

def treatment_response_ode(t, y, params, drug_conc_fn):
    V, E = y
    V = max(V, 1e-6)
    E = max(E, 0.0)
    alpha = params["alpha"]
    V_max = params["V_max"]
    delta_c = params.get("delta_c", 0.0)
    delta_t = params.get("delta_t", 0.0)
    s_E = params.get("s_E", 1.3e4)
    mu_E = params.get("mu_E", 0.041)
    rho = params.get("rho", 0.02)
    sigma = params.get("sigma", 2e7)
    delta_EV = params.get("delta_EV", 3.4e-10)
    k_immune = params.get("k_immune", 1.22e-4)
    E_ref = params.get("E_ref", 1.3e5)
    drug_type = params.get("drug_type", "none")
    C = drug_conc_fn(t, drug_type) if drug_type != "none" else 0.0
    C = max(C, 0.0)
    growth = alpha * V * np.log(V_max / V) if V > 0 else 0.0
    cytotoxic_kill = delta_c * C * V
    targeted_kill = delta_t * C * V
    if drug_type == "immuno" and C > 0:
        k_immuno = params.get("k_immuno", 2.5e-5)
        checkpoint_kill = k_immuno * C * V
        immune_boost = 1.0 + 0.02 * C
        delta_EV_eff = delta_EV * np.exp(-0.02 * C)
        rho_eff = rho * (1.0 + 0.01 * C)
    else:
        checkpoint_kill = 0.0
        immune_boost = 1.0
        delta_EV_eff = delta_EV
        rho_eff = rho
    immune_kill = k_immune * immune_boost * (E / E_ref) * V if V > 0 and E_ref > 0 else 0.0
    dVdt = growth - cytotoxic_kill - targeted_kill - immune_kill - checkpoint_kill
    V_cells = V * 1e6 if V > 0 else 0.0
    if E > 0:
        recruitment = rho_eff * E * V_cells / (sigma + V_cells) if V_cells > 0 else 0.0
        interaction_loss = delta_EV_eff * E * V_cells if V_cells > 0 else 0.0
        interaction_loss = min(interaction_loss, 10.0 * s_E)
        dEdt = s_E - mu_E * E + recruitment - interaction_loss
    else:
        dEdt = s_E
    return [dVdt, dEdt]

def make_cisplatin_fn(dose=75.0, k_el=0.552, V_d=30.0, cycle_interval=21.0, n_cycles=4):
    C_peak = dose / V_d
    def C(t, drug_type="chemo"):
        if t < 0: return 0.0
        cycle_idx = int(t // cycle_interval)
        if cycle_idx >= n_cycles: return 0.0
        t_in_cycle = t - cycle_idx * cycle_interval
        if t_in_cycle < 1/24:
            return C_peak * (1 - np.exp(-k_el * t_in_cycle)) / (1 - np.exp(-k_el / 24))
        return C_peak * np.exp(-k_el * (t_in_cycle - 1/24))
    return C

def make_pembrolizumab_fn(dose=200.0, k_el=0.0269, V_d=7.66, cycle_interval=21.0, n_cycles=6):
    C_peak = dose / V_d
    def C(t, drug_type="immuno"):
        if t < 0: return 0.0
        cycle_idx = int(t // cycle_interval)
        if cycle_idx >= n_cycles: return 0.0
        t_in_cycle = t - cycle_idx * cycle_interval
        return C_peak * np.exp(-k_el * t_in_cycle)
    return C

def make_osimertinib_fn(dose=80.0, k_el=0.346, k_a=2.77, V_d=986.0, F_bio=0.70):
    C_max_ss = 0.501
    def C(t, drug_type="targeted"):
        if t < 0: return 0.0
        t_in_day = t % 24.0
        if t_in_day < 1.0 / k_a:
            return C_max_ss * (1 - np.exp(-k_a * t_in_day))
        return C_max_ss * np.exp(-k_el * (t_in_day - 1.0 / k_a))
    return C

def make_no_drug_fn():
    return lambda t, drug_type: 0.0

DRUG_FNS = {
    "none": make_no_drug_fn,
    "chemo": make_cisplatin_fn,
    "immuno": make_pembrolizumab_fn,
    "targeted": make_osimertinib_fn,
}

def solve_trajectory(params, treatment="none", t_span=(0, 1095), n_points=100):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    defaults = {"alpha": 0.0008, "V_max": 1e6, "V0": 8000.0, "E0": 1.3e5,
                "delta_c": 0.0, "delta_t": 0.0, "s_E": 1.3e4, "mu_E": 0.041,
                "rho": 0.02, "sigma": 2e7, "delta_EV": 3.4e-10,
                "k_immune": 1.22e-4, "E_ref": 1.3e5, "k_immuno": 2.5e-5}
    p = dict(params)
    for key, default_val in defaults.items():
        val = p.get(key, default_val)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            p[key] = default_val
    V0 = np.clip(p.get("V0", 8000.0), 100.0, 500000.0)
    E0 = p.get("E0", 1.3e5)
    if treatment == "none":
        def ode_fn(t, y):
            return [gompertz_growth(t, y[0], p["alpha"], p["V_max"])]
        sol = solve_ivp(ode_fn, t_span, [V0], t_eval=t_eval, method="RK45",
                        rtol=1e-8, atol=1e-10, max_step=10.0)
    else:
        p["drug_type"] = treatment
        drug_fn = DRUG_FNS.get(treatment, make_no_drug_fn)()
        def ode_fn(t, y):
            return treatment_response_ode(t, y, p, drug_fn)
        sol = solve_ivp(ode_fn, t_span, [V0, E0], t_eval=t_eval, method="Radau",
                        rtol=1e-6, atol=1e-8, max_step=10.0)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    V = np.clip(sol.y[0], 0, None)
    return {"t": sol.t, "V": V, "E": sol.y[1] if treatment != "none" else None}

def compute_ttp(traj, threshold=2.0):
    V0 = traj["V"][0]
    idx = np.where(traj["V"] >= V0 * threshold)[0]
    if len(idx) == 0:
        return float(traj["t"][-1])
    return float(traj["t"][idx[0]])

# ═════════════════════════════════════════════════════════════
# FEATURE BUILDING
# ═════════════════════════════════════════════════════════════
GENE_FEATURES = ["SPP1", "COL1A1", "MMP1", "CXCL8", "EGLN3", "ANGPTL4",
                 "MELTF", "CD109", "MYEOV", "LAMC2", "RHOV", "SORCS2", "COL22A1"]
ENSG_MAP = {
    "SPP1": "ENSG00000118785", "COL1A1": "ENSG00000108821",
    "MMP1": "ENSG00000196611", "CXCL8": "ENSG00000169429",
    "EGLN3": "ENSG00000104708", "ANGPTL4": "ENSG00000148677",
    "MELTF": "ENSG00000073564", "CD109": "ENSG00000095752",
    "MYEOV": "ENSG00000185619", "LAMC2": "ENSG00000048342",
    "RHOV": "ENSG00000117322", "SORCS2": "ENSG00000171004",
    "COL22A1": "ENSG00000182898",
}

STAGE_V0 = {"I": 1000, "IA": 800, "IB": 1500, "II": 5000,
            "IIA": 4000, "IIB": 8000, "III": 30000,
            "IIIA": 25000, "IIIB": 50000, "IV": 100000}

def build_features(age, cancer_type, stage, smoking, gene_expr, feature_names, scaler, age_median):
    row = {}
    try:
        row["Age"] = float(age)
    except (ValueError, TypeError):
        row["Age"] = age_median

    ct = None
    if cancer_type and "adenocarcinoma" in cancer_type.lower():
        ct = "LUAD_Adenocarcinoma"
    elif cancer_type and ("squamous" in cancer_type.lower() or "scc" in cancer_type.lower()):
        ct = "LUSC_SquamousCell"

    st = stage if stage in STAGE_V0 else None
    sm = None
    if smoking and "former" in smoking.lower(): sm = "Former"
    elif smoking and "current" in smoking.lower(): sm = "Current"
    elif smoking and "never" in smoking.lower(): sm = "Never"

    for fname in feature_names:
        if fname == "Age": continue
        elif fname.startswith("Cancer_Type="):
            row[fname] = 1.0 if ct == fname.split("=", 1)[1] else 0.0
        elif fname.startswith("Stage="):
            row[fname] = 1.0 if st == fname.split("=", 1)[1] else 0.0
        elif fname.startswith("Smoking_Status="):
            row[fname] = 1.0 if sm == fname.split("=", 1)[1] else 0.0

    for g in GENE_FEATURES:
        if gene_expr and g in gene_expr:
            row[g] = float(gene_expr[g])
        else:
            row[g] = 0.0

    X = np.array([[row.get(f, 0.0) for f in feature_names]])
    return scaler.transform(X)

def predict_risk(age, cancer_type, stage, smoking, gene_expr, config, scaler, models):
    X = build_features(age, cancer_type, stage, smoking, gene_expr,
                       config["feature_names"], scaler, config["age_median"])
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        preds = [m(X_t).cpu().numpy()[0] for m in models]
    return float(np.mean(preds)), float(np.std(preds))

# ═════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Lung Cancer Digital Twin", page_icon="🫁", layout="wide")

    # Mandatory banner
    st.markdown(f"### {CLINICAL_USE_WARNING}")
    st.markdown("---")

    st.title("🫁 Personalized Neural-Mechanistic Digital Twin")
    st.markdown("*Level 1 Computational Digital Twin — Research Prototype*")

    # Load artifacts
    config, scaler, models = load_models()
    priors = load_population_priors()
    twin_summaries = load_twin_summaries()

    # ═══ Sidebar: Patient Input ═══
    st.sidebar.header("Patient Input")

    # Mode: select existing patient or manual input
    mode = st.sidebar.radio("Input Mode", ["Select Existing Patient", "Manual Input"])

    if mode == "Select Existing Patient" and len(twin_summaries) > 0:
        pids = twin_summaries["patient_id"].tolist()
        selected_pid = st.sidebar.selectbox("Select Patient", pids[:100])
        row = twin_summaries[twin_summaries["patient_id"] == selected_pid].iloc[0]
        age = float(row["age"]) if pd.notna(row["age"]) else 65
        cancer_type = str(row["cancer_type"])
        stage = str(row["stage"])
        smoking = str(row["smoking_status"])
        alpha = float(row["alpha"])
        V0 = float(row["V0"])
        V_max = float(row["V_max"])
        deepsurv_risk = float(row["deepsurv_risk"])
        deepsurv_risk_std = float(row["deepsurv_risk_std"])
    else:
        age = st.sidebar.number_input("Age", 30, 90, 65)
        cancer_type = st.sidebar.selectbox("Cancer Type", ["LUAD_Adenocarcinoma", "LUSC_SquamousCell"])
        stage = st.sidebar.selectbox("Stage", list(STAGE_V0.keys()))
        smoking = st.sidebar.selectbox("Smoking Status", ["Never", "Former", "Current"])

        # ODE parameters
        st.sidebar.subheader("Tumor Biology Parameters")
        alpha = st.sidebar.slider("Growth rate α (day⁻¹)", 1e-5, 5e-3, 8e-4, format="%.6f")
        V0 = st.sidebar.slider("Initial volume V₀ (mm³)", 100, 500000, 8000, step=100)
        V_max = st.sidebar.slider("Carrying capacity V_max (mm³)", 1e5, 5e6, 1e6, step=1e5)

        # Predict risk
        deepsurv_risk, deepsurv_risk_std = predict_risk(
            age, cancer_type, stage, smoking, None, config, scaler, models)

    # ═══ Simulation Controls ═══
    st.sidebar.subheader("Simulation Controls")
    sim_horizon = st.sidebar.slider("Simulation horizon (months)", 6, 36, 36)
    selected_treatments = st.sidebar.multiselect(
        "Treatment Scenarios",
        ["none", "chemo", "immuno", "targeted"],
        default=["none", "chemo", "immuno", "targeted"]
    )

    # ═══ Main Content ═══
    col1, col2 = st.columns(2)

    with col1:
        st.header("Patient State")
        st.write(f"**Patient ID:** {selected_pid if mode == 'Select Existing Patient' else 'Manual Input'}")
        st.write(f"**Age:** {age:.0f} years")
        st.write(f"**Cancer Type:** {cancer_type}")
        st.write(f"**Stage:** {stage}")
        st.write(f"**Smoking:** {smoking}")

        st.subheader("AI-Derived State")
        st.write(f"**DeepSurv Risk Score:** {deepsurv_risk:.4f}")
        st.write(f"**Risk Std (ensemble):** {deepsurv_risk_std:.4f}")
        risk_group = "High" if deepsurv_risk > 0 else "Low"
        st.write(f"**Risk Group:** {risk_group}")

        # Survival probability
        for years in [1, 2, 3]:
            S = np.exp(-0.001 * np.exp(deepsurv_risk) * years * 365)
            st.write(f"**{years}-year survival probability:** {S:.1%}")

    with col2:
        st.header("Biological Tumor State")
        st.write(f"**Growth rate (α):** {alpha:.6f} day⁻¹")
        st.write(f"**Initial volume (V₀):** {V0:.0f} mm³")
        st.write(f"**Carrying capacity (V_max):** {V_max:.0f} mm³")
        st.write(f"**Immune kill rate (k_immune):** {priors.get('k_immune', 1.22e-4):.6f}")

    st.markdown("---")

    # ═══ Simulation ═══
    st.header("Disease Evolution Simulation")

    ode_params = {
        "alpha": alpha, "V_max": V_max, "V0": V0,
        "k_immune": priors.get("k_immune", 1.22e-4),
        "mu_E": priors.get("mu_E", 0.041),
        "rho": priors.get("rho", 0.02),
        "delta_c": priors.get("delta_c", 0.028),
        "delta_t": priors.get("delta_t", 0.15),
        "E0": priors.get("E0", 1.3e5),
        "s_E": 1.3e4, "sigma": 2e7, "delta_EV": 3.4e-10,
        "E_ref": 1.3e5, "k_immuno": 2.5e-5,
    }

    t_span = (0, sim_horizon * 30)
    trajectories = {}
    ttp_results = {}

    for treatment in selected_treatments:
        try:
            traj = solve_trajectory(ode_params, treatment=treatment, t_span=t_span)
            trajectories[treatment] = traj
            ttp_results[treatment] = compute_ttp(traj)
        except Exception as e:
            st.error(f"Simulation failed for {treatment}: {e}")

    # Tumor volume plot
    fig = go.Figure()
    labels = {"none": "Natural History", "chemo": "Chemotherapy",
              "immuno": "Immunotherapy", "targeted": "Targeted Therapy"}
    colors = {"none": "gray", "chemo": "blue", "immuno": "green", "targeted": "red"}

    for treatment, traj in trajectories.items():
        t_months = traj["t"] / 30.44
        V = traj["V"]
        fig.add_trace(go.Scatter(
            x=t_months, y=V, mode="lines",
            name=labels.get(treatment, treatment),
            line=dict(color=colors.get(treatment, "black"), width=2)
        ))

    fig.update_layout(
        title="Tumor Volume Trajectory",
        xaxis_title="Time (months)",
        yaxis_title="Tumor Volume (mm³)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Survival probability plot
    fig2 = go.Figure()
    for treatment in trajectories:
        traj = trajectories[treatment]
        t_months = traj["t"] / 30.44
        S = np.exp(-0.001 * np.exp(deepsurv_risk) * traj["t"])
        fig2.add_trace(go.Scatter(
            x=t_months, y=S, mode="lines",
            name=labels.get(treatment, treatment),
            line=dict(color=colors.get(treatment, "black"), width=2)
        ))

    fig2.update_layout(
        title="Survival Probability",
        xaxis_title="Time (months)",
        yaxis_title="S(t)",
        template="plotly_white",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # TTP comparison table
    st.subheader("Time to Progression (TTP) Comparison")
    ttp_data = []
    for treatment, ttp in ttp_results.items():
        ttp_data.append({
            "Scenario": labels.get(treatment, treatment),
            "TTP (days)": f"{ttp:.0f}",
            "TTP (months)": f"{ttp / 30.44:.1f}",
            "Benefit vs no treatment (months)": f"{(ttp - ttp_results.get('none', ttp)) / 30.44:.1f}" if treatment != "none" else "—",
        })
    st.table(pd.DataFrame(ttp_data))

    # ═══ Uncertainty ═══
    st.markdown("---")
    st.header("Uncertainty Quantification")
    st.write(f"**Method:** DeepSurv ensemble std + Monte Carlo over ODE parameter bounds (α, V_max, V₀ ±30%)")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.write(f"**DeepSurv risk:** {deepsurv_risk:.4f} ± {deepsurv_risk_std:.4f}")
        st.write(f"**95% CI:** [{deepsurv_risk - 1.96 * deepsurv_risk_std:.4f}, {deepsurv_risk + 1.96 * deepsurv_risk_std:.4f}]")
    with col_u2:
        for years in [1, 2, 3]:
            S = np.exp(-0.001 * np.exp(deepsurv_risk) * years * 365)
            S_lo = np.exp(-0.001 * np.exp(deepsurv_risk + 1.96 * deepsurv_risk_std) * years * 365)
            S_hi = np.exp(-0.001 * np.exp(deepsurv_risk - 1.96 * deepsurv_risk_std) * years * 365)
            st.write(f"**{years}-yr survival:** {S:.1%} (CI: {S_lo:.1%}–{S_hi:.1%})")

    # ═══ Explainability ═══
    st.markdown("---")
    st.header("Explainability")

    st.subheader("AI Feature Importance (Global, from Phase 3 Permutation Importance)")
    perm_imp_path = PROJECT_ROOT / "PHASE3_DEEP_LEARNING" / "tables" / "final_permutation_importance.csv"
    if perm_imp_path.exists():
        perm_imp = pd.read_csv(perm_imp_path).head(10)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=perm_imp.iloc[:, 1],
            y=perm_imp.iloc[:, 0],
            orientation="h",
            marker_color="steelblue",
        ))
        fig3.update_layout(
            title="Top 10 Features (Permutation Importance)",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Mechanistic Drivers")
    st.write(f"**Growth rate (α):** {alpha:.6f} day⁻¹ — {'fast' if alpha > 0.001 else 'moderate' if alpha > 0.0005 else 'slow'} growth")
    st.write(f"**Initial volume (V₀):** {V0:.0f} mm³ — {'large' if V0 > 50000 else 'moderate' if V0 > 5000 else 'small'} tumor")
    st.write(f"**Immune activity (k_immune):** {priors.get('k_immune', 1.22e-4):.6f} day⁻¹")

    # ═══ Footer ═══
    st.markdown("---")
    st.markdown(f"**{CLINICAL_USE_WARNING}**")
    st.markdown("*Framework: Streamlit | Fidelity: Live backend | "
                "Out of scope: Real-time monitoring, clinical decision support, regulatory approval*")


if __name__ == "__main__":
    main()
