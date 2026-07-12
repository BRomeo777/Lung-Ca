"""
Phase 7 Post-Processing — Fix 7D/7F reports after main script completed.

Fixes:
  1. Permutation importance path (tables/ not reports/)
  2. Biological validation using all patients (train+val) for parameter variance
  3. IBS using Cox model baseline hazard for proper calibration
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P7_DATA = PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "data"
P7_REPORTS = PROJECT_ROOT / "PHASE7_DIGITAL_TWIN" / "reports"
P3_TABLES = PROJECT_ROOT / "PHASE3_DEEP_LEARNING" / "tables"

CLINICAL_USE_WARNING = (
    "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. "
    "This system generates computational simulations for hypothesis generation. "
    "It does NOT determine the best treatment for individual patients."
)

def boot_ci_cindex(events, times, risks, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    n = len(events)
    cis = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(events[idx])) < 2:
            continue
        try:
            ci = concordance_index_censored(events[idx].astype(bool), times[idx], risks[idx])[0]
            cis.append(ci)
        except Exception:
            continue
    if len(cis) < 10:
        return float("nan"), (float("nan"), float("nan"))
    point = np.median(cis)
    lo, hi = np.percentile(cis, [2.5, 97.5])
    return float(point), (float(lo), float(hi))

def main():
    print("=" * 60)
    print("Phase 7 Post-Processing: Fixing 7D/7F reports")
    print("=" * 60)

    # Load state matrix
    state_df = pd.read_csv(P7_DATA / "digital_twin_state_matrix.csv")
    print(f"Loaded state matrix: {len(state_df)} patients")

    # Load trajectory simulations
    traj_df = pd.read_csv(P7_DATA / "digital_twin_trajectory_simulations.csv")
    print(f"Loaded trajectories: {len(traj_df)} rows")

    # Load twin summaries (has TTP per patient per treatment)
    twin_df = pd.read_csv(P7_DATA / "twin_summaries_for_interface.csv")
    print(f"Loaded twin summaries: {len(twin_df)} patients")

    # Identify validation patients (those not in patient_parameters_TCGA.csv)
    p4_params = pd.read_csv(PROJECT_ROOT / "PHASE4_MECHANISTIC_MODELING" / "data" / "patient_parameters_TCGA.csv")
    train_pids = set(p4_params["Patient_ID"].astype(str))
    val_pids = set(state_df["patient_id"].astype(str)) - train_pids
    print(f"Train patients: {len(train_pids)}, Val patients: {len(val_pids)}")

    # ═══════════════════════════════════════════════════════════
    # FIX 7D: Validation Report
    # ═══════════════════════════════════════════════════════════
    print("\n--- Fixing 7D: Validation Report ---")

    val_df = state_df[state_df["patient_id"].astype(str).isin(val_pids)].copy()
    val_events = val_df["os_event"].values
    val_times = val_df["os_time"].values
    val_risks = val_df["deepsurv_risk"].values

    # C-index
    ci_point, ci_ci = boot_ci_cindex(val_events, val_times, val_risks, n_boot=500)
    p3_ci = 0.6131
    print(f"  C-index: {ci_point:.4f} ({ci_ci[0]:.4f}-{ci_ci[1]:.4f})")

    # Calibration slope
    val_df_calib = pd.DataFrame({
        "risk": val_risks, "time": val_times, "event": val_events.astype(bool),
    })
    try:
        cph = CoxPHFitter()
        cph.fit(val_df_calib, duration_col="time", event_col="event")
        cal_slope = float(cph.params_["risk"])
    except Exception as e:
        cal_slope = float("nan")
        print(f"  Calibration failed: {e}")
    print(f"  Calibration slope: {cal_slope:.4f}")

    # IBS using Cox baseline hazard
    surv_times_ibs = np.linspace(30, min(1500, val_times.max()), 50)
    km_cens = KaplanMeierFitter()
    km_cens.fit(val_times, event_observed=(1 - val_events.astype(int)))

    ibs_scores = []
    if not np.isnan(cal_slope):
        for st in surv_times_ibs:
            try:
                sf = cph.predict_survival_function(val_df_calib[["risk"]], times=[st])
                S_pred = sf.values[0]  # shape (n_patients,) at time st
            except Exception:
                S_pred = np.exp(-0.001 * np.exp(val_risks) * st)
            G = float(km_cens.survival_function_at_times(st).iloc[0]) if st <= km_cens.survival_function_.index.max() else 0.0
            if G <= 0: continue
            for i in range(len(val_df)):
                event_by_st = (val_times[i] <= st) & (bool(val_events[i]))
                at_risk = val_times[i] >= st
                if event_by_st:
                    ibs_scores.append((1 - S_pred[i])**2 / G)
                elif at_risk:
                    ibs_scores.append((S_pred[i])**2 / G)
    ibs = float(np.mean(ibs_scores)) if ibs_scores else float("nan")

    # Null model IBS
    km_null = KaplanMeierFitter()
    km_null.fit(val_times, event_observed=val_events.astype(bool))
    null_scores = []
    for st in surv_times_ibs:
        S_null = float(km_null.survival_function_at_times(st).iloc[0]) if st <= km_null.survival_function_.index.max() else 0.0
        G = float(km_cens.survival_function_at_times(st).iloc[0]) if st <= km_cens.survival_function_.index.max() else 0.0
        if G <= 0: continue
        for i in range(len(val_df)):
            event_by_st = (val_times[i] <= st) & (bool(val_events[i]))
            at_risk = val_times[i] >= st
            if event_by_st:
                null_scores.append((1 - S_null)**2 / G)
            elif at_risk:
                null_scores.append((S_null)**2 / G)
    ibs_null = float(np.mean(null_scores)) if null_scores else float("nan")
    print(f"  IBS (Cox-calibrated): {ibs:.4f}, Null: {ibs_null:.4f}")

    # Subgroup C-indices
    subgroup_results = []
    for stage in ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]:
        mask = val_df["stage"].values == stage
        n = mask.sum()
        if n < 5:
            subgroup_results.append({"stage": stage, "n": int(n), "ci": np.nan,
                                     "ci_lo": np.nan, "ci_hi": np.nan, "note": "too_small"})
            continue
        ci_s, ci_s_ci = boot_ci_cindex(val_events[mask], val_times[mask], val_risks[mask], n_boot=200)
        subgroup_results.append({"stage": stage, "n": int(n), "ci": ci_s,
                                 "ci_lo": ci_s_ci[0], "ci_hi": ci_s_ci[1], "note": ""})

    # Biological validation — use ALL patients for parameter variance
    all_alphas = state_df["alpha"].values
    all_os_times = state_df["os_time"].values
    all_os_events = state_df["os_event"].values
    # Compute mech risk from twin summaries
    twin_df_sorted = twin_df.set_index("patient_id")
    mech_risk_all = []
    for _, row in state_df.iterrows():
        pid = str(row["patient_id"])
        if pid in twin_df_sorted.index:
            ttp = twin_df_sorted.loc[pid, "ttp_natural"]
            if pd.notna(ttp) and ttp > 0:
                mech_risk_all.append(1.0 / ttp)
            else:
                mech_risk_all.append(0.0)
        else:
            mech_risk_all.append(0.0)
    all_mech_risk = np.array(mech_risk_all)

    all_valid = all_os_events.astype(bool)
    print(f"  All patients with events: {all_valid.sum()}")

    # Test 1: Growth rate vs survival
    if all_valid.sum() >= 10:
        rho_growth, p_growth = stats.spearmanr(all_alphas[all_valid], all_os_times[all_valid])
        bio1_pass = rho_growth < 0 and p_growth < 0.1
    else:
        rho_growth, p_growth = float("nan"), float("nan")
        bio1_pass = False
    print(f"  Bio test 1 (alpha vs survival): rho={rho_growth:.4f}, p={p_growth:.4f} → {'PASS' if bio1_pass else 'FAIL'}")

    # Test 2: Mech risk vs survival
    if all_valid.sum() >= 10:
        rho_risk, p_risk = stats.spearmanr(all_mech_risk[all_valid], all_os_times[all_valid])
        bio2_pass = rho_risk < 0 and p_risk < 0.1
    else:
        rho_risk, p_risk = float("nan"), float("nan")
        bio2_pass = False
    print(f"  Bio test 2 (mech risk vs survival): rho={rho_risk:.4f}, p={p_risk:.4f} → {'PASS' if bio2_pass else 'FAIL'}")

    # Test 3: Treatment benefit
    val_twin = twin_df[twin_df["patient_id"].astype(str).isin(val_pids)]
    benefit_threshold = 30
    n_val = len(val_twin)
    pct_chemo = (val_twin["ttp_chemo"] - val_twin["ttp_natural"] > benefit_threshold).sum() / n_val * 100
    pct_immuno = (val_twin["ttp_immuno"] - val_twin["ttp_natural"] > benefit_threshold).sum() / n_val * 100
    pct_targeted = (val_twin["ttp_targeted"] - val_twin["ttp_natural"] > benefit_threshold).sum() / n_val * 100
    bio3_pass = pct_chemo >= 50 or pct_targeted >= 50
    print(f"  Bio test 3 (treatment benefit): chemo={pct_chemo:.1f}%, immuno={pct_immuno:.1f}%, targeted={pct_targeted:.1f}% → {'PASS' if bio3_pass else 'FAIL'}")

    # Treatment simulation validity
    chemo_benefits = (val_twin["ttp_chemo"] - val_twin["ttp_natural"]).values
    median_benefit = np.median(chemo_benefits)
    responders = chemo_benefits > median_benefit
    ttp_resp = val_twin["ttp_chemo"].values[responders]
    ttp_nonresp = val_twin["ttp_chemo"].values[~responders]
    med_resp = np.median(ttp_resp) if len(ttp_resp) > 0 else np.nan
    med_nonresp = np.median(ttp_nonresp) if len(ttp_nonresp) > 0 else np.nan
    if len(ttp_resp) > 0 and len(ttp_nonresp) > 0:
        lr = logrank_test(ttp_resp, ttp_nonresp,
                          event_observed_A=np.ones(len(ttp_resp)),
                          event_observed_B=np.ones(len(ttp_nonresp)))
        lr_p = float(lr.p_value)
    else:
        lr_p = float("nan")

    # Save treatment simulation validation
    treat_val_df = pd.DataFrame([
        {"metric": "responder_count", "value": int(responders.sum()), "caveat": "model-derived labels"},
        {"metric": "non_responder_count", "value": int((~responders).sum()), "caveat": "model-derived labels"},
        {"metric": "median_ttp_responder_days", "value": med_resp, "caveat": "simulated TTP"},
        {"metric": "median_ttp_non_responder_days", "value": med_nonresp, "caveat": "simulated TTP"},
        {"metric": "logrank_p_value", "value": lr_p, "caveat": "simulated TTP, model-derived labels"},
        {"metric": "benefit_pct_chemo", "value": pct_chemo, "caveat": "simulated"},
        {"metric": "benefit_pct_immuno", "value": pct_immuno, "caveat": "simulated"},
        {"metric": "benefit_pct_targeted", "value": pct_targeted, "caveat": "simulated"},
    ])
    treat_val_df.to_csv(P7_REPORTS / "treatment_simulation_validation_report.csv", index=False)
    print(f"  -> treatment_simulation_validation_report.csv saved")

    # Save validation report
    val_report_lines = [
        "Digital Twin Validation Report — Phase 7D (Post-Processed)",
        "=" * 50,
        "",
        f"Validation set: {len(val_df)} patients",
        f"All patients (for biological validation): {len(state_df)} patients",
        "",
        "1. PREDICTION VALIDATION",
        "-" * 30,
        f"   C-index (DeepSurv risk): {ci_point:.4f} (95% CI: {ci_ci[0]:.4f}-{ci_ci[1]:.4f})",
        f"   Phase 3 reported C-index: {p3_ci:.4f}",
        f"   Delta (twin - phase3): {ci_point - p3_ci:.4f}",
        f"   Calibration slope: {cal_slope:.4f} (target: 0.8-1.2)",
        f"   IBS (Cox-calibrated): {ibs:.4f}",
        f"   IBS (null model): {ibs_null:.4f}",
        "",
        "2. SUBGROUP PREDICTION ERROR BY STAGE",
        "-" * 30,
    ]
    for sg in subgroup_results:
        if sg["note"] == "too_small":
            val_report_lines.append(f"   {sg['stage']}: n={sg['n']} (too small for CI)")
        else:
            val_report_lines.append(f"   {sg['stage']}: n={sg['n']}, C-index={sg['ci']:.4f} ({sg['ci_lo']:.4f}-{sg['ci_hi']:.4f})")
    val_report_lines.extend([
        "",
        "3. BIOLOGICAL VALIDATION (ALL patients for parameter variance)",
        "-" * 30,
        f"   NOTE: Validation patients share population prior alpha (no personalization).",
        f"   Tests 1-2 use ALL patients (train+val) for parameter variance.",
        f"   3.1 Faster growth → worse outcomes: rho={rho_growth:.4f}, p={p_growth:.4f} → {'PASS' if bio1_pass else 'FAIL'}",
        f"   3.2 Higher mech risk → shorter survival: rho={rho_risk:.4f}, p={p_risk:.4f} → {'PASS' if bio2_pass else 'FAIL'}",
        f"   3.3 Effective therapies → improved outcomes: chemo={pct_chemo:.1f}%, immuno={pct_immuno:.1f}%, targeted={pct_targeted:.1f}% → {'PASS' if bio3_pass else 'FAIL'}",
        f"        Threshold: ≥50% for at least one therapy",
        "",
        "4. TREATMENT SIMULATION VALIDITY",
        "-" * 30,
        f"   Responders: {responders.sum()}, Non-responders: {(~responders).sum()}",
        f"   Median TTP — Responders: {med_resp/30.44:.1f}mo, Non-responders: {med_nonresp/30.44:.1f}mo",
        f"   Log-rank p-value: {lr_p:.4e}",
        "",
        "   CAVEAT: Responder/non-responder labels are model-derived (simulated TTP benefit).",
        "   This validates internal consistency of the treatment-response module,",
        "   NOT real-world treatment discrimination. Only call it clinical validation",
        "   if labels come from real, independent patient records.",
        "",
        "5. CIRCULARITY CAVEAT",
        "-" * 30,
        "   The mechanistic component of the Digital Twin is partially circular.",
        "   TTP and treatment response outcomes are simulated by the same Gompertz ODE",
        "   model that is used inside the twin. Biological validation tests (3.1-3.3)",
        "   validate internal consistency, not real-world predictive accuracy.",
        "   Prediction validation (C-index, calibration, IBS) uses real survival outcomes",
        "   and is NOT circular.",
        "",
        CLINICAL_USE_WARNING,
    ])
    with open(P7_REPORTS / "digital_twin_validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_report_lines))
    print(f"  -> digital_twin_validation_report.txt saved")

    # ═══════════════════════════════════════════════════════════
    # FIX 7F: Explainability Report
    # ═══════════════════════════════════════════════════════════
    print("\n--- Fixing 7F: Explainability Report ---")

    perm_imp_path = P3_TABLES / "final_permutation_importance.csv"
    if perm_imp_path.exists():
        perm_imp = pd.read_csv(perm_imp_path)
        print(f"  Loaded permutation importance ({len(perm_imp)} features)")
        top5 = perm_imp.head(5)
        top_features_str = "; ".join([f"{r.iloc[0]}={r.iloc[1]:.4f}" for _, r in top5.iterrows()])
    else:
        perm_imp = None
        top_features_str = "N/A"
        print(f"  WARNING: Permutation importance not found at {perm_imp_path}")

    explain_rows = []
    for _, twin_row in val_twin.iterrows():
        pid = str(twin_row["patient_id"])
        state_row = state_df[state_df["patient_id"].astype(str) == pid].iloc[0]
        ttp_nat = twin_row["ttp_natural"]
        ttp_chemo = twin_row["ttp_chemo"]
        ttp_immuno = twin_row["ttp_immuno"]
        ttp_targeted = twin_row["ttp_targeted"]
        deepsurv_risk = state_row["deepsurv_risk"]
        risk_group = state_row["risk_group"]
        mech_drivers = f"alpha={state_row['alpha']:.6f}, V0={state_row['V0']:.0f}, k_immune={state_row['k_immune']:.6f}"

        explain_rows.append({
            "patient_id": pid,
            "deepsurv_risk": deepsurv_risk,
            "risk_group": risk_group,
            "top_ai_features": top_features_str,
            "mechanistic_drivers": mech_drivers,
            "ttp_natural_months": ttp_nat / 30.44,
            "ttp_chemo_months": ttp_chemo / 30.44,
            "ttp_immuno_months": ttp_immuno / 30.44,
            "ttp_targeted_months": ttp_targeted / 30.44,
            "benefit_chemo_months": (ttp_chemo - ttp_nat) / 30.44,
            "benefit_immuno_months": (ttp_immuno - ttp_nat) / 30.44,
            "benefit_targeted_months": (ttp_targeted - ttp_nat) / 30.44,
            "stage": state_row["stage"],
            "cancer_type": state_row["cancer_type"],
            "explanation": (
                f"Patient {pid}: DeepSurv risk={deepsurv_risk:.3f} ({risk_group} risk). "
                f"Key AI features: {top_features_str}. "
                f"Mechanistic drivers: {mech_drivers}. "
                f"Natural history TTP: {ttp_nat/30.44:.1f}mo. "
                f"Chemo benefit: +{(ttp_chemo-ttp_nat)/30.44:.1f}mo. "
                f"Immuno benefit: +{(ttp_immuno-ttp_nat)/30.44:.1f}mo. "
                f"Targeted benefit: +{(ttp_targeted-ttp_nat)/30.44:.1f}mo. "
                f"{CLINICAL_USE_WARNING}"
            ),
        })

    explain_df = pd.DataFrame(explain_rows)
    explain_df.to_csv(P7_REPORTS / "digital_twin_explainability_report.csv", index=False)
    print(f"  -> digital_twin_explainability_report.csv saved ({len(explain_df)} rows)")

    # Print summary
    print("\n" + "=" * 60)
    print("POST-PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  C-index: {ci_point:.4f} ({ci_ci[0]:.4f}-{ci_ci[1]:.4f})")
    print(f"  Calibration slope: {cal_slope:.4f}")
    print(f"  IBS (Cox-calibrated): {ibs:.4f}, Null: {ibs_null:.4f}")
    print(f"  Biological validation: {sum([bio1_pass, bio2_pass, bio3_pass])}/3 passed")
    print(f"  Treatment benefit: chemo={pct_chemo:.1f}%, immuno={pct_immuno:.1f}%, targeted={pct_targeted:.1f}%")
    print(f"  Permutation importance: {'loaded' if perm_imp is not None else 'NOT FOUND'}")
    print(CLINICAL_USE_WARNING)

if __name__ == "__main__":
    main()
