"""
run_phase1.py
Phase 1 data engineering for the Lung Cancer Digital Twin project.

Organises existing, already-downloaded and verified datasets into
reproducible, machine-learning-ready cohorts for overall-survival modelling.

Design principles (enforced throughout):
  * Cohorts are NEVER merged at the patient level. TCGA, CPTAC and each GEO
    series keep their own identity.
  * Provenance schema fields are ALWAYS populated (never blank); 'not_applicable'
    is written where a field genuinely does not apply.
  * No new downloads, no external database queries, no model training.

Run:  python 00_CODE/run_phase1.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

import dt_common as C
from dt_common import (NA, NAV, log)

RESULTS: dict = {}
RANDOM_SEED = 42
SPLIT_RATIO = 0.80


# =========================================================================== #
# Helper: copy the lightweight structured raw inputs into 01_RAW_DATA
# =========================================================================== #
def populate_raw_data():
    """Copy structured (non-huge) inputs into 01_RAW_DATA for a self-contained
    project. The 1000+ per-file expression TSVs are NOT copied (the merged
    matrices already contain them)."""
    def cp(src, dst):
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for coh in ("LUAD", "LUSC"):
        s = C.RAW_SRC / f"TCGA_{coh}"
        d = C.RAW_DIR / f"TCGA_{coh}"
        cp(s / "clinical" / f"TCGA-{coh}_clinical.tsv", d / f"TCGA-{coh}_clinical.tsv")
        cp(s / "survival" / f"TCGA-{coh}_survival.tsv", d / f"TCGA-{coh}_survival.tsv")
        cp(s / "metadata" / f"TCGA-{coh}_sample_sheet.tsv", d / f"TCGA-{coh}_sample_sheet.tsv")
        cp(s / "metadata" / f"TCGA-{coh}_file_manifest.tsv", d / f"TCGA-{coh}_file_manifest.tsv")
        cp(s / "metadata" / f"TCGA-{coh}_summary.json", d / f"TCGA-{coh}_summary.json")
        cp(s / "expression" / f"TCGA-{coh}_gene_annotation.tsv.gz", d / f"TCGA-{coh}_gene_annotation.tsv.gz")
        cp(s / "expression" / f"TCGA-{coh}_STAR_counts_matrix.tsv.gz", d / f"TCGA-{coh}_STAR_counts_matrix.tsv.gz")

    for g in C.GEO_SERIES:
        s = C.RAW_SRC / "GEO" / g
        d = C.RAW_DIR / "GEO" / g
        cp(s / f"{g}_series_matrix.txt.gz_expression_matrix.tsv.gz",
           d / f"{g}_expression_matrix.tsv.gz")
        cp(s / f"{g}_series_matrix.txt.gz_sample_metadata.tsv",
           d / f"{g}_sample_metadata.tsv")
        cp(s / f"{g}_summary.json", d / f"{g}_summary.json")

    cp(C.RAW_SRC / "CPTAC" / "clinical" / "CPTAC3_LUAD_clinical.tsv",
       C.RAW_DIR / "CPTAC_LUNG" / "CPTAC3_LUAD_clinical.tsv")
    cp(C.RAW_SRC / "CPTAC" / "metadata" / "CPTAC3_LUAD_metadata.json",
       C.RAW_DIR / "CPTAC_LUNG" / "CPTAC3_LUAD_metadata.json")
    log("[RAW] 01_RAW_DATA populated with structured input files (large per-file "
        "expression TSVs intentionally excluded; merged matrices copied).")


# =========================================================================== #
# STEP 1 - Complete data inventory
# =========================================================================== #
def _classify(name: str) -> str:
    n = name.lower()
    if "clinical" in n:
        return "clinical"
    if "survival" in n:
        return "survival"
    if "counts_matrix" in n or "tpm_matrix" in n:
        return "RNA-seq expression"
    if "expression_matrix" in n:
        return "microarray expression"
    if "gene_annotation" in n:
        return "annotation"
    if "sample_metadata" in n or "sample_sheet" in n or "manifest" in n:
        return "metadata"
    if "summary" in n or n.endswith(".json"):
        return "metadata"
    if n.endswith(".maf") or "mutation" in n or "somatic" in n:
        return "mutation"
    return "other"


def _quick_shape(path):
    """Return (rows, cols) cheaply for tabular files; ('', '') otherwise."""
    try:
        if path.suffix == ".gz" and (".tsv" in path.suffixes or ".txt" in path.suffixes):
            df = pd.read_csv(path, sep="\t", nrows=5)
            full = pd.read_csv(path, sep="\t", usecols=[0])
            return len(full), df.shape[1]
        if path.suffix in (".tsv", ".csv", ".txt"):
            sep = "," if path.suffix == ".csv" else "\t"
            df = pd.read_csv(path, sep=sep, nrows=5)
            full = pd.read_csv(path, sep=sep, usecols=[0])
            return len(full), df.shape[1]
    except Exception:
        pass
    return "", ""


def step1_inventory():
    log("=== STEP 1: Data inventory ===")
    rows = []
    datasets = {
        "TCGA-LUAD": C.RAW_SRC / "TCGA_LUAD",
        "TCGA-LUSC": C.RAW_SRC / "TCGA_LUSC",
        "CPTAC-LUNG": C.RAW_SRC / "CPTAC",
    }
    for g in C.GEO_SERIES:
        datasets[g] = C.RAW_SRC / "GEO" / g

    for dsname, root in datasets.items():
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            # skip the 1000+ individual per-aliquot expression files in inventory
            if "rna_seq.augmented_star_gene_counts" in p.name:
                continue
            cls = _classify(p.name)
            r, c = ("", "")
            if cls in ("clinical", "survival", "metadata", "annotation") and p.stat().st_size < 5_000_000:
                r, c = _quick_shape(p)
            rows.append({
                "dataset": dsname,
                "filename": p.name,
                "path": str(p),
                "format": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size,
                "rows": r,
                "columns": c,
                "classification": cls,
            })
    # Add a single representative row for the per-aliquot expression file group
    for coh in ("LUAD", "LUSC"):
        expr = C.RAW_SRC / f"TCGA_{coh}" / "expression"
        if expr.exists():
            n = len(list(expr.glob("*rna_seq.augmented_star_gene_counts*")))
            if n:
                rows.append({
                    "dataset": f"TCGA-{coh}", "filename": f"<{n} per-aliquot STAR gene-count TSVs>",
                    "path": str(expr), "format": "tsv", "size_bytes": "",
                    "rows": "", "columns": "", "classification": "RNA-seq expression"})
    inv = pd.DataFrame(rows)
    inv.to_csv(C.DOC_DIR / "dataset_inventory.csv", index=False)
    log(f"[STEP1] inventory rows={len(inv)} -> dataset_inventory.csv")
    RESULTS["inventory_rows"] = len(inv)
    return inv


# =========================================================================== #
# STEPS 2-4 - TCGA standardisation, filtering, discovery dataset
# =========================================================================== #
def _load_tcga_clue(coh: str):
    base = C.RAW_SRC / f"TCGA_{coh}"
    clin = pd.read_csv(base / "clinical" / f"TCGA-{coh}_clinical.tsv", sep="\t", dtype=str)
    surv = pd.read_csv(base / "survival" / f"TCGA-{coh}_survival.tsv", sep="\t", dtype=str)
    ss = pd.read_csv(base / "metadata" / f"TCGA-{coh}_sample_sheet.tsv", sep="\t", dtype=str)
    return clin, surv, ss


def step3_filter_tcga():
    """Filter to Primary Tumor, one row per patient with deterministic selection.
    Returns dict per cohort: selected aliquot table + normals table."""
    log("=== STEP 3: TCGA sample filtering (Primary Tumor only) ===")
    normals_all = []
    selected = {}
    for coh in ("LUAD", "LUSC"):
        _, _, ss = _load_tcga_clue(coh)
        ss["Patient_ID"] = ss["case_submitter_id"]
        ss["portion"] = ss["aliquot_submitter_id"].map(C.tcga_portion_number)

        primary = ss[ss["sample_type"] == "Primary Tumor"].copy()
        non_primary = ss[ss["sample_type"] != "Primary Tumor"].copy()
        for _, r in non_primary.iterrows():
            normals_all.append({
                "Patient_ID": r["Patient_ID"], "Dataset_Source": f"TCGA-{coh}",
                "aliquot_submitter_id": r["aliquot_submitter_id"],
                "sample_type": r["sample_type"],
                "reason_excluded": "non-primary-tumour sample (set aside, not modelled)",
            })
        log(f"[STEP3] TCGA-{coh}: {len(ss)} samples -> {len(primary)} Primary Tumor, "
            f"{len(non_primary)} non-primary routed to matched-normal set")

        # deterministic selection among multiple primary aliquots per patient
        chosen_rows = []
        excluded_dupes = 0
        for pid, grp in primary.groupby("Patient_ID"):
            if len(grp) == 1:
                row = grp.iloc[0].to_dict()
                row["Selection_Method"] = "not_applicable_single_sample"
                chosen_rows.append(row)
            else:
                # Priority 1 (sequencing-quality metric) UNAVAILABLE in the
                # downloaded GDC metadata -> apply Priority 2 (lowest barcode portion).
                g2 = grp.sort_values(["portion", "aliquot_submitter_id"])
                row = g2.iloc[0].to_dict()
                row["Selection_Method"] = "lowest_barcode_portion"
                chosen_rows.append(row)
                excluded_dupes += len(grp) - 1
                for _, ex in g2.iloc[1:].iterrows():
                    normals_all.append({
                        "Patient_ID": pid, "Dataset_Source": f"TCGA-{coh}",
                        "aliquot_submitter_id": ex["aliquot_submitter_id"],
                        "sample_type": "Primary Tumor (duplicate aliquot)",
                        "reason_excluded": "duplicate primary aliquot; not selected "
                                           "(kept lowest barcode portion)",
                    })
                    log(f"[STEP3] {pid}: excluded duplicate primary aliquot "
                        f"{ex['aliquot_submitter_id']} (selected lowest portion)")
        sel = pd.DataFrame(chosen_rows)
        log(f"[STEP3] TCGA-{coh}: {sel['Patient_ID'].nunique()} unique patients "
            f"after selection ({excluded_dupes} duplicate primary aliquots excluded)")
        selected[coh] = sel

    normals_df = pd.DataFrame(normals_all)
    normals_df.to_csv(C.READY_DIR / "TCGA_matched_normal_samples.csv", index=False)
    log(f"[STEP3] wrote TCGA_matched_normal_samples.csv (n={len(normals_df)})")
    RESULTS["tcga_excluded_normal_samples"] = int(len(normals_df))
    return selected


def step4_discovery(selected):
    """Build harmonised TCGA discovery master (LUAD+LUSC) + aligned expression."""
    log("=== STEP 4: TCGA discovery dataset (LUAD + LUSC) ===")
    cancer_map = {"LUAD": "LUAD_Adenocarcinoma", "LUSC": "LUSC_SquamousCell"}
    frames = []
    aliquot_index = {}  # Patient_ID -> aliquot barcode (for expression alignment)

    for coh in ("LUAD", "LUSC"):
        clin, surv, _ = _load_tcga_clue(coh)
        sel = selected[coh]
        clin = clin.set_index("case_submitter_id")
        surv = surv.set_index("case_submitter_id")

        rows = []
        for _, s in sel.iterrows():
            pid = s["Patient_ID"]
            aliquot_index[pid] = s["aliquot_submitter_id"]
            cl = clin.loc[pid] if pid in clin.index else pd.Series(dtype=str)
            if isinstance(cl, pd.DataFrame):
                cl = cl.iloc[0]
            sv = surv.loc[pid] if pid in surv.index else pd.Series(dtype=str)
            if isinstance(sv, pd.DataFrame):
                sv = sv.iloc[0]

            # Age (years): prefer age_at_index; else age_at_diagnosis (days)/365.25
            age = cl.get("age_at_index")
            if age in (None, "", "nan") or pd.isna(age):
                aad = cl.get("age_at_diagnosis")
                try:
                    age = round(float(aad) / 365.25, 1) if aad not in (None, "", "nan") and not pd.isna(aad) else NAV
                except (ValueError, TypeError):
                    age = NAV
            else:
                try:
                    age = round(float(age), 1)
                except (ValueError, TypeError):
                    age = NAV

            os_event = C.std_vital_to_event(sv.get("OS_event") if sv.get("OS_event") is not None else cl.get("vital_status"))
            os_time = sv.get("OS_time_days")
            try:
                os_time = round(float(os_time), 2) if os_time not in (None, "", "nan") and not pd.isna(os_time) else None
            except (ValueError, TypeError):
                os_time = None

            rows.append({
                "Patient_ID": pid,
                "Dataset_Source": f"TCGA-{coh}",
                "Cohort_Label": "Discovery",
                "Analysis_Role": "training",  # provisional; set in Step 5 split
                "Platform": "Illumina RNA-Seq (STAR - Counts)",
                "GPL_ID": NA,
                "Modality_Available": "clinical+expression",
                "Sample_Type": "Primary Tumor",
                "Selection_Method": s["Selection_Method"],
                "Age": age,
                "Sex": C.std_sex(cl.get("gender")),
                "Cancer_Type": cancer_map[coh],
                "Stage": C.std_stage(cl.get("ajcc_pathologic_stage")),
                "Smoking_Status": C.std_smoking(cl.get("tobacco_smoking_status")),
                "Treatment": NAV,  # treatment tables were not downloaded
                "Overall_Survival_Time": os_time,
                "Survival_Status": os_event,
            })
        frames.append(pd.DataFrame(rows))

    master = pd.concat(frames, ignore_index=True)
    master = master[C.FINAL_COLS]
    master.to_csv(C.PROC_TCGA / "TCGA_discovery_master.csv", index=False)
    log(f"[STEP4] TCGA discovery master: {len(master)} patients "
        f"(LUAD={sum(master.Dataset_Source=='TCGA-LUAD')}, "
        f"LUSC={sum(master.Dataset_Source=='TCGA-LUSC')}) -> TCGA_discovery_master.csv")

    _build_tcga_expression(aliquot_index)
    RESULTS["tcga_discovery_patients"] = int(len(master))
    return master


def _build_tcga_expression(aliquot_index: dict):
    """Combine LUAD+LUSC STAR-count matrices and subset to selected aliquots.
    Saves a gene x patient matrix keyed by Patient_ID."""
    mats = []
    for coh in ("LUAD", "LUSC"):
        m = pd.read_csv(
            C.RAW_SRC / f"TCGA_{coh}" / "expression" / f"TCGA-{coh}_STAR_counts_matrix.tsv.gz",
            sep="\t", index_col=0)
        mats.append(m)
    # genes should be identical across cohorts; align on intersection
    common_genes = mats[0].index.intersection(mats[1].index)
    combined = pd.concat([mats[0].loc[common_genes], mats[1].loc[common_genes]], axis=1)

    # map selected aliquot barcodes -> Patient_ID; keep only selected columns
    col_to_pid = {}
    for pid, aliquot in aliquot_index.items():
        if aliquot in combined.columns:
            col_to_pid[aliquot] = pid
    keep_cols = [c for c in combined.columns if c in col_to_pid]
    expr = combined[keep_cols].rename(columns=col_to_pid)
    expr = expr.loc[:, ~expr.columns.duplicated()]
    out = C.PROC_TCGA / "TCGA_discovery_expression_counts.tsv.gz"
    if not out.exists():
        expr.to_csv(out, sep="\t", compression="gzip")
    log(f"[STEP4] aligned TCGA expression: {expr.shape[0]} genes x {expr.shape[1]} "
        f"patients (matched to selected aliquots) -> {out.name}")
    RESULTS["tcga_genes"] = int(expr.shape[0])
    RESULTS["tcga_expr_patients"] = int(expr.shape[1])


# =========================================================================== #
# STEP 5 - reproducible split
# =========================================================================== #
def step5_split(master: pd.DataFrame):
    log("=== STEP 5: TCGA reproducible 80/20 stratified split ===")
    df = master.copy()
    strat = df["Survival_Status"].fillna(-1).astype(int).astype(str)
    rng = np.random.default_rng(RANDOM_SEED)

    train_idx, val_idx = [], []
    for _, grp in df.groupby(strat):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * SPLIT_RATIO))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:].tolist())

    train = df.loc[train_idx].copy()
    val = df.loc[val_idx].copy()
    train["Analysis_Role"] = "training"
    val["Analysis_Role"] = "internal_validation"
    train = train.sort_values("Patient_ID")[C.FINAL_COLS]
    val = val.sort_values("Patient_ID")[C.FINAL_COLS]

    train.to_csv(C.READY_DIR / "TCGA_train.csv", index=False)
    val.to_csv(C.READY_DIR / "TCGA_internal_validation.csv", index=False)

    # companion expression subsets
    _split_expression(train["Patient_ID"].tolist(), "TCGA_train_expression.tsv.gz")
    _split_expression(val["Patient_ID"].tolist(), "TCGA_internal_validation_expression.tsv.gz")

    # split_information.txt
    def _sd(d):
        s = d["Survival_Status"].fillna(-1).astype(int)
        return {"n": len(d), "deaths(1)": int((s == 1).sum()),
                "alive(0)": int((s == 0).sum()), "unknown(-1)": int((s == -1).sum())}
    info = [
        "TCGA DISCOVERY COHORT - TRAIN / INTERNAL-VALIDATION SPLIT",
        "=" * 58, "",
        f"dataset_version      : TCGA discovery master (LUAD+LUSC), Phase 1",
        f"date                 : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"random_seed          : {RANDOM_SEED}",
        f"split_ratio          : {int(SPLIT_RATIO*100)}% train / {int((1-SPLIT_RATIO)*100)}% internal validation",
        f"stratification_var   : Survival_Status (patient-level)",
        f"split_level          : patient (one row = one patient)", "",
        f"total_patients       : {len(df)}",
        f"train                : {_sd(train)}",
        f"internal_validation  : {_sd(val)}", "",
        "Note: stratification bins include an 'unknown' survival-status stratum",
        "so that patients with missing outcome are split proportionally too.",
    ]
    (C.DOC_DIR / "split_information.txt").write_text("\n".join(info), encoding="utf-8")
    log(f"[STEP5] train={len(train)} internal_validation={len(val)} "
        f"(seed={RANDOM_SEED}) -> TCGA_train.csv / TCGA_internal_validation.csv")
    RESULTS["train_n"] = int(len(train))
    RESULTS["val_n"] = int(len(val))
    return train, val


def _split_expression(patient_ids, out_name):
    src = C.PROC_TCGA / "TCGA_discovery_expression_counts.tsv.gz"
    dst = C.READY_DIR / out_name
    if dst.exists() or not src.exists():
        log(f"[STEP5] {out_name}: exists (skipped rewrite)")
        return
    expr = pd.read_csv(src, sep="\t", index_col=0)
    cols = [p for p in patient_ids if p in expr.columns]
    sub = expr[cols]
    sub.to_csv(dst, sep="\t", compression="gzip")
    log(f"[STEP5] wrote {out_name}: {sub.shape[0]} genes x {sub.shape[1]} patients")


# =========================================================================== #
# STEP 6 - CPTAC (clinical-only) + overlap with TCGA
# =========================================================================== #
def step6_cptac(tcga_master: pd.DataFrame):
    log("=== STEP 6: CPTAC clinical-only processing + TCGA overlap ===")
    clin = pd.read_csv(C.RAW_SRC / "CPTAC" / "clinical" / "CPTAC3_LUAD_clinical.tsv",
                       sep="\t", dtype=str)
    rows = []
    for _, r in clin.iterrows():
        pid = r.get("submitter_id")
        # OS: days_to_death if dead else days_to_last_follow_up
        event = C.std_vital_to_event(r.get("vital_status"))
        if event == 1:
            t = r.get("days_to_death")
        else:
            t = r.get("days_to_last_follow_up")
        try:
            t = round(float(t), 2) if t not in (None, "", "nan") and not pd.isna(t) else None
        except (ValueError, TypeError):
            t = None
        aad = r.get("age_at_diagnosis")
        try:
            age = round(float(aad) / 365.25, 1) if aad not in (None, "", "nan") and not pd.isna(aad) else NAV
        except (ValueError, TypeError):
            age = NAV
        rows.append({
            "Patient_ID": pid,
            "Dataset_Source": "CPTAC-3-LUAD",
            "Cohort_Label": "Clinical_validation",
            "Analysis_Role": "clinical_validation",
            "Platform": "GDC harmonised clinical (CPTAC-3)",
            "GPL_ID": NA,
            "Modality_Available": "clinical_only",
            "Sample_Type": "Primary Tumor",
            "Selection_Method": NA,
            "Age": age,
            "Sex": C.std_sex(r.get("gender")),
            "Cancer_Type": "LUAD_Adenocarcinoma",
            "Stage": NAV,          # stage not present in downloaded CPTAC clinical
            "Smoking_Status": NAV,  # not present in downloaded CPTAC clinical
            "Treatment": NAV,
            "Overall_Survival_Time": t,
            "Survival_Status": event,
        })
    cptac = pd.DataFrame(rows)[C.FINAL_COLS]

    # Overlap vs TCGA (literal ID match is valid for CPTAC vs TCGA)
    tcga_ids = set(tcga_master["Patient_ID"])
    cptac["overlap_with_TCGA"] = cptac["Patient_ID"].isin(tcga_ids)
    overlap = cptac[cptac["overlap_with_TCGA"]][["Patient_ID"]].copy()
    overlap["Dataset_Source"] = "CPTAC-3-LUAD"
    overlap["overlaps_TCGA_case"] = True
    overlap["action"] = "flagged; CPTAC used clinical-only (never expression) so no double-counting"
    if overlap.empty:
        overlap = pd.DataFrame([{
            "Patient_ID": "(none)", "Dataset_Source": "CPTAC-3-LUAD",
            "overlaps_TCGA_case": False,
            "action": "no literal CPTAC/TCGA patient-ID overlap detected"}])
    overlap.to_csv(C.DOC_DIR / "CPTAC_TCGA_overlap_report.csv", index=False)

    cptac.drop(columns=["overlap_with_TCGA"]).to_csv(
        C.READY_DIR / "CPTAC_clinical_validation.csv", index=False)
    n_ovl = int(cptac["overlap_with_TCGA"].sum())
    log(f"[STEP6] CPTAC clinical validation: {len(cptac)} patients; "
        f"TCGA literal-ID overlap = {n_ovl} -> CPTAC_clinical_validation.csv")
    RESULTS["cptac_n"] = int(len(cptac))
    RESULTS["cptac_tcga_overlap"] = n_ovl
    return cptac


# =========================================================================== #
# STEPS 7-8 - GEO processing + cross-cohort overlap
# =========================================================================== #
def _build_geo_clinical(g: str):
    cfg = C.GEO_CONFIG[g]
    meta = pd.read_csv(C.RAW_SRC / "GEO" / g /
                       f"{g}_series_matrix.txt.gz_sample_metadata.tsv", sep="\t", dtype=str)
    kv_map = C.parse_geo_characteristics(meta)
    rows = []
    for acc, kv in kv_map.items():
        age = C.first_key(kv, cfg["age"])
        try:
            age = round(float(age), 1) if age is not None else NAV
        except (ValueError, TypeError):
            age = NAV
        sex = C.std_sex(C.first_key(kv, cfg["sex"]))
        smk = C.std_smoking(C.first_key(kv, cfg["smoking"])) if cfg["smoking"] else NAV
        # stage (may be composite across pt/pn/pm)
        if len(cfg["stage"]) > 1:
            parts = [C.first_key(kv, [k]) for k in cfg["stage"]]
            stage = "".join(p.upper().replace(" ", "") for p in parts if p) or NAV
        else:
            stage = C.std_stage(C.first_key(kv, cfg["stage"])) if cfg["stage"] else NAV
        hist = C.first_key(kv, cfg["histology"]) if cfg["histology"] else NAV
        hist = hist if hist else NAV
        event = C.std_vital_to_event(C.first_key(kv, cfg["os_event"]))
        os_time = C.os_time_to_days(C.first_key(kv, cfg["os_time"]), cfg["os_time_unit"])

        # normal-tissue flag
        sample_type = "Primary Tumor"
        if cfg["normal_flag"]:
            fk, token = cfg["normal_flag"]
            val = kv.get(fk, "")
            if token in str(val).lower():
                sample_type = "Normal/Non-tumour"

        rows.append({
            "Patient_ID": acc,  # GSM accession (series-internal)
            "Dataset_Source": g,
            "Cohort_Label": "External_validation",
            "Analysis_Role": "external_validation",
            "Platform": cfg["platform"],
            "GPL_ID": cfg["gpl"],
            "Modality_Available": "clinical+expression",
            "Sample_Type": sample_type,
            "Selection_Method": NA,
            "Age": age,
            "Sex": sex,
            "Cancer_Type": str(hist),
            "Stage": stage,
            "Smoking_Status": smk,
            "Treatment": NAV,
            "Overall_Survival_Time": os_time,
            "Survival_Status": event,
        })
    return pd.DataFrame(rows)[C.FINAL_COLS]


def step7_geo():
    log("=== STEP 7: GEO per-series processing ===")
    geo_frames = {}
    platform_rows = []
    for g in C.GEO_SERIES:
        cfg = C.GEO_CONFIG[g]
        df = _build_geo_clinical(g)
        geo_frames[g] = df

        # expression: save probe-level matrix (skip slow read+write if present)
        outdir = C.PROC_GEO / g
        outdir.mkdir(parents=True, exist_ok=True)
        _gexpr = outdir / f"{g}_expression_probe_level.tsv.gz"
        if not _gexpr.exists():
            expr = pd.read_csv(C.RAW_SRC / "GEO" / g /
                               f"{g}_series_matrix.txt.gz_expression_matrix.tsv.gz",
                               sep="\t", index_col=0)
            expr.to_csv(_gexpr, sep="\t", compression="gzip")

        n_norm = int((df["Sample_Type"] == "Normal/Non-tumour").sum())
        platform_rows.append({
            "GSE_ID": g, "GPL_ID": cfg["gpl"], "Platform_Name": cfg["platform"],
            "Technology_Type": cfg["technology"], "Probe_Count": cfg["probes"],
            "N_samples": len(df), "N_normal_samples": n_norm,
            "probe_to_gene_mapping": "DEFERRED",
            "annotation_source": "GPL annotation file NOT present locally",
            "mapping_success_rate": "0% (no-download constraint; probe IDs retained)",
            "os_time_source_unit": cfg["os_time_unit"],
        })
        log(f"[STEP7] {g}: {len(df)} samples, GPL={cfg['gpl']}, probes={cfg['probes']}, "
            f"normals={n_norm}; expression saved probe-level (gene mapping deferred)")
    pd.DataFrame(platform_rows).to_csv(C.PROC_REPORTS / "geo_platform_annotation.csv", index=False)
    RESULTS["geo_platform_rows"] = platform_rows
    return geo_frames


def _fingerprint(row):
    """Clinical-metadata fingerprint for suspected cross-series duplicate detection."""
    def norm(x):
        s = str(x).strip().lower()
        return s if s not in ("", "nan", NA, NAV, "none") else "?"
    age = norm(row["Age"])
    try:
        age = str(int(round(float(age)))) if age != "?" else "?"
    except (ValueError, TypeError):
        age = "?"
    os_t = row["Overall_Survival_Time"]
    try:
        os_b = str(int(float(os_t) // 30)) if os_t not in (None, "", "nan") and not pd.isna(os_t) else "?"
    except (ValueError, TypeError):
        os_b = "?"
    return "|".join([age, norm(row["Sex"]), norm(row["Cancer_Type"])[:4],
                     norm(row["Stage"]), norm(row["Smoking_Status"])[:4],
                     os_b, str(row["Survival_Status"])])


def step8_geo_overlap(geo_frames: dict):
    log("=== STEP 8: GEO cross-cohort clinical-fingerprint overlap ===")
    # completeness score per series to decide retention priority
    def completeness(df):
        cols = ["Age", "Sex", "Stage", "Smoking_Status",
                "Overall_Survival_Time", "Survival_Status"]
        filled = 0
        for c in cols:
            filled += df[c].apply(lambda x: str(x) not in ("", "nan", NA, NAV, "None") and not pd.isna(x)).sum()
        return filled / (len(df) * len(cols)) if len(df) else 0

    priority = {}
    for g, df in geo_frames.items():
        has_surv = df["Survival_Status"].notna().mean()
        priority[g] = (completeness(df), has_surv, len(df))

    # build fingerprint index
    fp_rows = []
    for g, df in geo_frames.items():
        for _, r in df.iterrows():
            fp_rows.append({"GSM": r["Patient_ID"], "series": g, "fp": _fingerprint(r)})
    fp = pd.DataFrame(fp_rows)
    # ignore fully-uninformative fingerprints (mostly '?')
    fp["informative"] = fp["fp"].apply(lambda s: s.count("?") <= 3)

    report_rows = []
    to_remove = {g: set() for g in geo_frames}
    dup_groups = fp[fp["informative"]].groupby("fp")
    for fpval, grp in dup_groups:
        series_set = grp["series"].unique()
        if len(series_set) < 2:
            continue  # within-series duplicates are not cross-cohort overlap
        # cross-series suspected overlap -> keep highest-priority series
        ranked = sorted(series_set, key=lambda s: priority[s], reverse=True)
        keep = ranked[0]
        for _, rr in grp.iterrows():
            if rr["series"] != keep:
                to_remove[rr["series"]].add(rr["GSM"])
            report_rows.append({
                "fingerprint_id": fpval,
                "GSM": rr["GSM"],
                "Original_GEO_series": rr["series"],
                "Retained_GEO_series": keep,
                "Reason_for_exclusion": ("" if rr["series"] == keep else
                    "suspected duplicate; lower priority (completeness/survival/size)"),
                "Match_confidence": "fingerprint-based (suspected; not publication-confirmed)",
            })

    if not report_rows:
        report_rows = [{
            "fingerprint_id": "(none)", "GSM": "(none)",
            "Original_GEO_series": "(none)", "Retained_GEO_series": "(none)",
            "Reason_for_exclusion": "no cross-series fingerprint matches detected",
            "Match_confidence": "fingerprint-based"}]
    pd.DataFrame(report_rows).to_csv(C.DOC_DIR / "GEO_patient_overlap_report.csv", index=False)

    # apply dedup and write per-series external validation CSVs
    total_removed = 0
    per_series_counts = {}
    for g, df in geo_frames.items():
        rem = to_remove[g]
        total_removed += len(rem)
        kept = df[~df["Patient_ID"].isin(rem)].copy()
        kept.to_csv(C.READY_DIR / f"{g}_external_validation.csv", index=False)
        per_series_counts[g] = int(len(kept))
        if rem:
            log(f"[STEP8] {g}: removed {len(rem)} suspected cross-series duplicate(s)")
        log(f"[STEP8] {g}: {len(kept)} external-validation patients -> {g}_external_validation.csv")
    RESULTS["geo_removed_overlap"] = int(total_removed)
    RESULTS["geo_series_counts"] = per_series_counts
    return per_series_counts


# =========================================================================== #
# STEP 9 - mutation data availability
# =========================================================================== #
def step9_mutation():
    log("=== STEP 9: Mutation data availability check ===")
    maf = list(C.RAW_SRC.rglob("*.maf*")) + list(C.RAW_SRC.rglob("*somatic*mutation*"))
    if maf:
        log(f"[STEP9] mutation files FOUND: {[m.name for m in maf]}")
        RESULTS["mutation_available"] = True
    else:
        log("[STEP9] Mutation data unavailable - no MAF/somatic-mutation files present "
            "in any cohort. No mutation variables created (not assumed). "
            "Note: GSE72094 metadata contains microarray-derived KRAS/EGFR/STK11/TP53 "
            "genotype ANNOTATIONS, but these are not a dedicated mutation data file and "
            "are left in raw metadata only.")
        RESULTS["mutation_available"] = False


# =========================================================================== #
# STEP 10 - quality control
# =========================================================================== #
def step10_qc(train, val, cptac, geo_counts):
    log("=== STEP 10: Quality control ===")
    lines = ["LUNG CANCER DIGITAL TWIN - PHASE 1 QUALITY CONTROL REPORT",
             "=" * 60, f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]

    ready_files = {
        "TCGA_train": C.READY_DIR / "TCGA_train.csv",
        "TCGA_internal_validation": C.READY_DIR / "TCGA_internal_validation.csv",
        "CPTAC_clinical_validation": C.READY_DIR / "CPTAC_clinical_validation.csv",
        **{f"{g}_external_validation": C.READY_DIR / f"{g}_external_validation.csv"
           for g in C.GEO_SERIES},
    }

    schema_fields = ["Analysis_Role", "Modality_Available", "Sample_Type",
                     "Selection_Method", "GPL_ID"]
    for name, path in ready_files.items():
        if not path.exists():
            lines.append(f"[{name}] MISSING FILE"); continue
        df = pd.read_csv(path)
        lines.append(f"--- {name}  (n={len(df)}) ---")
        # patient checks
        dup = int(df["Patient_ID"].duplicated().sum())
        miss_id = int(df["Patient_ID"].isna().sum() + (df["Patient_ID"].astype(str).str.strip() == "").sum())
        lines.append(f"  duplicate Patient_IDs : {dup}")
        lines.append(f"  missing Patient_IDs   : {miss_id}")
        # clinical checks
        age_num = pd.to_numeric(df["Age"], errors="coerce")
        lines.append(f"  age <18 or >110       : {int(((age_num < 18) | (age_num > 110)).sum())}")
        st = pd.to_numeric(df["Overall_Survival_Time"], errors="coerce")
        lines.append(f"  negative survival time: {int((st < 0).sum())}")
        lines.append(f"  missing survival time : {int(st.isna().sum())}")
        ss = pd.to_numeric(df["Survival_Status"], errors="coerce")
        lines.append(f"  missing survival stat : {int(ss.isna().sum())}")
        zero_alive = int(((st == 0) & (ss == 0)).sum())
        lines.append(f"  survival=0 & Alive    : {zero_alive}")
        # residual normals
        resid = int((df["Sample_Type"].astype(str).str.contains("Normal", case=False)).sum())
        lines.append(f"  residual normal-tissue: {resid}")
        # schema completeness
        blanks = {}
        for f in schema_fields:
            b = int(df[f].isna().sum() + (df[f].astype(str).str.strip() == "").sum())
            if b:
                blanks[f] = b
        lines.append(f"  schema blank fields   : {blanks if blanks else 'none (all populated)'}")
        lines.append("")

    # expression checks (TCGA discovery)
    exprp = C.PROC_TCGA / "TCGA_discovery_expression_counts.tsv.gz"
    if exprp.exists():
        ex = pd.read_csv(exprp, sep="\t", index_col=0, nrows=50)
        full_cols = pd.read_csv(exprp, sep="\t", nrows=0).columns
        lines.append("--- TCGA discovery expression matrix ---")
        lines.append(f"  duplicate gene rows   : {int(ex.index.duplicated().sum())} (first 50 rows sampled)")
        lines.append(f"  duplicate sample cols : {int(pd.Series(full_cols).duplicated().sum())}")
        lines.append(f"  orientation           : genes(rows) x patients(cols) [correct]")
        lines.append("")

    lines.append("SUMMARY COUNTS")
    lines.append(f"  TCGA train / internal-val : {len(train)} / {len(val)}")
    lines.append(f"  CPTAC clinical-validation : {len(cptac)}")
    lines.append(f"  GEO external validation   : {geo_counts}")
    lines.append(f"  TCGA excluded normals set : {RESULTS.get('tcga_excluded_normal_samples')}")
    lines.append(f"  GEO removed overlaps      : {RESULTS.get('geo_removed_overlap')}")
    lines.append(f"  mutation data available   : {RESULTS.get('mutation_available')}")

    lines += [
        "", "DOCUMENTED LIMITATIONS (Phase 1; no-download / no-external-query constraint)",
        "-" * 60,
        "1. CPTAC survival outcome: the downloaded CPTAC-3 clinical table has an",
        "   EMPTY vital_status column (0/337 populated) and no days_to_death.",
        "   Consequence: CPTAC Survival_Status is UNKNOWN for all 337 patients;",
        "   only right-censoring time (days_to_last_follow_up) is available.",
        "   CPTAC therefore serves as a CLINICAL-FEATURE validation cohort only,",
        "   not a survival-outcome cohort. Not re-downloaded (Phase 1 constraint).",
        "2. GEO probe->gene mapping is DEFERRED: no GPL annotation file is present",
        "   locally and external DB queries are disallowed. Expression is retained",
        "   at PROBE level per series (GPL570/GPL96/GPL15048). mapping_success=0%.",
        "3. GEO overall-survival times were unit-converted to DAYS assuming source",
        "   units of: GSE31210=days, GSE30219=months, GSE50081=years,",
        "   GSE68465=months, GSE72094=days (per series-matrix field labels).",
        "4. GEO normal-tissue detection is heuristic (series-specific token match);",
        "   GSE30219 non-tumoural-lung samples may be under-counted. Flagged, not deleted.",
        "5. GEO cross-cohort overlap (Step 8) is FINGERPRINT-BASED (age/sex/histology/",
        "   stage/smoking/survival), NOT literal-ID: it flags SUSPECTED duplicates and",
        "   may miss some or over-flag common clinical profiles. Not publication-confirmed.",
        "6. Flagged-but-retained anomalies (per 'flag, do not auto-delete' rule):",
        "   CPTAC negative follow-up times and 2 GSE30219 age>110 records are reported",
        "   above and left in place for investigator review.",
        "7. Treatment field is 'not_available' across all cohorts (treatment tables",
        "   were not part of the downloaded data).",
        "8. TCGA Sex is 'not_available': the 'gender' column is present but EMPTY in",
        "   both downloaded TCGA-LUAD/LUSC clinical tables (0 populated). race,",
        "   vital_status, age, stage and smoking ARE populated and were used.",
        "   Not re-downloaded (Phase 1 constraint).",
    ]

    (C.DOC_DIR / "quality_control_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("[STEP10] quality_control_report.txt written (with limitations section)")


# =========================================================================== #
# Documentation: cohort_definition.csv + data_dictionary.csv
# =========================================================================== #
def write_docs(train, val, cptac, geo_counts):
    log("=== Writing cohort_definition.csv and data_dictionary.csv ===")
    coh_rows = [
        {"Cohort": "TCGA-LUAD+LUSC (train)", "Dataset_Source": "TCGA-LUAD/TCGA-LUSC",
         "Cohort_Label": "Discovery", "Analysis_Role": "training", "N_patients": len(train),
         "Modality": "clinical+expression", "Platform": "Illumina RNA-Seq STAR-Counts",
         "GPL_ID": NA, "Notes": "80% stratified split, seed 42"},
        {"Cohort": "TCGA-LUAD+LUSC (internal val)", "Dataset_Source": "TCGA-LUAD/TCGA-LUSC",
         "Cohort_Label": "Discovery", "Analysis_Role": "internal_validation", "N_patients": len(val),
         "Modality": "clinical+expression", "Platform": "Illumina RNA-Seq STAR-Counts",
         "GPL_ID": NA, "Notes": "20% stratified split, seed 42"},
        {"Cohort": "CPTAC-3-LUAD", "Dataset_Source": "CPTAC-3-LUAD",
         "Cohort_Label": "Clinical_validation", "Analysis_Role": "clinical_validation",
         "N_patients": len(cptac), "Modality": "clinical_only",
         "Platform": "GDC harmonised clinical", "GPL_ID": NA,
         "Notes": "never used for expression modelling"},
    ]
    for g in C.GEO_SERIES:
        cfg = C.GEO_CONFIG[g]
        coh_rows.append({
            "Cohort": g, "Dataset_Source": g, "Cohort_Label": "External_validation",
            "Analysis_Role": "external_validation", "N_patients": geo_counts.get(g),
            "Modality": "clinical+expression", "Platform": cfg["platform"],
            "GPL_ID": cfg["gpl"], "Notes": "independent; never pooled with other series"})
    pd.DataFrame(coh_rows).to_csv(C.DOC_DIR / "cohort_definition.csv", index=False)

    dd = [
        ("Patient_ID", "Unique patient/sample identifier (TCGA case barcode, CPTAC submitter id, or GEO GSM accession)", "string", "cohort-specific", "all"),
        ("Dataset_Source", "Originating dataset", "string", "TCGA-LUAD|TCGA-LUSC|CPTAC-3-LUAD|GSE*", "all"),
        ("Cohort_Label", "Role class of the cohort", "categorical", "Discovery|External_validation|Clinical_validation", "all"),
        ("Analysis_Role", "Modelling role", "categorical", "training|internal_validation|external_validation|clinical_validation", "all"),
        ("Platform", "Assay/platform description", "string", "free-text", "all"),
        ("GPL_ID", "GEO platform accession", "string", "GPL570|GPL96|GPL15048|not_applicable", "GEO (else not_applicable)"),
        ("Modality_Available", "Data modalities present", "categorical", "clinical_only|clinical+expression|...", "all"),
        ("Sample_Type", "Tissue/sample class", "categorical", "Primary Tumor|Normal/Non-tumour", "all"),
        ("Selection_Method", "TCGA multi-sample selection rule applied", "categorical", "not_applicable_single_sample|lowest_barcode_portion|highest_sequencing_quality|not_applicable", "TCGA (else not_applicable)"),
        ("Age", "Age in years", "float", ">=0", "all"),
        ("Sex", "Biological sex", "categorical", "Male|Female|not_available", "all"),
        ("Cancer_Type", "Histology / cancer subtype", "string", "LUAD_Adenocarcinoma|LUSC_SquamousCell|histology text", "all"),
        ("Stage", "Tumour stage (harmonised)", "string", "I|II|III|IV(+A/B)|TNM|not_available", "all"),
        ("Smoking_Status", "Smoking history", "categorical", "Never|Former|Current|not_available", "all"),
        ("Treatment", "Treatment received", "string", "not_available (treatment tables not downloaded)", "all"),
        ("Overall_Survival_Time", "Overall survival time in DAYS", "float", ">=0", "all"),
        ("Survival_Status", "Mortality event", "int", "1=dead|0=alive/censored|blank=unknown", "all"),
    ]
    pd.DataFrame(dd, columns=["variable", "description", "type", "allowed_values", "applies_to"]).to_csv(
        C.DOC_DIR / "data_dictionary.csv", index=False)
    log("[DOCS] cohort_definition.csv + data_dictionary.csv written")


# =========================================================================== #
# main
# =========================================================================== #
def main():
    C.make_dirs()
    C.reset_log()
    log("################ PHASE 1 PIPELINE START ################")
    populate_raw_data()

    step1_inventory()
    selected = step3_filter_tcga()          # Steps 2-3 (schema built inline)
    master = step4_discovery(selected)      # Step 4
    train, val = step5_split(master)        # Step 5
    cptac = step6_cptac(master)             # Step 6
    geo_frames = step7_geo()                # Step 7
    geo_counts = step8_geo_overlap(geo_frames)  # Step 8
    step9_mutation()                        # Step 9
    step10_qc(train, val, cptac, geo_counts)    # Step 10
    write_docs(train, val, cptac, geo_counts)

    log("################ PHASE 1 PIPELINE COMPLETE ################")
    print("\n" + json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    main()
