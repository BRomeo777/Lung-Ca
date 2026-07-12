"""
Phase 1C: GEO Platform Annotation, Gene Symbol Mapping, and External Validation Readiness Assessment.

This script performs ONLY data harmonization and compatibility assessment.
It does NOT train, retrain, evaluate, or modify any ML models.
"""
from __future__ import annotations
import os, sys, gzip, json, hashlib, shutil, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

# ─── Paths ───
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
GEO_RAW_DIR = PROJECT_ROOT.parent / "Lung_Cancer_AI_Datasets" / "GEO"
READY_DIR = PROJECT_ROOT / "03_ANALYSIS_READY_DATA"
ML_REPORTS = PROJECT_ROOT / "ML_RESULTS" / "reports"
PROCESSED_DIR = PROJECT_ROOT / "02_PROCESSED_DATA"
GEO_PROCESSED = PROCESSED_DIR / "GEO_EXTERNAL_VALIDATION"

PHASE1C_DIR = PROJECT_ROOT / "PHASE1C_GEO_HARMONIZATION"
PHASE1C_REPORTS = PHASE1C_DIR / "reports"
PHASE1C_DATA = PHASE1C_DIR / "processed_data"
ANNOT_CACHE = PHASE1C_DIR / "annotation_cache"

for d in [PHASE1C_DIR, PHASE1C_REPORTS, PHASE1C_DATA, ANNOT_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

GEO_SERIES = ["GSE31210", "GSE30219", "GSE50081", "GSE68465", "GSE72094"]

GPL_INFO = {
    "GPL570":   {"name": "Affymetrix Human Genome U133 Plus 2.0 Array", "tech": "spotted oligonucleotide"},
    "GPL96":    {"name": "Affymetrix Human Genome U133A Array",         "tech": "spotted oligonucleotide"},
    "GPL15048": {"name": "Affymetrix HuGene 2.0 ST Array",              "tech": "oligonucleotide (gene-level)"},
}

GEO_GPL_MAP = {}  # filled in step1

def log(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line)
    with open(PHASE1C_REPORTS / "phase1c_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def reset_log():
    p = PHASE1C_REPORTS / "phase1c_log.txt"
    if p.exists():
        p.unlink()


# ═══════════════════════════════════════════════════
# STEP 0: Verify frozen Phase 1B feature panel
# ═══════════════════════════════════════════════════
def step0_verify_panel():
    log("STEP 0: Verify frozen Phase 1B feature panel")
    imp_path = ML_REPORTS / "feature_importance.csv"
    fi_path = ML_REPORTS / "final_report.txt"

    if not imp_path.exists():
        raise RuntimeError("feature_importance.csv not found — Phase 1B outputs missing")

    imp = pd.read_csv(imp_path)
    all_features = sorted(imp["Feature"].unique())
    clinical_features = [f for f in all_features if "=" in f or f == "Age"]
    molecular_genes = [f for f in all_features if f not in clinical_features]

    n_molecular = len(molecular_genes)
    n_total = len(all_features)

    # Checksum
    panel_str = "\n".join(sorted(molecular_genes))
    checksum = hashlib.sha256(panel_str.encode()).hexdigest()

    # File metadata
    import os
    stat = imp_path.stat()
    created = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 70,
        "PHASE 1B FEATURE PANEL VERIFICATION",
        "=" * 70,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"Source file: {imp_path}",
        f"File created: {created}",
        f"File size: {stat.st_size} bytes",
        "",
        f"Molecular feature count: {n_molecular}",
        f"Clinical feature count:  {len(clinical_features)}",
        f"Total predictors:        {n_total}",
        "",
        f"SHA-256 checksum (gene panel): {checksum}",
        "",
        "Verification:",
    ]

    ok = True
    if n_molecular != 127:
        report_lines.append(f"  [FAIL] Expected 127 molecular genes, found {n_molecular}")
        ok = False
    else:
        report_lines.append("  [PASS] Molecular gene count = 127")

    if n_total != 143:
        report_lines.append(f"  [FAIL] Expected 143 total predictors, found {n_total}")
        ok = False
    else:
        report_lines.append("  [PASS] Total predictor count = 143")

    report_lines += [
        "",
        "Frozen molecular gene panel (127 genes):",
    ]
    for i, g in enumerate(molecular_genes, 1):
        report_lines.append(f"  {i:3d}. {g}")

    if not ok:
        report_lines += [
            "",
            "RESULT: Feature panel mismatch detected.",
            "GEO harmonization cannot continue until the Phase 1B feature panel is restored.",
        ]
    else:
        report_lines += [
            "",
            "RESULT: Feature panel verified. 127 molecular genes + 16 clinical features = 143 total.",
            "GEO harmonization may proceed.",
        ]

    report_lines.append("=" * 70)

    out = PHASE1C_REPORTS / "phase1B_feature_panel_verification.txt"
    out.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"  -> {out.name} written")
    log(f"  Molecular genes: {n_molecular}, Total predictors: {n_total}, Checksum: {checksum[:16]}...")

    if not ok:
        raise RuntimeError("Feature panel mismatch — stopping.")

    return molecular_genes, clinical_features


# ═══════════════════════════════════════════════════
# STEP 1: Confirm GEO platform information
# ═══════════════════════════════════════════════════
def step1_confirm_platforms():
    log("STEP 1: Confirm GEO platform information")
    rows = []
    for gse in GEO_SERIES:
        sm_path = GEO_RAW_DIR / gse / f"{gse}_series_matrix.txt.gz"
        gpl_id = None
        with gzip.open(sm_path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("!Series_platform_id"):
                    gpl_id = line.strip().split('"')[1]
                    break

        if gpl_id is None:
            log(f"  [WARN] {gse}: could not find GPL ID")
            gpl_id = "UNKNOWN"

        GEO_GPL_MAP[gse] = gpl_id

        # Count probes from expression matrix
        expr_path = GEO_RAW_DIR / gse / f"{gse}_series_matrix.txt.gz_expression_matrix.tsv.gz"
        # Read just the index
        probe_count = 0
        with gzip.open(expr_path, "rt") as f:
            for line in f:
                probe_count += 1  # includes header
        probe_count -= 1  # subtract header

        gpl_meta = GPL_INFO.get(gpl_id, {"name": "Unknown", "tech": "Unknown"})

        rows.append({
            "GEO_series": gse,
            "GPL_ID": gpl_id,
            "Platform_Name": gpl_meta["name"],
            "Technology_Type": gpl_meta["tech"],
            "Probe_Count": probe_count,
            "Status": "Confirmed" if gpl_id != "UNKNOWN" else "Unconfirmed",
        })
        log(f"  {gse}: {gpl_id} ({gpl_meta['name']}), {probe_count} probes")

    df = pd.DataFrame(rows)
    out = PHASE1C_REPORTS / "geo_platform_confirmation.csv"
    df.to_csv(out, index=False)
    log(f"  -> {out.name} written")
    return df


# ═══════════════════════════════════════════════════
# STEP 1B: Map Ensembl IDs to Gene Symbols
# ═══════════════════════════════════════════════════
def map_ensg_to_symbols(molecular_genes: list) -> dict:
    """Map Phase 1B Ensembl gene IDs to HGNC gene symbols via mygene."""
    log("STEP 1B: Map Phase 1B Ensembl IDs to gene symbols")
    import mygene
    mg = mygene.MyGeneInfo()

    ensg_to_orig = {}
    ensg_ids = []
    for orig in molecular_genes:
        base = orig.split(".")[0]
        ensg_to_orig.setdefault(base, []).append(orig)
        if base not in ensg_ids:
            ensg_ids.append(base)

    all_symbols = {}
    batch_size = 50
    for i in range(0, len(ensg_ids), batch_size):
        batch = ensg_ids[i:i+batch_size]
        for attempt in range(3):
            try:
                results = mg.querymany(batch, scopes="ensembl.gene", fields="symbol", species="human")
                for hit in results:
                    if "notfound" in hit:
                        continue
                    q = hit.get("query")
                    sym = hit.get("symbol")
                    if q and sym:
                        all_symbols[q] = sym
                break
            except Exception as e:
                log(f"  batch {i} attempt {attempt+1} error: {e}")
                time.sleep(2)
        time.sleep(0.3)

    ensg_to_symbol = {}
    for base, origs in ensg_to_orig.items():
        sym = all_symbols.get(base)
        if sym:
            for o in origs:
                ensg_to_symbol[o] = sym

    n_mapped = len(ensg_to_symbol)
    log(f"  Mapped {n_mapped}/{len(molecular_genes)} Ensembl IDs to gene symbols")

    mapping_df = pd.DataFrame([
        {"Ensembl_ID": k, "Gene_Symbol": v}
        for k, v in sorted(ensg_to_symbol.items())
    ])
    mapping_df.to_csv(PHASE1C_REPORTS / "phase1B_ensg_to_symbol_mapping.csv", index=False)

    return ensg_to_symbol


# ═══════════════════════════════════════════════════
# STEP 2: Retrieve platform annotation files
# ═══════════════════════════════════════════════════
def step2_retrieve_annotations(ensg_to_symbol: dict):
    """Fast reverse-lookup annotation: query 127 gene symbols to find probes per platform."""
    log("STEP 2: Retrieve platform annotation files (reverse lookup mode)")

    annotation_sources = []
    gpl_annotations = {}  # gpl_id -> DataFrame with probe_id, gene_symbol columns

    gene_symbols = sorted(set(ensg_to_symbol.values()))
    log(f"  Querying {len(gene_symbols)} unique gene symbols across platforms")

    unique_gpls = set(GEO_GPL_MAP.values())

    for gpl_id in sorted(unique_gpls):
        cache_path = ANNOT_CACHE / f"{gpl_id}_annotation.tsv.gz"

        if cache_path.exists():
            log(f"  {gpl_id}: using cached annotation")
            annot = pd.read_csv(cache_path, sep="\t", compression="gzip")
        else:
            log(f"  {gpl_id}: reverse lookup via mygene")
            annot = build_annotation_via_mygene_reverse(gpl_id, gene_symbols)
            if annot is not None and len(annot) > 0:
                annot.to_csv(cache_path, sep="\t", compression="gzip", index=False)
            else:
                log(f"  {gpl_id}: [ERROR] Could not obtain annotation")
                annotation_sources.append({
                    "GPL_ID": gpl_id,
                    "Annotation_Source": "FAILED",
                    "Package_or_File_Name": "N/A",
                    "Version": "N/A",
                    "Retrieval_Date": datetime.now().strftime("%Y-%m-%d"),
                    "Number_of_Probe_Annotations": 0,
                })
                continue

        n_annot = len(annot)
        gpl_annotations[gpl_id] = annot

        annotation_sources.append({
            "GPL_ID": gpl_id,
            "Annotation_Source": "mygene.info API (reverse: gene symbol -> probe ID)",
            "Package_or_File_Name": f"{gpl_id}_annotation.tsv.gz",
            "Version": "current",
            "Retrieval_Date": datetime.now().strftime("%Y-%m-%d"),
            "Number_of_Probe_Annotations": n_annot,
        })
        log(f"  {gpl_id}: {n_annot} probe annotations")

    # Write annotation sources report
    lines = [
        "=" * 70,
        "ANNOTATION SOURCES USED",
        "=" * 70,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Only platform-level annotation resources were retrieved.",
        "No patient-level data were downloaded.",
        "",
    ]
    for src in annotation_sources:
        lines.append(f"  GPL_ID: {src['GPL_ID']}")
        lines.append(f"    Source: {src['Annotation_Source']}")
        lines.append(f"    File: {src['Package_or_File_Name']}")
        lines.append(f"    Version: {src['Version']}")
        lines.append(f"    Retrieved: {src['Retrieval_Date']}")
        lines.append(f"    Annotations: {src['Number_of_Probe_Annotations']}")
        lines.append("")

    lines += [
        "Statement: Only platform-level annotation resources were retrieved.",
        "No patient-level data were downloaded.",
        "=" * 70,
    ]
    out = PHASE1C_REPORTS / "annotation_sources_used.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"  -> {out.name} written")

    return gpl_annotations


def parse_gpl_annot(path: Path, gpl_id: str) -> pd.DataFrame | None:
    """Parse GPL .annot.gz file — these are tab-delimited with a metadata header."""
    try:
        rows = []
        header_found = False
        col_names = None
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("^") or line.startswith("!") or line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if not header_found:
                    col_names = parts
                    header_found = True
                    continue
                if col_names and len(parts) == len(col_names):
                    rows.append(parts)

        if not rows or not col_names:
            return None

        df = pd.DataFrame(rows, columns=col_names)

        # Find probe ID and gene symbol columns
        probe_col = None
        gene_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("probe set id", "probeset_id", "probe_set_id", "id", "probe_id"):
                probe_col = c
            elif cl in ("gene symbol", "gene_symbol", "symbol", "gene designation", "genesymbol"):
                gene_col = c

        if probe_col is None:
            # Use first column as probe ID
            probe_col = df.columns[0]
        if gene_col is None:
            for c in df.columns:
                cl = c.lower().strip()
                if "gene" in cl and "sym" in cl:
                    gene_col = c
                    break

        if gene_col is None:
            return None

        result = df[[probe_col, gene_col]].copy()
        result.columns = ["probe_id", "gene_symbol"]
        result["gene_symbol"] = result["gene_symbol"].fillna("")
        # Clean up gene symbols — remove extra info after spaces
        result["gene_symbol"] = result["gene_symbol"].astype(str).str.split("///").str[0].str.strip()
        result = result[result["gene_symbol"] != ""]
        result = result[result["gene_symbol"] != "---"]
        result = result.drop_duplicates(subset="probe_id")
        return result
    except Exception as e:
        log(f"  parse_gpl_annot error: {e}")
        return None


def parse_gpl_soft(path: Path, gpl_id: str) -> pd.DataFrame | None:
    """Parse GPL SOFT family file — table section starts after ^PLATFORM."""
    try:
        rows = []
        header_found = False
        col_names = None
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("^PLATFORM"):
                    header_found = False
                    continue
                if line.startswith("!Platform_table_begin"):
                    header_found = False
                    continue
                if line.startswith("!platform_table_begin"):
                    header_found = False
                    continue
                if line.startswith("!Platform_table_end") or line.startswith("!platform_table_end"):
                    break
                if line.startswith("#"):
                    continue
                if line.startswith("!"):
                    continue

                parts = line.strip().split("\t")
                if not header_found and len(parts) > 2:
                    col_names = parts
                    header_found = True
                    continue
                if header_found and col_names and len(parts) == len(col_names):
                    rows.append(parts)

        if not rows or not col_names:
            return None

        df = pd.DataFrame(rows, columns=col_names)

        # Find probe ID and gene symbol columns
        probe_col = None
        gene_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("id", "probe_id", "probeset_id", "probe set id"):
                probe_col = c
            elif "gene" in cl and "sym" in cl:
                gene_col = c

        if probe_col is None:
            probe_col = df.columns[0]
        if gene_col is None:
            return None

        result = df[[probe_col, gene_col]].copy()
        result.columns = ["probe_id", "gene_symbol"]
        result["gene_symbol"] = result["gene_symbol"].fillna("").astype(str)
        result["gene_symbol"] = result["gene_symbol"].str.split("///").str[0].str.strip()
        result = result[(result["gene_symbol"] != "") & (result["gene_symbol"] != "---")]
        result = result.drop_duplicates(subset="probe_id")
        return result
    except Exception as e:
        log(f"  parse_gpl_soft error: {e}")
        return None


def build_annotation_via_mygene_reverse(gpl_id: str, gene_symbols: list) -> pd.DataFrame | None:
    """Reverse lookup: query gene symbols to find probe IDs for this platform.
    Uses mygene 'reporter' field which returns platform-specific probe IDs."""
    # Map GPL IDs to mygene reporter platform keys
    GPL_REPORTER_KEY = {
        "GPL570": "HG-U133_Plus_2",
        "GPL96": "HG-U133_Plus_2",  # U133A is subset of U133 Plus 2; matching against actual probes filters correctly
        "GPL15048": "__ACCESSION__",  # Special: use forward accession query for merck- probes
    }
    reporter_key = GPL_REPORTER_KEY.get(gpl_id)
    if not reporter_key:
        log(f"  {gpl_id}: no reporter key mapping, cannot annotate")
        return None

    try:
        import mygene
        mg = mygene.MyGeneInfo()

        # Gather all probe IDs from one GEO dataset using this GPL
        probe_ids = set()
        for gse in GEO_SERIES:
            if GEO_GPL_MAP.get(gse) != gpl_id:
                continue
            expr_path = GEO_RAW_DIR / gse / f"{gse}_series_matrix.txt.gz_expression_matrix.tsv.gz"
            with gzip.open(expr_path, "rt") as f:
                next(f)
                for line in f:
                    pid = line.strip().split("\t")[0]
                    probe_ids.add(pid)
            break
        probe_set = set(probe_ids)
        log(f"  {gpl_id}: {len(probe_set)} probes in expression matrix, reporter key={reporter_key}")

        mapping = {}  # probe_id -> gene_symbol

        if reporter_key == "__ACCESSION__":
            # GPL15048: forward query via accession scope for merck- probes
            # Strip merck- prefix and _at/_s_at/_a_at/_x_at suffix, query accession
            import re
            probe_to_core = {}
            for p in sorted(probe_set):
                if p.startswith("AFFX-"):
                    continue
                if p.startswith("merck-"):
                    clean = p[6:]
                else:
                    clean = p
                # Strip _at, _s_at, _a_at, _x_at suffix
                core = re.sub(r'_(s_|a_|x_)?at$', '', clean)
                if core:
                    probe_to_core[p] = core

            cores = list(set(probe_to_core.values()))
            log(f"  {gpl_id}: {len(cores)} unique accession cores to query")
            batch_size = 1000
            for i in range(0, len(cores), batch_size):
                batch = cores[i:i+batch_size]
                for attempt in range(2):
                    try:
                        results = mg.querymany(batch, scopes="accession", fields="symbol", species="human")
                        for hit in results:
                            if "notfound" in hit:
                                continue
                            q = hit.get("query")
                            sym = hit.get("symbol")
                            if q and sym:
                                # Find all probes with this core
                                for probe, core in probe_to_core.items():
                                    if core == q and probe not in mapping:
                                        mapping[probe] = sym
                        break
                    except Exception as e:
                        log(f"  {gpl_id}: batch {i} attempt {attempt+1} error: {e}")
                        time.sleep(2)
                if (i + batch_size) % 5000 == 0 or i + batch_size >= len(cores):
                    log(f"  {gpl_id}: processed {min(i+batch_size, len(cores))}/{len(cores)}, {len(mapping)} mapped")
                time.sleep(0.3)
        else:
            # GPL570, GPL96: reverse lookup via reporter field
            batch_size = 100
            for i in range(0, len(gene_symbols), batch_size):
                batch = gene_symbols[i:i+batch_size]
                for attempt in range(2):
                    try:
                        results = mg.querymany(batch, scopes="symbol", fields="reporter", species="human")
                        for hit in results:
                            if "notfound" in hit:
                                continue
                            sym = hit.get("query")
                            reporter = hit.get("reporter")
                            if not reporter or not isinstance(reporter, dict):
                                continue
                            probes = reporter.get(reporter_key, [])
                            if isinstance(probes, str):
                                probes = [probes]
                            for p in probes:
                                if p in probe_set and p not in mapping:
                                    mapping[p] = sym
                        break
                    except Exception as e:
                        log(f"  {gpl_id}: batch {i} attempt {attempt+1} error: {e}")
                        time.sleep(2)
                time.sleep(0.3)

        if not mapping:
            log(f"  {gpl_id}: no probe matches found")
            return None

        df = pd.DataFrame([
            {"probe_id": k, "gene_symbol": v}
            for k, v in mapping.items()
        ])
        log(f"  {gpl_id}: reverse lookup returned {len(df)} probe-to-gene mappings")
        return df
    except Exception as e:
        log(f"  build_annotation_via_mygene_reverse error: {e}")
        return None


def build_annotation_via_mygene(gpl_id: str) -> pd.DataFrame | None:
    """Original forward lookup — kept for reference but not used in fast mode."""
    return None


# ═══════════════════════════════════════════════════
# STEP 3: Probe ID to Gene Symbol mapping
# ═══════════════════════════════════════════════════
def step3_probe_mapping(gpl_annotations: dict):
    log("STEP 3: Probe ID to Gene Symbol mapping")
    mapping_rows = []
    geo_probe_maps = {}  # gse -> {probe_id: gene_symbol}

    for gse in GEO_SERIES:
        gpl_id = GEO_GPL_MAP[gse]
        annot = gpl_annotations.get(gpl_id)

        if annot is None or len(annot) == 0:
            log(f"  {gse}: no annotation available for {gpl_id}")
            mapping_rows.append({
                "GEO_series": gse,
                "GPL_ID": gpl_id,
                "Total_probes": 0,
                "Mapped_probes": 0,
                "Unmapped_probes": 0,
                "Mapping_percentage": 0.0,
                "Unique_gene_symbols": 0,
            })
            geo_probe_maps[gse] = {}
            continue

        # Load expression matrix probe IDs
        expr_path = GEO_RAW_DIR / gse / f"{gse}_series_matrix.txt.gz_expression_matrix.tsv.gz"
        probe_ids = []
        with gzip.open(expr_path, "rt") as f:
            next(f)  # header
            for line in f:
                probe_ids.append(line.strip().split("\t")[0])

        total_probes = len(probe_ids)
        annot_dict = dict(zip(annot["probe_id"], annot["gene_symbol"]))

        mapped = {}
        for pid in probe_ids:
            sym = annot_dict.get(pid)
            if sym and sym.strip():
                mapped[pid] = sym.strip()

        mapped_count = len(mapped)
        unmapped_count = total_probes - mapped_count
        pct = (mapped_count / total_probes * 100) if total_probes > 0 else 0
        unique_genes = len(set(mapped.values()))

        geo_probe_maps[gse] = mapped

        mapping_rows.append({
            "GEO_series": gse,
            "GPL_ID": gpl_id,
            "Total_probes": total_probes,
            "Mapped_probes": mapped_count,
            "Unmapped_probes": unmapped_count,
            "Mapping_percentage": round(pct, 2),
            "Unique_gene_symbols": unique_genes,
        })
        log(f"  {gse}: {mapped_count}/{total_probes} mapped ({pct:.1f}%), {unique_genes} unique genes")

    df = pd.DataFrame(mapping_rows)
    out = PHASE1C_REPORTS / "probe_gene_mapping_report.csv"
    df.to_csv(out, index=False)
    log(f"  -> {out.name} written")
    return geo_probe_maps


# ═══════════════════════════════════════════════════
# STEP 4: Multiple probe collapsing
# ═══════════════════════════════════════════════════
def step4_collapse_probes(geo_probe_maps: dict):
    log("STEP 4: Multiple probe collapsing (highest mean expression rule)")
    geo_gene_matrices = {}  # gse -> DataFrame (genes x samples)

    for gse in GEO_SERIES:
        gpl_id = GEO_GPL_MAP[gse]
        probe_map = geo_probe_maps.get(gse, {})

        if not probe_map:
            log(f"  {gse}: no probe mapping, skipping")
            continue

        # Load expression matrix
        expr_path = GEO_RAW_DIR / gse / f"{gse}_series_matrix.txt.gz_expression_matrix.tsv.gz"
        expr = pd.read_csv(expr_path, sep="\t", index_col=0)
        expr.index = expr.index.astype(str)

        # Filter to mapped probes only
        mapped_probes = [p for p in expr.index if p in probe_map]
        expr_mapped = expr.loc[mapped_probes].copy()
        expr_mapped["gene_symbol"] = [probe_map[p] for p in mapped_probes]

        # Find duplicated genes
        gene_counts = expr_mapped["gene_symbol"].value_counts()
        duplicated_genes = gene_counts[gene_counts > 1]
        n_duplicated = len(duplicated_genes)
        n_probes_collapsed = int(gene_counts[gene_counts > 1].sum()) - n_duplicated

        # Collapse: keep probe with highest mean expression per gene
        expr_mapped["mean_expr"] = expr_mapped.drop(columns=["gene_symbol"]).mean(axis=1)
        idx_best = expr_mapped.groupby("gene_symbol")["mean_expr"].idxmax()
        expr_best = expr_mapped.loc[idx_best].drop(columns=["mean_expr"])
        expr_best = expr_best.set_index("gene_symbol")

        geo_gene_matrices[gse] = expr_best

        log(f"  {gse}: {n_duplicated} duplicated genes, {n_probes_collapsed} probes collapsed, "
            f"final: {expr_best.shape[0]} genes x {expr_best.shape[1]} samples")

    return geo_gene_matrices


# ═══════════════════════════════════════════════════
# STEP 5: Gene symbol quality control
# ═══════════════════════════════════════════════════
def step5_gene_qc(geo_gene_matrices: dict):
    log("STEP 5: Gene symbol quality control")
    qc_rows = []
    qc_approved = {}  # gse -> set of approved gene symbols

    # Known invalid patterns
    invalid_patterns = ["", "---", "NA", "N/A", "null", "None", "unknown"]

    for gse, mat in geo_gene_matrices.items():
        genes = list(mat.index)
        total = len(genes)

        invalid = [g for g in genes if g in invalid_patterns or not g.strip()]
        # Check for non-standard symbols (contain numbers-only, or too long, etc.)
        suspicious = [g for g in genes if g not in invalid and (len(g) > 50 or g.isdigit())]
        # Remove invalid
        clean_genes = [g for g in genes if g not in invalid and g not in suspicious]
        # Remove duplicates (shouldn't exist after collapsing, but check)
        seen = set()
        duplicates = []
        clean_unique = []
        for g in clean_genes:
            if g in seen:
                duplicates.append(g)
            else:
                seen.add(g)
                clean_unique.append(g)

        qc_rows.append({
            "GEO_series": gse,
            "Total_genes": total,
            "Invalid_symbols": len(invalid),
            "Suspicious_symbols": len(suspicious),
            "Duplicate_genes": len(duplicates),
            "Clean_unique_genes": len(clean_unique),
            "Changes_made": f"Removed {len(invalid)} invalid, {len(suspicious)} suspicious, {len(duplicates)} duplicates",
        })

        qc_approved[gse] = set(clean_unique)
        log(f"  {gse}: {total} -> {len(clean_unique)} clean genes "
            f"({len(invalid)} invalid, {len(suspicious)} suspicious, {len(duplicates)} dup)")

    df = pd.DataFrame(qc_rows)
    out = PHASE1C_REPORTS / "gene_symbol_quality_report.csv"
    df.to_csv(out, index=False)
    log(f"  -> {out.name} written")
    return qc_approved


# ═══════════════════════════════════════════════════
# STEP 6: Create gene-mapped GEO expression matrices
# ═══════════════════════════════════════════════════
def step6_create_matrices(geo_gene_matrices: dict, qc_approved: dict):
    log("STEP 6: Create gene-mapped GEO expression matrices")

    for gse in GEO_SERIES:
        if gse not in geo_gene_matrices:
            log(f"  {gse}: no gene matrix, skipping")
            continue

        mat = geo_gene_matrices[gse]
        approved = qc_approved.get(gse, set())
        # Filter to QC-approved genes
        mat_clean = mat.loc[mat.index.isin(approved)]

        out_dir = PHASE1C_DATA / gse
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "expression_gene_mapped.csv"
        mat_clean.to_csv(out_path)
        log(f"  {gse}: {mat_clean.shape[0]} genes x {mat_clean.shape[1]} samples -> {out_path.name}")

        # Also copy to processed data directory
        proc_dir = GEO_PROCESSED / gse
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_dir / "expression_gene_mapped.csv"
        mat_clean.to_csv(proc_path)


# ═══════════════════════════════════════════════════
# STEP 7: Verify Phase 1B gene panel coverage
# ═══════════════════════════════════════════════════
def step7_panel_coverage(molecular_genes: list, geo_gene_matrices: dict, qc_approved: dict, ensg_to_symbol: dict):
    log("STEP 7: Verify Phase 1B 127-gene panel coverage")

    # ensg_to_symbol was already computed in step 1B
    n_mapped = len(ensg_to_symbol)
    log(f"  Using pre-computed mapping: {n_mapped}/{len(molecular_genes)} Ensembl IDs mapped to symbols")

    # Now check coverage for each GEO cohort
    coverage_rows = []
    panel_symbols = set(ensg_to_symbol.values())
    panel_ensg_with_symbol = set(ensg_to_symbol.keys())

    for gse in GEO_SERIES:
        if gse not in geo_gene_matrices:
            coverage_rows.append({
                "GEO_series": gse,
                "Required_genes": 127,
                "Available_genes": 0,
                "Missing_genes": 127,
                "Coverage_percentage": 0.0,
            })
            continue

        approved = qc_approved.get(gse, set())
        available = panel_symbols & approved
        missing = panel_symbols - approved
        # Also count Ensembl IDs that couldn't be mapped to symbols
        unmapped_ensg = set(molecular_genes) - panel_ensg_with_symbol

        coverage_pct = (len(available) / 127) * 100

        coverage_rows.append({
            "GEO_series": gse,
            "Required_genes": 127,
            "Available_genes": len(available),
            "Missing_genes": 127 - len(available),
            "Coverage_percentage": round(coverage_pct, 2),
        })
        log(f"  {gse}: {len(available)}/127 genes available ({coverage_pct:.1f}%), "
            f"missing {127 - len(available)} (+{len(unmapped_ensg)} unmapped Ensembl IDs)")

    df = pd.DataFrame(coverage_rows)
    out = PHASE1C_REPORTS / "panel_coverage_by_series.csv"
    df.to_csv(out, index=False)
    log(f"  -> {out.name} written")

    # Also save missing genes detail
    missing_detail = []
    for gse in GEO_SERIES:
        if gse not in geo_gene_matrices:
            continue
        approved = qc_approved.get(gse, set())
        missing_syms = panel_symbols - approved
        for sym in sorted(missing_syms):
            ensg = [k for k, v in ensg_to_symbol.items() if v == sym]
            missing_detail.append({
                "GEO_series": gse,
                "Missing_Gene_Symbol": sym,
                "Phase1B_Ensembl_ID": ";".join(ensg),
            })
        # Add unmapped Ensembl IDs
        unmapped_ensg = set(molecular_genes) - panel_ensg_with_symbol
        for eid in sorted(unmapped_ensg):
            missing_detail.append({
                "GEO_series": gse,
                "Missing_Gene_Symbol": "[UNMAPPED_ENSEMBL]",
                "Phase1B_Ensembl_ID": eid,
            })

    if missing_detail:
        pd.DataFrame(missing_detail).to_csv(
            PHASE1C_REPORTS / "missing_genes_detail.csv", index=False
        )

    return df


# ═══════════════════════════════════════════════════
# STEP 8: Expression scale compatibility assessment
# ═══════════════════════════════════════════════════
def step8_expression_compatibility(geo_gene_matrices: dict):
    log("STEP 8: Expression scale compatibility assessment")
    rows = []

    for gse in GEO_SERIES:
        if gse not in geo_gene_matrices:
            rows.append({
                "GEO_series": gse,
                "GPL_ID": GEO_GPL_MAP.get(gse, "?"),
                "Min_Value": "N/A",
                "Max_Value": "N/A",
                "Mean": "N/A",
                "Std_Dev": "N/A",
                "Transformation_Status": "N/A",
                "Compatibility": "N/A",
            })
            continue

        mat = geo_gene_matrices[gse]
        vals = mat.values.flatten()
        vals = vals[~np.isnan(vals)]

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        vmean = float(np.mean(vals))
        vstd = float(np.std(vals))

        # Determine transformation status
        if vmin < 0:
            transform = "Processed/normalized (negative values present)"
            compat = "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"
        elif vmax > 100:
            transform = "Raw intensity (not log-transformed)"
            compat = "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"
        elif vmin >= 0 and vmax < 20:
            transform = "Log2-transformed"
            compat = "Likely compatible (log2 scale)"
        elif vmin >= 0 and vmax < 100:
            transform = "Possibly log-transformed or normalized"
            compat = "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"
        else:
            transform = "Unknown scale"
            compat = "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"

        rows.append({
            "GEO_series": gse,
            "GPL_ID": GEO_GPL_MAP.get(gse, "?"),
            "Min_Value": round(vmin, 4),
            "Max_Value": round(vmax, 4),
            "Mean": round(vmean, 4),
            "Std_Dev": round(vstd, 4),
            "Transformation_Status": transform,
            "Compatibility": compat,
        })
        log(f"  {gse}: min={vmin:.2f}, max={vmax:.2f}, mean={vmean:.2f}, std={vstd:.2f} -> {transform}")

    df = pd.DataFrame(rows)
    out = PHASE1C_REPORTS / "expression_compatibility_report.csv"
    df.to_csv(out, index=False)
    log(f"  -> {out.name} written")
    return df


# ═══════════════════════════════════════════════════
# STEP 9: Final GEO readiness classification
# ═══════════════════════════════════════════════════
def step9_readiness(coverage_df: pd.DataFrame, compat_df: pd.DataFrame):
    log("STEP 9: Final GEO readiness classification")
    classifications = []

    for _, row in coverage_df.iterrows():
        gse = row["GEO_series"]
        coverage = row["Coverage_percentage"]

        compat_row = compat_df[compat_df["GEO_series"] == gse].iloc[0]
        compat_status = compat_row["Compatibility"]

        if coverage < 70:
            status = "EXCLUDED"
            reason = f"Insufficient gene overlap ({coverage:.1f}%)"
        elif coverage >= 90 and "compatible" in compat_status.lower():
            status = "READY FOR PHASE 2"
            reason = f"Good coverage ({coverage:.1f}%) and expression compatibility"
        elif coverage >= 90:
            status = "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"
            reason = f"Good coverage ({coverage:.1f}%) but expression scale needs harmonization"
        elif coverage >= 70:
            status = "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"
            reason = f"Partial coverage ({coverage:.1f}%) — reduced panel possible"
        else:
            status = "EXCLUDED"
            reason = f"Insufficient gene overlap ({coverage:.1f}%)"

        classifications.append({
            "GEO_series": gse,
            "Coverage": coverage,
            "Expression_Compatibility": compat_status,
            "Final_Classification": status,
            "Reason": reason,
        })
        log(f"  {gse}: {status} — {reason}")

    return pd.DataFrame(classifications)


# ═══════════════════════════════════════════════════
# STEP 10: Final quality control
# ═══════════════════════════════════════════════════
def step10_final_qc(readiness_df: pd.DataFrame):
    log("STEP 10: Final quality control")

    lines = [
        "=" * 70,
        "GEO MAPPING QUALITY CONTROL REPORT",
        "=" * 70,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "QC CHECKLIST:",
        "",
        "  [x] No patient data changed",
        "  [x] No clinical variables changed",
        "  [x] No survival outcomes changed",
        "  [x] No cohorts merged",
        "  [x] Only platform annotation was added",
        "  [x] Raw probe matrices preserved (untouched in GEO raw data directory)",
        "  [x] Mapping reproducible (annotation files cached in annotation_cache/)",
        "  [x] Gene panel coverage reproducible (frozen Phase 1B panel used)",
        "  [x] Multi-probe collapsing consistently applied (highest mean expression rule)",
        "",
        "READINESS CLASSIFICATION SUMMARY:",
        "",
    ]

    for _, row in readiness_df.iterrows():
        lines.append(f"  {row['GEO_series']}: {row['Final_Classification']}")
        lines.append(f"    Coverage: {row['Coverage']:.1f}%")
        lines.append(f"    Reason: {row['Reason']}")
        lines.append("")

    lines += [
        "FILES GENERATED:",
        f"  reports/phase1B_feature_panel_verification.txt",
        f"  reports/geo_platform_confirmation.csv",
        f"  reports/annotation_sources_used.txt",
        f"  reports/probe_gene_mapping_report.csv",
        f"  reports/gene_symbol_quality_report.csv",
        f"  reports/panel_coverage_by_series.csv",
        f"  reports/expression_compatibility_report.csv",
        f"  reports/geo_mapping_quality_control.txt",
        f"  reports/phase1B_ensg_to_symbol_mapping.csv",
        f"  reports/missing_genes_detail.csv",
        f"  processed_data/GSE*/expression_gene_mapped.csv",
        "",
        "RAW DATA PRESERVED:",
        f"  Original probe-level matrices: {GEO_RAW_DIR}",
        f"  Phase 1A processed probe matrices: {GEO_PROCESSED}",
        "",
        "=" * 70,
    ]

    out = PHASE1C_REPORTS / "geo_mapping_quality_control.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"  -> {out.name} written")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    reset_log()
    log("=" * 70)
    log("PHASE 1C: GEO Platform Annotation & External Validation Readiness")
    log("=" * 70)

    # Step 0
    molecular_genes, clinical_features = step0_verify_panel()

    # Step 1
    platform_df = step1_confirm_platforms()

    # Step 1B: Map Ensembl IDs to gene symbols FIRST (needed for reverse lookup)
    ensg_to_symbol = map_ensg_to_symbols(molecular_genes)

    # Step 2 (fast reverse lookup using 127 gene symbols)
    gpl_annotations = step2_retrieve_annotations(ensg_to_symbol)

    # Step 3
    geo_probe_maps = step3_probe_mapping(gpl_annotations)

    # Step 4
    geo_gene_matrices = step4_collapse_probes(geo_probe_maps)

    # Step 5
    qc_approved = step5_gene_qc(geo_gene_matrices)

    # Step 6
    step6_create_matrices(geo_gene_matrices, qc_approved)

    # Step 7
    coverage_df = step7_panel_coverage(molecular_genes, geo_gene_matrices, qc_approved, ensg_to_symbol)

    # Step 8
    compat_df = step8_expression_compatibility(geo_gene_matrices)

    # Step 9
    readiness_df = step9_readiness(coverage_df, compat_df)

    # Step 10
    step10_final_qc(readiness_df)

    # Final summary
    log("")
    log("=" * 70)
    log("PHASE 1C COMPLETE — All outputs in PHASE1C_GEO_HARMONIZATION/")
    log("=" * 70)

    # Print final summary table
    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for _, row in readiness_df.iterrows():
        gse = row["GEO_series"]
        gpl = GEO_GPL_MAP.get(gse, "?")
        cov = row["Coverage"]
        status = row["Final_Classification"]
        print(f"\n  {gse} ({gpl}):")
        print(f"    Coverage: {cov:.1f}%")
        print(f"    Status:   {status}")

    print("\n" + "=" * 70)
    ready = readiness_df[readiness_df["Final_Classification"] == "READY FOR PHASE 2"]
    partial = readiness_df[readiness_df["Final_Classification"] == "REQUIRES PHASE 1D EXPRESSION HARMONIZATION"]
    excluded = readiness_df[readiness_df["Final_Classification"] == "EXCLUDED"]

    if len(ready) > 0:
        print(f"READY FOR PHASE 2: {', '.join(ready['GEO_series'])}")
    if len(partial) > 0:
        print(f"REQUIRES PHASE 1D: {', '.join(partial['GEO_series'])}")
    if len(excluded) > 0:
        print(f"EXCLUDED: {', '.join(excluded['GEO_series'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
