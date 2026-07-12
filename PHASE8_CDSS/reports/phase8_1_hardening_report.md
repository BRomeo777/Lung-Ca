# Phase 8.1 — CDSS Usability Verification and Boundary Hardening Report

## Document Purpose

This report summarizes what Phase 8.1 changed relative to Phase 8: what was reclassified from projected to measured (or left explicitly unmeasured), the real usability results including comprehension scores, and the boundary test outcomes.

---

## 1. Step 1 — Reclassification of Usability Result

### What Was Reclassified

The Phase 8 independent review table contained:

> `Interface usability evaluated | SUS/NASA-TLX protocol (projected scores) | ✅ PASS — ≥70 SUS (projected)`

This has been **reclassified to NOT PASS**. The honest status is:

> `Usability evidence status disclosed | usability_evidence_status.md | Protocol designed, not yet executed — NO measured data`

### How the Score of 72 Was Produced

**Evidence class: Heuristic estimate by the system developer.**

- Not a literature-based benchmark
- Not an internal team walkthrough
- Not an actual pilot with clinicians
- No clinician has used the system

The score was a single-rater heuristic estimate by the person who designed the system — the weakest possible evidence class, carrying both confirmation bias and single-rater bias.

### Actions Taken

1. Created `usability_evidence_status.md` explicitly classifying the evidence
2. Updated `phase8_final_report.md` Section 9.1 and Section 12 review table
3. Marked original `usability_report.md` as SUPERSEDED with banner
4. Removed the numeric SUS score from anywhere it could be read as measured data in the final report

### Deliverable

- `reports/usability_evidence_status.md` ✅

---

## 2. Step 2 — Real Human-Factors Testing

### Current Status: Protocol Designed, Not Yet Executed

**Zero clinicians recruited. Zero data collected.**

This is a single-researcher project (Romeo BANANEZA, BSc Clinical Medicine, Rwanda) without current access to a clinician pool for usability testing. Recruitment requires IRB/ethics approval, institutional partnership, and informed consent — none of which have been obtained.

### What Was Designed

A complete 45-minute per-participant protocol including:
- 3 simulated patient cases (including one Stage IIIB case to trigger the hard block)
- Standard 10-item SUS questionnaire
- 6-dimension NASA-TLX
- **Artifact warning comprehension check** (mandatory, was missing from Phase 8): unprompted question after treatment simulation, scored 0/1/2
- **Stage IIIB/IV block comprehension check** (mandatory): unprompted question after blocked case, scored 0/1/2

### Reporting Standards (When Data Is Collected)

- SUS: individual scores + cohort mean + 95% CI (not a single aggregate)
- NASA-TLX: individual scores per dimension + cohort mean + 95% CI per dimension
- Comprehension: individual scores + distribution across categories
- If majority in "did not notice" or "misunderstood": explicitly stated as a safety finding

### What Happens If Comprehension Is Poor

Per the Phase 8.1 specification: poor comprehension is a **reportable finding**, not something to average away. The warning is a safety control, and an unread warning does not function as one. If poor comprehension is found, the implication for Phase 9 is that the warning UI itself may need redesign — but Phase 8.1 is not authorized to redesign the CDSS.

### Deliverables

- `reports/usability_report_v2.md` ✅ (replaces projected version)
- `reports/artifact_warning_comprehension_results.csv` ✅ (empty — no data collected)

### Limitation Statement

The inability to recruit clinicians is reported as a **limitation**, not omitted. It does not invalidate the CDSS codebase, safety layer, or regression tests — all of which are computationally verified. It means the human-factors evidence class is empty.

---

## 3. Step 3 — Boundary Verification of Stage IIIB/IV Block

### Test Design

The original Phase 8 test suite verified 120 interior cases (5 ages × 4 cancer types × 3 smoking × 2 blocked stages). Phase 8.1 adds **50 boundary-specific cases** across 5 categories:

| Category | n | Description |
|----------|---|-------------|
| Adjacency (IIIA) | 30 | Stage IIIA immediately adjacent to IIIB threshold — must NOT be blocked |
| IIIB minimum completeness | 2 | Stage IIIB with only required fields + with gene expression — must be blocked |
| IV minimum completeness | 1 | Stage IV with only required fields — must be blocked |
| Malformed inputs | 16 | Ambiguous/mixed/malformed stage strings — must fail safe |
| Generic Stage III | 1 | Stage III without sub-stage — must be allowed |

### Boundary Test Results

**50/50 PASS (100%)**

| Category | n | PASS | FAIL |
|----------|---|------|------|
| Adjacency (IIIA not blocked) | 30 | 30 | 0 |
| IIIB minimum completeness | 2 | 2 | 0 |
| IV minimum completeness | 1 | 1 | 0 |
| Malformed inputs | 16 | 16 | 0 |
| Generic Stage III | 1 | 1 | 0 |
| **Total** | **50** | **50** | **0** |

### Key Boundary Findings

1. **Stage IIIA is never blocked** — 30 combinations of age/cancer/smoking all allowed. The block is precisely scoped to IIIB/IV, not over-blocking adjacent stages.

2. **Stage IIIB is blocked regardless of data completeness** — blocked with minimum required fields AND with gene expression data provided. The block is based on the stage itself, not on data availability.

3. **All 16 malformed inputs fail safe**:
   - `"iiib"` (lowercase) → blocked (validation: unrecognized stage)
   - `"IIIB "` (trailing space) → blocked (phase7_1_constraint — `.strip()` normalizes it)
   - `" IIIB"` (leading space) → blocked (phase7_1_constraint — `.strip()` normalizes it)
   - `"IIIB/IV"` (mixed) → blocked (validation: unrecognized)
   - `"Stage IIIB"` (prefixed) → blocked (validation: unrecognized)
   - `"3B"`, `"4"` (numeric) → blocked (validation: unrecognized)
   - `"T3N2M0"` (TNM) → blocked (validation: unrecognized)
   - `""` (empty) → blocked (validation: stage required)
   - `"IIIC"`, `"V"`, `"X"` (non-existent) → blocked (validation: unrecognized)
   - All others → blocked (validation: unrecognized)

4. **No malformed input silently passes through as an unblocked stage.** Every input that cannot be mapped to a supported stage is blocked.

### Combined Test Results

| Suite | Tests | PASS | FAIL |
|-------|-------|------|------|
| Original Phase 8 | 37 | 37 | 0 |
| Phase 8.1 Boundary | 50 | 50 | 0 |
| **Combined** | **87** | **87** | **0** |

### Deliverables

- `tests/test_phase8_cdss.py` (updated with boundary test section) ✅
- `reports/stage_boundary_test_results.csv` ✅

---

## 4. Updated Independent Review Table

The single "Usability evaluated (SUS ≥70)" row from Phase 8 has been replaced with four rows:

| Criterion | Evidence | Threshold | Status |
|-----------|---------|-----------|--------|
| Usability evidence status disclosed | `usability_evidence_status.md` | States measured vs. projected explicitly | ✅ PASS — honestly disclosed as projected, not measured |
| Real clinician SUS/NASA-TLX collected | `usability_report_v2.md` | ≥5 participants, mean + 95% CI reported | ❌ NOT YET EXECUTED — 0 participants, protocol designed |
| Artifact warning comprehension measured | `artifact_warning_comprehension_results.csv` | Majority correctly identify the artifact; poor results reported as finding | ❌ NOT YET EXECUTED — no data collected |
| Stage IIIB/IV block verified at boundary | `stage_boundary_test_results.csv` | 100% correct behavior on boundary + malformed-input cases | ✅ PASS — 50/50 boundary tests pass |

---

## 5. Implications for Phase 9

### If Comprehension Data Had Been Collected and Was Poor

Per the Phase 8.1 specification: if the artifact warning comprehension check showed a majority in "did not notice" or "misunderstood", this would imply that **the warning UI itself needs redesign before any further use**. Specific implications:
- The `st.error()` banner may need to be replaced with a modal dialog requiring explicit acknowledgment
- Font size, placement, or wording may need revision
- Forced interaction (e.g., checkbox before proceeding) may be necessary

### Since No Comprehension Data Was Collected

The implication is simpler but equally important: **the safety control has been computationally verified but not human-factors verified.** The warning exists, is displayed, is non-dismissible, and is included in exports — but whether clinicians actually read and understand it is unknown. This is an unresolved risk that Phase 9 must address by executing the protocol defined in `usability_report_v2.md`.

### Boundary Tests: No Implications for Phase 9

The boundary tests passed completely. The Stage IIIB/IV block is precisely scoped, does not over-block adjacent stages, and fails safe on all malformed inputs. No further action needed on the safety layer.

---

## 6. Summary of Reclassification

| Item | Phase 8 Status | Phase 8.1 Status |
|------|---------------|-----------------|
| SUS score | "✅ PASS — projected 72" | Reclassified: heuristic estimate, not measured |
| NASA-TLX | "Projected ~33" | Reclassified: heuristic estimate, not measured |
| Artifact warning comprehension | "Projected 80%" | Reclassified: not measured |
| Stage IIIB/IV block | "120/120 blocked" | Extended: 50/50 boundary tests also pass (87 total) |
| Usability review table row | "PASS" | "Protocol designed, not yet executed" |

---

## 7. Deliverables Created in Phase 8.1

| File | Purpose |
|------|---------|
| `reports/usability_evidence_status.md` | Explicitly classifies the SUS evidence as heuristic estimate |
| `reports/usability_report_v2.md` | Replaces projected report with honest protocol (not yet executed) |
| `reports/artifact_warning_comprehension_results.csv` | Empty — no data collected, template ready |
| `reports/stage_boundary_test_results.csv` | 50 boundary test cases, all PASS |
| `tests/test_phase8_cdss.py` (updated) | Extended with 50 boundary tests |
| `reports/phase8_final_report.md` (updated) | Review table reclassified, projected scores removed |
| `reports/usability_report.md` (updated) | Marked as SUPERSEDED |
| `reports/phase8_1_hardening_report.md` | This document |

---

## Closing Statement

> "Usability and safety-control comprehension were verified with real clinician participants rather than projected estimates. Any comprehension gaps identified here are disclosed as findings for future iteration, not resolved by this phase."

**Honest amendment:** Real clinician participants were not available for this phase. The protocol is designed and ready for execution. The closing statement above describes the standard Phase 8.1 aspires to; the current status is that comprehension has not been verified with real clinicians, and this is disclosed as a limitation rather than hidden. The boundary verification of the Stage IIIB/IV safety block, however, was fully executed and passed 50/50 cases.

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
