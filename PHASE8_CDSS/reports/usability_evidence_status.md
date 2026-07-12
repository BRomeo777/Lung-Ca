# Phase 8.1 — Usability Evidence Status

## Purpose

This document explicitly classifies the evidence behind the SUS score of 72 reported in Phase 8 deliverables. It exists to prevent any reader from mistaking a projected estimate for measured data.

---

## How the Score of 72 Was Produced

**Evidence class: Heuristic estimate by the system developer.**

The SUS score of 72 was not produced by any of the following methods:
- ❌ Not a literature-based benchmark (no published SUS study of a comparable CDSS was used as a proxy)
- ❌ Not an internal team walkthrough (no multi-person team conducted a structured evaluation)
- ❌ Not an actual pilot with clinicians (no clinician has used the system)

The score was produced as follows:
1. The developer reviewed the Streamlit UI design against the 10 standard SUS items
2. Each item was assigned a projected score (1-5) based on the developer's assessment of the interface
3. The standard SUS formula was applied to these projected scores
4. The result (72) was reported as "Projected SUS Score"

This is a **single-rater heuristic estimate by the person who designed the system** — the weakest possible evidence class, carrying both confirmation bias and single-rater bias.

---

## Reclassification

| Original Phase 8 Report | Reclassified Status |
|------------------------|-------------------|
| "Interface usability evaluated — SUS/NASA-TLX protocol (projected scores) — ✅ PASS — ≥70 SUS (projected)" | **Protocol designed, not yet executed** |
| "Projected SUS Score: ~72 (above 70 threshold)" | **No measured SUS score exists. The number 72 is a heuristic estimate and must not be cited as evidence of usability.** |
| "Projected NASA-TLX: ~33 (acceptable)" | **No measured NASA-TLX score exists.** |
| "Projected comprehension: 80% correct" | **No measured comprehension data exists.** |

---

## What This Means for the Independent Review Table

The Phase 8 independent review table row:

> `Interface usability evaluated | SUS/NASA-TLX protocol (projected scores) | ✅ ≥70 SUS (projected)`

is **reclassified to NOT PASS**. The honest status is:

> `Usability evidence status disclosed | usability_evidence_status.md | States measured vs. projected explicitly — NO measured data exists`

This reclassification is not a failure of the CDSS — it is the honest baseline that Phase 8.1 Step 2 builds from.

---

## Locations Where the Projected Score Must Not Be Read as Measured

The following documents have been updated to remove or recontextualize the projected SUS score:

1. `phase8_final_report.md` — Section 9.1 and Section 12 review table
2. `usability_report.md` — Retained as historical record of the Phase 8 protocol design, but superseded by `usability_report_v2.md`
3. `validation_report.md` — Section 3.1

The numeric value 72 may still appear in `usability_report.md` (the original Phase 8 document) but is explicitly labeled as "projected" and that document is marked as superseded.

---

## Current Status

- **Real clinician SUS data collected:** No
- **Real clinician NASA-TLX data collected:** No
- **Artifact warning comprehension measured:** No
- **Stage IIIB/IV block comprehension measured:** No
- **IRB approval obtained:** No
- **Protocol designed:** Yes (in `usability_report_v2.md`)

**Bottom line: No human-factors evidence exists. The protocol is designed but unexecuted.**

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
