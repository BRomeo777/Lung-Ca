# Phase 8 CDSS — Usability Report

> **⚠️ SUPERSEDED by `usability_report_v2.md` (Phase 8.1)**
> This document is retained as a historical record of the Phase 8 protocol design.
> All projected scores in this document are heuristic estimates by the system developer,
> not measured data. No clinician has used the system as of this writing.
> See `usability_evidence_status.md` for the full reclassification.

## Evaluation Framework

This report documents the usability evaluation protocol for the Phase 8 CDSS. Full clinician evaluation requires IRB approval and prospective enrollment. The protocol and projected outcomes are documented here for completeness.

## 1. System Usability Scale (SUS)

### Protocol
- **Participants:** Minimum 10 clinicians (oncologists, pulmonologists, clinical researchers)
- **Task:** Complete a full patient workflow (input → predict → simulate → explain → export)
- **Survey:** 10-item SUS questionnaire administered post-task
- **Scoring:** Standard SUS calculation (0-100, ≥70 = acceptable)

### Projected Results
Based on system design analysis:
- **Projected SUS Score:** 72 (above 70 threshold)
- **Strengths:** Clear navigation, persistent disclaimers, inline warnings
- **Weaknesses:** Multiple pages for a single workflow may increase cognitive load

### Justification if <70
If the projected score were below 70, the following mitigations would be recommended:
1. Consolidate workflow into fewer pages (single-page dashboard)
2. Add guided tour / onboarding tutorial
3. Simplify export options

## 2. NASA Task Load Index (NASA-TLX)

### Protocol
- **Same participants as SUS**
- **Dimensions:** Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration
- **Scoring:** 0-100 per dimension, weighted average

### Projected Results

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Mental Demand | 35 | Multiple outputs (risk, survival, trajectory, explanation) require interpretation |
| Physical Demand | 10 | Computer-based, minimal physical effort |
| Temporal Demand | 20 | Predictions complete in <10 seconds |
| Performance | 70 | System produces correct outputs with appropriate warnings |
| Effort | 40 | Understanding calibration and artifact warnings requires attention |
| Frustration | 25 | Clear interface, but blocked stages may frustrate some users |

**Overall TLX:** ~33 (acceptable)

## 3. Treatment-Simulation Artifact Warning Comprehension

### This is a scored item, not an assumption.

### Protocol
1. Clinician views a treatment simulation (e.g., chemotherapy vs immunotherapy)
2. Post-task question: "What does the artifact warning shown above the treatment comparison mean?"
3. Scoring:
   - **Correct:** Mentions that the model shows benefit for all therapies due to missing resistance/toxicity/discontinuation
   - **Partial:** Recognizes the warning but cannot explain the specific limitation
   - **Incorrect:** Does not notice the warning or misinterprets it

### Implementation Details
- **Warning placement:** `st.error()` banner directly below the "Simulation Result" header
- **Warning content:** Full `TREATMENT_ARTIFACT_WARNING` text (non-dismissible)
- **Warning in exports:** Included verbatim in simulation report TXT export
- **Warning on comparison plots:** Additional `st.error()` below the comparison bar chart

### Projected Comprehension Rate
- **Correct interpretation:** 80% (warning is prominently displayed in red)
- **Partial:** 15%
- **Incorrect:** 5%
- **Design rationale:** Red error box + persistent display + verbatim text in exports maximizes visibility

## 4. Recommendations for Full Clinical Evaluation

1. Obtain IRB approval for clinician usability study
2. Enroll ≥10 clinicians from diverse specialties (medical oncology, radiation oncology, thoracic surgery)
3. Include both tech-savvy and tech-naive participants
4. Record screen interactions to verify warning visibility
5. Administer SUS, NASA-TLX, and artifact warning comprehension assessment
6. Collect qualitative feedback on workflow integration
7. Target SUS ≥70; if not achieved, iterate on UI design

## 5. Limitations of This Assessment

- Scores are projected based on design analysis, not actual clinician feedback
- No real interaction data to verify warning comprehension
- Sample size not yet determined (minimum 10 recommended)
- Cultural and workflow differences not assessed

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
