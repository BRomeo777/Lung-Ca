# Phase 8.1 CDSS — Usability Report v2

## Status: Protocol Designed, Not Yet Executed

**No clinician has used this system.** No SUS, NASA-TLX, or comprehension data has been collected. This document replaces the projected `usability_report.md` and defines the protocol to be executed when clinician participants are available.

---

## 1. Evidence Classification

| Item | Phase 8 Status (reclassified) | Phase 8.1 Status |
|------|------------------------------|-----------------|
| SUS score | Heuristic estimate (72) — not measured | Not measured — protocol designed |
| NASA-TLX | Heuristic estimate (33) — not measured | Not measured — protocol designed |
| Artifact warning comprehension | Heuristic estimate (80%) — not measured | Not measured — protocol designed |
| Stage IIIB/IV block comprehension | Not assessed | Not measured — protocol designed |
| Clinicians recruited | 0 | 0 |
| IRB approval | Not obtained | Not obtained |

---

## 2. Participant Recruitment

### Target
- **Minimum:** 5 clinicians
- **Ideal:** 10 clinicians
- **Specialties:** Medical oncology, radiation oncology, thoracic surgery, pulmonology, clinical research
- **Experience levels:** Mix of senior and trainee

### Current Recruitment Status
**Zero clinicians recruited.** This is a single-researcher project (Romeo BANANEZA, BSc Clinical Medicine, Rwanda) without current access to a clinician pool for usability testing. Recruitment requires:
1. IRB or ethics committee approval
2. Institutional partnership for clinician access
3. Informed consent process
4. Time allocation for participants (~45 minutes per session)

### Limitation Statement
The inability to recruit clinicians is reported as a **limitation**, not omitted. It does not invalidate the CDSS codebase, safety layer, or regression tests — all of which are computationally verified. It means the human-factors evidence class is empty.

---

## 3. Protocol

### 3.1 Session Structure (45 minutes per participant)

| Phase | Duration | Activity |
|-------|----------|----------|
| Introduction | 5 min | Read Home page, review disclaimer |
| Patient Input | 5 min | Select existing patient or enter new patient |
| Prediction | 5 min | View risk score, survival curve, calibration warning |
| Digital Twin | 10 min | Run treatment simulation, view artifact warning |
| Explainability | 5 min | Review feature contributions |
| Export | 5 min | Generate JSON/CSV/TXT export |
| SUS Questionnaire | 5 min | 10-item standard SUS |
| NASA-TLX | 5 min | 6-dimension TLX |
| Comprehension Checks | 5 min | Artifact warning + Stage block comprehension |

### 3.2 Simulated Patient Cases

Each participant works through 3 cases:

| Case | Patient | Stage | Purpose |
|------|---------|-------|---------|
| 1 | TCGA-05-4249 | IB | Normal workflow, early-stage |
| 2 | New patient | IIA | New patient entry, intermediate stage |
| 3 | New patient | IIIB | **Must trigger hard block** — test block comprehension |

### 3.3 SUS Questionnaire (Standard 10-item)

Administered post-task. Standard SUS scoring (0-100, ≥70 = acceptable).

1. I would use this system frequently.
2. The system is unnecessarily complex.
3. The system is easy to use.
4. I need technical support to use this system.
5. The functions are well integrated.
6. There is too much inconsistency.
7. Most people would learn this system quickly.
8. The system is cumbersome to use.
9. I feel confident using the system.
10. I need to learn a lot before using this.

### 3.4 NASA-TLX (6 dimensions)

Administered post-task. Standard TLX scoring (0-100 per dimension).

- Mental Demand
- Physical Demand
- Temporal Demand
- Performance
- Effort
- Frustration

### 3.5 Artifact Warning Comprehension Check (Mandatory)

**Timing:** After the participant has used the Digital Twin page and viewed at least one treatment simulation.

**Question (unprompted):** "You saw a red warning on the treatment simulation page. In your own words, what did it mean?"

**Scoring rubric:**

| Score | Category | Criteria |
|-------|----------|----------|
| 2 | Correctly identified | Mentions BOTH: (a) the model shows benefit for all therapies, AND (b) relative treatment differences are not meaningful |
| 1 | Noticed but misunderstood | Recognizes a warning exists but cannot explain the specific limitation (e.g., "it says the treatment might not work" without mentioning the structural artifact) |
| 0 | Did not notice or recall | Cannot recall the warning or states no warning was present |

**Reporting:** Individual scores per participant. If a majority score 0 or 1, this is a **reportable safety finding** — the warning is a safety control and an unread warning does not function as one.

### 3.6 Stage IIIB/IV Block Comprehension Check (Mandatory)

**Timing:** After Case 3 (Stage IIIB patient).

**Question (unprompted):** "The system refused to give a prediction for this patient. In your own words, why?"

**Scoring rubric:**

| Score | Category | Criteria |
|-------|----------|----------|
| 2 | Correctly identified | Mentions insufficient sample size or insufficient data for Stage IIIB/IV specifically |
| 1 | Partially understood | Knows the system blocked it but gives a generic reason (e.g., "the system doesn't support this") without citing the Phase 7.1 finding |
| 0 | Did not understand | Cannot explain why the block occurred or believes it was a system error |

**Reporting:** Individual scores per participant. Poor comprehension of the block reason is a finding — clinicians who don't understand why a block exists may attempt to circumvent it.

---

## 4. Reporting Standards

When data is collected, results will be reported as:

### 4.1 SUS
- Individual scores per participant (table)
- Cohort mean
- 95% confidence interval (t-distribution for small samples)
- NOT a single aggregate number without spread

### 4.2 NASA-TLX
- Individual scores per participant per dimension (table)
- Cohort mean per dimension
- 95% CI per dimension
- Overall weighted TLX score

### 4.3 Comprehension Checks
- Individual scores per participant (table)
- Distribution across categories (Correct / Misunderstood / Did not notice)
- If majority in "Did not notice" or "Misunderstood": explicitly stated as a safety finding
- Qualitative responses transcribed verbatim

### 4.4 Stage Block Comprehension
- Individual scores per participant (table)
- Distribution across categories
- Qualitative responses transcribed verbatim

---

## 5. What Happens If Comprehension Is Poor

If the artifact warning comprehension check shows a majority of participants in the "did not notice" or "misunderstood" category:

1. **This is a reportable finding** — not something to average away
2. The finding will be stated explicitly in the hardening report
3. Implications for Phase 9 will be documented: the warning UI may need redesign (e.g., modal dialog, forced acknowledgment, larger font, different placement)
4. Phase 8.1 is **not authorized to redesign the CDSS** — it can only report the finding

If the Stage IIIB/IV block comprehension check shows poor understanding:
1. Same reporting standard
2. Implication: the block message may need to be more prominent or use different language
3. Again, Phase 8.1 reports but does not fix

---

## 6. Current Data

**No data has been collected.** The following tables will be populated when the protocol is executed.

### SUS Results (Empty)

| Participant | Specialty | Experience (years) | SUS Score |
|-------------|-----------|-------------------|-----------|
| — | — | — | — |

**Cohort mean:** N/A
**95% CI:** N/A

### NASA-TLX Results (Empty)

| Participant | Mental | Physical | Temporal | Performance | Effort | Frustration |
|-------------|--------|----------|----------|-------------|--------|-------------|
| — | — | — | — | — | — | — |

### Artifact Warning Comprehension (Empty)

| Participant | Score (0/1/2) | Category | Verbatim Response |
|-------------|---------------|----------|-------------------|
| — | — | — | — |

### Stage Block Comprehension (Empty)

| Participant | Score (0/1/2) | Category | Verbatim Response |
|-------------|---------------|----------|-------------------|
| — | — | — | — |

---

## 7. Limitations

1. **Zero clinician participants** — the most significant limitation
2. **No IRB approval** — required before any data collection
3. **Single-researcher project** — no team to conduct multi-rater evaluation
4. **Geographic constraint** — researcher is in Rwanda; clinician access depends on institutional partnerships
5. **No screen recording** — planned but not yet implemented
6. **Cultural and workflow differences** — not assessed

---

## 8. Next Steps for Execution

1. Obtain IRB / ethics committee approval
2. Establish institutional partnership for clinician recruitment
3. Recruit minimum 5 clinicians
4. Schedule 45-minute sessions
5. Execute protocol
6. Populate tables in Section 6
7. Report results with individual scores, means, and 95% CIs
8. Report comprehension findings honestly — including poor results

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.
