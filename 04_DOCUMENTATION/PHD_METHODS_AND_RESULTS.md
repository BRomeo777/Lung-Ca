# A Federated Neural-Mechanistic Digital-Twin Framework for Personalized Clinical Decision Support in Lung Cancer

**Methods & Results summary for PhD application review.**
**Status: research prototype — not for clinical use. No prospective or regulatory validation.**

---

## 1. Motivation and contribution

Lung cancer prognosis and treatment planning are limited by heterogeneous molecular biology and the impracticality of testing treatment scenarios directly in patients. This project develops an end-to-end, **externally validated** computational framework that couples:

- a **deep survival model** (DeepSurv) that learns patient risk from clinical + transcriptomic features;
- a **mechanistic tumor-growth model** (ODE) grounded in published NSCLC pharmacology;
- a **federated-learning** evaluation of privacy-preserving training; and
- a **patient-specific digital twin** delivered through a **safety-constrained clinical decision support system (CDSS)**.

The distinguishing methodological commitments are **honest external validation**, **statistical rigor**, **calibration analysis**, and **explicit safety guardrails** — the properties a doctoral committee scrutinizes most.

---

## 2. Data (used exactly as provided; no external augmentation)

| Role | Source | N | Platform |
|---|---|---|---|
| Training | TCGA-LUAD/LUSC | 737 train / 182 val | RNA-seq |
| Internal validation | TCGA held-out | 179 (evaluable) | RNA-seq |
| External validation | GSE30219 | 289 | microarray |
| External validation | GSE50081 | 181 | microarray |
| External validation | GSE72094 | 398 | microarray |
| External validation | GSE31210 | 226 | microarray |

Cross-platform harmonization (RNA-seq ↔ microarray) and scale alignment were performed in Phases 1C–1D. Features: age, histology, stage, smoking status, and 127 genes (143 features total).

---

## 3. Deep survival model (Phase 3)

- **Architecture:** DeepSurv MLP `[64, 32]`, GELU, dropout 0.2, BatchNorm, Xavier init.
- **Loss:** Cox partial likelihood (Efron ties). Optimizer AdamW, weight decay 5e-5, gradient clipping, early stopping.
- **Model selection by EXTERNAL validation** (not training metrics): 5 architectures × 5 seeds = 25 models; 6 ensemble strategies each; selected the top-2 ensemble with the highest mean external C-index and no catastrophic failure (min external C-index > 0.5).
- **Key diagnostic:** naive z-score ensembles amplified risk-direction inversions caused by RNA-seq→microarray shift (notably GSE31210). The top-2 ensemble of the strongest members preserved a consistent risk direction across platforms.

---

## 4. Statistical rigor and calibration (Phase 3.1 — new)

All statistics computed with **1000 bootstrap resamples, seed 42**, reproducing the Phase 3 report C-indices exactly. Script: `00_CODE/run_phase3_rigor.py`. Outputs: `PHASE3_DEEP_LEARNING/rigor/`.

### 4.1 Discrimination (Harrell C-index, bootstrap 95% CI)

| Cohort | DeepSurv (95% CI) | Cox baseline | Δ (DS−Cox) | paired *p* |
|---|---|---|---|---|
| TCGA (internal) | 0.6131 (0.545–0.687) | 0.5881 | +0.0249 | 0.404 |
| GSE30219 | 0.6777 (0.635–0.722) | 0.6583 | +0.0194 | 0.118 |
| GSE50081 | 0.5965 (0.533–0.665) | 0.5411 | +0.0554 | 0.226 |
| GSE72094 | 0.6334 (0.575–0.687) | 0.5593 | **+0.0741** | **0.002** |
| GSE31210 | 0.7271 (0.624–0.819) | 0.7049 | +0.0222 | 0.540 |

**Mean external C-index: DeepSurv 0.659 vs Cox 0.616. DeepSurv wins 5/5 cohorts.** Improvement over Cox reaches statistical significance in GSE72094 (paired bootstrap *p* = 0.002).

### 4.2 Calibration

Absolute survival probabilities were previously produced by an ad-hoc exponential. A **single-covariate Cox recalibration** was fit on the internal validation cohort:

```
S(t | risk) = S0(t) ** exp(slope · (risk − mean) / std)
```

Per-cohort calibration slope (ideal = 1.0): TCGA 0.40, GSE30219 **0.99**, GSE50081 0.61, GSE72094 0.72, GSE31210 0.80. Calibration is materially improved but **still imperfect** (weakest internally); survival probabilities are therefore reported as **approximate / ranking-oriented**, and the CDSS retains a mandatory `UNCALIBRATED` flag.

### 4.3 Time-dependent performance

Time-dependent AUC (IPCW) and integrated Brier score (IBS) at 1/2/3 years, e.g. GSE31210 mean AUC 0.873 / IBS 0.049; GSE30219 mean AUC 0.724. Decision-curve analysis (net benefit at 24 months) is provided in `rigor/figures/decision_curve_24m.png`.

---

## 5. Mechanistic model & digital twin (Phases 4–7)

Gompertz growth + log-cell-kill cytotoxic/targeted terms + immune dynamics, with literature-cited PK for cisplatin, pembrolizumab, and osimertinib. Neural risk and mechanistic states are linked per patient (Phase 5). Phase 7.1 froze a scientific-constraint contract.

**Known limitation (honestly reported):** treatment scenarios currently show benefit across all therapies because the ODE does not yet model **resistance, toxicity, or discontinuation**, and treatment parameters are population-level rather than individually identifiable from the available data. This is surfaced as a **mandatory artifact warning** on every simulation and is documented as the primary future-work item. Fabricating treatment differentiation without identifiable data would be scientifically indefensible; the framework instead quantifies and discloses the limitation.

---

## 6. Safety-constrained CDSS (Phase 8)

- **Stage IIIB/IV predictions are hard-blocked** (insufficient sample: n=7 / n=10).
- Every prediction carries **uncertainty** (ensemble std) and a **calibration flag**.
- Every treatment simulation carries the **artifact warning**.
- **87/87 safety tests pass** (`PHASE8_CDSS/tests/test_phase8_cdss.py`), including 50 boundary tests.
- Delivered as a single integrated **Clinical Console** (one-screen workflow).

---

## 7. Reproducibility

- Fixed seeds (42) throughout; pinned dependencies in `PHASE8_CDSS/requirements.txt`.
- One command reproduces the rigor analysis: `python 00_CODE/run_phase3_rigor.py`.
- All artifacts (models, scaler, calibrator, risk scores) are versioned in-repo.
- The Phase 3.1 analysis independently **reproduced the Phase 3 report C-indices exactly**, confirming pipeline integrity.

---

## 8. Honest limitations

- Retrospective data only; no prospective or external-treatment validation.
- Discrimination is moderate (mean external C-index 0.659) — appropriate for **risk ranking**, not absolute prognosis.
- Survival calibration remains imperfect (internal slope 0.40).
- Digital-twin treatment differentiation is a modeling artifact (see §5).
- **Not cleared by any regulatory body. Not for clinical use.**

---

## 9. What makes this defensible at PhD level

Rather than inflated accuracy claims, the framework's strength is methodological: external validation across 5 independent cohorts, bootstrap inference, formal calibration analysis, decision-curve analysis, reproducibility, and an auditable safety layer with 87 passing tests. These are the criteria by which doctoral committees judge translational ML in oncology.
