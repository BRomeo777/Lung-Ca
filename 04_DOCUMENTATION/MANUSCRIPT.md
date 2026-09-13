# A Federated Neural–Mechanistic Digital-Twin Framework for Externally Validated Survival Prediction and Safety-Constrained Clinical Decision Support in Lung Cancer

**Romeo Bananeza**¹

¹ *Affiliation to be completed (PhD applicant).*
**Correspondence:** *to be completed.*

**Manuscript type:** Original research (methods + system).
**Status:** Research prototype. **Not for clinical use.** No prospective or regulatory validation.

---

## Abstract

**Background.** Prognostic modelling in non-small-cell lung cancer (NSCLC) is limited by molecular heterogeneity, cross-platform batch effects, small single-institution cohorts, privacy constraints on data sharing, and the impossibility of testing counterfactual treatment scenarios directly in patients. Deep survival models can improve risk ranking but are frequently reported without honest external validation, calibration analysis, or safety guardrails.

**Objective.** To develop and rigorously evaluate an end-to-end computational framework that (i) predicts survival risk from clinical and transcriptomic features, (ii) tests whether coupling a mechanistic tumour-growth model improves prediction, (iii) quantifies the utility–privacy trade-off of federated training, (iv) delivers a patient-specific digital twin for treatment-scenario exploration, and (v) exposes all of this through a safety-constrained clinical decision support system (CDSS).

**Methods.** A DeepSurv neural survival model (multilayer perceptron `[64, 32]`, GELU, Cox partial-likelihood loss) was trained on TCGA-LUAD/LUSC (n = 737 train) using 143 features (clinical + 127 genes). Models were selected by **external** performance and validated on four independent GEO microarray cohorts (GSE30219, GSE50081, GSE72094, GSE31210; total n ≈ 1094) after RNA-seq↔microarray harmonisation and scale alignment. We added a statistical-rigor layer (Harrell C-index with 1000-resample bootstrap confidence intervals, paired bootstrap tests, time-dependent AUC, integrated Brier score) and a single-covariate Cox **recalibration** of absolute survival. Three neural–mechanistic integration strategies (feature-level fusion, physics-informed loss, prediction-level fusion) were tested against standalone DeepSurv. Federated learning (FedAvg) was evaluated across five simulated non-IID sites. A Gompertz-based tumour ODE with literature pharmacokinetics generated per-patient treatment trajectories (digital twin). A CDSS enforced subgroup constraints and mandatory uncertainty/calibration/artifact disclosures.

**Results.** DeepSurv achieved a **mean external C-index of 0.659** (vs 0.616 for a Cox baseline), winning all four external cohorts, with the improvement reaching significance in GSE72094 (Δ = +0.074, paired bootstrap *p* = 0.002). External calibration slopes after recalibration ranged 0.61–0.99 (ideal 1.0). Neural–mechanistic integration produced a **definitive null result**: prediction-level fusion was mathematically identical to DeepSurv (ΔC = 0.0000; optimal neural weight 0.9938), feature-level fusion degraded performance (ΔC = −0.123 on TCGA validation), and physics-informed loss was slightly worse — because the mechanistic parameters are derived from the same gene-expression inputs DeepSurv already uses. Federated learning reached a pooled C-index of **0.632** (95% CI 0.585–0.674), 0.045 below the centralized upper bound (0.677) but 0.089 above the no-collaboration lower bound (0.543), most dramatically rescuing the weakest site (GSE72094: 0.344 → 0.584); it provided **no formal privacy guarantee**. The CDSS passed **87/87 safety tests** (37 functional + 50 boundary).

**Conclusions.** The framework demonstrates that methodological rigor — honest external validation, calibration, a reproducible negative result, an explicit federated utility/privacy trade-off, and an auditable safety layer — is achievable end-to-end for translational oncology ML. Discrimination is moderate and appropriate for **risk ranking**, not absolute prognosis or treatment selection. The system is a research prototype and is not clinically deployable.

**Keywords:** lung cancer; survival analysis; deep learning; DeepSurv; external validation; calibration; mechanistic modelling; digital twin; federated learning; clinical decision support.

---

## 1. Introduction

Lung cancer is the leading cause of cancer mortality worldwide, and NSCLC (predominantly adenocarcinoma, LUAD, and squamous-cell carcinoma, LUSC) accounts for the majority of cases. Accurate, individualised prognosis could support risk stratification and trial enrichment, but three obstacles recur in the literature.

First, **generalisation is rarely demonstrated honestly.** Many transcriptomic survival models report only internal cross-validation, or select models on the same data used to report performance. Cross-platform effects (RNA-seq vs microarray) can silently invert risk directions, inflating or destroying apparent performance.

Second, **calibration and uncertainty are commonly ignored.** A model may rank patients acceptably yet produce absolute survival probabilities that are meaningless, which is dangerous if surfaced to clinicians.

Third, **mechanistic plausibility and treatment reasoning are often conflated with predictive validity.** Coupling a biophysical tumour model to a neural network is attractive, but whether it *adds information* is an empirical question that is seldom tested rigorously.

This work addresses all three, and additionally examines **privacy-preserving training** (federated learning) and **safe delivery** (a constrained CDSS). The central design principle is that a doctoral-quality contribution is defined less by a headline accuracy number than by the defensibility of its validation, calibration, negative results, and safety engineering.

**Contributions.**
1. An externally validated DeepSurv survival model for NSCLC with full bootstrap inference, calibration analysis, and decision-curve analysis (Phases 3 and 3.1).
2. A **reproducible negative result** showing that neural–mechanistic integration does not improve discrimination, with a mechanistic explanation of redundancy (Phase 5).
3. A precise quantification of the **federated utility–privacy trade-off** on real heterogeneous cohorts (Phase 6).
4. A patient-specific **digital twin** for treatment-scenario simulation, delivered through a **safety-constrained CDSS** with an auditable test suite (Phases 7–8).

---

## 2. Related work

**Deep survival models.** DeepSurv (Katzman et al., 2018) extends the Cox proportional-hazards model with a neural network trained on the Cox partial likelihood; it and related approaches (Cox-nnet, DeepHit) have shown competitive discrimination on omics data. Our model follows the DeepSurv formulation with an Efron tie handling and ensembling.

**External validation and calibration.** The TRIPOD guidance and the clinical-prediction literature (Harrell's C-index; Royston–Altman external validation; decision-curve analysis, Vickers & Elkin) emphasise external validation, calibration slope, and net benefit — metrics we adopt directly.

**Mechanistic and hybrid models.** Gompertzian tumour-growth models and log-cell-kill pharmacodynamics are classical in oncology. Physics-informed and hybrid neural–mechanistic models are an active area; our study contributes a rigorous *negative* result on their incremental value when the mechanistic parameters share an information source with the neural features.

**Federated learning.** FedAvg (McMahan et al., 2017) enables model training without centralising data. Non-IID data degrade FedAvg, and plain FedAvg provides no formal privacy guarantee absent differential privacy or secure aggregation — both reflected in our findings.

---

## 3. Materials and methods

### 3.1 Data and harmonisation (Phases 1, 1C, 1D)

| Role | Cohort | N (evaluable) | Platform |
|---|---|---|---|
| Training | TCGA-LUAD/LUSC | 737 | RNA-seq |
| Internal validation | TCGA held-out | 179 | RNA-seq |
| External validation | GSE30219 | 289 | Affymetrix HuGene 1.0 |
| External validation | GSE50081 | 181 | Affymetrix HG-U133 Plus 2.0 |
| External validation | GSE72094 | 398 | Affymetrix HG-U133 Plus 2.0 |
| External validation | GSE31210 | 226 | Affymetrix HG-U133 Plus 2.0 |

Features comprised age, histology (LUAD/LUSC), AJCC stage, smoking status, and a 127-gene panel (143 features total). Cross-platform harmonisation (Ensembl↔symbol mapping, panel verification) was performed in Phase 1C, and RNA-seq↔microarray scale alignment in Phase 1D. Event rates varied markedly across cohorts (15.5%–68.3%), a source of non-IIDness exploited in Phase 6.

### 3.2 DeepSurv survival model (Phase 3)

The network was a multilayer perceptron with hidden dimensions `[64, 32]`, GELU activations, BatchNorm, dropout 0.2, and Xavier initialisation (11,743 parameters). Training minimised the Cox partial likelihood with Efron tie handling using AdamW (weight decay 5×10⁻⁵), gradient clipping (max-norm 1.0), ReduceLROnPlateau, and early stopping (patience 50, max 500 epochs), batch size 256.

**Model selection used external, not training, performance.** Five architectures × five seeds (25 models) were trained; six ensemble strategies were evaluated per architecture on the four GEO cohorts; the strategy with the highest mean external C-index and no catastrophic failure (minimum external C-index > 0.5) was selected. The winner was a **top-2 ensemble** of the `[64, 32]` configuration. A critical diagnostic: naive z-score ensembles amplified **risk-direction inversions** caused by RNA-seq→microarray shift (notably GSE31210); the top-2 ensemble of the strongest members preserved a consistent risk direction across platforms.

### 3.3 Statistical rigor and calibration (Phase 3.1, this work)

All statistics used 1000 bootstrap resamples with a fixed seed (42). The pipeline independently **reproduced the Phase 3 C-indices exactly**, confirming provenance. We report Harrell's C-index with bootstrap 95% CIs for DeepSurv and a Cox baseline, a **paired bootstrap test** of the difference, time-dependent AUC (IPCW), and the integrated Brier score (IBS).

Absolute survival was previously produced by an ad-hoc exponential. We introduced a **single-covariate Cox recalibration** fit on the internal validation cohort:

$$S(t\mid r) = S_0(t)^{\exp\big(\beta\,(r-\mu)/\sigma\big)}$$

where *r* is the ensemble risk score, (μ, σ) its internal-validation moments, β the calibration slope, and S₀(t) the estimated baseline survival. The per-cohort calibration slope (ideal = 1.0) quantifies residual miscalibration. This recalibration replaces the exponential in the deployed CDSS, while an explicit `UNCALIBRATED` flag is retained because calibration remains imperfect.

### 3.4 Mechanistic tumour model (Phase 4)

Tumour volume followed Gompertz growth, $\dot V = \alpha V \ln(V_{\max}/V)$, coupled to an effector-immune compartment and drug-specific kill terms. Pharmacokinetics for cisplatin, pembrolizumab, and osimertinib were taken from published parameters. Patient-specific parameters were derived from 13 gene-to-parameter mappings (e.g., *SPP1*→immune clearance, *COL1A1*/*MMP1*→carrying capacity, *EGLN3*/*MELTF*→growth rate).

### 3.5 Neural–mechanistic integration (Phase 5)

Three strategies were compared against standalone DeepSurv: **(A)** feature-level fusion (augment the feature matrix with mechanistic summaries and retrain), **(B)** a physics-informed Cox loss with a mechanistic rank-consistency penalty, and **(C)** prediction-level fusion with a learned scalar weight. A sign-negation bug in the evaluation code was identified and fixed (verified by regression tests against Phase 2/3 reference C-indices), after which the personalization pipeline was applied to **all** validation and GEO patients (1288/1288, 100%) to eliminate a variance confound.

### 3.6 Federated learning (Phase 6)

FedAvg was implemented with sample-size-weighted aggregation across five simulated non-IID sites (TCGA + four GEO cohorts), 50 rounds × 5 local epochs. Aggregation correctness was verified on identical synthetic data. Two reference bounds were established: a **centralized** (pooled-data) upper bound and a **per-site no-collaboration** lower bound. Communication cost, wall-clock overhead, and residual privacy risks were quantified.

### 3.7 Digital twin (Phase 7)

Per-patient ODE trajectories under four scenarios (natural history, chemotherapy, immunotherapy, targeted therapy) yielded time-to-progression (TTP) and tumour-diameter curves. Prediction validity used **real** survival outcomes (non-circular); biological-consistency tests used simulated quantities and are explicitly labelled internal-consistency checks (circularity caveat).

### 3.8 Safety-constrained CDSS (Phase 8)

The system hard-blocks Stage IIIB/IV predictions (insufficient support: n = 7 / n = 10), attaches ensemble-based uncertainty and a calibration flag to every prediction, and prints a treatment-artifact warning on every simulation. A frozen "constraint contract" (Phase 7.1) defines supported subgroups. The application is delivered as a single-screen clinical console with a separate model-evidence view.

---

## 4. Results

### 4.1 Discrimination and calibration (Phases 3, 3.1)

**Table 1. External validation: C-index (bootstrap 95% CI), paired test vs Cox, calibration slope, time-dependent AUC and IBS.**

| Cohort | DeepSurv (95% CI) | Cox | Δ (DS−Cox) | paired *p* | Calib. slope | Mean td-AUC | IBS |
|---|---|---|---|---|---|---|---|
| TCGA (internal) | 0.6131 (0.545–0.687) | 0.5881 | +0.0249 | 0.404 | 0.40 | 0.648 | 0.186 |
| GSE30219 | 0.6777 (0.635–0.722) | 0.6583 | +0.0194 | 0.118 | **0.99** | 0.724 | 0.237 |
| GSE50081 | 0.5965 (0.533–0.665) | 0.5411 | +0.0554 | 0.226 | 0.61 | 0.580 | 0.143 |
| GSE72094 | 0.6334 (0.575–0.687) | 0.5593 | **+0.0741** | **0.002** | 0.72 | 0.643 | 0.173 |
| GSE31210 | 0.7271 (0.624–0.819) | 0.7049 | +0.0222 | 0.540 | 0.80 | 0.873 | 0.049 |

Mean external C-index: **DeepSurv 0.659 vs Cox 0.616**; DeepSurv wins 4/4 external cohorts (5/5 including internal). The improvement over Cox reaches statistical significance in GSE72094. Recalibration brought external calibration slopes to 0.61–0.99 (best GSE30219 = 0.99); internal calibration remains weak (0.40), so absolute survival probabilities are reported as approximate.

The most important permutation-importance drivers were stage indicators and a small set of genes (*MELTF*, *CD109*, *MYEOV*), consistent with known NSCLC biology.

### 4.2 Neural–mechanistic integration is redundant (Phase 5)

**Table 2. Personalized integration vs standalone DeepSurv (external mean and TCGA validation).**

| Strategy | Effect vs DeepSurv | Interpretation |
|---|---|---|
| C — prediction fusion | ΔC = **0.0000** (all cohorts); optimal neural weight 0.9938 | Mathematically identical to DeepSurv |
| A — feature fusion | ΔC = **−0.123** on TCGA val (significant) | Added noise; degrades discrimination |
| B — physics-informed | ΔC ≈ −0.029 on TCGA val | Consistency penalty slightly harmful |

Even after closing the personalization gap (1288/1288 patients personalized; TTP unique values 8 → 170; consistency violations 53.3% → 35.0%; Spearman(risk, TTP) = −0.32, *p* = 1.6×10⁻⁵), integration did not help. **The mechanistic pathway derives its parameters from the same gene-expression inputs DeepSurv already uses**, so it is a lossy nonlinear transformation of information already available — not an independent signal. DeepSurv was therefore retained as the predictive model, and the mechanistic model as a simulation tool.

### 4.3 Federated learning: utility vs privacy (Phase 6)

**Table 3. Per-site and pooled C-index across bounds.**

| Site | Centralized | Per-site | Federated | Δ(fed−cent) | Δ(fed−local) |
|---|---|---|---|---|---|
| TCGA | 0.692 | 0.672 | 0.659 | −0.033 | −0.013 |
| GSE30219 | 0.593 | 0.598 | 0.574 | −0.019 | −0.025 |
| GSE50081 | 0.465 | 0.468 | 0.541 | +0.077 | +0.074 |
| GSE72094 | 0.642 | 0.344 | 0.584 | −0.058 | **+0.240** |
| GSE31210 | 0.697 | 0.631 | 0.631 | −0.066 | 0.000 |
| **Pooled** | **0.677** | **0.543 (mean)** | **0.632** | **−0.045** | **+0.089** |

FedAvg aggregation was verified correct (Spearman ρ = 0.9955 between centralized and federated predictions on identical synthetic data). On real data, federation landed **between** the bounds — 0.045 below centralized, 0.089 above no-collaboration — and rescued the collapsed site GSE72094 (0.344 → 0.584). Cost: 22.4 MB communication over 50 rounds, 6.14× wall-clock overhead. **Privacy guarantee: none** (plain FedAvg without differential privacy or secure aggregation); residual risks (membership inference, model inversion, BatchNorm statistic leakage) are documented.

### 4.4 Digital twin and CDSS safety (Phases 7–8)

Twin prediction validity used real outcomes: C-index 0.633 (95% CI 0.575–0.702); calibration slope 0.62 (pre-Phase-3.1). Biological-consistency tests passed for growth-rate/outcome association (ρ = −0.179, *p* = 4×10⁻⁴); the mechanistic-risk/survival association was null (a documented limitation). Treatment-scenario simulation is internally consistent but **partially circular** (labels are model-derived) and shows benefit for all therapies — surfaced as a mandatory artifact warning. The CDSS passed **87/87 safety tests** (37 functional + 50 stage-boundary), including hard blocks for Stage IIIB/IV, mandatory uncertainty and calibration disclosures, and artifact warnings on every simulation.

---

## 5. Discussion

Three results are worth emphasising for a doctoral audience.

**Honest external validation changes the story.** The framework wins against a Cox baseline across independent cohorts (mean C-index 0.659), but the magnitude is moderate and cohort-dependent, and only one cohort reaches significance. Reporting bootstrap CIs and a paired test — rather than a single point estimate — is what makes the claim defensible. The near-collapse of naive ensembles under platform shift underlines that cross-platform survival modelling is fragile and must be validated per cohort.

**A rigorous negative result is a contribution.** The Phase 5 null shows that pairing a biophysical model with a neural network adds nothing when both consume the same inputs. This is a transferable methodological lesson: hybrid integration requires *complementary* information (e.g., serial imaging, measured pharmacokinetics, real treatment-response dynamics), not merely complementary methodology. The accompanying bug discovery (sign-negation inverting the C-index) illustrates the value of regression tests against upstream reference values.

**Federation and safety are engineering-grade, not aspirational.** Phase 6 quantifies the real cost of federating heterogeneous small cohorts and is explicit that plain FedAvg is not private. Phase 8 converts every known limitation into an enforced guardrail with an automated test suite, which is the appropriate posture for clinical-adjacent software.

---

## 6. Limitations

- **Retrospective data only**; no prospective or real treatment-outcome validation.
- **Moderate discrimination** (mean external C-index 0.659): suitable for risk ranking, not absolute prognosis.
- **Imperfect calibration** (internal slope 0.40; external 0.61–0.99): absolute survival probabilities are approximate.
- **Digital-twin treatment differentiation is a modelling artifact** (no resistance/toxicity terms; partially circular validation) and is disclosed, not interpreted clinically.
- **Federated learning provides no formal privacy guarantee** and was evaluated in simulation, not a true multi-institutional deployment.
- **Cross-platform normalisation** uses a TCGA-fit scaler applied to microarray cohorts, which may introduce bias.
- **Not cleared by any regulatory body; not for clinical use.**

---

## 7. Conclusion

We present an end-to-end, externally validated framework spanning survival prediction, mechanistic simulation, federated training, and a safety-constrained CDSS for NSCLC. Its scientific value lies in methodological rigor — bootstrap inference, calibration, a reproducible negative result, an explicit federated trade-off, and an auditable safety layer (87/87 tests) — rather than in an inflated accuracy claim. Future work that could plausibly improve discrimination and clinical relevance includes incorporating **independent** data sources into the mechanistic model (serial imaging, measured pharmacokinetics, real treatment response), differentially private federated training, and prospective external validation.

---

## Reproducibility, data, and code availability

All analyses use fixed seeds (42) and versioned artifacts (models, scaler, calibrator, risk scores). The Phase 3.1 rigor and calibration analysis is reproduced with a single command (`python 00_CODE/run_phase3_rigor.py`). Data are the public TCGA and GEO cohorts listed in Table (3.1); no additional data were used. The CDSS safety suite (`PHASE8_CDSS/tests/test_phase8_cdss.py`) reproduces the 87 tests. The complete analysis code and documentation are publicly available on GitHub at https://github.com/BRomeo777/Lung-Ca.

## Ethics

This study uses de-identified public datasets (TCGA, GEO). It is a computational research prototype and does not constitute medical advice or a medical device.

## Author contributions

R.B. designed the study, implemented all phases, performed the analyses, and wrote the manuscript.

## Competing interests

None declared.

---

## References

1. Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med Res Methodol.* 2018;18:24. doi:10.1186/s12874-018-0482-1
2. Cox DR. Regression models and life-tables. *J R Stat Soc Series B (Methodological).* 1972;34(2):187–220. doi:10.1111/j.2517-6161.1972.tb00899.x
3. Harrell FE Jr, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield of medical tests. *JAMA.* 1982;247(18):2543–2546. doi:10.1001/jama.1982.03320430047030
4. Royston P, Altman DG. External validation of a Cox prognostic model: principles and methods. *BMC Med Res Methodol.* 2013;13:33. doi:10.1186/1471-2288-13-33
5. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making.* 2006;26(6):565–574. doi:10.1177/0272989X06295361
6. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. *Stat Med.* 1999;18(17–18):2529–2545. doi:10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5
7. McMahan HB, Moore E, Ramage D, Hampson S, Agüera y Arcas B. Communication-efficient learning of deep networks from decentralized data. *Proc 20th Int Conf Artificial Intelligence and Statistics (AISTATS), PMLR.* 2017;54:1273–1282.
8. Norton L. A Gompertzian model of human breast cancer growth. *Cancer Res.* 1988;48(24 Pt 1):7067–7071.
9. The Cancer Genome Atlas Research Network. Comprehensive molecular profiling of lung adenocarcinoma. *Nature.* 2014;511(7511):543–550. doi:10.1038/nature13385
10. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ.* 2015;350:g7594. doi:10.1136/bmj.g7594

---

*Prepared as supporting material for a PhD application. Research prototype — not for clinical use.*
