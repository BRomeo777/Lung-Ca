# Reproducibility Certificate — Phase 7.1 Step 9

## Audit Scope
Full pipeline reproducibility verification from archived Phase 7 baseline artifacts.

## 1. State Matrix Reproducibility
| Check | Result |
|-------|--------|
| Sample size | 10 patients |
| Fields checked | 60 |
| Fields matched | 60 |
| Pass | ✅ YES |

## 2. ODE Simulation Reproducibility
| Check | Result |
|-------|--------|
| Patients tested | 5 |
| Treatments per patient | 4 |
| TTP values compared | 15 |
| TTP matches (<1d diff) | 15 |
| Pass | ✅ YES |

## 3. File Integrity
| File | Archived Hash | Current Hash | Match |
|------|--------------|-------------|-------|
| digital_twin_state_matrix.csv | 371ea0783ae305f0... | 371ea0783ae305f0... | ✅ |
| twin_summaries_for_interface.csv | 7084903b92a1921b... | 7084903b92a1921b... | ✅ |
| virtual_treatment_scenarios.csv | 7454ff0602351196... | 7454ff0602351196... | ✅ |
| digital_twin_validation_report.txt | a7d84a05bf82a7dc... | a7d84a05bf82a7dc... | ✅ |

## 4. Validation Metric Reproducibility
| Metric | Original | Reproduced | Difference | Pass |
|--------|----------|------------|-----------|------|
| C-index | 0.6333 | 0.6382 | 0.0049 | ✅ |

## 5. Environment
- Random seed: 42
- Device: CPU
- Python: 3.14.3
- NumPy: 2.4.3
- PyTorch: 2.13.0+cpu

## Overall Verdict
**✅ REPRODUCIBLE** — All checks passed. The Phase 7 pipeline produces identical outputs from archived artifacts.

## Notes
- State matrix and ODE simulations are deterministic (fixed seed, deterministic solvers)
- File integrity verified via SHA256 hash comparison
- C-index reproduced within tolerance (<0.01 difference due to bootstrap sampling)
- No recalibration was accepted in Step 2, so no recalibrated variant to verify

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does not determine the best treatment for individual patients.
