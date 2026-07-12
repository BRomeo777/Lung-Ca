# Treatment Simulation Audit — Phase 7.1 Step 3

## Finding: Universal Treatment Benefit
100% benefit is a **structural property of the ODE model**, not a clinical finding.
All treatment kill terms are strictly positive when C>0 and V>0.

## Code Inspection
The treatment_response_ode() applies:
- Chemo: cytotoxic_kill = delta_c * C * V (Skipper log-cell kill)
- Immuno: checkpoint_kill = k_immuno * C * V + immune boost
- Targeted: targeted_kill = delta_t * C * V

All terms are positive when drug is active. No mechanism for zero/negative response exists.

## Documented Assumptions (8)
See treatment_assumption_table.csv for full details.

## Benefit Distribution (Val, n=182)
| Treatment | Median (mo) | IQR | % >30d |
|-----------|-------------|-----|--------|
| chemo | 19.6 | [17.4-21.8] | 100.0% |
| immuno | 30.2 | [29.4-31.2] | 100.0% |
| targeted | 30.2 | [29.4-31.2] | 100.0% |

## Decision
Model is **correct as designed** — simulates average treatment effect under ideal conditions.
100% benefit is explicitly documented as a modeling assumption. No code changes needed.
Probabilistic response modifiers not implemented (requires biomarker data not available).

RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. This system generates computational simulations for hypothesis generation. It does not determine the best treatment for individual patients.
