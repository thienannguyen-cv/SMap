# Experimental Validation Branch: AI Governed Update Trial

## Objective
Demonstrate AI-assisted open-source maintenance under explicit verification gates.

## Workflow
Human maintainer
    |
    v
reasoning/review layer
    |
    v
coding agent
    |
    v
validated PR artifacts

## Current Completed Example
**Bug A: FUT convergence filter fix.**

### Problem Identification
The `FUT` rectification mode lacked the convergence filter present in `DEP` mode (`y_flow*=(1−case)`). Without this filter, `y_flow==1` alignment triggered non-zero gradient flow (`g_m ≠ 0`), violating the strict mask fixed-point absorbing property dictated by Lemma 1 in the mathematical spec (`math_model.md`).

### Evidence Protocol
Instead of relying on historical test artifacts or tautological mocks, a fresh, independent minimal regression probe (`t_bug_a_minimal.py`) was created. It strictly enforces PyTorch's native execution without simulator testbot side-effects, mapped to an exact identity projection grid.

### Regression Validation
A strict 3-stage validation (fail-before / pass-after / revert-fail) proved causality:
- **Fail-Before:** Baseline source incorrectly produced reinforcing mask gradients (`g_m = +0.3`).
- **Pass-After:** Patched source became strictly absorbing (`g_m = 0.0`), with max difference in spatial flow being exact `0.0`.
- **Revert-Fail:** Reverting the patch restored the bug exactly (`g_m = +0.3`).

### Consistency Check
- **Math Model:** `math_model.md` Lemma 1 dictates matched targets are mask-absorbing. The source fix fulfills this.
- **Simulator:** `smap_simulator.jsx` perfectly models the bug behavior via the `futFilter` UI toggle. Both "source (buggy)" and "fixed (applied)" states are correctly documented in the JSX strings. No external UI modifications were required.
- **Source:** Patched `smap/rectify.py` aligns exactly with the mathematical theory.

---
**Disclaimer:** This is an experimental validation branch demonstrating a repeatable AI-assisted engineering process. AI contributions are validated through human-defined evidence standards. The branch does not claim autonomous merging, autonomous software ownership, or replacement of human maintainers.
