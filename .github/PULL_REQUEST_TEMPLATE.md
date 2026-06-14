## SMap Bug Reporting Standard: Pull Request Template

Thank you for contributing to SMap! Before we can review your PR, you must adhere to the SMap Evidence Standard. Please fill out the sections below.

### 1. Bug Classification
*Check exactly one of the following:*
- [ ] **App Inconsistency**: This is a logic/UI bug strictly within the JSX Simulator.
- [ ] **Source Bug**: This is a mathematical or core codebase deviation (modifying `smap/`, `rectify.py`, or `math_model.md`).
- [ ] **By-Design + Honesty Cue**: The behavior is correct by design, but visually confusing. This PR adds UI honesty-cues to clarify the behavior to the user.

### 2. POLA Violation Analysis (Principle of Least Astonishment)
*(Required for "App Inconsistency" and "By-Design + Honesty Cue". Optional/NA for "Source Bug" where mathematical divergence is the primary evidence)*

Explain exactly how the previous behavior violated user expectations. What was astonishing?

> **⚠️ CRITICAL WARNING FOR AI CONTRIBUTORS:**
> "Your AI is answering from memory, not from your code."
> When analyzing this bug, ensure you are reasoning *strictly* from the SMap math model and the provided code context. Do not hallucinate fixes based on common programming patterns you have seen in your training data. Ground your reasoning in the actual codebase structure.

**Your Analysis Here:**
<!-- Describe the POLA violation here -->

### 3. Snapshot-First Evidence
*You must provide a reproducible state. Did you include a snapshot JSON?*
- [ ] Yes, `snapshot_*.json` is attached/included in this PR folder.
- [ ] N/A (Only for Harness improvements)

### 4. Verification
*How did you verify this fix?*
- [ ] I ran `node harness/simulator/_repro_check.mjs` and it passed.
- [ ] I provided the headless probe log (`t_*.py`) demonstrating the fix.
- [ ] (If Math Model fix) I provided formal logical/mathematical derivations in this PR description.
