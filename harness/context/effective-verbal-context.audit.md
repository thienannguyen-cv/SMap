# Fresh-Agent Audit — effective-verbal-context.md

**Audited document:** `effective-verbal-context.md` (project root)
**Audit mode:** fresh-agent audit (general-purpose agent, no prior-chat access; read-only + ran cited commands)
**Date:** 2026-06-04
**Provenance:** the agent was given ONLY the handoff path + workspace and a neutral cold-start sufficiency task; it was NOT primed with conclusions or suspected gaps. Response below is verbatim.

> **★ SUPERSEDED — annotated 2026-06-12 (later session; does NOT alter the verbatim audit below).** This audit predates later handoff patches; treat it as a historical critique, not current state. Its 5 *Recommended Patches* have since been APPLIED:
> 1. The 14 `SMap_Z_10_*.npy` dumps are now inventoried → `harness/audit/kit/data/` (role: regenerable testbot reference dumps; NOT the m-fixed real-data seed).
> 2. Headless `far, SOURCE k=1.0 pinned` case added to `_repro_check.mjs` (CONVERGES in 12 steps) — resolving the apparent convergence contradiction.
> 3. Cache toolkit committed → `harness/simulator/_cache_toolkit.js` (no longer "lost / re-derivable").
> 4. The L62-72 nest now has source probes (`t_cam_matched.py`, `t_indep_matched.py`, `t_a1a4.py`); the `far/k_const` lead was **RE-FRAMED from a "deterministic stall" to a precision/RATE effect** (Bug B precision-only; real-failure risk only in the m-fixed regime).
> 5. ipynb L432/L435 are in `depth-estimation.ipynb`, now confirmed **EXTERNAL** to this workspace ⇒ unverifiable here (re-check upstream).
>
> **Path reorg since this audit:** `_audit_sandbox/` → `harness/audit/sandbox/`; root `t_*.py` → `harness/audit/kit/probes/`; root `_repro_check.mjs` → `harness/simulator/`; the audited handoff itself moved from project root → `harness/context/effective-verbal-context.md`.

---

**Can Continue?** — **Yes (with one caveat).** A fresh session can recover the objective, re-verify every cited source anchor, and run both the Node and PyTorch recipes successfully; the only real friction is that the #1 priority lead (far/k=1.0 stall) has no headless repro and needs a partly-undocumented live-app toolkit.

**Recovered Objective** — Two nested goals, clearly stated:
1. *Stated meta-task (current AUDIT phase):* iteratively make the SMap source algorithm correct (Φ→0) by auditing it through the React app `smap_simulator.jsx` against the proved spec `math_model.md`, treating source as single-source-of-truth; source files stay unfixed by design (bugs A/B/C live).
2. *★ Real next arc (the actual highest-priority next action):* debug the real source against real training data, where the output mask is "perceptually close but not clean." The single highest-priority concrete action is to **source-validate the far/k_const coverage-by-routing stall tied to `smap.py` L53–63** — specifically instrument L51/L54 `np_rand` + L57 `temp_rand` + k_const and verify whether geometry (`weights_b`, global `np_rand` gate L54) and mask (`weights_m`, per-cell `temp_rand` L57) using *different* randoms is intended, especially in the **m-fixed camera-calibration regime**.

**Verification Done**
- Re-opened anchors, all matched current revision:
  - `smap.py:11` `self.k_const = 1.` ✓
  - `smap.py:51` `np_rand = (np.random.rand(1)[0])` (single global scalar) ✓; `smap.py:50` `temp_rand = torch.rand_like(...)` (per-cell) ✓ — confirms the desync mechanism is source-accurate.
  - `smap.py:54` `if (self.k_const>np_rand)` global gate ✓; `smap.py:57/58` weights_m/weights_x split ✓; `smap.py:62/63` is_last==False intermediate inversion ✓.
  - `smap.py:176-180` inactive-winner→center fallback (Lemma 3) ✓.
  - `rectify.py:294` DEP convergence filter `y_flow = y_flow*(1.-((pre_mask>OFF)==target))` ✓, and it is the ONLY occurrence (FUT lacks it → Bug A confirmed structurally).
  - `math_model.md:155` Setting-assumption (mask-optimization scope; m-fixed must be re-derived) ✓; `mechanism_specs.md:212` Spec 5 ✓.
- Ran `node _repro_check.mjs` from project root: **success** — all presets converge; occlusion converges at k=1.0 (12 steps) and starves at k=0 (Φ=1, stuck); no Φ-rise. Matches handoff.
- Ran PyTorch probe `C:\Users\Admin\miniconda3\envs\smap-env\python.exe t_a2.py` from `_audit_sandbox/`: **success** — built the real SMap, ran `backward()`, output exactly matches documented A2 (DEP self mask-force = +0.0 absorbing; FUT = +0.3 reinforcing). Interpreter, working-dir, and hook-disabling recipe all reproduce.

**Missing Information**
- The 14 `_audit_sandbox/SMap_Z_10_*.npy` files (input/target representation + `wbl`/`ml`/`xl`/`zb`/`rnpl`/`tmpl` testbot dumps — exactly the L53–63 `weights_b/weights_m/np_rand` tensors) are **not in the Artifact Inventory**. A fresh agent cannot tell if these are the "real data" for the next arc, captured probe outputs, or stale leftovers.
- The full definition of the interactive cache toolkit (`window.__P()`, `__clickCell`, `__btn`, `__preset`) is explicitly described as "in the chat log / re-derivable" (handoff L111). The chat log is gone; "re-derivable" assumes a non-trivial reconstruction the handoff does not spell out.
- The exact stall coordinates for the prime lead — "routes (0,0)→(6,6), stalls one cell short of (7,7), coverage 0/1" — exist only as prose in the Open-Questions row; there is no headless fixture encoding them.

**Ambiguities / Hidden Context**
- "**far/k_const stall**" is described as DETERMINISTIC and "App-observed (interactive agent)," yet `_repro_check.mjs` runs the far preset only with `kSchedFixed:true` and shows it CONVERGING (`_repro_check.mjs:322`). A cold reader sees an apparent contradiction unless they realize the stall is only reproducible in the *live app at pinned k=1.0*, which the headless port does not exercise.
- "G1/G2/G3," "S_A–S_D / S1–S7," "FUT-filter & B-faithful irrelevant," "pull-model split-transport" are used as settled terms; they are defined in math_model.md/mechanism_specs.md/jsx, but the bisection result ("`k_const schedule` is the SOLE lever") is asserted from a prior interactive run with no reproducible trace in the workspace.
- Bug-fix anchors "insert L294 filter in FUT" / "notebook L435→k_const" reference ipynb line numbers (L432/L435) not re-verified here; the handoff itself warns line numbers drift.

**Actionability Gaps**
- **Prime lead lacks a headless repro.** To act on the #1 item a fresh agent must stand up the Vite server (`cp smap_simulator.jsx preview/src/SMapSimulator.jsx`, start server `smap`, port 5188), re-install the cache toolkit (definition lost), and drive the live app — multi-step, partly undocumented, and `preview_screenshot` reportedly times out. A `far + kConst=1.0` case should exist in `_repro_check.mjs` but does not.
- The proposed "source validation" for L53–63 (instrument L51/L54/L57 np_rand + gate firing on a far run) has **no probe script** — unlike A1/A2/Force-3, there is no `t_*.py` for the desync. A fresh agent must write one from scratch.
- The m-fixed camera-calibration re-derivation names *what* to re-derive (Lemma 2, Theorem′, Bug-B) but provides no input data, no harness, and no concrete first file to edit — gated on "when real data arrives," with the data's location unspecified (possibly the undocumented .npy files).

**Recommended Patches**
1. Add the 14 `_audit_sandbox/SMap_Z_10_*.npy` files to the Artifact Inventory with their role (testbot dumps of L53–63 tensors?) and state whether they are the real-data seed for the next arc or disposable.
2. Add a headless `far, SOURCE k_const=1.0 pinned` case to `_repro_check.mjs` (or document explicitly that the stall is live-app-only and the Node port cannot reproduce it) — resolve the apparent convergence contradiction in the same row.
3. Inline the cache-toolkit definition (`window.__P/__clickCell/__btn/__preset`) into the handoff or a committed `_cache_toolkit.js`, instead of "in the chat log / re-derivable."
4. Add a `_audit_sandbox/t_nest.py` probe (or a TODO stub) that logs `np_rand`(L51), `temp_rand`(L50), k_const, and weights_b/m/x on a far run — the missing instrument for the prime lead.
5. Re-verify and pin `depth-estimation.ipynb` L432/L435 line numbers in the Bug-B fix row.

**Residual Risk**
- Even patched, the **prime dirty-output lead remains app-observed, not source-confirmed** — the deterministic stall trace lives only in a lost interactive run; reproducing it independently is non-trivial.
- The **m-fixed regime is entirely un-analyzed**; correctness, Lemma 2, Theorem′, and Bug-B classification may all change there, and the workspace contains no real-training harness to test it.
- Line-number drift: every anchor matched *today*, but any edit invalidates them (handoff's standing caveat).
- Interactive validation depends on finite evaluators + a toolkit that must be re-installed per reload and a screenshot path that times out, making live-app verification fragile.
