# Mechanism Test-Spec Catalogue (tri-anchored: app ⟷ source ⟷ math_model)

**Purpose.** For each *key mechanism* of the SMap algorithm, provide ONE canonical
**operation sequence** that is simultaneously:
1. a scripted scenario an agent can **execute on the jsx app** (`smap_simulator.jsx`) when
   UI-manipulation is explicitly requested (and to surface runtime blockers found in use);
2. described precisely enough that an agent can **independently write a unittest against the
   real algorithm source** (PyTorch, via the execution recipe), *from this spec, without
   reading the app* — to keep the verify independent; and
3. framed at a **semantic level** (Φ / 3 forces / meta-states / fixed points) high enough to
   serve as a building block for the `math_model.md` proof and for source bug-fixes.

Each spec therefore establishes a per-mechanism **equivalence verdict**:
`app-observable-under-scenario  ≡  source-behavior-under-equivalent-test  ≡  math_model claim`,
at the load-bearing precision (**sign + zero/non-zero**, never raw magnitude).

> **Status:** SPEC-ONLY. The `source_test` and `app_scenario` blocks below are *specifications*;
> they have NOT been executed in this pass (per user scope). Line numbers are revision-sensitive
> — **re-grep before relying** (handoff standard). Default verification remains *reading the jsx*;
> the executable layers are exercised only on explicit request.

---

## Schema (every mechanism spec uses these fields)

| Field | Content |
|---|---|
| `id` | Mechanism id, matching `mechanisms.json` (e.g. `R9_dep_weight_constr_FILTER`). |
| `source_anchor` | `file:lines` + the exact op string. The single source of truth. |
| `semantic_claim` | What the mechanism *means* in `math_model.md` terms: which force/meta-state, the Φ effect, and the **expected sign/zero** of the load-bearing gradient. The thing a proof relies on. |
| `app_scenario` | `{ init, steps[], expected }` — `init`: preset + control states; `steps`: ordered **atomic** UI actions, each mapped to a real control (toggle id / selector / cell-select / step); `expected`: the rendered observable (inspector/cert field + sign/zero). |
| `source_test` | `{ setup, action, assert }` — written from the execution recipe (handoff §"Running the real source"), reproducing the *same regime* on the PyTorch source. Independent of the app. |
| `equivalence` | The cross-check: which app observable must agree with which source assertion, at sign/zero precision, and the provenance of each side. |
| `independence_note` | What must NOT be copied from the app when writing the source_test (to keep the unprimed property). |
| `residual_risk` | What this scenario does *not* cover. |

**App execution blockers (apply to every `app_scenario`)** — from handoff §"Verification gotchas":
- step via **separate `preview_eval` round-trips** (multiple synchronous `.click()` batch into ONE
  React render and silently corrupt convergence measurement);
- read state/text from the **DOM** via `preview_eval`, not screenshots; headers are CSS
  `text-transform:uppercase` ⇒ match **case-insensitively**;
- before running: `cp smap_simulator.jsx preview/src/SMapSimulator.jsx`, start server `smap`, open
  `http://127.0.0.1:5188`.

**Source execution recipe (shared `setup` preamble for every `source_test`)** — verified working,
see `_audit_sandbox/t_a2.py`:
- interpreter `C:\Users\Admin\miniconda3\envs\smap-env\python.exe`; run **from `_audit_sandbox/`**;
- disable backward hooks: `nn.Module.register_backward_hook = nn.Module.register_full_backward_hook = lambda self,h: None`;
- `SMap(window_h=8, window_w=8, camera_matrix=np.eye(3), rectify_type=…, device="cpu", n=NLEVEL≥1)`;
- stub `model.smap3x3.vtestcase.testbot_input/.testbot_target` to no-ops; set `model.smap3x3.k_const`;
- input `[B,4,H,W]` channels `(x,y,z,mask)`, identity-cam projection `z>0, x=c*z, y=r*z`, mask>0.5 at a target cell;
- `out = model(x, target=tgt, zoom=0)` → `[B,9,H,W]`; isolate a cell's self-force via center slice
  `loss = out[0,4,r,c]; loss.backward(); g = x.grad[0,3,r,c]` (= d(rectified weight)/dm;
  **0 = absorbing · >0 = reinforcing m↑ · <0 = deactivating Φ↑-risk**).

---

## Pilot spec 1 — `R9_dep_weight_constr_FILTER` (DEP convergence filter = S1 absorbing)

| Field | Content |
|---|---|
| `source_anchor` | `rectify.py:294` — `y_flow = y_flow*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float())`. (Block `R9` = `rectify.py:238-310`.) |
| `semantic_claim` | This is the **load-bearing absorber** (math_model Lemma 1). At a **matched target cell = meta-state S1** (`m>τ` and `t=1` ⇒ `case=1`), the filter zeros `y_flow`, which gates off **Force 1 (alignment, α₁)**. With F2 also gated by `(1−case)=0`, the **mask-gradient `g_m = 0`** ⇒ S1 is a strict fixed point ⇒ Φ cannot rise from S1 under rectification. Expected load-bearing value: **`g_m == 0` exactly** at S1 for DEP. |

`app_scenario`:
- `init`: any preset; set Rectify selector = **DEP** (`setRectify("DEP")`); ensure a cell is a matched
  target (`selOrig` over a cell with `m>τ` and `t=1`). Default `selOrig={3,4}`; pick/seed an S1 cell.
- `steps`:
  1. click Rectify tab **DEP**.
  2. select an S1 cell (matched target) → confirm inspector shows `meta = S1`, `case/caseVal = 1`.
  3. read the m-gradient panel for that cell.
- `expected`: in `gradAt(...)` for that cell, `filterApplies = true` (DEP ∈ filter set,
  `smap_simulator.jsx:318`) ⇒ `yflowGate = 0` (since `caseVal===1`, `:319`) ⇒ `t1 = 0` and `t2 = 0`
  ⇒ **`gm == 0`**. Inspector "m-gradient — mask channel = F1+F2" must display **0** (absorbing); the
  next-state preview must keep the cell at S1 (no Φ increase).

`source_test`:
- `setup`: recipe with `rectify_type=rectify.types.DEP`, NLEVEL=1, one matched-target cell at `(r,c)`
  (`z>0`, mask>0.5, `tgt[r,c]=1`).
- `action`: `loss = out[0,4,r,c]; loss.backward()`.
- `assert`: `x.grad[0,3,r,c] == 0` (self mask-force absorbing). Already empirically confirmed in
  `t_a2.py` (DEP self-force = 0).

`equivalence`: app `gm == 0` at S1 (DEP)  ≡  source `x.grad[0,3,r,c] == 0` (DEP, S1)  ≡  Lemma 1
"S1 mask-absorbing". Provenance: app = `gradAt` `smap_simulator.jsx:311-358`; source = `rectify.py:294`
executed; math = `math_model.md` Lemma 1.

`independence_note`: write the source assertion from `semantic_claim` ("S1 self mask-force is zero"),
NOT from the app's `yflowGate` formula. The app must be a *prediction to falsify*, not the oracle.

`residual_risk`: covers the **rectify-layer** mask-gradient only. It does **not** rule out the
**routing layer** vacating a matched S1 cell (occlusion-drop) — that is exactly the open **A1/A4**
question; this spec must not be read as closing it.

---

## Pilot spec 2 — `R13_fut_weight_constr_BUG` (Bug A: FUT missing the L294 filter)

| Field | Content |
|---|---|
| `source_anchor` | `rectify.py:354-430` (FUT weight construction). **BUG:** no analog of `rectify.py:294` — `weights_grdf` is otherwise byte-identical to DEP (`:305` vs `:425`). FUT `y_flow` is produced at `R12` (`rectify.py:341-352`) with **no `*(1−case)`** and **no target multiply**. |
| `semantic_claim` | At **S1 with `y_flow==1`**, the missing filter leaves **Force 1 (alignment) ungated** ⇒ `g_m ≠ 0` ⇒ S1 **loses its strict mask fixed point** (deviation from Lemma 1). **Crucial sign correction:** the tap is **`+α₁` (reinforcing, m→1)**, so **Φ does NOT numerically rise** — confirmed by real-source backward (`t_a2.py`: FUT self-force = +α₁ ≈ +0.3). Severity: CORRECTNESS but **latent** (training uses DEP, ipynb cell-11). Expected load-bearing value: **`g_m > 0`** at S1 under SOURCE; **`g_m == 0`** under FIXED. |

`app_scenario`:
- `init`: Rectify = **FUT**; Bug-A toggle `futFilter` (`futFilterFixed`) = **OFF (SOURCE)**;
  select an S1 cell with `yc==1`.
- `steps`:
  1. click Rectify tab **FUT** (tab shows ⚠ when `!futFilterFixed`, `smap_simulator.jsx:672-673`).
  2. ensure Bug toggle **"FUT convergence filter"** is OFF (source).
  3. select an S1 cell; read m-gradient + sign.
  4. flip the toggle **ON (FIX)** via the same round-trip discipline; re-read.
- `expected`:
  - OFF: `filterApplies=false` (`:318`) ⇒ `yflowGate = (yc===1?1:0) = 1` ⇒ `t1 = +α₁ > 0` ⇒
    **`gm > 0`** (reinforcing). Cert flags `bugA` (`:420`). The m-gradient must show **positive**,
    and the cause/fix text must say "reinforcing (+α₁), Φ does NOT rise" — **not** "Φ rises".
  - ON: `filterApplies=true` ⇒ `yflowGate=0` ⇒ **`gm == 0`** (S1 absorbing, same as DEP).

`source_test`:
- `setup`: recipe with `rectify_type=rectify.types.FUT`, matched-target S1 cell.
- `action`: `loss = out[0,4,r,c]; loss.backward()`.
- `assert`: `x.grad[0,3,r,c] > 0` (sign **positive**, reinforcing) — NOT merely "≠0", and NOT "<0".
  This is the precise claim `t_a2.py` confirmed (FUT ≈ +0.3). Optionally compare to a DEP run in the
  same harness: `grad_FUT > 0 == grad_DEP` should be `False` (DEP is 0).

`equivalence`: app `gm>0` (FUT/SOURCE/S1) ≡ source `x.grad[0,3,r,c] > 0` ≡ math "Lemma 1 fails for
FUT, but tap sign is +α₁ ⇒ no Φ-rise". The **sign agreement (+)** is the whole point: a faithful
mirror must reproduce *reinforcing*, not *deactivating*.

`independence_note`: the source_test must assert the **sign (+)**, written from `semantic_claim`,
independently of the app's `t1` formula. A bug here would be the app showing `gm≠0` but with the
wrong sign (e.g. claiming "Φ rises") — the source_test is what catches that.

`residual_risk`: latent because training path is DEP; this spec proves the *mechanism's* deviation
and its sign, not a numerically observable training failure. Does not address whether a routing-layer
interaction could change the picture (A4).

---

## Pilot spec 3 — `M7_einsum_select` (detached routing: gradient flows to xyz, not to selection)

| Field | Content |
|---|---|
| `source_anchor` | `smap.py:83-136` — `einsum(weights_x.detach(), x_z_value)`: gathers the selected neighbor's xyz. The selection one-hot `weights_x` is **`.detach()`-ed** ⇒ gradient flows into the **xyz/mask values only**, NOT into *which* neighbor is chosen. |
| `semantic_claim` | **Separation of the two meta-state layers** (math_model §3): the **routing layer (S_A–S_D)** decides *participation/occlusion* via a **non-differentiable, structurally recomputed** selection; the **rectify layer (S1–S7)** moves the *continuous* xyz/mask via gradient. A gradient step must therefore change `(x,y,z,m)` of a cell **without** the gradient itself altering the routing choice. Load-bearing property: the routing selection is a **detached / zero-gradient** path. |

`app_scenario`:
- `init`: any preset; pick a cell whose routing class is well-defined (`classifyRoute`, e.g. S_C
  participating); note its routed-to neighbor.
- `steps`:
  1. select a cell; record inspector `route` (S_A–S_D) and current `(x,y,z,m)`.
  2. perform ONE gradient **step** (the Step control → `oneStep`, `smap_simulator.jsx:371-383`) in a
     single round-trip.
  3. re-read the same cell's `(x,y,z,m)` and `route`.
- `expected`: in `oneStep`, the routing (`routeStep`, `:224`/`:373`) is recomputed **structurally**
  each pass and the gradient (`gradAt → gx,gy,gz,gm`, `:339-358`) updates `x += lr·gx`, … ,
  `m += lr·gm` (`:383`). The cell's **xyz/m change by the gradient**, while the routing *decision*
  is **not nudged by the gradient** (it changes only if the recomputed structure changes). The app
  must show xyz/m moving while attributing any cell-shift to ROUTING, not to the gradient
  (inspector grad line `:475` says "LOCAL only — far targets reached by ROUTING").

`source_test`:
- `setup`: recipe; build `out`, take any differentiable scalar loss.
- `action`: `loss.backward()`.
- `assert`: the routing-selection tensor carries **no gradient** — i.e. `weights_x` is detached
  (`weights_x.requires_grad is False`, or `weights_x.grad_fn is None` at the einsum input), while
  `x.grad` (the xyz/mask channels) is populated and finite. Equivalent black-box check: a step that
  changes input xyz must change `out` smoothly, but the **argmax selection index** must be a step
  function of structure, not of an infinitesimal gradient nudge.

`equivalence`: app "gradient step moves xyz/m, routing unchanged-by-gradient" ≡ source
"`weights_x.detach()` ⇒ no grad to selection, grad to xyz only" ≡ math "two-layer separation:
routing non-differentiable, rectify differentiable".

`independence_note`: write the source assertion as "the selection path is non-differentiable",
from `semantic_claim` — do not infer it from the app's separate `routeStep`/`gradAt` functions.

`residual_risk`: this proves the *gradient does not flow into selection*; it does **not** prove the
routing layer never deactivates a matched S1 (occlusion vacate) — again the **A1/A4** dynamics, out
of scope for this static-mechanism spec.

---

## How this plugs into the existing pipeline (proposed, not yet wired)

- `mechanisms.json` already carries `lines` + `desc` + a one-line `scenario` per mechanism. This file
  **upgrades** the `scenario` of the 3 pilots into the structured schema above; the remaining
  M0–M16 / R1–R13 keep their one-liners until promoted.
- Step-5 unprimed validation can later gain a per-mechanism layer: (B) a **source-side unittest
  suite** generalizing `t_a2.py` (one `source_test` per mechanism), and (C) an optional
  **UI-driving agent** that runs `app_scenario` on the live preview and compares to `expected` —
  both **deferred** in this pass.
- Mechanisms whose current `scenario` is "Implicit in any routing step" (`M0`, `M6`) must, when
  promoted, either get a concrete eliciting scenario or be explicitly marked
  *not-elicitable-in-isolation* with a reason (per the "one-way reflection = defect" standard).

---

## Pilot spec 4 — Force 3 (coordinate gradient) — OPEN GAP from interactive validation

| Field | Content |
|---|---|
| `source_anchor` | `rectify.py:305` last term (`key_query_grdf` → backprops to x/y/z), conditioned by `k_t = k_d/pre_z` (`rectify.py:130-131`), CAM/DEP/FUT only (not Default). App: `gradAt` `smap_simulator.jsx:333-356` + applied in `oneStep:383`. |
| `semantic_claim` | Force 3 is a **coordinate** force: it moves a point's (x,y,z) toward a vacant target's back-projection; it must **not** touch the mask `[m>τ]` (math_model Lemma 1). Load-bearing: sign/zero of the xyz move for cells in the eliciting state. |
| **GAP (found by interactive audit, confirmed in source)** | Two independent agents observed **xyz never changes under a Markov step**. Source read confirms: (1) the displayed **`spread(F3)` bar `t3`** is gated on `t===1 && B && mm===0` (inactive target) but is **never applied** — `oneStep` applies `gx,gy,gz`, not `t3`, so the bar is **cosmetic / decoupled**; (2) the **applied** `gx,gy` are gated on `mm===1 && z>0 && adjacent-target && coords-misaligned` — a narrow state **no preset elicits**; (3) **`gz ≡ 0`** always — the z-coordinate is **never rectified**. So Force 3 is currently **one-way / not reflectable** (the bar shows a force that drives nothing; the applied force is unreachable; z is inert). |
| `app_scenario` (to ADD when fixed) | A preset placing an **active** point at a cell **adjacent** to a target, with its (x,y) **not** already projecting onto that target → Forward/step must show `gx`/`gy ≠ 0` and the point's x/y measurably move; the displayed Force-3 bar must correspond to the **applied** coordinate move (not a separate gate). |
| `source_test` (to ADD) | Recipe, CAM/DEP/FUT; active cell adjacent to a target, misaligned coords; `loss.backward()`; assert `x.grad[0,0:3,r,c]` (x,y,z channels) has the expected sign/zero — **including z** (decides whether `gz≡0` is faithful or a gap). |
| `equivalence` | app applied-xyz-move (sign/zero, incl. z) ≡ source `x.grad` on x/y/z channels ≡ math "coordinate force moves geometry, not mask". |
| `residual_risk` | Until fixed: the app **cannot demonstrate Force 3 faithfully**. Decision needed (math_model-level): is `gz≡0` correct (z fixed by identity-cam) or a missing term? Should `t3` bar be re-bound to the applied `gx/gy` gate, or removed? This is a **semantic** item, not a quick code fix.

---

## Spec 5 — `smap.py` L53–L63 warming/routing-selection nest (COHESIVE region, pre-investigation map)

> **Why this spec exists.** User's **prime suspect** for the real-data "dirty output" (perceptually-close, not perfect). High cohesion ⇒ a bug here is only pinpointable *within the whole mechanism*. This is a **source-exact map + app-fidelity gap list + candidate anomalies (HYPOTHESES, not confirmed bugs — no real data yet)**, to bootstrap the fresh session.

**Source-exact line map.** Three chained selection tensors, all built from `weights_b` (argmax `-key_query` one-hot, `L45–47`):
- `L53` `weights_b = where(z>0, weights_b, temp)` — z-validity: no-depth cell can't route ⇒ falls to center (`temp` = slot-4 one-hot).
- `L54–55` `if (self.k_const > np_rand):` where **`np_rand = np.random.rand()` is a SINGLE SCALAR per forward pass** ⇒ with field-wide probability `k_const`, `weights_b = where(active∧z>0, weights_b, where(z>0, temp, 0))` (inactive-with-depth → center; no-depth → nothing). **Global coherent gate.**
- `L57` `weights_m = where(active, where(~z>0, temp, weights_b), where(temp_rand>k, temp, weights_b))` — `weights_m` (drives the **mask / z-buffer** transport): active→route (or center if no depth); inactive→**per-cell** `P(route)=k_const`.
- `L58` `weights_x = where(active, where(~z>0, 0, weights_m), weights_b)` — `weights_x` (drives the **xyz / value** einsum, detached, M7): active→`weights_m` (or 0 if no depth); inactive→**`weights_b`** (NOT `weights_m`).
- `L61–63` `if is_last==False:` override `weights_m`,`weights_x` with **inverted** direction (`P(route)=1−k_const` for inactive) — the L62/L63 inversion.

**Downstream roles:** `weights_m.detach()` → mask gather (`r_mask`, → Φ); `weights_x.detach()` → xyz gather (→ geometry). They can pick **different source cells** for the same inactive target.

**Candidate anomalies — HYPOTHESES to verify against real data (NOT confirmed; do not treat as bugs yet):**
1. **Global `np_rand` scalar gate (L54):** one per-pass random gating a field-wide `weights_b` edit, mixed with per-cell `temp_rand` (L57). All-or-nothing coherent randomness is hard to motivate and is a plausible source of frame-level noise ⇒ "dirty" output.
2. **`weights_m` ⟂ `weights_x` divergence for inactive cells (L57 vs L58):** mask-transport and xyz-transport may select **different neighbours** ⇒ a cell's displayed mask and its geometry come from different points ⇒ mask/geometry desync (dirtiness).
3. **Intermediate inversion (L62/L63):** participation direction flips vs finest; benign at pinned `k=1.0`, bites under a ramp (see Bug B / quick-win).

**App-fidelity gaps — STATUS (upgraded via `inactiveNest` + inspector "L53–63 nest" readout):**
- (G1) ✅ global per-pass `np_rand` gate modelled (scalar `passRand`, fires w.p. k_const) — surfaced as diagnostic.
- (G2) ✅ `weights_m` (mask, per-cell) vs `weights_x` (xyz, via global gate) computed per cell with a **DESYNC flag** in the inspector. *Verified live:* at `k=1.0` inactive cells show mask-routes-but-xyz-stays ⇒ systematic desync. **Deferred:** applying the split to the *dynamics* (pull-model rewrite) → real-data arc, to avoid baking a wrong model into the prime suspect.
- (G3) ✅ intermediate-level inversion `P=1−k` applied to the **mask-route dynamics** at `l>0` (finest `l=0` unchanged ⇒ Lemma 2 / proven warming preserved).
The region is now **observable** (bidirectional-reflection standard met for diagnosis); full split-dynamics is the remaining real-data-arc item.

`source_test` (for the fresh session): instrument the real source to log `weights_b/weights_m/weights_x` and the per-pass `np_rand` on a real-data batch; check (i) whether `weights_m≠weights_x` for any inactive target, (ii) the variance contributed by the `np_rand` gate, (iii) per-level participation under the trained `k_const`. Compare to the (upgraded) app.

**INTERACTIVE-VALIDATION LEAD — CORRECTED (was overstated as a "stall"):** on the **far** preset at pinned `k_const=1.0`, an interactive agent saw the active point route (0,0)→(6,6) then sit at cov 0/1 for the ~3–4 steps it ran, and reported a "deterministic stall." **Headless re-check (`_repro_check.mjs` `far, SOURCE k=1.0 pinned`, up to 80 steps) CORRECTS this: it CONVERGES in 12 steps (cov 1/1)** — vs **1 step** with the `k_const schedule` fix. Mechanism: at k=1.0 the dark target's warming-claim fires every step ⇒ the incoming active point is blocked (coverage-by-**ROUTING** stalls), BUT the target still **warms IN PLACE** (Force-2) and crosses τ after ~12 steps. So this is a **RATE/precision effect, NOT a correctness stall** — the few-step interactive read was premature. **Reaffirms Bug B = precision-only** (schedule speeds 12→1 steps) and does **not** contradict halt⟺correct. The mask/xyz desync (#1+#2, real at source) is the *routing-cover* mechanism that's slow here. **Where it COULD become a real failure:** the **m-fixed camera-calibration regime** (warming inert — no in-place rescue) ⇒ still the prime "dirty-output" lead for that arc. **Proposed source validation (next arc):** instrument L51/L54 `np_rand`+gate on a far run; confirm the geometry(L54 global)/mask(L57 per-cell) random asymmetry is intended; test in the m-fixed regime.

`residual_risk` / scope: anomalies are **unconfirmed at source level**; confirming/refuting needs the **real training data + camera-calibration (m-fixed) setting** + the source instrumentation above. This spec is the semantic anchor, not a verdict.

---

## Equivalence-Validation STANDARD (the "definition of done") + maturity triage

A mechanism is **equivalence-complete** — the bar an audit must hit before claiming it validated — iff ALL four hold:
1. **app_scenario** — a *deterministic* operation sequence on the LIVE app with a scalar/observable readout.
2. **source_test (independent)** — a unittest against the REAL PyTorch source, written **from the semantic claim, not copied from the app**, and actually run.
3. **equivalence** — the app observable and the source result **agree at the load-bearing precision (sign + zero/non-zero)**; magnitudes are labeled abstractions and need not match.
4. **reproducible** — source script committed (`_audit_sandbox/t_*.py`) AND the app scenario scripted (cache toolkit).

> Note on prior wording: the interactive validation runs done so far were **faithfulness/no-astonishment** audits (interaction-only) + separate main-session source probes — they did NOT close this app↔source equivalence loop per-mechanism. R9/R13 below is the **first** mechanism taken to the full standard.

### Reference (PASS) — R9/R13: S1 mask-force / FUT convergence filter — verified 2026-06-04
- **source_test:** `_audit_sandbox/t_a2.py` (independent; isolates a cell's own center-slice self mask-force via `backward()`). Result: DEP S1 = **+0.000** (absorbing); FUT S1 = **+0.300** (reinforcing).
- **app_scenario:** select S1 cell (default preset (2,5)); read inspector **"m TOTAL"** under Rectify=DEP, FUT(filter SOURCE), FUT(filter FIXED). (Cache toolkit: `__clickCell(2,5)`, `__btn('DEP')`/`__btn('FUT')`, `__src(0)`; read `__mg().mTotal`.)
- **equivalence:**

| Config @ S1 | App "m TOTAL" | Source self mask-force | sign/zero |
|---|---|---|---|
| DEP | +0.000 (absorbing) | +0.000 (absorbing) | ✅ both 0 |
| FUT, filter SOURCE | +0.100 (reinforcing) | +0.300 (reinforcing) | ✅ both >0 |
| FUT, filter FIXED | +0.000 (absorbing) | (=DEP by construction) 0 | ✅ both 0 |

⇒ **EQUIVALENCE HOLDS** at sign/zero (magnitudes 0.10 vs 0.30 = labeled abstraction). **This is the gold standard an audit must reproduce to call a mechanism "equivalence-validated."**

**INDEPENDENTLY re-validated (2026-06-04)** — a fresh independent agent (no prior context, source = ground truth, app/docs = claims) **wrote its OWN source test `_audit_sandbox/t_indep_matched.py`** (reads source → states expectation → calls real `SMap` forward+backward), **independently found the matched-target cell in the app**, and reproduced the same equivalence (DEP 0 / FUT-source +0.3↔app +0.1 / FUT-fixed 0; all sign/zero EQUIVALENT). This is what the standard requires (independent test-generation + independent interaction mapping), vs the earlier main-session self-check which reused a pre-written probe. **The independent run also surfaced a real defect (now FIXED):** the RECTIFY-layer META.S1 static label "ABSORBING / m-grad→0" was shown even in FUT-source where the live m-grad = +0.1 ≠ 0 ⇒ label contradicted the load-bearing number. Fixed: the panel now flags **"⚠ NOT absorbing here (g_m≠0)"** when an S1 cell's actual m-grad is non-zero (Bug A active). (Lesson: independent validation catches overclaims the self-check missed.)

### Maturity triage (honest — a REAL reason for anything below COMPLETE)
| Mechanism | Status | Reason |
|---|---|---|
| **R9/R13** (S1 mask-force) | ✅ **COMPLETE (reference)** | — |
| **Force-3 / coord gradient (gz)** | ◑ NEAR | source `t_force3.py` (z-grad≡0; x/y real) + app gx/gy bars both exist; gap = the app's **hand-coded** gx/gy and source's **autograd** x/y don't yet share an identical eliciting loss (app = direction; source = l1 field loss) ⇒ needs ONE scenario/loss alignment to compare cleanly at sign/zero. |
| **M7** (detached-routing identity) | ◑ PARTIAL | the claim is **structural** (no gradient to the selection; `weights_x.detach()`), not a scalar ⇒ equivalence needs a graph-connectivity assert (`weights_x.grad_fn is None` / grad reaches xyz only), a *different form* than the sign/zero scalar; that source probe isn't written yet. |
| **A1 single-forward** (Lemma 3 no-uncover) | ◑ closable | source `t_a1a4.py` overwrite test (`pred_m@target=1`) ↔ app occlusion-preset cell stays active — pairable; just not yet scripted as a paired app_scenario+equivalence. |
| **A4 multi-step dynamics** | ✗ BLOCKED | the app/proxy per-step **stochastic routing ≠ the real training dynamics**; GD-on-field is an unfaithful proxy (Φ-flicker = instrument artifact) ⇒ a faithful dynamics equivalence needs the **real training harness (absent)**. |
| **L53-63 nest / G1·G2 / far-k_const** | ✗ INCOMPLETE | (a) **no source probe yet** (`t_nest.py` missing); (b) the app reflects G1/G2 **diagnostically only** — the mask/xyz split is NOT applied to dynamics (pull-model rewrite deferred), so there is **no app *outcome*** to compare; (c) the deciding **m-fixed regime + real data are absent**. |
| **M1 / M11 / M14 + ~27 un-promoted** | ✗ INCOMPLETE | not yet promoted to the spec schema; no `source_test`; app reflection partial — deferred coverage work. |
| **Strict-Lyapunov / convergence rate** | ✗ BLOCKED | needs the real training harness (network + SGD + data); not in the workspace. |

**Honest read:** exactly **one** mechanism (R9/R13) is equivalence-complete today; Force-3 and A1-single-forward are a short step away; the rest are blocked for *structural* reasons (different equivalence form, unfaithful dynamics proxy, missing source probe, missing real data/harness) — not for lack of effort. An audit should treat R9/R13 as the template and only mark a mechanism "validated" when it reproduces that loop.
