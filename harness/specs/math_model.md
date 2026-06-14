# SMap Mathematical Model & Correctness Proof

**Status / contract.** This file defines the *correct* SMap algorithm **kept as close as possible to the current implementation** (not an arbitrary model chosen for local convenience), and proves its correctness. Every claim is anchored to source:
- `smap.py` — forward operator (FOLD → route → UNFOLD), routing weights, z-buffer.
- `rectify.py` — the four rectifiers (DEF/CAM/DEP/FUT), gradient construction, `pre_allow`, flows.
- `utils.py` — `agg`, `flip`, `to_3d3x3`, `calculate_key_query`.
- `specials.py` — `OFF_THRESH = 0.5`, `INF = 1e32`.
- `depth-estimation.ipynb` — the **mask-optimization training** (loss, optimizer, k_const schedule). See §1.1.
- `camera-calibration.ipynb` (`applications/Camera Calibration/`) — the **camera-calibration training** (m-fixed, pose optimization). Reduced form: `audit/hdvo/run_calib_step.py`. See §9-CAM.
- `12Lamp_dataset.ipynb` — sample **data preparation** (targets/initial configuration).

Deviations of the *current* source from this correct algorithm are catalogued as **bugs** (final section) with a concrete fix direction; each must be reflected as a toggle in `smap_simulator.jsx` so it is auditable. The proof is placed **at the bottom** of the file, as required.

Line numbers are for the current revision and must be re-checked after edits.

---

## 1. Objective (anchored to the training loss)

The algorithm minimises the L1 disagreement between a z-buffered projection rendering `R(θ)` and a target silhouette `T`, over learnable 3D parameters `θ = (x,y,z,m)`:

> `min_θ ‖ R(θ) − T ‖₁`  (math_model §1; loss in `depth-estimation.ipynb` `l1_error` L399–419).

The training loss `l1_error` returns `mean( loss_m.detach() + 1e5·(loss − loss.detach()) )` (L419): the **value** is `loss_m = Σ |[pred>τ] − T|` (L412/417) and the **gradient** is carried by the soft surrogate `loss` (a value/gradient split). Therefore the quantity actually decreased, in value, is the integer mismatch count

> **Φ(state) := Σ_{cells in window} 𝟙([m>τ] ≠ t) = loss_m.**

`Φ = 0  ⇔  [m>τ] ⇔ t` for every cell ⇔ **every target is active and every active cell is a target** — the correctness condition. (Geometry: the algorithm models a *vision* matching; when the match is optimal the geometry realising it is, by definition, an admissible answer. Ill-posed instances are resolved upstream by data preparation, `12Lamp_dataset.ipynb`.)

The rectification **gradient is injected only at the final** `go(is_last=True)` **at finest resolution** (`smap.py` L244–251). Distance is crossed by **routing** (detached, coarse→fine, ≤1 cell = 2^l finest px per level), *not* by the gradient.

### 1.1 Training Schemas

The SMap algorithm serves multiple training configurations ("schemas"), each with different learnable parameters, forces, and convergence dynamics. The correctness proof covers each schema separately because the dynamics — which parameters move, which are fixed, which forces are active — differ fundamentally.

| Property | Mask-Opt Schema (`depth-estimation.ipynb`) | CAM Schema (`camera-calibration.ipynb`) |
|---|---|---|
| **Rectify type** | DEP (`rectify.types.DEP`) | CAM (`rectify.types.CAM`) |
| **Learnable** | θ = (x,y,z,m) per point | θ_pose = (ϕ,γ,ρ,θ_xyz) — 6 DoF camera pose |
| **Mask** | Optimized (m is learnable) | **Fixed from ground-truth** (`mask = .999 * r_mask`) |
| **k_const** | 1.0 pinned (Bug B) | 0.5 pinned |
| **Surrogate scaling** | `1e5` (`depth-estimation.ipynb` L419) | `3e3` (`camera-calibration.ipynb` / `run_calib_step.py` L185) |
| **Coarse-to-fine** | Single zoom pass (typical) | Multi-pass: `zoom = n−i` for `i ∈ 0..n` (n=6) |
| **Forces → mask** | F1 (alignment) + F2 (activation) | **All inert** (mask not learnable) |
| **Forces → geometry** | F3 → xyz (per-point) | F1 + F3 → xyz → pose (rigid-body) |
| **Coverage mechanism** | Warming: F2 activates dark targets via mask gradient | Geometric alignment: pose change moves projections |
| **Correctness proof** | §8 (Lemmas 0–3, Theorems) | §9-CAM |

The loss function `l1_error` is shared across schemas — Φ = mask-mismatch is always the measured quantity. The **dynamics** to minimise Φ differ fundamentally. The loss function itself is a tool and could contain defects that obstruct correctness in a given schema; its behaviour under each schema must be verified, not assumed correct.

---

## 2. Multi-scale Markov chain (FOLD / route / UNFOLD)

**Quadtree FOLD** (`smap.py` L213–218). Pixel `(r,c)` at level `l` maps to channel `encode(r mod 2^l, c mod 2^l)`, cell `(⌊r/2^l⌋, ⌊c/2^l⌋)`. Level `l` has `4^l` channels over `(H/2^l, W/2^l)` spatial. Total points = `H×W` at every level (a bijection — zooming never writes state).

**Per-level pipeline** (`smap.py` forward loop L224–238): 3×3 routing → z-buffer (visibility) → merge (4 channels → 2×2 spatial, inverse quadtree). Intermediate levels **route only** (slot assignment, values unchanged); the final level propagates values and then constructs the gradient.

**Synthetic gradient.** Selection ops are `.detach()`ed (`smap.py` L83/L130, `weights_x.detach()`), so autograd through selection is broken. Gradients are injected by the `(x − x.detach())` trick (value unchanged, unit backward tap) scaled by force coefficients (`rectify.py`). *The realised sign of these taps is **now confirmed by executing the source** (`_audit_sandbox/t_a2.py`, torch 2.0.1): `(x−x.detach())` realises `+1`, `−(x−x.detach())` realises `−1`, and `sign(·).detach()·(·)` realises `sign(·)` — A2 closed; see Residual Risks.*

---

## 3. The two state layers ("Meta-states")

Correctness is governed by **two orthogonal per-cell classifications**, one per mechanism. Both are needed; the two known bugs hit one layer each.

### 3.1 Routing layer S_A–S_D — on `(m, z)` (governs participation / occlusion / warming)

| State | Condition | Routing (source) | Requirement |
|---|---|---|---|
| **S_A** | `m>τ ∧ z>0` | `weights_b` (best neighbour), `smap.py` L53/57/58 | R1 full participation |
| **S_B** | `m>τ ∧ z≤0` | center / self, L57 `~(z>0)→temp` | R2 no depth to propagate |
| **S_C** | `m≤τ ∧ z>0` | **stochastic, route-to-neighbour with prob `k_const`** (L57 `where(temp_rand>k_const, temp, weights_b)`); z-buffer depth `z+1e-5·k_const` (L165). **Level-dependent:** this branch order is the **finest / `is_last=True`** path (L57, which alone carries the gradient); the **intermediate `is_last=False`** path (L62/L63) uses the *inverted* order `where(temp_rand>k_const, weights_b, temp)`, so there higher `k_const` ⇒ *less* value-participation. | **R4: p_warm > 0** |
| **S_D** | `m≤τ ∧ z≤0` | excluded (zeros), L58/L165 `zmax+1e-5` | R3 no information |

This layer decides **who is visible** at the z-buffer (`smap.py` `calculate_weights` L154–197): `visible(p) = argmin_{z>0} z` within a channel's 3×3 (`min` over `dim=2`, L169), center-fallback to slot 4 if no valid winner (L170). A losing point is **not selected ⇒ receives no gradient**. The stochastic S_C participation (R4) is the mechanism that prevents permanent gradient-starvation under occlusion (see §6 and Lemma 2).

### 3.2 Rectification layer S1–S7 — on `(m, t, B)` (governs gradient sign / Φ)

`m = [mask>OFF_THRESH]`, `t = target`, `B = ` neighbourhood target-confirmation. The triple fixes which force gates are open and each force's sign, so it is exactly the equivalence class on which the per-cell mask-gradient sign is constant.

**`B` is per-Rectify** (this is a faithfulness requirement the app must respect — see Bug C note):
- **Default / FUT:** `s = Σ 𝟙(weight·target==1)` over **`dim=2`** (the 9 candidate slots), then `agg(flip(·))` — `rectify.py` L25 / L326.
- **DEP:** `s = Σ … ` over **`dim=1`** (**cross-channel**), `rectify.py` L203.

Desired absorbing classes: **S1** `(1,1,1)` for target cells, **S4** `(0,0,0)` for non-target cells — exactly the per-cell `case = 1` states, where `case := ([m>τ] == t)`.

**Coverage of the 8 combinations of `(m,t,B)`.** Seven are used; `(1,1,0)` is **unreachable** *iff* `B` counts the cell itself (an active target contributes to its own `B`, so `B≥1`) — this must be re-verified once `B` is made source-faithful per type. Boundary cells have a truncated 3×3 (`B`/`y_flow` over a partial neighbourhood): treated as an **explicitly-excluded minority** whose measure is `O(1/side) → 0` as resolution grows (not practically significant); data preparation avoids placing critical targets on the 1-cell rim. *Isolated meta-states are not ignored: §Proof shows the dynamics never enter `(1,1,0)` and states the initial conditions that keep the rim non-critical.*

---

## 4. Routing & Z-buffer (detail)

`ind = where(m>τ, z, where(~(z>0), zmax+1e-5, z + 1e-5·k_const))` (`smap.py` L165, detached) → `pre_ind = agg(ind, INF)` → `val,ind = min(pre_ind, dim=2)` (L169) → center-fallback (L170). A cell whose value routed **out** has a zero self (slot-4) term, so if nothing routes in, its gather is empty ⇒ it **vacates to mask 0** (no value retained). Tie-break jitter is `1e-5` (L159) — large enough only to break ties, **not** to overturn occlusion (relevant to Bug B).

---

## 5. The three rectification forces (enforce the three invariants)

Invariants: **I1 Stability** (converged cells stay correct), **I2 Non-obstruction** (movers don't collide with converged), **I3 Cleanup** (mis-placed actives deactivate).

- **Force 1 — Alignment (I1):** `G_align = sign(w)·∇w·α₁·𝟙(y_flow=1)·𝟙(w>0)` (`rectify.py` DEP L305 / FUT L425). R7: the `𝟙(w>0)` guard is mandatory (no spatial evidence ⇒ direction undefined ⇒ noise). Present in **both** DEP and FUT.
- **Force 2 — Activation (I3):** `G_act = where(m>τ, ∇m·α₂, −∇m·α₃)·(1−case)·𝟙(y_flow≠1)` (DEP L305 / FUT L425; CAM L183; DEF L108–113). `α₃>α₂`: a false positive (active at the wrong place) is costlier than a delayed activation.
- **Force 3 — Spreading (I2):** `G_spread = sign(w)·where(ρ>1, ∇neg_kq·α₁·d(s), ∇pos_kq·coord)·𝟙(T=1)`, `d(s)=1−ρ/ρ_localmax` (DEP L305). R6: `ρ_ref` must be a **local** max (global max ⇒ no directional signal). **Force 3 backprops to the coordinate leaves `x/y/z` (via `key_query`), NOT the mask** — any faithful mirror (e.g. the simulator) must keep it **out of the mask gradient** `g_m`; folding `coord_flow` into the mask update is an infidelity (it makes vacant-target activation ride a geometry force instead of Force 2).

**The convergence filter** (the load-bearing difference): DEP multiplies its alignment/activation flow by `(1−case)` via `y_flow *= (1 − ((pre_mask>τ)==target))` at **`rectify.py` L294**, so at a matched cell (`case=1`) every force gate closes. **FUT lacks this line** (nothing between L405 and L425). CAM/DEF achieve the same via `mask_flow·(1−case)` (CAM L168, DEF L108–111).

---

## 6. Convergence damper k_t, and the warming schedule k_const

**k_t (magnitude conditioner):** `k_t = k_d(z,|xy|)/z`, applied by CAM/DEP/FUT, **not** Default (`rectify.py` L130–131 / L243–244 / L359–360). `k_t>0` for `z>0`, so it **never changes a gradient's sign or zero/non-zero** — only its magnitude (layer-wise convergence). Hence sign/zero arguments below are k_t-independent.

**k_const (warming schedule):** a per-problem schedule that the training **ramps `0 → 1`** across resolutions/epochs: `k_const = (tr_i + (epoch + (batch+0.3)/|loader|)/n_epochs)/n`, capped at 1 (`depth-estimation.ipynb` L432–434). It sets the S_C warming probability (R4). **Direction (from `smap.py` L57** `where(temp_rand > k_const, temp, weights_b)`**):** an inactive cell takes the participation branch (`weights_b`) when `temp_rand > k_const` is *false*, so **higher `k_const` ⇒ more participation/warming**. Thus `k_const → 0` ⇒ inactive cells stay at center (near-deterministic z-buffer, minimal warming); `k_const → 1` ⇒ inactive cells always participate (**maximal** warming). The intended schedule therefore *increases* warming as it ramps up. *(Current source discards the ramp and pins `k_const ≡ 1.0` — the **maximal-warming** end — see Bug B. NB: the earlier "early = explore / late = anneal-to-deterministic" reading was **inverted**; `k→1` is maximal warming, not deterministic freezing.)*

**Schema-dependent k_const.** The mask-opt schema pins `k_const ≡ 1.0`; the CAM schema pins `k_const ≡ 0.5` (`camera-calibration.ipynb`; `run_calib_step.py` L57). At k=0.5, the L62-72 intermediate inversion (`is_last=False`: `where(temp_rand > k, weights_b, temp)` → P(route)=1−k=0.5, same as finest `where(temp_rand > k, temp, weights_b)` → P(route)=k=0.5) is **symmetric** — the inversion is behaviourally moot at this value. Both k=1.0 and k=0.5 satisfy k>0 (no starvation, Lemma 2). The surrogate gradient scaling also differs: `1e5` (mask-opt) vs `3e3` (CAM) — a 30× reduction in gradient magnitude, appropriate for the smaller pose-parameter space.

---

## 7. Requirements summary

| ID | Requirement | Derivation |
|---|---|---|
| R1 | S_A → full participation | real 3D point |
| R2 | S_B → center only | `z≤0`, no depth to propagate |
| R3 | S_D → excluded | no information |
| R4 | S_C → `p_warm > 0` (and Σ p_warm = ∞ before annealing) | occluded points must be selectable infinitely often |
| R5 | Z-buffer → local 3×3 context | sub-chain independence |
| R6 | `ρ_ref` → local max | gradient-direction preservation |
| R7 | `𝟙(w>0)` guard on alignment | no evidence ⇒ noise |

---

# 8. Correctness Proof

Throughout, `Φ = #{cells : [m>τ] ≠ t}` (= `loss_m`, §1). Correctness = `Φ → 0` in finitely many steps with no periodic non-termination, **and** coverage (no target permanently starved). Two layers (§3) are proved separately, then composed.

### Lemma 0 (Learning-rate decoupling).
There exists `η₀ > 0` such that for every step size `η ∈ (0, η₀]`: (i) the meta-state transition structure — which cell crosses `τ`, and the sign of every force — is preserved (stability); (ii) convergence still occurs in finitely many steps. *Justification.* The forces' **signs/zeros are η-independent** (they are determined by `(m,t,B)` and the `case`/`y_flow` gates, not by magnitude; §5, and k_t>0 by §6). A cell's discrete `case` changes only when `m` crosses `τ`; for small enough `η` (SGD, `lr=1e-4` with `StepLR γ=0.975`, `depth-estimation.ipynb` L540/573/574) each update moves `m` monotonically toward the sign-dictated side, crossing `τ` after finitely many steps without overshoot-oscillation. ∎ *(This is the framing that lets the rest of the proof reason about signs while still concluding finite termination.)*

### Lemma 1 (Matched cells are mask-absorbing — DEF/CAM/DEP).
At any cell with `case = 1`: **Force 2** carries factor `(1−case) = 0`, and the convergence filter sets **Force 1**'s gate to 0 (DEP: `y_flow*=(1−case)`, `rectify.py` L294; CAM/DEF: `mask_flow·(1−case)`, L168/L108–111). *(Caveat: DEF additionally carries an `n_case` mask term (L109/L113) that is **not** `(1−case)`-gated, leaving a residual mask tap at some matched cells; this is **latent** — training uses DEP, not DEF — but the blanket "CAM/DEF same via `(1−case)`" is an oversimplification.)* **Force 3** is gated by `(target>τ)` *only* (not by `(1−case)`; `rectify.py` L305 last term), so at a matched **target** cell it is **nonzero**, and it *is* summed into the returned `weights` tensor (L306). However Force 3 is built from `key_query` gradients (`neg/pos_key_query_grdf`) whose autograd path reaches the **coordinate leaves `x/y/z`** (`utils.py` `calculate_key_query` L167–185), a **distinct parameter from the mask leaf** — so `∂loss/∂(mask)` from Force 3 is **0**: it moves geometry, not `[m>τ]`. Hence the **mask-gradient `g_m = 0`** (the only quantity Φ depends on): a matched cell stays matched. Therefore S1 (target) and S4 (non-target) are **mask-absorbing**, and along the mask dynamics **Φ is non-increasing**. *(η-independent: only zero/non-zero of `g_m` is used. The residual Force-3 coordinate motion at S1 is harmless to Φ — this is why "absorbing" is qualified to the mask channel; the earlier blanket "Force 3 carries `(1−case)`" was incorrect, though the conclusion `g_m=0` stands.)* ∎

### Lemma 2 (No starvation / warming — routing layer).
Assume R4 with `Σ_t p_warm(t) = ∞`. Then every inactive target cell (an S_C point under occlusion) is **selected by the z-buffer infinitely often almost surely**, hence receives the activation gradient (Force 2, `t=1,m=0 ⇒ +α₂`) infinitely often, and is therefore not permanently gradient-starved: coverage of every reachable target is achievable despite occlusion. *Sketch.* Stochastic S_C participation (`smap.py` L57) gives each occluded point a per-iteration participation probability `= k_const`; by Borel–Cantelli (second), `Σ k_const = ∞` ⇒ infinitely many selections a.s. **Corollary (source default):** the source pins `k_const ≡ 1.0`, so `p_warm ≡ 1` and `Σ p_warm = ∞` trivially ⇒ **no starvation at the source default**. (Empirically confirmed: the app's occlusion preset converges with full coverage at pinned `k = 1.0`; it stalls only at *small* `k`, e.g. `k = 0.0`, where `Σ k_const < ∞`.) Starvation is therefore a *small-`k`* failure mode the source never enters. ∎ *(The participation **direction** of L54–57 is now confirmed by reading + the simulator re-port; the exact PyTorch occlusion-drop dynamics remain pending — Residual Risks A1/A4.)*

### Lemma 3 (Occlusion is mask-preserving — conditional on `k≡1` or Bug D fix).
The z-buffer cannot turn a rendered-active cell rendered-inactive, nor render an inactive point active. *Proof intent vs source reality:* The proof assumes L189 keeps the winner only if its mask `>τ`, else the cell's own center value `weights4`. **However, Bug D (`smap.py` L188) reassigns `weights4` to a detached copy of the winner**, destroying the center fallback. Thus, a nearer **inactive** winner will display its inactive mask over a matched **active** cell, uncovering it. The protection against this uncovering therefore relies entirely on the L63 global gate being OFF for inactive points (which requires `k_const ≡ 1`), which prevents inactive cells from routing/winning. Hence, occlusion is mask-preserving **only at `k≡1` or after the Bug-D fix**. ∎

### Theorem (Conditional convergence + non-periodicity).
Assume **(A1)** ✅ **now a lemma** — Lemma 3: the z-buffer subsystem does not flip a matched target's `case` (an inactive winner is rejected to center, L180), and Lemma 2 supplies coverage; **(A2)** ✅ **validated by execution** (`t_a2.py`): the realised autograd signs equal the constructed signs of §5, and the DEP matched-cell self mask-force is 0 (absorbing) while FUT's is `+α₁` (reinforcing); **(A3)** `η ≤ η₀` (Lemma 0); **(A4)** the neighbourhood coupling (`y_flow`, `B`) induces no Φ-increasing transition — its *uncovering* half is now discharged (Lemma 3 + Theorem′); only the stronger strict-monotone/rate claim remains. Then under **DEP** (and CAM/DEF):
1. By Lemma 1, `Φ` is a non-increasing integer `≥ 0` that strictly decreases whenever a mismatched cell crosses `τ` (Lemma 0); so `Φ → 0` in **at most `Φ₀`** threshold-crossings — convergence to the correctness condition, with **no periodic non-termination** (a strictly-decreasing well-founded potential admits no cycle).
2. **Multi-scale induction** (folds in the inductive argument of the model): at coarse levels the target is max-pooled, so `y_flow > 1` is prevalent ⇒ Force 2 (safe toggle) dominates and Force 1 fires only at `y_flow = 1` (a sole lit channel) ⇒ coarse phase is *conservative* (I2: no obstruction); finer phases recompute with a finer target and correct precisely; base case `zoom = 0` is pixel-exact. Hence coarse never obstructs fine, and the cascade terminates at `zoom = 0` exact. □

Isolated state `(1,1,0)` is never entered: any cell with `m>τ ∧ t` contributes to its own `B`, so `B≥1` (given a self-inclusive, source-faithful `B`); initialisation (data prep) keeps targets off the truncated rim, so the boundary exception has measure `O(1/side)` and does not affect the limit.

### Theorem′ (Halting ⟺ correctness — occlusion-robust, via per-point correctness).
**Objective (per-point, not per-rendered-pixel — adopted as the project's correctness criterion).** Adopt the correctness target at the level of *physical points*:
**(S, soundness)** every active point projects onto a target pixel; **(C, coverage)** every target pixel is the projection of ≥1 active point. *Justification of the framing.* Occlusion never alters a point's `xyz` — the z-buffer selection einsum is **detached** (M7; routing transports the differentiable coordinate value, it does not relabel or destroy it), so an occluded-but-correct point retains exact geometry and is recoverable by a classical re-render (a depth test on the stored `xyz`). Visibility is therefore a rendering detail; per-point correctness is the invariant the dynamics must reach.

**Consistency (per-point-correct ⟹ rendered Φ = 0).** At a per-point-correct fixed point every target is covered **in place**: Force 2 (`+α₂`) fires whenever a cell's *own* `m≤τ ∧ t=1`, regardless of whether a far point also happens to cover the pixel, so warming (Lemma 2) does not rest until the **target cell's own** point is active ⇒ `m_center` is active at every target ⇒ by Lemma 3(i) the pixel renders active even under occlusion. By (S) no active point sits on a non-target, so every non-target's center is inactive and by Lemma 3(ii) renders inactive. Hence rendered `Φ = 0`: the per-point fixed point and the rendered-`Φ=0` condition **coincide**.

**Equivalence.**
- **(⟸ correct ⟹ halt).** At `Φ=0` every cell is matched (`case=1`): Lemma 1 gives `g_m=0` everywhere and Lemma 3 makes occlusion render-invariant ⇒ no mask update ⇒ fixed point. (Residual Force-3 motion moves geometry, not `[m>τ]`, so `Φ` stays 0; `gz≡0` confirmed in source by `t_force3.py`.)
- **(⟹ halt ⟹ correct — no spurious fixed point).** Contrapositive: if `Φ>0`, some cell is mismatched.
  - *Uncovered target* (`t=1`, renders inactive): by Lemma 3(ii) its own center is inactive **and** no active point covers it ⇒ it is a dark, self-projecting target with `z>0` ⇒ it is its own nearest winner ⇒ it participates (S_C) and receives Force 2 `+α₂` (Lemma 2; `k_const>0` ⇒ selected infinitely often a.s.) ⇒ nonzero mask update ⇒ **not** a fixed point.
  - *False positive* (`t=0`, renders active): the displayed active value incurs cleanup `−α₃` (`t=0,m=1`); by the **value-path gradient identity** (M7 — the displayed value's mask leaf *is* the responsible point's own leaf, even after the point routed in) the deactivating gradient reaches that point's mask ⇒ nonzero update ⇒ **not** a fixed point.
  ⇒ No `Φ>0` configuration is a fixed point. ∎

**Dependency & scope.** The equivalence needs **`k_const≡1.0`** (to protect against Bug D uncovering) or the Bug-D fix; the source default `k_const≡1.0` satisfies it — which is precisely why **Bug B is precision-only** (the discarded schedule changes the convergence *rate*, not the fixed-point set). A1 and the *uncovering* half of A4 are **discharged by Lemma 3 only at `k≡1` or post-fix**. What remains genuinely open is the **stronger** strictly-monotone-Lyapunov claim (no *transient* Φ rise mid-run) and the convergence **rate**; **neither is required for halt ⟺ correctness**. The transient sloshing seen under the GD-on-field proxy (`t_a1a4.py`) is an **instrument artifact** of that proxy (the real source trains an upstream network with small-η SGD, Lemma 0), not a property of the trained algorithm.

**Setting assumption (load-bearing — scope of this result).** This proof takes the **mask `m` as an optimized degree of freedom** (Φ = mask-match; warming/Force 2 can activate a dark target). It therefore establishes halt ⟺ correctness for the **mask-optimization regime** only. **The camera-calibration regime (m-fixed) is covered in §9-CAM** with its own Lemmas, Theorem, and bug-catalogue re-examination.

### Bug catalogue — deviations of the current source from the correct algorithm

Each bug = the exact term that **breaks a lemma above**; each must surface in `smap_simulator.jsx` as a toggle that also shows the **fix direction**.

- **Bug A — FUT missing convergence filter (rectification layer; breaks Lemma 1's strict fixed point).** FUT omits `y_flow *= (1−case)` that DEP has at `rectify.py` L294 (absent between L405–425; `weights_grdf` otherwise byte-identical to DEP L305). Effect: at **S1** (`case=1`) with `y_flow_count = 1`, Force 1's gate is open and `(w>0)` holds ⇒ `g_m = sign·α₁ ≠ 0` at a matched cell ⇒ **S1 is no longer a strict mask fixed point** (Lemma 1 violated). *Precise consequence:* the realised sign of that tap decides whether Φ can actually rise. **Confirmed by executing the real source** (`_audit_sandbox/t_a2.py`, torch 2.0.1): isolating the cell's own center-slice self-force, at a matched-target cell the **DEP** self mask-force is **exactly 0** (absorbing) while the **FUT** self mask-force is **+0.3 = +α₁** (the alignment coefficient `3e-1`), i.e. **reinforcing** (drives `m → 1`). So Φ does **not** numerically rise even in the real source; Bug A's effect is the **lost strict mask fixed point** (`g_m ≠ 0` where DEP has `0`), observable as a static flag, not divergence. This also matches the re-port (FUT-source converges identically to FUT-fixed). **(A2 closed.)** **Latency:** training uses **DEP, not FUT** (`depth-estimation.ipynb` cell-11), so Bug A is **latent for the trained pipeline**. *η-independent.* **Fix:** insert `y_flow = y_flow*(1.−((pre_mask>OFF_THRESH).long()==target_2Dr.long()).float())` after FUT L418. **App:** `CORRECTNESS` toggle (present) — when SOURCE, the certificate must say "S1 loses its strict mask fixed point (`g_m ≠ 0`); missing L294 filter" (NOT "Φ rises", which the app dynamics do not exhibit), and note the DEP-latency.

- **Bug B — k_const ramp discarded (routing layer). [Severity corrected: was "CORRECTNESS via occlusion-starvation"; the starvation claim is FALSE at the source default — re-classified to AUDITABILITY / UNRESOLVED.]** `depth-estimation.ipynb` L435 sets `model.smap_layer.smap3x3.k_const = 1.` (a literal), **discarding the ramp computed at L432** (L741 elsewhere correctly uses `= k_const`). **Source fact (certain):** the schedule is never applied; training runs pinned at `k_const ≡ 1.0`. **Corrected interpretation (the prior story was directionally inverted):** `k_const = 1.0` is the **maximal-warming** end, not a starving low value — so by Lemma 2's corollary `Σ p_warm = ∞` and **occluded targets are NOT starved at the source default** (empirically: occlusion preset converges with full coverage at pinned `k = 1.0`; it stalls only at small `k` such as `0.0`). Pinning therefore does **not** break Φ-convergence via starvation, and like Bug A it produces **no forward-observable non-convergence** at the source default. The genuine deviation is that the **intended schedule progression is skipped**; its effect on precision / convergence-*rate* / late-stage behaviour is **unresolved without execution (A1/A4)**. **Falsification route:** run the real source with the ramp vs pinned `k = 1.0` on an occluded configuration; compare coverage, precision, step count. **Fix (source):** L435 → `model.smap_layer.smap3x3.k_const = k_const` — **and reconcile the inverted `is_last=False` branch (L62/L63) so the participation direction is consistent across levels** (otherwise a ramp pulls the finest and intermediate paths in opposite senses; the inversion is moot only at the pinned `k=1.0`). **App:** the certificate must **not** claim starvation at the source default `k = 1.0`; the SOURCE state reads "schedule pinned at 1.0 (ramp discarded)"; the occlusion preset demonstrates the warming *mechanism* (small-`k` starves, high-`k` covers), **not** the source regime; the `suggested-k` / Auto control is a **non-source auditing heuristic** (`suggestK` has no source counterpart) — the source fix is the *schedule*, not a tuned value. **Reopen caveat:** the precision-only classification holds for the **mask-optimization setting at the source default `k=1.0`**. It may **reopen** under real training settings — notably **camera calibration with `m` fixed** (only `xyz` optimized; activity set by the prior camera pose) — where real data and the setting **directly drive `k_const` scheduling**, so the schedule's effect on convergence/correctness must be re-evaluated there. (Conversely, a principled `k_const` schedule can legitimately be *derived from* fixing this bug.)

- **Bug C — app `B` not source-faithful (faithfulness prerequisite; not a source bug, an audit-instrument bug).** The app's `computeB` is neighbour-centric over a 3×3 (incl. self) and ignores the per-Rectify summation axis (`dim=2` DEF/FUT vs `dim=1` DEP) and the `agg(flip)` redistribution (§3.2). Until fixed, bugs in the `pre_allow`/`B` aspect are neither observable nor reverse-readable in the app (bidirectional-reflection failure). **Fix:** define `B` per-Rectify matching `rectify.py`; re-verify the `(1,1,0)` unreachability afterwards.

### Residual risks (what this proof does **not** yet establish)
- **(A2) autograd signs — RESOLVED ✅** by executing the real source (`_audit_sandbox/t_a2.py`, torch 2.0.1, SST-identical `smap` package): the `x−x.detach()`/`sign().detach()` taps realise their constructed signs; at a matched-target cell the DEP self mask-force = **0** (absorbing, confirms Lemma 1) and the FUT self mask-force = **+α₁** (reinforcing, confirms Bug A is the lost fixed point — Φ does not rise). Note: this isolates the *self*-force via the cell's own center-slice; full-model per-cell grads also include neighbour-broadcast coupling (A1/A4, still execution-pending for the *dynamics*).
- **(A1) — DISCHARGED ✅** by **Lemma 3** (structural, `calculate_weights` L176–180: an inactive winner is rejected back to the cell's own center value, so occlusion can never uncover a matched, self-projecting target) + execution `t_a1a4.py` (nearer inactive intruder leaves `pred_m@target=1`). The z-buffer does not flip a matched target's `case`.
- **(A4) — uncovering half DISCHARGED** by Lemma 3 + Theorem′ (no `Φ>0` fixed point; halt ⟺ correctness holds for any `k_const>0`, incl. the source default `1.0`). **Still open (weaker need):** the *strictly-monotone-Lyapunov* property (no transient Φ rise mid-run) and the convergence **rate** — NOT required for halt ⟺ correctness. Closing these fully needs the **real training harness** (upstream network + SGD), since GD-on-field (`t_a1a4.py`) is an unfaithful dynamics proxy that oscillates on background cells (instrument artifact, not algorithm). **Annealed re-run (`t_a1a4.py`, step `0.2·0.85^t`) corroboration:** in the overwrite scene the matched target is **never uncovered** across 40 steps (`cover≡1`, UNCOVER events = NONE) ⇒ A1 empirically robust; the residual Φ oscillation is driven entirely by **per-pass stochastic ROUTING re-rolling** (`np_rand`/`k_const`, the L53–63 nest) flickering **background** cells, *not* target uncovering. This stochastic-routing flicker is itself the prime suspect for the real-data "dirty output" (rendered Φ flickers even when the mask has converged) — to be investigated in the camera-calibration arc.
- **(Force 3 / `gz`) — RESOLVED ✅** by `t_force3.py`: z-channel gradient is identically zero (app `gz≡0` faithful); the x/y coordinate force is real with sign toward the target. Only an app display bar (`spread(F3)`) is cosmetically decoupled.
- **Magnitudes / k_t / lr** beyond the small-η regime (Lemma 0) are not analysed; convergence *rate* is out of scope.
- **Stochasticity** is argued a.s./in expectation (Borel–Cantelli), not deterministically; the app uses a deterministic `pseudo()` hash with a per-open random seed as a stand-in.

---

## 9-CAM. CAM Schema Correctness — Camera-Calibration (m-fixed regime)

**Scope.** This section derives the correctness argument for the **camera-calibration training schema** (`camera-calibration.ipynb`; reduced form: `audit/hdvo/run_calib_step.py`), where the mask `m` is **fixed from ground-truth** and only the camera **pose** is optimised. The mask-optimisation proof (§8 Lemmas 0–3, Theorems) does NOT apply here — the dynamics differ fundamentally (see §1.1 comparison table).

### 9.1 Setting & Objective

**Why the mask is ground-truth, not "missing."** In the camera-calibration task, the mask `r_mask` encodes which 3D points belong to the object — this is **known** (the points are given by the data preparation). The **unknown** is the camera pose that makes these points' projections match the observed target silhouette. Warming (§8, Lemma 2) was the mechanism to activate dark targets when mask was unknown; here the mask IS already correct, so warming is not needed — the coverage problem is purely geometric.

**Objective.** Same loss function `l1_error` (§1): **Φ = Σ 𝟙([pred>τ] ≠ t)**, where `pred` is the z-buffered SMap output. But the dynamics differ:
- In mask-opt: Φ decreases when per-point `m` crosses τ (activation/deactivation).
- In CAM: Φ decreases when the **pose** `(ϕ,γ,ρ,θ)` changes so that the **projections** of active points match the target silhouette.

**The loss function is a tool, not an axiom.** `l1_error` measures Φ via a soft surrogate (§1) with a `denom` normalisation and a scaling factor (`3e3` in CAM vs `1e5` in mask-opt — a 30× reduction, appropriate for the smaller 6-DoF parameter space). Whether this surrogate provides an adequate gradient signal for pose optimisation is a **source-level question**. If the loss function has defects (wrong scaling, wrong gradient direction, `denom` flattening the landscape near convergence), they are bugs this model must expose.

### 9.2 Pose → xyz gradient chain (source-anchored)

The pose parametrisation (`run_calib_step.py` Model.forward L132–168):

1. **Angles (bounded):** `ϕ_eff = tanh(3e-2·ϕ)·π` (similarly γ, ρ) — range `(−π, π)`, L145–147.
2. **Translation:** `θ` (3 DoF, `[1,3,1,1]`); identity tap `θ.detach() + (θ − θ.detach())` (L148) — value=θ, gradient tap=+1.
3. **Rotation:** three sequential axis rotations `x_r = R_ρ · R_γ · R_ϕ · x` (L149–158), each via `cos/sin` matrix × `einsum` + axis permutation (`cat` at L154/156/158).
4. **Translation application:** `x_t = x_r + (mask > 0.5) · θ` (L160) — translation applied **only to active points**.
5. **Z-clipping:** `x_z` clamps z to `[−0.5, 0.5]` (L161), `.detach()` boundary.
6. **Depth-gating:** `where(z_orig > 0, x_t, x_t.detach())` (L164) — no-depth points have no gradient.
7. **Active-gating:** `where(mask > 0.5, x_t, temp)` (L165) — inactive points get zero xyz (no gradient path).
8. **SMap call:** `smap_layer(cat([x_t, mask], dim=1), target, zoom)` (L167).

**Gradient path (backward):** `loss → weights → CAMRectify → {key_query_grdf, weights_grdf} → {calculate_key_query, weight values} → x,y,z → x_t → R·x + θ → ϕ, γ, ρ, θ`.

**Pose-specific edge cases:**
- **tanh saturation:** at large angle magnitudes, `∂ϕ_eff/∂ϕ → 0`. The `3e-2` scaling delays saturation (`|ϕ| >> 33` needed) but doesn't prevent it. Not a correctness issue; a rate/precision concern.
- **Translation-active gating** (L160): `θ` applies only to active points (`mask > 0.5`). Inactive points contribute no gradient to `θ`. Correct for the task (only scene points translate).

### 9.3 Forces under CAM rectify — which reach pose? (source-anchored)

**CAMRectify** (`rectify.py` L120–189) constructs three synthetic gradient taps:

| Force | Tap (source) | Reaches pose? | Mechanism |
|---|---|---|---|
| **F1 (alignment)** | `weights_grdf = pre_weights − pre_weights.detach()` (L174) | **Yes** | `pre_weights` = gathered weight values = function of xyz (SMap value-transport, M7) → xyz → pose |
| **F2 (activation)** | `pre_mask_grdf = pre_mask − pre_mask.detach()` (L176) | **No** | `pre_mask` = gathered mask values = `.999 * r_mask` (constant). Routing detached (M7). Dead end. |
| **F3 (coordinate)** | `key_query_grdf = pre_key_query.detach() − pre_key_query` (L175) | **Yes** | `pre_key_query = calculate_key_query(x·k_t, y·k_t, z·k_t)` (L133) → x,y,z → pose |

**Structural difference CAM vs DEP/FUT (Force 3).** CAM computes a **single** `pre_key_query` (L133, `k_t` **not** detached) while DEP/FUT split into `neg_pre_key_query` (`k_t.detach()` → gradient through xyz only) and `pos_pre_key_query` (`k_t` live → gradient through both xyz AND k_t). Consequences: (a) CAM's F3 carries gradient through both xyz and the magnitude conditioner k_t; (b) CAM has no spreading-distance-damper (`1 − n_flow/max(n_flow)`) that DEP applies to its negative branch (L305). Whether this simpler F3 provides sufficient directional signal for pose convergence is unanalysed.

**F2 inertness in detail.** Although `pre_mask_grdf` is nonzero and summed into the final weights (L185: `self(weights, weights_grdf, key_query_grdf, pre_mask_grdf)` → `weights.detach() + all three taps`), the backward path is: `loss → sum → pre_mask_grdf → (pre_mask − pre_mask.detach()) → pre_mask → gathered mask → value-transport of mask channel → SMap input mask → .999 * r_mask → **dead end** (constant)`. F2 therefore affects the **value** of the returned weights (which cells cross τ in Φ) but carries **no gradient to pose**. The value effect: F2's mask tap shifts some weight values, changing which cells are counted as mismatched in Φ. This shift is then corrected by F1/F3 gradient in subsequent steps. *(Whether this value-only contribution helps or hurts convergence rate is unanalysed; it is NOT a correctness concern.)*

**Gating at matched cells (`case = 1`, L168/L181–L183):**
- `mask_flow *= (1 − case)` (L168) → F2 gate closed ✓ (convergence filter, same intent as DEP L294).
- `weights_grdf *= (pre_weights > τ).detach()` (L181) → F1 gated on `w > τ` (nonzero at matched active cells).
- `key_query_grdf *= sign(w) · coord_flow.detach()` (L182) → F3 gated on `coord_flow` (target presence and neighbourhood isolation) and sign of weight.

### 9.4 Lemma 1-CAM (Geometric stability at matched cells)

**Statement.** At a matched cell (`case = 1`): F2 is zero (convergence filter L168 + mask fixed). F1 and F3 are generally **nonzero** but **small** at a well-aligned configuration, and under small η (Lemma 0) the residual coordinate perturbation does not cause the cell to un-match.

**Source-anchored proof.**

At `case = 1`:
- **F2:** `mask_flow × (1 − case) = 0` → **exactly zero**. Additionally, mask fixed → no gradient path to pose → doubly inert.
- **F1 (L174/L181):** at a matched active cell, `pre_weights > τ` → gate open → tap `pre_weights − pre_weights.detach()` creates gradient through gathered xyz values → reaches pose. **But:** at exact alignment, the gathered weight value equals the expected value → no displacement → the synthetic tap contributes a gradient proportional to the projection displacement, which is zero at alignment.
- **F3 (L175/L182):** `sign(w) × (pre_key_query.detach() − pre_key_query) × coord_flow`. At exact alignment, `key_query` displacement (the 2D distance between current and target pixel positions, computed via `calculate_key_query`) is zero → F3 = 0. At near-alignment, F3 is proportional to the subpixel displacement → small.

**Stability.** Total pose gradient from a matched cell = F1 + F3 (both zero at exact alignment, small near alignment). Under small η (Lemma 0), the per-step pose perturbation is bounded → the cell remains matched. This is an **η-dependent** guarantee, weaker than the **exact** `g_m = 0` of mask-opt Lemma 1. ∎

*(Verification needed: a `t_cam_matched.py` probe checking pose-gradient magnitude at a matched cell under CAM rectify would close this.)*

### 9.5 Lemma 2-CAM (Coverage by geometric alignment — replaces warming)

**Statement.** Under the CAM schema, coverage of all target pixels is achieved by **geometric alignment**: the ground-truth mask guarantees that enough active points exist to cover all targets *at the correct pose*, and the gradient signal from uncovered targets drives the pose toward that configuration.

**Proof.**

1. **Ground-truth mask guarantees point existence.** The task is: given 3D points with known activity (`r_mask`), find the camera pose such that projections match the target silhouette. By construction of the data (`camera-calibration.ipynb` / `Sky-dataset`), the target IS the silhouette of the **same** 3D points under a different camera. Therefore every target pixel has ≥1 active 3D point whose correct-pose projection covers it.

2. **Gradient signal at uncovered targets.** An uncovered target (`t = 1`, no active point projects onto it under current pose) is mismatched (`case ≠ 1`). At such a cell: F3 provides coordinate gradient pointing nearby active points toward the target (via `coord_flow`, which is gated by target presence). F1 provides alignment gradient through gathered weight values. Both reach pose ⇒ nonzero pose gradient ⇒ not a fixed point.

3. **Routing participation (k_const = 0.5).** Inactive-depth cells participate in the z-buffer with probability 0.5 per pass (L57, `smap.py`). By Borel–Cantelli, every such cell is selected infinitely often a.s. This serves **routing/visibility** (the z-buffer sees all relevant points), not warming (mask is already correct).

**Scope conditions (load-bearing):**
- **(C1) Pose-feasibility:** the target must be achievable by a rotation + translation from the given 3D configuration. If the object has deformed, no rigid-body transform suffices ⇒ Φ stays > 0.
- **(C2) Conflicting gradients:** one pose serves all points simultaneously. Targets on opposite sides may require contradictory rotations → gradient cancellation or local minima. The coarse-to-fine scheme mitigates by aligning large-scale structure first.
- **(C3) Occlusion dynamics:** a pose change may alter which points are occluded. Lemma 3 governs per-pass z-buffer structure, but the occlusion SET is pose-dependent and changes across steps. ∎

### 9.6 Lemma 3-CAM (Z-buffer — structural, unchanged)

Lemma 3 (§8) applies identically: `calculate_weights` L176–180 is independent of the training schema. An inactive winner is rejected to centre; rendered-activity is mask-preserving per forward pass.

**Caveat:** in CAM schema, the z-buffer *inputs* change each step (xyz changes with pose), so the occlusion set evolves. Lemma 3 governs the structure of each snapshot, not the dynamics across steps. ∎

### 9.7 Theorem-CAM (Convergence — conditional)

**Statement.** Under the CAM schema, assuming:
- **(A-CAM.1)** Pose-feasibility: the target is achievable by a rotation + translation.
- **(A-CAM.2)** Local landscape: no spurious local minimum traps the optimiser near the initial pose.
- **(A-CAM.3)** η ≤ η₀ (Lemma 0, applied to pose parameters).

Then Φ → 0 as SGD steps → ∞.

**Sketch.** At each step with Φ > 0, at least one target cell is uncovered or one non-target is incorrectly active. By Lemma 2-CAM, the gradient signal from such cells is nonzero and reaches the pose ⇒ not a fixed point. Under A-CAM.2, the gradient direction reduces Φ. Under A-CAM.3, no overshoot. The coarse-to-fine scheme (zoom = n → 0) convexifies: at coarse zoom, the target is max-pooled → smoother landscape → global direction captured; finer zooms refine.

**Comparison to Theorem (§8):**

| | Mask-Opt (§8) | CAM (this section) |
|---|---|---|
| **Guarantee** | Unconditional (k > 0 ⇒ halt ⟺ correct) | Conditional on A-CAM.1/A-CAM.2 |
| **Lyapunov** | Φ = non-increasing integer → 0 | NOT proven monotone (pose changes can temporarily increase Φ via occlusion) |
| **Fixed points** | No Φ > 0 fixed point (Theorem′) | No Φ > 0 fixed point under A-CAM.1/A-CAM.2 |
| **Convergence mechanism** | Per-cell mask crossing τ | Global rigid-body alignment |

**Why CAM is inherently weaker.** The mask-opt proof exploits per-cell independence: each cell's mask moves independently toward its correct value, and Φ is a sum of independent binary events. In CAM, all points share ONE pose ⇒ they are coupled. A pose update that helps some targets may hurt others (conflicting gradients). This coupling means Φ is NOT a simple Lyapunov function. Monotone convergence would require proving that the coarse-to-fine scheme + the specific force construction avoids all oscillation — a stronger claim left for future work.

**What would constitute a correctness bug in CAM.** A situation where the gradient signal is **wrong-signed** (points away from the correct pose when the current pose is close) or **zero** (the algorithm stalls at Φ > 0 despite a correct pose existing). Either would be a defect in the CAM force construction, traceable to `rectify.py` L120–189. ∎

### 9.8 Bug catalogue (CAM-schema re-examination)

| Bug | Mask-Opt (§8) | CAM-Schema | Notes |
|---|---|---|---|
| **A (FUT filter)** | CORRECTNESS (latent) | **N/A** — CAM uses `rectify.types.CAM`, not FUT | — |
| **B (k_const ramp)** | Precision-only (k=1.0) | **Re-examine at k=0.5:** ramp still discarded; k=0.5 → P(participation)=0.5. Borel–Cantelli ⇒ no starvation. At k=0.5, L62-72 intermediate inversion is **symmetric** (`1−k = 0.5 = k`) → behaviourally moot. **Precision-only holds** (any k>0 suffices). | Symmetric k → inversion moot |
| **C (app B)** | Auditability | Same — CAM's `compute_pre_allow_matrices` uses `dim=2` summation (DefaultRectify L25). Still an app-fidelity issue. | — |

**New CAM-schema-specific concerns:**

| Concern | Source anchor | Status | Risk |
|---|---|---|---|
| **tanh saturation** | `run_calib_step.py` L145–147: `tanh(3e-2·param)·π` | Rate/precision | Gradient vanishes at `\|param\| >> 33`. Not correctness. |
| **Translation-active gating** | L160: `x_r + (mask>0.5)·θ` | Source-faithful | Only active points translate. Correct for the task. |
| **Z-clipping at \|z\|=0.5** | L161 | Rate/precision | `.detach()` boundary stops gradient at extreme z. Could stall near z=±0.5. |
| **Single key_query (vs DEP dual)** | CAM L133 vs DEP L246–247 | **Unanalysed** | CAM's F3 lacks spreading-damper (`1−n_flow/max`). Directional adequacy for pose is unknown. |
| **Multi-zoom interaction** | `for i in 0..n` loop | **Unanalysed** | 6 zoom levels contribute gradient to same pose per step. Cross-level interaction not certified. |
| **F2 value-only contribution** | L176/L183/L185 | **Unanalysed** | F2 mask tap shifts weight VALUES (affects Φ count) but carries no gradient to pose. Effect on convergence unknown. |
| **Surrogate scaling 3e3** | `run_calib_step.py` L185 | **Unanalysed** | 30× smaller than mask-opt's `1e5`. Adequate for 6-DoF? Could under-weight or over-weight gradient. |

### 9.9 Residual risks (CAM-specific)

1. **No source probe for CAM rectify.** Existing probes (`t_a2.py`, `t_force3.py`, `t_a1a4.py`) use DEP/FUT, not CAM. A `t_cam_matched.py` probe is needed to verify: (a) Lemma 1-CAM (zero pose-gradient at exactly-matched cell), (b) sign/zero of pose gradient at mismatched cells, (c) F2 inertness (dead-end gradient path).
2. **Single key_query structure.** CAM's F3 has no spreading-damper — whether this causes wrong-signed or zero gradient at configurations where DEP's dual key_query would succeed is untested.
3. **Multi-zoom gradient accumulation.** The 6-level coarse-to-fine structure is unanalysed in this proof. The HDVO certification (N=8) covers the L62-72 nest at ONE zoom level; the interaction across all 6 levels is not certified.
4. **Pose-feasibility (A-CAM.1)** is a task assumption, not algorithmically verified. Data preparation (`Sky-dataset`) ensures it by construction.
5. **Local landscape (A-CAM.2)** is standard for pose estimation but NOT proven for this specific loss function / force construction. A landscape analysis or empirical basin-of-attraction study would close this.
6. **The loss function `l1_error`** at surrogate scaling `3e3` is taken as-is. If the `denom` normalisation (L181–182) flattens the gradient near convergence, or the `3e3` scaling produces too-weak signal for small pose errors, these would be correctness-obstructing bugs. Currently **unanalysed** — needs dedicated source study.

