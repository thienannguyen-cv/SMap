# Instrument theory — L62-72 routing nest (m-fixed camera-calibration regime)

**Purpose.** Seed for the `theory` / `aiTheory` section of this instrument's HDVO config.
It is the source-grounded theory of the mechanism this instrument probes, taken from the
nearest-to-source correct model `math_model.md` (NOT the app's iteratively-accumulated
`aiTheory`). The instrument tests whether the **routing / z-buffer / warming nest** behaves
as this theory says — under the **m-fixed camera-calibration regime** (the regime the proof
flags as needing re-derivation).

Line numbers are the **current revision** (`smap.py`, post DEBUG-gate insert); re-grep before relying.

---

## 1. What this instrument probes

The cohesive nest **`smap.py` L62-72** (old "L53-63"), which builds the differentiable routing
weights and the z-buffer participation:

| Source (L62-72) | Tensor | Dump (artifacts/SMap_Z_5_*) | Role in the instrument |
|---|---|---|---|
| L56-58 | `weights_b` (best-neighbour one-hot) + `temp` (center) | `wbf` / `wbl` (first/last level) | base routing slot per cell |
| L59 | `temp_rand` (per-cell `rand_like`) | `tmpf` / `tmpl` | per-cell warming draw |
| L60 | `np_rand` (single global scalar) | `rnpf` / `rnpl` | global warming gate draw |
| L63 | `if (k_const > np_rand)` global warming gate | (gate firing) | whether the warming branch runs this pass |
| L66 | `weights_m` (mask routing) | `mf` / `ml` | which slot the **mask** routes to |
| L67 | `weights_x` (xyz routing) | `xf` / `xl` | which slot the **xyz** routes to |
| L70-72 | `is_last==False` **inverted** branch | (intermediate levels) | coarse levels route only (no gradient) |
| z-buffer (`calculate_weights`, ~L185-189) | visibility / center-fallback | `zb` / `zl` | argmin-z winner, inactive→center reject |

The **mask/xyz desync** is the prime suspect: L60 `np_rand` is **global** (one scalar for the whole
grid) while L59 `temp_rand` is **per-cell**, so the mask-routing gate (L66) and the xyz-routing gate
(L67) can be driven by *different* randoms — an inactive cell can show "mask routes but xyz stays".

---

## 2. Source-grounded theory (from `math_model.md`)

### 2.1 Objective (math_model §1)
`Φ = Σ 𝟙([m>τ] ≠ t) = loss_m`. The differentiable rendering matches a target silhouette; the gradient
is injected **only at the finest `go(is_last=True)`** level via the `(x − x.detach())` synthetic-tap trick
(value unchanged, unit backward tap), scaled by force coefficients. Selection ops are `.detach()`ed
(`weights_x.detach()`), so **distance is crossed by routing (detached, coarse→fine), not by the gradient**.
**(m-fixed caveat below — Φ is mask-match; under m-fixed it becomes geometric.)**

### 2.2 Routing layer S_A–S_D on `(m, z)` (math_model §3.1 — the instrument's domain)
| State | Condition | Routing (source) | Requirement |
|---|---|---|---|
| **S_A** | `m>τ ∧ z>0` | `weights_b` best-neighbour (L62/66/67) | R1 full participation |
| **S_B** | `m>τ ∧ z≤0` | center / self (`~(z>0)→temp`, L66) | R2 no depth to propagate |
| **S_C** | `m≤τ ∧ z>0` | **stochastic** route-to-neighbour with prob `k_const` (L66 `where(temp_rand>k_const, temp, weights_b)`); z-penalty `z+1e-5·k_const` | **R4 p_warm > 0** |
| **S_D** | `m≤τ ∧ z≤0` | excluded (zeros) | R3 no information |

**Level dependence (load-bearing):** the **finest / `is_last=True`** path (L66) alone carries the gradient;
the **intermediate `is_last=False`** path (L70-72) uses the *inverted* order `where(temp_rand>k_const,
weights_b, temp)`, so there higher `k_const` ⇒ *less* value-participation. The inversion is moot only at
pinned `k_const` extremes.

### 2.3 Z-buffer (math_model §4 + Lemma 3)
`ind = where(m>τ, z, where(~(z>0), zmax+1e-5, z + 1e-5·k_const))` (detached) → `min` over the 3×3
channel → center-fallback to slot 4 if no valid winner. **Lemma 3 (occlusion mask-preserving):** an
inactive z-buffer winner is rejected back to the cell's **own center value** ⇒ an inactive point can
never uncover an active cell, and a matched self-projecting target is never uncovered. (`calculate_weights`
L176-180 in the old numbering; re-grep — shifted ~+9.)

### 2.4 Warming schedule `k_const` (math_model §6 + Lemma 2)
Direction (from L66 `where(temp_rand > k_const, temp, weights_b)`): an inactive cell takes the
participation branch when `temp_rand > k_const` is **false** ⇒ **higher `k_const` ⇒ more warming**.
`k→0` minimal warming (near-deterministic z-buffer); `k→1` maximal warming. **Lemma 2 (no starvation):**
with `Σ p_warm = ∞` (Borel–Cantelli) every occluded target is selected infinitely often. Source default
pins `k_const ≡ 1.0` (max warming) — but **the camera-calibration run pins `k_const = 0.5`** (dumps named
`SMap_Z_5`), a genuinely stochastic mid value where the global-vs-per-cell desync (L60 vs L59) is *active*.

---

## 3. m-fixed setting assumption (CRITICAL — math_model §Theorem′ "Setting assumption")

The math_model proof is established for the **mask-optimization regime** (mask `m` is a learnable DOF;
Force-2 warming can activate a dark target). **It does NOT yet cover the m-fixed camera-calibration
regime** that this instrument's data comes from, where:
- `m` is **fixed** (`mask = .999*r_mask`); only `xyz` (camera pose) is optimized;
- each point's active/inactive status is decided **upstream** by the prior camera pose;
- Force-2 warming is **inert** (no mask update); `k_const`'s role shifts from mask-warming to
  **routing/occlusion participation for the geometric optimization**;
- the objective is **geometric alignment**, not mask-match.

⇒ **Lemma 2, Theorem′, and Bug B's precision-only classification must all be re-derived here.** This
instrument exists precisely to gather behavioral evidence (the L62-72 dumps) for that re-derivation.
The handoff's prime "dirty-output" lead — per-pass stochastic ROUTING re-rolling (`np_rand`/`temp_rand`,
this nest) flickering background cells even when the mask has converged — is the hypothesis to test.

---

## 4. How the instrument reads it (HDVO L1 hypothesis)

The L1 `aiFunctionBody` encodes the *expected* routing/gradient behavior of the nest (per `conditionalBlockX/Y/Z/R`
= `rnpl`/`tmpl`/`zl`/`rl`) and is compared against the actual dump (`rawComponentSlidesData = wbl`, etc.)
by exec.py + the Fold/Unfold app; mismatches drive refinement. The theory above is the standard the
hypothesis must converge to — i.e. the nest is *equivalence-complete* iff the L1 function reproduces this
source-grounded routing/z-buffer/warming behavior on the m-fixed dumps with no mismatch.
