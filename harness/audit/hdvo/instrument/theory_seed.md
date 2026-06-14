# Instrument theory seed — routing / gradient-flow nest (L62–72)

Seed for the `aiTheory` / `userHint` section of the instrument's config JSON. This is the
**theory of the correct algorithm nearest the source** (from `math_model.md`) that is *relevant to
this instrument* — i.e. the routing/gradient-flow nest the testbot dumps capture
(`weights_b/m/x`, `np_rand`, `temp_rand`, z-buffer, `r_mask`). It is a starting point the HDVO
turns refine; it is **not** the accumulated/optimized theory (that comes back in the new config JSON).

## What the instrument audits
The dumps are the **per-cell, 9-slice** state of one SMap forward at the finest level (`is_last=True`).
The 9 slices = the **8 neighbour shifts + centre (slice 4)** — the differentiable "physical point"
routed to its 9 candidate positions around its own cell (slice 0 = up-left … slice 8 = down-right).
Aggregation `agg(flip(·))`/`agg(·)` move information between the candidate plane and the point's own
cell; `agg(agg(flip(x))) = x` (math_model §3.2, utils `agg`/`flip`).

## Config-block ↔ source mapping (this instrument)
| Config field | Dump (`SMap_Z_5_*`) | Source tensor (smap.py) | Meaning |
|---|---|---|---|
| `rawComponentSlidesData` | `wbl` | `weights_b` (L62/64) | best-neighbour routing slides |
| `conditionalBlockX` | `rnpl` | `np_rand` (L60, **global scalar**) | warming gate roll (S_C) |
| `conditionalBlockY` | `tmpl` | `temp_rand` (L59, **per-cell**) | per-cell participation roll |
| `conditionalBlockZ` | `zl` | z-buffer `z_value` | visibility / routing depth |
| `conditionalBlockR` | `rl` | `r_mask` | mask (FIXED here — m-fixed regime) |
| `userDefinedGradientFlows` | `zvl` | routed z-coordinate value | reference "flow" array |
| `inputGridData` / `targetGridData` | `input_/target_representation` | input mask / coarsened target | grid in/target |

## The correct-algorithm theory (source-anchored, condensed)
1. **Multi-scale Markov chain** (math_model §2; smap.py FOLD L213–218 / loop). Quadtree FOLD →
   3×3 route → z-buffer → inverse-quadtree merge. **Intermediate levels route only** (values
   unchanged); the **finest level** propagates values and constructs the gradient. Routing is
   `.detach()`ed (synthetic-gradient `x − x.detach()` taps) — selection is non-differentiable.
2. **Routing layer S_A–S_D on `(m,z)`** (§3.1): S_A `m>τ∧z>0` full participation (`weights_b`);
   S_B `m>τ∧z≤0` centre/self; **S_C `m≤τ∧z>0` stochastic warming with prob `k_const`**
   (`where(temp_rand>k_const, temp, weights_b)`); S_D excluded. Z-buffer = argmin-z over the 3×3
   with centre-fallback; an inactive winner is rejected to the cell's own centre
   (occlusion is mask-preserving, Lemma 3).
3. **k_const warming direction** (§6): higher `k_const` ⇒ **more** participation/warming
   (`k→1` = maximal warming, not freezing). Source pins `k≡1.0` (mask regime); **this instrument's
   training pins `k=0.5`**.
4. **Force 3 → coordinates, not mask** (§5): the routing/coordinate force backprops to the leaves
   `x/y/z` via `key_query`, **never** to the mask. A faithful instrument keeps it out of `g_m`.

## ★ m-fixed camera-calibration caveat (load-bearing — this IS that regime)
This instrument runs the **m-fixed regime**: the mask `m` is fixed (`mask=.999*r_mask`), only the
camera pose (`xyz`) is optimized. Per math_model Theorem′ "Setting assumption", here:
- **Force-2 warming is inert** (no mask update),
- **`k_const`'s role shifts** from mask-warming to **routing/occlusion participation for the
  geometric optimization**,
- the objective is **geometric alignment**, not mask-match.
So **Lemma 2, Theorem′, and Bug-B's precision-only classification must be RE-DERIVED here** — the
HDVO turns on this instrument's dumps are the vehicle for that re-derivation. Do not assume the
mask-regime conclusions carry over.

(Authoritative source: `math_model.md` §2–§6, Lemma 2/3, Theorem′ setting assumption.)
