# SMap Simulator — Specification & Verification Report

Companion to `smap_simulator.jsx`. Self-contained: an auditor needs only this file, the app, and the source (`smap.py`, `rectify.py`, `utils.py`, `specials.py`). No conversation context required.

The simulator's purpose is to make every core mechanism of the source **observable**, and to make the gap between **current (source) behavior** and **fixed behavior** visible by stepping a live Markov chain. Correctness claims are anchored to source lines so any error — in the algorithm or in the app — is traceable and falsifiable.

---

## §1. What the app represents (and what it abstracts)

| Aspect | Faithful to source | Abstracted (with reason) |
|---|---|---|
| Meta-state classification (m,t,B) | Exact: rectify.py L22-27 logic | — |
| Routing vs gradient separation | Exact: routing detached (smap.py L83/L130), xyz carries gradient (utils.py L174-182) | — |
| xyz-gradient SCOPE | **Exact (corrected this round):** finest-3×3-local. `rectificate_flow` runs only at the final `go(is_last=True)` at finest resolution (smap.py L244-251), so the gradient pulls (x,y) toward a target in the **immediate 3×3 only** (utils.py L180-182). | — *(A prior version searched the whole grid for the nearest target; that was **less faithful** and has been reverted. Distance is crossed by **routing**, not the gradient.)* |
| Candidate cells ("ô khả dĩ") | **Exact:** a point's value is broadcast to its 3×3 (smap.py L70-75: `reshape(…1,1,1…) + zeros_like(weights_x)(…3,3…)` = broadcast, all 9 carry the same xyzm); each candidate's gradient = key_query between the shared (x,y) and the back-projection of that shifted position (utils.py L174-182); 9 contributions `agg(flip)` back to the physical point (smap.py L99-113). Slice order 0=↖…4=●…8=↘ per 2config1.json ("vị trí dịch khả dĩ"). | View shows candidates of one selected point at a time; gradient shown per-slice + aggregate |
| Routing crosses distance | **Exact:** detached (`weights_x.detach()` L83/L130), coarse→fine loop L224-238, ≤1 neighbor/level = 2^l finest px/level. This is how far points move. | — |
| Center-fallback ("hard") cells | **Exact:** with no valid z-buffer winner the gather falls back to the cell's own center (slice 4), smap.py L170 `where((val>0)&(val<INF), ind, 0*ind+4)`. A cell whose value routed OUT has a **zero** slice-4 self term, so it **vacates to mask 0** (it does NOT retain its old value); the app marks these `fb` (emptied one-pass trace, no fresh routing, no duplication). | Z-buffer value *duplication* across cells not modeled (move-and-mark approximation) |
| Movement arrows | Solid → = projection/routing direction; dashed ⇢ = gradient-toward-target. Two distinct vectors, shown in an inspector compass with magnitudes. | — *(fixes the prior single-arrow ambiguity)* |
| Zoom FOLD/UNFOLD | Exact bijection: 2×2 spatial↔4 channels (smap.py L213-218 / L235) | Single image, BATCH=1 |
| Rectify axis DEF/CAM/DEP/FUT | Exact dispatch: smap.py L146-152 | Gradient terms are scaled constants, not full einsums — preserves the structural zero/non-zero behavior, not magnitudes |
| FUT convergence-filter bug | Exact: DEP L294 present, FUT absent (L405–L425) | y_flow modeled as a count; Term1 fires when ==1 (matches `(y_flow==1.)` gate) |
| z-buffer | Exact: nearest (min depth) wins, +1e-5 tie-break (smap.py L156-180) | — |
| Camera | Identity (one global matrix), per app_spec.md | Real camera matrix not modeled; meta-states don't depend on it |

---

## §2. Mechanism → Scenario table (covers the whole source)

This table is the verification backbone. Coverage (computed by Agent C below) is **98.9%** of semantically-active `smap.py` lines and **98.6%** of `rectify.py`. Each row: a mechanism, its source lines, what it does, and the steps to observe it in the app.

### smap.py

| Mechanism | Source lines | What it does | How to observe in the app |
|---|---|---|---|
| `M1_routing_key_query` | smap.py L41–47 | Build weights_b: argmax(-key_query) one-hot over 3x3 → which neighbor each cell routes to. Detached selection. | Zoom-route step: active point with z>0 moves toward projection neighbor; observe display cell shift by 2^L. |
| `M2_zbuffer_seed_noise` | smap.py L48–53 | temp (center one-hot), temp_rand tie-break, z>0 gate selecting weights_b vs temp. | Two points routed to same cell: nearest z wins; equal-z broken by jitter (random preset, step). |
| `M3_kconst_stochastic` | smap.py L54–55 | k_const>np_rand: inactive-with-depth participates stochastically (per-pass random). | Bug toggle 'k_const inactive participation'; observe S_C-like cells warming slowly. |
| `M4_weights_m_x_final` | smap.py L57–58 | weights_m / weights_x for final pass: active+z>0 keep route, active+z<=0 → center, inactive → stochastic. | Inspect a cell's routing class in inspector (active/depth state). |
| `M5_weights_intermediate` | smap.py L61–63 | is_last==False branch: intermediate-level weights (BUG2 design: inactive→weights_b). | Zoom levels L>0 routing pass; compare intermediate vs final movement. |
| `M6_broadcast_3x3` | smap.py L70–90 | Broadcast x,y,z,mask into 3x3x3 neighbor tensor (intermediate path). | Implicit in any routing step; data prep before einsum. |
| `M7_einsum_select` | smap.py L83–136 | einsum(weights_x.detach, x_z_value): gather selected neighbor's xyz; weights DETACHED → gradient flows to xyz only, not selection. | Gradient step moves xyz without changing display cell; inspector shows xyz change, cell fixed. |
| `M8_is_last_agg_flip` | smap.py L99–113 | is_last==True: agg(broadcast) then agg(flip(.)) on x,y,z,mask → identity rearrange producing per-slice values for rectify. | Final-level state used by rectify; observe meta-state at finest level. |
| `M9_calc_weights_zmax` | smap.py L154–159 | new_z_max (per-channel global max), z+1e-5 noise. | z-buffer depth assignment inspector. |
| `M10_calc_weights_ind` | smap.py L165–175 | Depth-test discriminator: mask>OFF→z; else (z<=0→zmax+eps | z>0→z+eps*k). min over 3x3 → winner index; default center if invalid. | Depth conflict: point folded onto another; winner = min depth. Inspector shows 9 neighbor depths + winner. |
| `M11_calc_weights_select` | smap.py L176–197 | agg(ind) selects winner xyz+mask; fallback center; detach xyz when winner mask<=OFF. Final path: recover_size (UNFOLD to original res) + crop to original_size. | Winner selection visualized; detached-xyz case when inactive wins. |
| `M12_zoom_fold` | smap.py L199–218 | FOLD loop: spatial 2x2 → 4 channels, /2 each dim. Builds C_zoom = 4^level. Includes forward() zoom-setup: C_zoom=4^n, height_zoom init. | Zoom slider L0..Lmax: grid side halves, channel count x4. 'C_zoom' is the channel dim. |
| `M13_sign_prep` | smap.py L221–223 | pre_sign = sign(z); coords multiplied by sign (orientation normalization). | z sign handling in inspector (negative z flips). |
| `M14_forward_loop_unfold` | smap.py L224–238 | Per-level loop coarse→fine: go() routing + calculate_weights + UNFOLD (channel→spatial, x2 each dim). | Markov step runs all levels; binary zoom lets a point cross any distance via 2^l steps. |
| `M15_final_rectify_dispatch` | smap.py L244–254 | Final go(is_last=True); if target: rectificate_flow (gradient) else calculate_weights (render). | Gradient step (target present) vs render (no target). |
| `M0_go_entry` | smap.py L26–39 | go(): split (x,y,z,mask) channels, add_pad (pad borders for to_3d projection), call forward; forward setup reshapes z/r/x_z tensors. | Implicit at every step — initial channel split and padding before routing. |
| `M16_module_init_rectify_dispatch` | smap.py L138–152 | SMap.__init__: build SMap3x3; choose rectify_module by rectify_type (DEF/CAM/DEP/FUT) at L146-152. This is the Rectify-type axis. | Rectify-type selector in app (DEF/CAM/DEP/FUT) — drives which gradient construction runs. |

### rectify.py

| Mechanism | Source lines | What it does | How to observe in the app |
|---|---|---|---|
| `R1_default_pre_allow` | rectify.py L16–28 | DefaultRectify.compute_pre_allow_matrices: comp1=weight*target, sum slices, agg(flip) → comp2. Defines B. | B indicator in inspector; meta-state classification. |
| `R2_default_mask_flow` | rectify.py L30–45 | prepare_flows_for_mask (Default/CAM): mask_flow binary free-direction. | y_flow/mask_flow value in inspector. |
| `R3_default_coord_flow` | rectify.py L47–61 | prepare_flows_for_coord (Default/CAM): 1-max>0.9 gate x target. | coord_flow arrow in inspector. |
| `R4_default_weight_constr` | rectify.py L63–118 | DefaultRectify weight construction + gradient (case factor + n_case). | Rectify=DEF gradient decomposition. |
| `R5_cam_weight_constr` | rectify.py L120–189 | CAMRectify: weights=pre_weights*(>OFF) (inactive=0); mask_flow*(1-case) at L168; coord_flow gate. | Rectify=CAM; inactive weights zeroed; S1 absorbing via mask_flow filter. |
| `R6_dep_pre_allow` | rectify.py L191–206 | DEPRectify.compute_pre_allow: sum dim=1 (cross-channel); returns (pre_allow, pre_allow_s). | Rectify=DEP; cross-channel B. |
| `R7_dep_mask_flow` | rectify.py L208–225 | DEP prepare_flows_for_mask: n_flow + y_flow (binary transform). | DEP n_flow/y_flow inspector. |
| `R8_dep_coord_flow` | rectify.py L227–236 | DEP prepare_flows_for_coord: 1-max>0.9 gate x target. | DEP coord_flow. |
| `R9_dep_weight_constr_FILTER` | rectify.py L238–310 | DEP weight construction; KEY: L294 y_flow*=(1-case) convergence filter → S1 absorbing. weights=signed(-.5pm attraction). | Rectify=DEP, S1 cell: m-grad=0 (absorbing). CONFIRMED filter present. |
| `R10_fut_pre_allow` | rectify.py L312–329 | FUTRectify.compute_pre_allow: parallel to DEP. | Rectify=FUT; B computation. |
| `R11_fut_mask_flow` | rectify.py L331–339 | FUT prepare_flows_for_mask: n_flow only (no y_flow here). | FUT n_flow inspector. |
| `R12_fut_coord_flow` | rectify.py L341–352 | FUT prepare_flows_for_coord: computes y_flow=agg(max(pre_allow)) + coord_flow; NO target multiply. | FUT y_flow source; fires Term1 when ==1. |
| `R13_fut_weight_constr_BUG` | rectify.py L354–430 | FUT weight construction; BUG: MISSING y_flow*=(1-case) (no DEP-L294 analog) → S1 Term1 nonzero when y_flow==1. weights_grdf otherwise identical to DEP. | Rectify=FUT, S1 cell with y_flow==1: m-grad != 0 (NOT absorbing). Bug toggle ON/OFF. |

---

## §3. Rectify axis & bug panel (POLA)

The app exposes the real source axis **Rectify ∈ {DEF, CAM, DEP, FUT}** (smap.py L146-152). Each confirmed deviation is an **independent toggle**, gated to the setting where it applies, so multiple can be active at once:

| Toggle | Severity | Source anchor | Gated to | OFF (source) | ON (fixed) |
|---|---|---|---|---|---|
| FUT convergence filter | CORRECTNESS | rectify.py: DEP L294 vs FUT absent L405–L425 | Rectify = FUT | S1 with y_flow==1 not absorbing | S1 absorbing (DEP analog applied) |
| k_const inactive participation | IMPROVEMENT | smap.py L54-55 | always | inactive+depth excluded (k_const==0.5 path) | inactive+depth route stochastically |

Design rationale (per POLA): a toggle that does not apply to the current Rectify is shown disabled with the reason ("N/A — only applies when Rectify = FUT"), so the user is never misled into thinking a fix is active when it is not. Adding a new confirmed bug = appending one row to the `bugRows` list; the gating/labeling machinery is shared.

---

## §4. Use case walkthrough (default preset = response.txt config)

Default state on load: Target at (2,5); Point B at (2,5) already at target (S1); Point A at (3,4) whose xyz projects to (3,5).

1. **Open** — Rectify=FUT, FUT filter=SOURCE. Header shows S1 non-absorbing count > 0.
2. **Click (2,5)** — opens the **candidate view**: (2,5) becomes the dashed-yellow center, its 3×3 fills with its 9 "ô khả dĩ" (each carrying its xyzm). Inspector: meta=S1, y_flow#≥1. Commentary: "FUT, S1, y_flow=…: m-grad≠0 → NOT absorbing. FUT lacks the filter DEP has at L294."
3. **Toggle 'FUT convergence filter' → FIXED** — inspector m-grad → 0; header → "all S1 absorbing". This is the bug's effect, isolated.
4. **Markov step ×10 (SOURCE)** — watch mask at S1 drift (non-convergent). **Reset, FIXED, ×10** — S1 stable. Severity made visible over time.
5. **Candidate view of Point A (3,4)** — click (3,4); its 3×3 candidates appear. Click candidate **slice 2 (↗)** = position (2,5): the inspector marks it target-confirmed (⊕) and shows its per-slice key_query + pull. Click the dashed center (●) for the **aggregate** gradient (sum of the 9 via `agg(flip)`). Clicking the center again exits to the plain physical/`fb` grid. This makes the broadcast→per-slice-gradient→aggregate mechanism (the core of `2config1.json`) directly inspectable.
5. **Zoom L0→L1→L2** — grid side halves, channel count ×4 ("C_zoom"). Selected physical point (2,5) appears only on the channel that contains it; a hint shows which channel to slide to. Returning to L0 leaves all points exactly where they were (FOLD/UNFOLD bijection → no drift from zoom spam).
6. **Preset = Far target** — a point is AIMED at its target (xyz projects to (7,7)) but still LOCATED at (0,0): the "gradient done, routing pending" state. One Markov step routes all levels coarse→fine and carries it (0,0)→(4,4)→(6,6)→(7,7), i.e. 7 = 4+2+1 (L2+L1+L0). This is the binary-zoom distance-covering mechanism (M14) made visible — the single clearest demonstration that routing crosses arbitrary distance via 2^l steps.
7. **Edit z of a point across 0.5** — depth crosses the z-buffer threshold; routing winner changes.
8. **Preset = Convergence test** — FIXED converges all 4 points in ~20 steps; SOURCE leaves ≥1 non-absorbing.
9. **Preset = Random** — confirms the app is not hard-coded; mechanisms hold on arbitrary input.

---

## §5. Falsifiability (per step)

Every claim the app makes is checkable: (1) meta-state matches the displayed (m,t,B); (2) at S1 with the filter fixed, m-grad = 0; (3) at S1 in FUT-source with y_flow#==1, m-grad ≠ 0; (4) drift over N steps ≈ N·lr·Δ; (5) zoom round-trip leaves positions unchanged; (6) routing winner = min depth. If any observed behavior contradicts the commentary, that is a reportable defect (in the app or the stated mechanism) — this is the channel by which errors anywhere get traced and fixed.

---

## §6. Independent Verification Workflow & Consensus Report

### 6.1 How it was run (transparency)

Because separate LLM agents cannot be spawned from this environment, each "agent" is an **executable checker script** whose verdict comes from running it against the source code and the app — not from author judgment. Scripts live in `verify/`. Each derives its ground-truth by reading the source directly, then checks the app against it. The workflow was iterated until all three accepted:

- **Stage 0** (`stage0_reconcile.py`): reconcile `bug_list.txt` against the actual current source.
- **Agent A** (`agent_A_reflection.py`): does the app provide an observable scenario for each mechanism in the inventory?
- **Agent B** (`agent_B_correctness.py`): do the app's structural invariants match the source (routing detached, FUT-bug = missing filter, weight-guard present in both, zoom 4×, physical-cell tracking, z-buffer min, identity projection, batch update)?
- **Agent C** (`agent_C_coverage.py`): do the mechanism line-ranges cover the semantically-active source?
- **Agent D** (`agent_D_presentation.py`, NEW): presentation faithfulness — sweep every visual signal; if {what it encodes} can diverge from {what a viewer reads}, a disambiguation artifact must exist. Catches the *presentation* class (e.g. the arrow).
- **Agent E** (`agent_E_equivalence.py`, NEW): computational/structural equivalence — trace every core data structure / computed quantity to a specific source tensor op with matching semantics, checked **app⟷source directly**. Catches the *data-model* class (e.g. candidates vs physical neighbors).

**Why D and E:** A/B/C check surface properties (mechanism mentioned, textual invariants, lines covered). Two defects slipped past them this project — the arrow (presentation) and the candidate neighborhood (data-model). D and E are property sweeps over *all* signals / *all* structures, surfacing a list of candidate mismatches for human acceptance rather than needing a per-bug spec. E checks app⟷source directly (not app⟷accumulated-knowledge) so a gap in the knowledge itself is caught.

**Non-triviality proof (adversarial):** corrupting the app makes the relevant check FAIL — removing the FUT-filter toggle → INV2 FAIL; relabeling the bug as the weight guard → INV3 FAIL; removing `origToView` cell-tracking → INV5 FAIL; **stripping the candidate machinery → E1 FAIL (REJECT 5/7)**; **restoring the whole-grid nearest-target gradient → E4 FAIL (REJECT 6/7)**; **removing the compass + arrow clarifier → D P1+P2 FAIL (REJECT 6/8)**. The checkers detect real defects, including the two that previously slipped.

### 6.2 Stage 0 — bug_list.txt vs current source (executable verdicts)

| Claim | Verdict | Evidence |
|---|---|---|
| BUG 1 — FUT missing `(weight>0)` guard | **REFUTED** | DEP L305 and FUT L425 both contain `(weight>0.).detach().float()` (identical). |
| BUG 4 — DefaultRectify C_zoom undefined | **ALREADY FIXED** | Source defines `C_zoom = …shape[1]` at L70 (`[BUG 4 FIX]`). |
| FUT missing y_flow convergence filter | **STRUCTURALLY CONFIRMED** | DEP applies `y_flow*=(1-case)` at L294; FUT has no such line L405–L425. This is the app's CORRECTNESS toggle. |
| BUG 2 — intermediate inactive weights | DESIGN | Quick-fix is a deliberate trade-off (bug_list's own conclusion); not a correctness break. |
| BUG 3 — k_const==0.5 excludes inactive | IMPROVEMENT | Slows convergence only; modeled as the app's IMPROVEMENT toggle. |

Note: `bug_list.txt` line numbers are offset ~2 lines and BUG 1/BUG 4 do not match the current source — it was written for an older fork. Only source-confirmed deviations are given toggles.

### 6.3 Final consensus

| Agent | Criterion | Verdict |
|---|---|---|
| A | Every mechanism observable in app | **ACCEPT (30/30)** |
| B | App invariants match source | **ACCEPT (8/8)** |
| C | Mechanisms cover the source | **ACCEPT (smap.py 98.9%, rectify.py 98.6%)** |
| D | Presentation faithfulness (no astonishment) | **ACCEPT (8/8)** |
| E | Computational/structural equivalence app⟷source | **ACCEPT (7/7)** |

All five agents accept. The simulator faithfully reflects the source mechanisms — including the candidate ("ô khả dĩ") broadcast structure and the finest-3×3 gradient scope — demonstrates the one source-confirmed correctness bug (FUT convergence filter) plus one convergence-speed improvement (k_const), and lets the user observe each over a live Markov chain with full traceability.

### 6.4 Iteration log

**Round 1 — gradient freeze (over-corrected, see Round 2).** A prior version capped the xyz target search to the finest 3×3, freezing far points. It was "fixed" by expanding the search to the whole grid. *This expansion was itself wrong* — see Round 2.

**Round 2 — candidate semantics + gradient-scope correction (this round).** Two deeper issues surfaced during human review and were fixed, then the workflow re-run with the two new criteria:

- **Self-reported correction (algorithm fidelity):** the Round-1 whole-grid gradient was **less faithful**, not more. `rectificate_flow` runs only at the final `go(is_last=True)` at finest resolution (smap.py L244-251), so the real xyz-gradient is **finest-3×3-local**. Distance is crossed by **routing** (detached, coarse→fine, 2^l/level), not by the gradient. The whole-grid search was **reverted**; the Far preset (which relies on routing, not gradient) still demonstrates distance traversal correctly.
- **Defect found (data-model, human-flagged, slipped A/B/C):** the grid rendered independent **physical neighbors** around a selected cell, but the source's 3×3 is a **broadcast of the center point** (smap.py L70-75) — the "ô khả dĩ" of `2config1.json`. **Fix:** selecting a point opens a candidate view (9 broadcast copies, slice 0=↖…8=↘), each candidate inspectable with its per-slice key_query + pull, and the aggregate (`agg(flip)`, smap.py L99-113).
- **Defect found (presentation, human-flagged):** a single arrow conflated routing-direction with target-direction. **Fix:** the grid arrow is explicitly labeled projection/routing (not target), and the inspector adds a two-vector compass (solid routing + dashed gradient, with magnitudes).
- **Fix (fidelity):** vacated cells follow **center-fallback** (smap.py L170): a routed-away cell's slice-4 self term is 0, so it **vacates to empty** (mask 0), marked `fb` (one-pass trace, no fresh routing) — not a fabricated default and not a retained value.
- **Process improvement:** added **Agent D** (presentation faithfulness) and **Agent E** (computational/structural equivalence). These are the criteria that *would have caught* both slipped defects, expressed as property sweeps so future review needs no per-bug spec.
- **Re-verification:** A 30/30 · B 8/8 · C 98.9%/98.6% · **D 8/8 · E 7/7** → all ACCEPT. Adversarial: strip candidates → E1 FAIL; restore whole-grid gradient → E4 FAIL; remove compass/clarifier → D P1+P2 FAIL; plus prior INV2/INV3/INV5 still bite. Headless replay confirms slice order (s0=↖…s8=↘), candidate (3,4)/s2 = target (2,5), and the zoom bijection over all 64 cells × 4 levels.
- **Multi-bug status:** the two source-confirmed deviations remain independent toggles at different stages (FUT filter → gradient; k_const → routing); both can be active at once; the header reports "N/M toggles ≠ source". No third correctness bug is fabricated — Stage 0 confirms only one exists on the current source.
