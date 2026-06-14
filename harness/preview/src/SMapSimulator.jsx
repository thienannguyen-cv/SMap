import { useState, useCallback, useMemo, useEffect, useRef } from "react";

// ════════════════════════════════════════════════════════════════════
// SMap Simulator — auditing instrument for smap.py / rectify.py / utils.py
// anchored to math_model.md (the correct algorithm nearest to source + its proof).
//
// SOURCE-OF-TRUTH (SST) = the point grid at FINEST resolution. Zoom levels are DERIVED VIEWS
// computed by FOLD on the fly (smap.py L213-218); FOLD/UNFOLD is a pure bijection (L235) → zoom
// never writes state.
//
// TWO META-STATE LAYERS (math_model.md §3), one per mechanism — both needed for correctness:
//   • ROUTING layer S_A..S_D on (m,z): participation / occlusion / warming (smap.py L53-58, L165).
//   • RECTIFY layer S1..S7 on (m,t,B): gradient sign / the potential Φ (rectify.py forces).
// Correctness condition (math_model.md §1): Φ = #{[m>τ] ≠ t} → 0 ⇔ every target active ∧ every
// active on a target. The app shows a CONVERGENCE CERTIFICATE (Φ, termination bound, non-periodicity)
// and, on stall, points to the responsible bug.
//
// DEVIATION TOGGLES (each = OFF:source(buggy) / ON:fix; fix direction shown):
//   • Bug A — FUT convergence filter (rectify.py: DEP y_flow*=(1-case) L294; FUT absent L405-425).
//   • Bug B — k_const schedule: source discards the intended 0→1 ramp (notebook L435) and pins k_const≡1.0
//     = MAX warming (smap.py L57: higher k = more participation). At 1.0 occluded targets ARE covered, so
//     this is NOT starvation at the source default — the deviation is the missing schedule; impact pending
//     execution. The occlusion preset shows the small-k starvation MECHANISM, not the source regime.
//   • Bug C — app B must be per-Rectify (dim=2 DEF/FUT vs dim=1/cross-channel DEP; rectify.py
//     L25/L326 vs L203). Faithfulness prerequisite for auditing the pre_allow aspect.
// ════════════════════════════════════════════════════════════════════

const OFF_THRESH = 0.5;            // specials.py L2
const FINE = 8;                    // finest grid FINE×FINE (power of 2)
const L_MAX = Math.log2(FINE);     // zoom levels 0..3 (0 = finest)
const RAMP = 16;                   // steps over which the FIXED k_const schedule anneals 0→1

// ── APP SELF-AUDIT STANDARDS (process-level health — DISTINCT from the Φ certificate) ──
// The standards the AUDIT INSTRUMENT itself must meet. This turns the vague "something's off"
// hunch into a structured signal: a count + the few most urgent items (ranked by an Effort–Impact
// matrix, impact/effort ∈ {1=low,2=med,3=high}). Flip `met:true` when an item is closed; when ALL
// are met the header badge goes green. Keep labels terse — the bar must stay one line.
const AUDIT_STANDARDS = [
  { id: "a1a4", label: "A1/A4 occlusion-uncover", met: false, impact: 3, effort: 3,
    note: "Closed by the 'A1/A4 occlusion-uncover' framing toggle: Lemma 3 (smap.py L176-180 — inactive winner→center) ⇒ a matched target is never uncovered ⇒ certificate routing-region unconditional. Confirmed by t_a1a4.py (cover≡1/40). Default RESOLVED; toggle OFF to see the conditional framing." },
  { id: "l62", label: "L62/L63 k_const inversion in routing", met: false, impact: 2, effort: 1,
    note: "App routeStep now models BOTH senses (G3 via inactiveNest); SOURCE reconciliation of L62/L63 still pending. Closes when the k_const schedule (Bug B) is assumed fixed (it is the prerequisite for that fix)." },
  { id: "bugB", label: "Bug B precision-vs-correctness", met: false, impact: 2, effort: 3,
    note: "Precision-only at the proof level (A1/A4 resolved), BUT interactive validation found a deterministic coverage-by-routing stall at pinned k=1.0 (far preset; k_const-schedule the sole lever) ⇒ forward-observable. Closes when the schedule is assumed fixed; source-validation pending — prime 'dirty-output' lead." },
];
const stdPriority = s => s.impact * 10 - s.effort;   // higher = act sooner (impact-dominant, low-effort tiebreak)
const stdQuadrant = s => s.impact >= 3 ? (s.effort <= 1 ? "quick-win" : s.effort >= 3 ? "major" : "high-pri")
                                       : (s.effort <= 1 ? "quick-win" : "later");

// Per-open random seed (math_model.md residual: pseudo() PRNG re-seeded each open).
const SEED = Math.random() * 1000;
function pseudo(r, c) {
  const s = Math.sin((r + SEED) * 12.9898 + (c + SEED * 0.37) * 78.233) * 43758.5453;
  return s - Math.floor(s);
}
// stochastic draw per (cell, salt) — seeded per open; salt folds in level/step so draws differ.
function rand01(a, b, salt) {
  const s = Math.sin((a + SEED) * 12.9898 + b * 78.233 + salt * 37.719) * 43758.5453;
  return s - Math.floor(s);
}

// ── RECTIFY layer S1..S7 on (m,t,B) — gradient sign / Φ (math_model.md §3.2) ──
const META = {
  S1: { name: "ABSORBING",   color: "#10b981", note: "m>τ ∧ t=1 ∧ B — at target. Correct absorbing class (m-grad→0)." },
  S2: { name: "RECURRENT",   color: "#f59e0b", note: "m>τ ∧ t=0 ∧ B — active, target nearby. xyz pulled toward it." },
  S3: { name: "EXPLORING",   color: "#3b82f6", note: "m>τ ∧ t=0 ∧ ¬B — active, free." },
  S4: { name: "FREE",        color: "#475569", note: "m≤τ ∧ t=0 ∧ ¬B — empty, non-target. Correct absorbing class." },
  S5: { name: "INFLUENCED",  color: "#8b5cf6", note: "m≤τ ∧ t=0 ∧ B — empty, near activity." },
  S6: { name: "VACANT_NEAR", color: "#06b6d4", note: "m≤τ ∧ t=1 ∧ B — target waiting, point approaching." },
  S7: { name: "VACANT_FAR",  color: "#f97316", note: "m≤τ ∧ t=1 ∧ ¬B — target waiting, nothing near." },
};
function classify(m, t, B) {
  const mm = m > OFF_THRESH ? 1 : 0;
  if (mm && t && B) return "S1";
  if (mm && !t && B) return "S2";
  if (mm && !t && !B) return "S3";
  if (!mm && !t && !B) return "S4";
  if (!mm && !t && B) return "S5";
  if (!mm && t && B) return "S6";
  if (mm && t && !B) return "S7"; // (1,1,0) UNREACHABLE (self-inclusive B ⇒ B≥1, math_model Prop. P.0) — guarded fallback, not silent
  return "S7"; // (0,1,0): m≤τ ∧ t ∧ ¬B — VACANT_FAR
}

// ── ROUTING layer S_A..S_D on (m,z) — participation / occlusion / warming (math_model.md §3.1) ──
const META_ROUTE = {
  SA: { name: "PARTICIPATE", color: "#34d399", note: "m>τ ∧ z>0 — real 3D point; full routing weights_b (smap.py L53/57/58)." },
  SB: { name: "NO_DEPTH",    color: "#fbbf24", note: "m>τ ∧ z≤0 — center/self only; no depth to propagate (L57 ~(z>0)→temp)." },
  SC: { name: "WARMING",     color: "#a78bfa", note: "m≤τ ∧ z>0 — inactive+depth; participates with prob k_const → occlusion relief (R4)." },
  SD: { name: "EXCLUDED",    color: "#64748b", note: "m≤τ ∧ z≤0 — no information; excluded (L58/L165 zmax+1e-5)." },
};
function classifyRoute(m, z) {
  const mm = m > OFF_THRESH;
  if (mm && z > 0) return "SA";
  if (mm && z <= 0) return "SB";
  if (!mm && z > 0) return "SC";
  return "SD";
}

// Projection (utils.py to_3d3x3, identity camera): stored (x,y,z) projects to (x/z, y/z);
// pixel (r,c) at depth z back-projects to 3D (z·c, z·r, z).
function projPixel(p) {
  if (Math.abs(p.z) < 1e-6) return { pr: 0, pc: 0 };
  return { pr: p.y / p.z, pc: p.x / p.z };
}

// ── Presets ──────────────────────────────────────────────────────────
function initDefault() {
  const pts = [], tgt = [];
  for (let r = 0; r < FINE; r++) {
    pts.push([]); tgt.push([]);
    for (let c = 0; c < FINE; c++) {
      pts[r].push({ x: c * 2.0, y: r * 2.0, z: 2.0, m: 0.05 + 0.12 * pseudo(r, c) });
      tgt[r].push(0);
    }
  }
  tgt[2][5] = 1;
  pts[2][5] = { x: 5 * 2.0, y: 2 * 2.0, z: 2.0, m: 0.82 };   // Point B at target → S1
  pts[3][4] = { x: 5 * 2.0, y: 3 * 2.0, z: 2.0, m: 0.85 };   // Point A → projects to (3,5)
  return { pts, tgt };
}
function initRandom() {
  const pts = [], tgt = [];
  for (let r = 0; r < FINE; r++) {
    pts.push([]); tgt.push([]);
    for (let c = 0; c < FINE; c++) {
      const z = 1 + 3 * Math.random();
      pts[r].push({ x: (c + (Math.random() - 0.5) * 3) * z, y: (r + (Math.random() - 0.5) * 3) * z, z, m: Math.random() });
      tgt[r].push(Math.random() < 0.1 ? 1 : 0);
    }
  }
  return { pts, tgt };
}
function initConvergence() {
  const pts = [], tgt = [];
  for (let r = 0; r < FINE; r++) {
    pts.push([]); tgt.push([]);
    for (let c = 0; c < FINE; c++) {
      pts[r].push({ x: c * 2.0, y: r * 2.0, z: 2.0, m: 0.05 + 0.1 * pseudo(r, c) });
      tgt[r].push(0);
    }
  }
  const pairs = [[[1, 1], [1, 2]], [[1, 6], [2, 6]], [[6, 1], [5, 1]], [[6, 6], [6, 5]]];
  for (const [[tr, tc], [pr, pc]] of pairs) {
    tgt[tr][tc] = 1;
    pts[pr][pc] = { x: tc * 2.0, y: tr * 2.0, z: 2.0, m: 0.8 };
    pts[tr][tc] = { x: tc * 2.0, y: tr * 2.0, z: 2.0, m: 0.1 };
  }
  return { pts, tgt };
}
function initFar() {
  // ROUTING traversal across distance via coarse zoom levels. The point is AIMED at its target
  // (xyz projects to (7,7)) but LOCATED at (0,0). One Markov step routes coarse→fine:
  // (0,0)→(4,4)→(6,6)→(7,7) = 4+2+1 (L2+L1+L0). ONE point ends at (7,7); cells it passes through
  // vacate to empty (fb), so it does NOT duplicate.
  const pts = [], tgt = [];
  for (let r = 0; r < FINE; r++) {
    pts.push([]); tgt.push([]);
    for (let c = 0; c < FINE; c++) {
      pts[r].push({ x: c * 2.0, y: r * 2.0, z: 2.0, m: 0.05 + 0.1 * pseudo(r, c) });
      tgt[r].push(0);
    }
  }
  tgt[7][7] = 1;
  pts[0][0] = { x: 14, y: 14, z: 2.0, m: 0.9 };
  return { pts, tgt };
}
function initOcclusion() {
  // OCCLUSION test (math_model.md §3.2 / Lemma 2) — demonstrates the warming MECHANISM, not the source regime.
  // A dark target cell (3,3) ringed by NEARER active points (small z) that keep winning the z-buffer. At
  // SMALL k_const the dark target is never selected ⇒ gradient-starved ⇒ never covered (Φ stuck). At the
  // SOURCE DEFAULT k_const=1.0 (max warming) it IS covered (no starvation). Bug B = the discarded 0→1 schedule.
  const pts = [], tgt = [];
  for (let r = 0; r < FINE; r++) {
    pts.push([]); tgt.push([]);
    for (let c = 0; c < FINE; c++) {
      pts[r].push({ x: c * 2.0, y: r * 2.0, z: 2.0, m: 0.05 + 0.08 * pseudo(r, c) });
      tgt[r].push(0);
    }
  }
  tgt[3][3] = 1;
  pts[3][3] = { x: 3 * 0.8, y: 3 * 0.8, z: 0.8, m: 0.1 };          // dark target; projects to itself (3,3) → must WARM in place
  for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {     // ring of nearer bright occluders (project to self → stay put)
    const r = 3 + dr, c = 3 + dc;
    pts[r][c] = { x: c * 0.5, y: r * 0.5, z: 0.5, m: 0.9 };        // smaller z (0.5<0.8) → win the z-buffer → starve the dark target unless it warms
  }
  return { pts, tgt };
}

// ── FOLD mapping (smap.py L213-218) ───────────────────────────────────
function levelDims(l) { return { side: FINE >> l, chans: 1 << (2 * l), per: 1 << l }; }
function origToView(orig_r, orig_c, l) {
  const per = 1 << l;
  return { ph: Math.floor(orig_r / per), pw: Math.floor(orig_c / per), channel: (orig_r % per) * per + (orig_c % per) };
}
function viewToOrig(ph, pw, channel, l) {
  const per = 1 << l;
  return { orig_r: ph * per + Math.floor(channel / per), orig_c: pw * per + (channel % per) };
}

// ── CANDIDATE cells ("ô khả dĩ", 2config1.json) — broadcast of the center (smap.py L70-75) ──
const SLICE_GLYPH = ["↖", "↑", "↗", "←", "●", "→", "↙", "↓", "↘"];
function sliceShift(s) { return { sr: ((s / 3) | 0) - 1, sc: (s % 3) - 1 }; }
function candPos(cr, cc, s, l) {
  const per = 1 << l, { sr, sc } = sliceShift(s);
  return { pr: Math.max(0, Math.min(FINE - 1, cr + sr * per)), pc: Math.max(0, Math.min(FINE - 1, cc + sc * per)) };
}
function candInfo(p, cr, cc, s, l, tgt) {
  const { pr, pc } = candPos(cr, cc, s, l);
  const confirmed = tgt[pr][pc] === 1;
  const qx = pc * p.z, qy = pr * p.z;
  const kq = Math.abs(p.x - qx) + Math.abs(p.y - qy);
  const gradates = l === 0;                            // gradient only at finest (is_last)
  const gx = (confirmed && gradates) ? -Math.sign(p.x - qx) * 0.5 * p.z : 0;
  const gy = (confirmed && gradates) ? -Math.sign(p.y - qy) * 0.5 * p.z : 0;
  return { s, pr, pc, confirmed, kq, gx, gy, isCenter: s === 4, shift: SLICE_GLYPH[s] };
}

// ── Mechanism B: ROUTING (smap.py L45 + z-buffer L165-170) ────────────
// Per level l each cell moves ≤1 step (=2^l finest px) toward the NEAREST folded cell to its
// projection (argmin key_query, L45), clamped to the folded grid (no coarse over-shoot). A cell
// whose value routes OUT vacates to m=0 (center-fallback, L165-170) — no ghost, no duplication.
// S_C WARMING (math_model.md R4): an inactive target cell (m≤τ,z>0,t) participates / resists overwrite
// with probability k_const → occlusion relief; k_const=0 ⇒ always overwritten (starved).

// ── smap.py L53–L63 warming/routing-selection nest (cohesive; mechanism_specs.md Spec 5) ──
// The source builds THREE coupled selections (weights_b→weights_m→weights_x). This reflects the three
// facets the prior single per-cell rule collapsed:
//   G1 — a GLOBAL per-pass scalar gate `np_rand` (L54): with field-wide prob k_const it forces inactive
//        cells' weights_b→weights_x (XYZ) to center (stay). Coherent (all-or-nothing) per pass.
//   G2 — weights_m (MASK, per-cell L57) vs weights_x (XYZ, via weights_b L58) can DIVERGE for inactive
//        cells ⇒ a cell's mask and its geometry get sourced from different neighbours (suspected "dirty").
//   G3 — intermediate-level INVERSION (L62/L63): P(participate)=k_const at finest, (1−k_const) coarser.
// NOTE: G3 affects the (mask) dynamics below; G1/G2 are surfaced as inspector diagnostics — faithfully
// applying the mask/xyz SPLIT to the dynamics needs a pull-model rewrite, deferred to the real-data arc.
function inactiveNest(r, c, l, kEff, salt) {
  const pThresh = (l === 0) ? kEff : (1 - kEff);                       // G3: L62/L63 inversion
  const maskRoutes = rand01(r * 7 + c, l * 13 + 1, salt) < pThresh;    // weights_m, per-cell (L57)
  const passRand = rand01(101, l * 97 + 7, salt);                      // G1: single per-pass np_rand (L54)
  const globalSuppress = kEff > passRand;                              // L54 fires w.p. k ⇒ inactive XYZ→center
  const xyzRoutes = !globalSuppress;                                   // weights_x via weights_b (L53–55/L58)
  return { pThresh, maskRoutes, passRand, globalSuppress, xyzRoutes, desync: maskRoutes !== xyzRoutes };
}

function routeStep(pts, tgt, l, kEff, salt) {
  const per = 1 << l, side = FINE >> l;
  const next = pts.map(row => row.map(p => ({ ...p })));
  const moves = [];
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      const p = pts[r][c];
      if (p.stale) continue;
      const active = p.m > OFF_THRESH;
      // S_A always routes; S_C (inactive+depth) participates via the mask gate weights_m — P=k_const at
      // finest, (1−k_const) at coarser levels (G3 inversion, smap.py L57/L62-63). The XYZ gate (G1/G2)
      // is surfaced as a diagnostic, not applied to this unit-point move (see inactiveNest).
      const mayRoute = (active && p.z > 0) || (!active && p.z > 0 && inactiveNest(r, c, l, kEff, salt).maskRoutes);
      if (!mayRoute) continue;
      const { pr, pc } = projPixel(p);
      // Binary refinement toward the NEAREST cell to the projection (smap.py L45 argmin). At level l
      // move by exactly `per` only if it does NOT overshoot (|dist| ≥ per); otherwise defer to a finer
      // level. Handles: Far binary (0→4→6→7), fractional-nearest (5.7→6), a point already at its
      // projection stays (no oscillation), and NO lateral overshoot for a point adjacent to its target.
      const targR = Math.max(0, Math.min(FINE - 1, Math.round(pr)));
      const targC = Math.max(0, Math.min(FINE - 1, Math.round(pc)));
      const dR = targR - r, dC = targC - c;
      const stepR = Math.abs(dR) >= per ? Math.sign(dR) * per : 0;
      const stepC = Math.abs(dC) >= per ? Math.sign(dC) * per : 0;
      if (stepR === 0 && stepC === 0) continue;
      const nr = Math.max(0, Math.min(FINE - 1, r + stepR));
      const nc = Math.max(0, Math.min(FINE - 1, c + stepC));
      moves.push({ r, c, nr, nc, z: p.z });
    }
  const taken = {}, vacated = {};
  // S_C warming: a dark target cell claims (resists overwrite) with prob k_const → survives to accrue
  // the activation gradient (Force 2). With k_const→0 it never claims ⇒ occlusion-starved.
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      const p = pts[r][c];
      if (!p.stale && p.m <= OFF_THRESH && p.z > 0 && tgt[r][c] === 1 && rand01(r * 5 + c, l * 11 + 3, salt) < kEff)
        taken[r + "_" + c] = true;
    }
  moves.sort((a, b) => a.z - b.z);                     // nearest (min z) wins (smap.py L165-169)
  for (const mv of moves) {
    const key = mv.nr + "_" + mv.nc;
    if (taken[key]) continue;                          // closer point, or a warming dark-target, already claimed
    if (pts[mv.nr][mv.nc].m > OFF_THRESH && !(mv.nr === mv.r && mv.nc === mv.c)) continue; // occlusion
    taken[key] = true;
    next[mv.nr][mv.nc] = { ...pts[mv.r][mv.c], stale: false };
    if (!(mv.nr === mv.r && mv.nc === mv.c)) vacated[mv.r + "_" + mv.c] = true;
  }
  for (const k in vacated) {
    if (taken[k]) continue;
    const [r, c] = k.split("_").map(Number);
    next[r][c] = { ...pts[r][c], m: 0, stale: true, id: null };  // vacate to 0 (center-fallback); particle id lives at the destination, not the empty origin
  }
  return next;
}

// ── B per-Rectify (Bug C, math_model.md §3.2) ─────────────────────────
// DEF/FUT: confirmation over the 9 candidate slices (dim=2) → 3×3 neighbour. DEP: cross-channel
// (dim=1) → at the finest single image this collapses to the cell's OWN (active∧target).
// bFaithful=false reverts to the uniform neighbour approximation (the prior Bug-C behaviour).
function computeB(pts, tgt, rectify, bFaithful) {
  const B = pts.map(row => row.map(() => false));
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      if (bFaithful && rectify === "DEP") {
        B[r][c] = pts[r][c].m > OFF_THRESH && tgt[r][c] === 1;   // dim=1 cross-channel → self at finest
      } else {
        let cnt = 0;
        for (let dr = -1; dr <= 1; dr++)
          for (let dc = -1; dc <= 1; dc++) {
            const nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < FINE && nc >= 0 && nc < FINE && pts[nr][nc].m > OFF_THRESH && tgt[nr][nc] === 1) cnt++;
          }
        B[r][c] = cnt > 0;                                       // dim=2 over the 9 slices → 3×3 neighbour
      }
    }
  return B;
}

function yflowCount(r, c, pts, tgt) {
  let cnt = 0;
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < FINE && nc >= 0 && nc < FINE && pts[nr][nc].m > OFF_THRESH && tgt[nr][nc] === 1) cnt++;
    }
  return cnt;
}

// rectify gradient: three forces (math_model.md §5). futFilterFixed: FUT applies DEP's L294 filter.
function gradAt(r, c, pts, tgt, B, rectify, futFilterFixed) {
  const p = pts[r][c];
  const t = tgt[r][c];
  const mm = p.m > OFF_THRESH ? 1 : 0;
  const caseVal = mm === t ? 1 : 0;
  const yc = yflowCount(r, c, pts, tgt);
  // Convergence filter (DEP L294): zero the alignment gate where mask matches target.
  const filterApplies = rectify === "DEF" || rectify === "CAM" || rectify === "DEP" || (rectify === "FUT" && futFilterFixed);
  const yflowGate = filterApplies ? (caseVal === 1 ? 0 : (yc === 1 ? 1 : 0)) : (yc === 1 ? 1 : 0);

  const sign = p.m >= OFF_THRESH ? 1 : -1;             // attraction sign (rectify L424)
  const weightPos = Math.abs(p.m) > 0.01;              // (weight>0) guard — present in DEP L305 & FUT L425
  // Force hierarchy faithful to source: cleanup α₃(1e1) ≫ activation α₂(3e0) ≫ align α₁(3e-1).
  // The SIGN of the sum is load-bearing — cleanup must dominate align so a misplaced active deactivates.
  const t1 = (yflowGate === 1 && weightPos) ? sign * 0.10 : 0;   // Force 1 — alignment α₁

  let t2 = 0;
  if (caseVal === 0) {
    if (t === 1 && mm === 0) t2 = 0.30;                // Force 2 — activation α₂ (activate a vacant target)
    if (t === 0 && mm === 1) t2 = -0.50;              // Force 2 — cleanup α₃ > α₂ (false positive costlier)
  }

  let coordFlow = 0;
  if (t === 1 && B[r][c] && mm === 0) coordFlow = 1;
  const t3 = coordFlow * 0.10;   // Force 3 (spread) — a COORDINATE force in source (key_query → x/y/z, rectify.py L305 last term); NOT a mask gradient.

  const gm = t1 + t2;            // mask channel = Force 1 (align) + Force 2 (activate) only. Force 3 (t3) moves geometry, not [m>τ] — see math_model Lemma 1.

  let gx = 0, gy = 0, gz = 0, wantR = null, wantC = null;
  if (mm === 1 && p.z > 0) {
    let best = null, bestD = 1e9;
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < FINE && nc >= 0 && nc < FINE && tgt[nr][nc] === 1) {
          const d = dr * dr + dc * dc;
          if (d < bestD) { bestD = d; best = { nr, nc }; }
        }
      }
    if (best) {
      wantR = best.nr; wantC = best.nc;
      const qx = best.nc * p.z, qy = best.nr * p.z;
      gx = -Math.sign(p.x - qx) * 0.5 * p.z;
      gy = -Math.sign(p.y - qy) * 0.5 * p.z;
    }
  }
  const pp = projPixel(p);
  return { gm, t1, t2, t3, coordFlow, yc, gx, gy, gz, total: gm, yflowGate, caseVal, wantR, wantC,
           arrowDx: pp.pc - c, arrowDy: pp.pr - r };
}

function simulate(pts, tgt, rectify, futFilterFixed, bFaithful) {
  const B = computeB(pts, tgt, rectify, bFaithful);
  const meta = pts.map((row, r) => row.map((p, c) => classify(p.m, tgt[r][c], B[r][c])));
  const route = pts.map((row) => row.map((p) => classifyRoute(p.m, p.z)));
  const grad = pts.map((row, r) => row.map((p, c) => gradAt(r, c, pts, tgt, B, rectify, futFilterFixed)));
  return { B, meta, route, grad };
}

// One forward pass (pure) — used by Markov step AND by the next-meta-state preview.
function oneStep(pts, tgt, rectify, futFilterFixed, kEff, bFaithful, lr, salt, freezeFinest = false, schema = "MASK_OPT") {
  let cur = pts.map(row => row.map(p => ({ ...p, stale: false })));
  for (let l = L_MAX; l >= (freezeFinest ? 1 : 0); l--) cur = routeStep(cur, tgt, l, kEff, salt);
  const s = simulate(cur, tgt, rectify, futFilterFixed, bFaithful);
  return cur.map((row, r) => row.map((p, c) => {
    if (p.stale) return { ...p };
    const g = s.grad[r][c];
    let gm = g.gm;
    // S_C warming gate (math_model Lemma 2 / R4): an occluded inactive target receives its activation
    // gradient only when it WINS z-buffer selection this step — probability = effective k_const.
    // k_const=0 ⇒ permanently gradient-starved (Bug B); k_const>0 ⇒ selected infinitely often a.s. ⇒ warms.
    if (gm > 0 && p.m <= OFF_THRESH && tgt[r][c] === 1 && p.z > 0 && rand01(r * 5 + c, salt * 3 + 9, salt) >= kEff) gm = 0;
    const nextM = schema === "CAM" ? p.m : Math.max(0, Math.min(1, p.m + lr * gm));
    return { ...p, x: p.x + lr * g.gx, y: p.y + lr * g.gy, z: Math.max(0.01, p.z + lr * g.gz), m: nextM };
  }));
}

// ── Particle identity through routing ─────────────────────────────────
// smap.py L83 einsum: weights_x (selection) is DETACHED but x_z_value (the coordinates) is NOT —
// gradient at the post-routing destination flows back to the SAME differentiable coordinate variable.
// So a routed value is the same physical particle that moved, not a fresh unlinked point. The UI gives
// each point a stable id, preserves it through routeStep, and follows it so the particle is traceable.
function stampIds(sst) {
  return { tgt: sst.tgt, pts: sst.pts.map((row, r) => row.map((p, c) => ({ ...p, id: r * FINE + c }))) };
}
function findById(pts, id) {
  if (id == null) return null;
  for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++) if (pts[r][c].id === id) return { r, c };
  return null;
}
// Forward (routing transport only, no rectification gradient) = forward() in render mode. Applies the
// projection of the current coordinates to the on-screen position without a training gradient step.
function routeOnly(pts, tgt, kEff, salt, freezeFinest = false) {
  let cur = pts.map(row => row.map(p => ({ ...p, stale: false })));
  for (let l = L_MAX; l >= (freezeFinest ? 1 : 0); l--) cur = routeStep(cur, tgt, l, kEff, salt);
  return cur;
}

// Φ and the two correctness sub-conditions (math_model.md §1, §5).
function phiStats(pts, tgt) {
  let phi = 0, covered = 0, totT = 0, clean = 0, totA = 0;
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      const a = pts[r][c].m > OFF_THRESH, t = tgt[r][c] === 1;
      if (t) { totT++; if (a) covered++; }
      if (a) { totA++; if (t) clean++; }
      if (a !== t) phi++;
    }
  return { phi, covered, totT, clean, totA };
}

// Effective k_const for a step: FIXED = explore→anneal ramp 0→1 (math_model L94); SOURCE = constant base.
function kEffective(kBase, kSchedFixed, step) {
  return kSchedFixed ? Math.min(1, 0.1 + 0.9 * step / RAMP) : kBase;
}
// Suggested base k_const for a setting: more depth spread ⇒ more occlusion ⇒ more warming.
function suggestK(pts) {
  let zmin = 1e9, zmax = -1e9;
  for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++) { const z = pts[r][c].z; if (z > 0) { zmin = Math.min(zmin, z); zmax = Math.max(zmax, z); } }
  if (zmax <= 0) return 0.3;
  const spread = (zmax - zmin) / Math.max(zmax, 1e-6);   // 0 = coplanar, →1 = very varied depth
  return Math.round(Math.max(0.2, Math.min(0.95, 0.2 + 0.8 * spread)) * 100) / 100;
}

// Convergence certificate (math_model.md §5 + §P): valid ⇔ no active correctness bug breaks the
// Lyapunov / coverage argument; otherwise point to the responsible bug + fix direction.
function certificate({ pts, tgt, rectify, futFilterFixed, kSchedFixed, bFaithful, kBaseEff, kSuggest, sim, phi0, phiNow, phiHist, schema = "MASK_OPT" }) {
  // Bug A: FUT source with an actually non-absorbing S1 present.
  let s1bad = 0;
  for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++)
    if (sim.meta[r][c] === "S1" && Math.abs(sim.grad[r][c].total) > 1e-6) s1bad++;
  const bugA = rectify === "FUT" && !futFilterFixed && s1bad > 0;
  // Bug C: DEP without source-faithful B → certificate computed on the wrong partition.
  const bugC = rectify === "DEP" && !bFaithful;
  // Bug B: occlusion present (a dark target with a strictly-nearer active neighbour) and schedule pinned.
  let occluded = 0;
  for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++) {
    if (tgt[r][c] === 1 && pts[r][c].m <= OFF_THRESH && pts[r][c].z > 0) {
      let ring = false;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr, nc = c + dc;
        if ((dr || dc) && nr >= 0 && nr < FINE && nc >= 0 && nc < FINE && pts[nr][nc].m > OFF_THRESH && pts[nr][nc].z > 0 && pts[nr][nc].z < pts[r][c].z) ring = true;
      }
      if (ring) occluded++;
    }
  }
  const bugB = occluded > 0 && !kSchedFixed && kBaseEff < kSuggest;
  // monotonicity over the recent trace
  let roseAt = -1;
  for (let i = 1; i < phiHist.length; i++) if (phiHist[i] > phiHist[i - 1]) roseAt = i;

  if (bugA) return { valid: false, sev: "CORRECTNESS", cause: `Bug A — FUT lacks the convergence filter (rectify.py L294). ${s1bad} S1 cell(s) have m-grad≠0 ⇒ S1 is NOT a strict mask fixed point (Lemma 1 violated). CONFIRMED by executing the real source (t_a2.py): DEP matched-cell self-force = 0 (absorbing); FUT = +α₁ (reinforcing, m→1), so Φ does NOT numerically rise — the defect is the lost absorbing property, not divergence. Latent: training uses DEP, not FUT. Fix: toggle 'FUT convergence filter = FIXED'.` };
  if (bugC) return { valid: false, sev: "AUDITABILITY", cause: `Bug C — under DEP the app B must be cross-channel (source dim=1), not 3×3-neighbour. The certificate is computed on the wrong meta-state partition. Fix: toggle 'B = source-faithful'.` };
  if (bugB) return { valid: false, sev: "CORRECTNESS", cause: `Warming insufficient — ${occluded} occluded target(s) with k_const pinned at ${kBaseEff.toFixed(2)} < ${kSuggest.toFixed(2)} needed: at this SMALL k the dark target loses z-buffer selection ⇒ gradient-starved ⇒ never covered. NOTE: this is the small-k MECHANISM, not the source regime — the source pins k=1.0 (MAX warming) which covers fine (no starvation). The real Bug B is the discarded 0→1 schedule (notebook L435), not a low value. Fix: apply the schedule (toggle 'k_const schedule = FIXED'); raising k also covers in-app.` };
  if (roseAt >= 0) {
    if (schema === "CAM") {
      return { valid: true, sev: "NON_MONOTONE", cause: `Φ rose (step ${roseAt}: ${phiHist[roseAt - 1]}→${phiHist[roseAt]}). Note: In CAM schema (m-fixed), coordinate dynamics are not proven strictly monotone due to projection alignment and occlusion coupling; transient rises are mathematically expected.` };
    }
    return { valid: false, sev: "OBSERVED", cause: `Φ rose (step ${roseAt}: ${phiHist[roseAt - 1]}→${phiHist[roseAt]}). With no a-priori bug this should not happen — inspect the cell whose meta-state regressed.` };
  }
  const W = 8;
  if (phiNow > 0 && phiHist.length > W + 1) {
    const win = phiHist.slice(-(W + 1));
    if (!win.some((v, i) => i > 0 && v < win[i - 1]))
      return { valid: false, sev: "STALLED", cause: `Φ stalled at ${phiNow} over the last ${W} steps with no a-priori bug (A/B/C) active ⇒ convergence NOT observed. This is the routing↔gradient coupling region (math_model residual A1/A4 — e.g. a point cluster all projecting onto one cell); not an a-priori bug — pending the deeper execution audit (backward() + real source), or adjust the configuration / resolution.` };
  }
  return { valid: true, sev: "OK", cause: schema === "CAM"
    ? `Φ convergence depends on geometric alignment (Lemma 2-CAM). Active points project to cover targets; since mask is fixed, convergence is conditional on pose-feasibility and avoids local minima (Theorem-CAM).`
    : `Φ is a non-increasing integer ≥0 (Lemma 1) that strictly drops at each τ-crossing ⇒ converges in ≤ Φ₀ = ${phi0} crossings; a strictly-decreasing well-founded potential admits no cycle ⇒ NO periodic non-termination (math_model Theorem).` };
}

function commentary(sel, pts, tgt, l, channel, rectify, simNow, simFixed, nextMetaNow, nextRouteNow, schema = "MASK_OPT") {
  const { r, c } = sel;
  const p = pts[r][c];
  const t = tgt[r][c];
  const B = simNow.B[r][c];
  const meta = simNow.meta[r][c];
  const route = simNow.route[r][c];
  const gN = simNow.grad[r][c], gF = simFixed.grad[r][c];
  const view = origToView(r, c, l);
  const lines = [];
  if (schema === "CAM") {
    lines.push({ k: "rule", t: `CAM Schema (m-fixed): mask is fixed, no F2 gradient. Rectify-state ${meta} (m=${p.m.toFixed(2)}, t=${t}, B=${B ? 1 : 0}, y_flow=${gN.yc}) · Routing-state ${route} (z=${p.z.toFixed(2)}).` });
  } else {
    lines.push({ k: "rule", t: `Rectify-state ${meta} (m=${p.m.toFixed(2)} ${p.m > OFF_THRESH ? ">τ" : "≤τ"}, t=${t}, B=${B ? 1 : 0}, y_flow=${gN.yc}) · Routing-state ${route} (z=${p.z.toFixed(2)}). Next step → ${meta}/${route} ⇒ ${nextMetaNow}/${nextRouteNow}.` });
  }
  lines.push({ k: "view", t: `At L${l}: channel ${view.channel}/${levelDims(l).chans}, cell (${view.ph},${view.pw}). ${channel === view.channel ? "(visible now)" : `Slide to channel ${view.channel}.`}` });
  const pp = projPixel(p);
  lines.push({ k: "proj", t: `xyz→finest pixel (row ${pp.pr.toFixed(1)}, col ${pp.pc.toFixed(1)}); cell=(row ${r}, col ${c}). Routing moves ≤1 neighbor PER LEVEL (=${1 << l}px at L${l}); but ONE forward pass runs all levels coarse→fine, so a single Forward/Markov step can cross Σ2^l cells at once.` });
  if (meta === "S1") {
    if (rectify === "FUT") {
      if (Math.abs(gN.total) > 1e-6) {
        lines.push({ k: "bug", t: `FUT, S1, y_flow=${gN.yc}: m-grad=${gN.total.toFixed(3)}≠0 → S1 not a strict mask fixed point (Lemma 1). Tap reinforcing (m→1) ⇒ Φ doesn't rise — CONFIRMED via real-source backward (t_a2.py: DEP self-force 0, FUT +α₁). FUT lacks y_flow*=(1−case) (DEP L294; absent L405-425). Latent: training uses DEP.` });
        lines.push({ k: "fix", t: `'FUT convergence filter = FIXED' → m-grad ${gF.total.toFixed(3)}. Source: add y_flow*=(1.−((pre_mask>OFF)==target)) after rectify.py L418.` });
      } else lines.push({ k: "ok", t: `FUT, S1: m-grad=${gN.total.toFixed(3)} (filter fixed, or y_flow≠1 here).` });
    } else lines.push({ k: "ok", t: `${rectify}, S1: m-grad=${gN.total.toFixed(3)} ✓ absorbing (filter present DEF L108 / CAM L168 / DEP L294).` });
  } else if (gN.wantR !== null) {
    lines.push({ k: "grad", t: `xyz gradient pulls (x,y) toward target (${gN.wantR},${gN.wantC}) in the finest 3×3: gx=${gN.gx.toFixed(2)}, gy=${gN.gy.toFixed(2)} (key_query, utils.py L180-182; is_last L244-251). LOCAL only — far targets reached by ROUTING.` });
  }
  if (route === "SC")
    lines.push({ k: "grad", t: `Routing-state S_C (warming): inactive+depth participates with prob k_const (smap.py L54-58). This is the occlusion-relief knob (Bug B). Raise k_const / FIX the schedule if an occluded target stalls.` });
  return lines;
}

export default function SMapSimulator() {
  const [{ pts, tgt }, setSST] = useState(() => stampIds(initDefault()));
  const [rawSST, setRawSST] = useState(() => stampIds(initDefault()));   // retained step-0 snapshot for the "show original" compare view
  const [schema, setSchema] = useState("MASK_OPT");              // MASK_OPT | CAM
  const [rectify, setRectify] = useState("FUT");
  const [futFilterFixed, setFutFilterFixed] = useState(false);   // Bug A
  const [kSchedFixed, setKSchedFixed] = useState(false);         // Bug B (schedule pinned vs ramp)
  const [bFaithful, setBFaithful] = useState(false);            // Bug C (B per-Rectify)
  const [a1a4Fixed, setA1a4Fixed] = useState(true);             // A1/A4 framing: RESOLVED by Lemma 3 (proof-default ON); toggle OFF to see the pre-Lemma-3 conditional framing
  const [kBase, setKBase] = useState(1.0);                       // k_const constant (source default 1.0)
  const [kMode, setKMode] = useState("manual");                 // manual | auto
  const [zoom, setZoom] = useState(0);
  const [channel, setChannel] = useState(0);
  const [selOrig, setSelOrig] = useState({ r: 3, c: 4 });
  const [candOn, setCandOn] = useState(false);
  const [candSlice, setCandSlice] = useState(4);
  const [lrMode, setLrMode] = useState("manual");
  const [lrManual, setLrManual] = useState(0.1);
  const [step, setStep] = useState(0);
  const [preset, setPreset] = useState("default");
  const [hist, setHist] = useState([]);
  const [phiHist, setPhiHist] = useState([]);
  const [lastMove, setLastMove] = useState(null);   // trace of where the selected particle just routed
  const [freezeFinest, setFreezeFinest] = useState(false);   // skip the finest (L0) routing pass — inspect pre-finest-routing state (source: n−zoom, zoom=1)
  const [showOriginal, setShowOriginal] = useState(false);   // pure VIEW: render the step-0 raw snapshot (any zoom) to compare positions before/after routing
  const [snapshotPanelOpen, setSnapshotPanelOpen] = useState(false);
  const [snapshotText, setSnapshotText] = useState("");

  const handleSchemaChange = (newSchema) => {
    setSchema(newSchema);
    if (newSchema === "CAM") {
      setRectify("CAM");
      setKBase(0.5);
    } else {
      setRectify("DEP");
      setKBase(1.0);
    }
  };

  const simNow = useMemo(() => simulate(pts, tgt, rectify, futFilterFixed, bFaithful), [pts, tgt, rectify, futFilterFixed, bFaithful]);
  const simFixed = useMemo(() => simulate(pts, tgt, rectify, true, true), [pts, tgt, rectify]);
  const sim = simNow;

  const { side, chans, per } = levelDims(zoom);
  const safeChannel = Math.min(channel, chans - 1);

  const kSuggest = useMemo(() => suggestK(pts), [pts]);
  const kBaseEff = kMode === "auto" ? kSuggest : kBase;
  const kEff = kEffective(kBaseEff, kSchedFixed, step);

  const lrAuto = useMemo(() => {
    let mx = 0;
    for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++) mx = Math.max(mx, Math.abs(sim.grad[r][c].total));
    return mx > 1e-6 ? 0.05 / mx : 0.1;
  }, [sim]);
  const lr = lrMode === "auto" ? lrAuto : lrManual;

  // Next-state preview (one pure forward pass on current state).
  const nextPts = useMemo(() => oneStep(pts, tgt, rectify, futFilterFixed, kEff, bFaithful, lr, step + 1, freezeFinest, schema),
    [pts, tgt, rectify, futFilterFixed, kEff, bFaithful, lr, step, freezeFinest, schema]);
  const nextSim = useMemo(() => simulate(nextPts, tgt, rectify, futFilterFixed, bFaithful), [nextPts, tgt, rectify, futFilterFixed, bFaithful]);

  const ps = useMemo(() => phiStats(pts, tgt), [pts, tgt]);
  const phi0 = useMemo(() => (phiHist.length ? phiHist[0] : ps.phi), [phiHist, ps.phi]);
  const cert = useMemo(() => certificate({ pts, tgt, rectify, futFilterFixed, kSchedFixed, bFaithful, kBaseEff, kSuggest, sim: simNow, phi0, phiNow: ps.phi, phiHist, schema }),
    [pts, tgt, rectify, futFilterFixed, kSchedFixed, bFaithful, kBaseEff, kSuggest, simNow, phi0, ps.phi, phiHist, schema]);

  const selVisible = safeChannel === origToView(selOrig.r, selOrig.c, zoom).channel;

  const toggleShowOriginal = () => {
    setShowOriginal(prev => {
      const nextVal = !prev;
      if (nextVal) {
        // Toggling ON: map live selection to original position
        const selP = pts[selOrig.r]?.[selOrig.c];
        if (selP && selP.id !== null) {
          const origR = Math.floor(selP.id / FINE);
          const origC = selP.id % FINE;
          setSelOrig({ r: origR, c: origC });
        }
      } else {
        // Toggling OFF: map original selection to live position
        const origId = selOrig.r * FINE + selOrig.c;
        const livePos = findById(pts, origId);
        if (livePos) {
          setSelOrig(livePos);
        }
      }
      return nextVal;
    });
  };

  const editSel = (key, v) => {
    if (showOriginal) {
      setRawSST(({ pts: rPts, tgt: rTgt }) => {
        const NP = rPts.map(row => row.map(p => ({ ...p })));
        NP[selOrig.r][selOrig.c][key] = v;
        return { pts: NP, tgt: rTgt };
      });
      if (step === 0) {
        setSST(({ pts: sPts, tgt: sTgt }) => {
          const NP = sPts.map(row => row.map(p => ({ ...p })));
          NP[selOrig.r][selOrig.c][key] = v;
          return { pts: NP, tgt: sTgt };
        });
      }
    } else {
      setSST(({ pts: sPts, tgt: sTgt }) => {
        const NP = sPts.map(row => row.map(p => ({ ...p })));
        NP[selOrig.r][selOrig.c][key] = v;
        return { pts: NP, tgt: sTgt };
      });
      if (step === 0) {
        setRawSST(({ pts: rPts, tgt: rTgt }) => {
          const NP = rPts.map(row => row.map(p => ({ ...p })));
          NP[selOrig.r][selOrig.c][key] = v;
          return { pts: NP, tgt: rTgt };
        });
      }
    }
  };

  const toggleT = () => {
    if (showOriginal) {
      setRawSST(({ pts: rPts, tgt: rTgt }) => {
        const NT = rTgt.map(row => [...row]);
        NT[selOrig.r][selOrig.c] = NT[selOrig.r][selOrig.c] ? 0 : 1;
        return { pts: rPts, tgt: NT };
      });
      if (step === 0) {
        setSST(({ pts: sPts, tgt: sTgt }) => {
          const NT = sTgt.map(row => [...row]);
          NT[selOrig.r][selOrig.c] = NT[selOrig.r][selOrig.c] ? 0 : 1;
          return { pts: sPts, tgt: NT };
        });
      }
    } else {
      setSST(({ pts: sPts, tgt: sTgt }) => {
        const NT = sTgt.map(row => [...row]);
        NT[selOrig.r][selOrig.c] = NT[selOrig.r][selOrig.c] ? 0 : 1;
        return { pts: sPts, tgt: NT };
      });
      if (step === 0) {
        setRawSST(({ pts: rPts, tgt: rTgt }) => {
          const NT = rTgt.map(row => [...row]);
          NT[selOrig.r][selOrig.c] = NT[selOrig.r][selOrig.c] ? 0 : 1;
          return { pts: rPts, tgt: NT };
        });
      }
    }
  };

  // Commit a new SST, follow the selected particle by id, and record its routing trace.
  // isStep=true: full Markov pass (advances step + Φ history). isStep=false: Forward (route only).
  const commitPts = (newPts, isStep) => {
    let currentSelOrig = selOrig;
    if (showOriginal) {
      const origId = selOrig.r * FINE + selOrig.c;
      const livePos = findById(pts, origId);
      if (livePos) {
        currentSelOrig = livePos;
      }
    }
    const selId = pts[currentSelOrig.r]?.[currentSelOrig.c]?.id;
    const moved = findById(newPts, selId);
    setHist(h => [...h, { pts: pts.map(row => row.map(p => ({ ...p }))), step, sel: { ...currentSelOrig }, lastMove, phiHist }]);
    setSST({ pts: newPts, tgt });
    if (isStep) {
      setPhiHist(h => [...(h.length ? h : [phiStats(pts, tgt).phi]), phiStats(newPts, tgt).phi]);
      setStep(s => s + 1);
    }
    setShowOriginal(false);
    if (moved && (moved.r !== currentSelOrig.r || moved.c !== currentSelOrig.c)) {
      setSelOrig(moved); setCandOn(false);
      setLastMove({ from: { ...currentSelOrig }, to: moved, occluded: false });
    } else if (!moved && selId != null) {
      setLastMove({ from: { ...currentSelOrig }, to: null, occluded: true });
      setSelOrig(currentSelOrig);
    } else {
      setLastMove(null);
      setSelOrig(currentSelOrig);
    }
  };
  const stepMarkov = () => commitPts(nextPts, true);
  const forward = () => commitPts(routeOnly(pts, tgt, kEff, step + 1, freezeFinest), false);

  const undo = useCallback(() => {
    if (!hist.length) return;
    const last = hist[hist.length - 1];
    setSST(prev => ({ pts: last.pts, tgt: prev.tgt }));
    setStep(last.step); setHist(h => h.slice(0, -1));
    setPhiHist(last.phiHist || []);
    if (last.sel) setSelOrig(last.sel);
    setLastMove(last.lastMove || null);
    setShowOriginal(false);
  }, [hist]);

  const reset = useCallback(() => {
    const init = stampIds(preset === "default" ? initDefault() : preset === "random" ? initRandom()
      : preset === "far" ? initFar() : preset === "occlusion" ? initOcclusion() : initConvergence());
    setSST(init); setRawSST(init); setStep(0); setHist([]); setPhiHist([phiStats(init.pts, init.tgt).phi]); setLastMove(null);
    setSelOrig({ r: 3, c: 4 }); setCandOn(false); setCandSlice(4);   // reset selection too (was stale across resets)
    setShowOriginal(false);
  }, [preset]);

  const getSnapshotJSON = () => {
    const state = {
      pts,
      tgt,
      rawSST,
      schema,
      rectify,
      futFilterFixed,
      kSchedFixed,
      bFaithful,
      a1a4Fixed,
      kBase,
      kMode,
      zoom,
      channel,
      selOrig,
      candOn,
      candSlice,
      lrMode,
      lrManual,
      step,
      preset,
      hist,
      phiHist,
      lastMove,
      freezeFinest,
      showOriginal
    };
    return JSON.stringify(state);
  };

  const loadSnapshot = (jsonStr) => {
    try {
      const state = JSON.parse(jsonStr);
      if (!state.pts || !state.tgt || !Array.isArray(state.pts)) {
        alert("Invalid snapshot format: missing pts or tgt");
        return;
      }
      setSST({ pts: state.pts, tgt: state.tgt });
      if (state.rawSST) setRawSST(state.rawSST);
      if (state.schema) setSchema(state.schema);
      if (state.rectify) setRectify(state.rectify);
      if (state.futFilterFixed !== undefined) setFutFilterFixed(state.futFilterFixed);
      if (state.kSchedFixed !== undefined) setKSchedFixed(state.kSchedFixed);
      if (state.bFaithful !== undefined) setBFaithful(state.bFaithful);
      if (state.a1a4Fixed !== undefined) setA1a4Fixed(state.a1a4Fixed);
      if (state.kBase !== undefined) setKBase(state.kBase);
      if (state.kMode) setKMode(state.kMode);
      if (state.zoom !== undefined) setZoom(state.zoom);
      if (state.channel !== undefined) setChannel(state.channel);
      if (state.selOrig) setSelOrig(state.selOrig);
      if (state.candOn !== undefined) setCandOn(state.candOn);
      if (state.candSlice !== undefined) setCandSlice(state.candSlice);
      if (state.lrMode) setLrMode(state.lrMode);
      if (state.lrManual !== undefined) setLrManual(state.lrManual);
      if (state.step !== undefined) setStep(state.step);
      if (state.preset) setPreset(state.preset);
      if (state.hist) setHist(state.hist);
      if (state.phiHist) setPhiHist(state.phiHist);
      if (state.lastMove !== undefined) setLastMove(state.lastMove);
      if (state.freezeFinest !== undefined) setFreezeFinest(state.freezeFinest);
      if (state.showOriginal !== undefined) setShowOriginal(state.showOriginal);
      setSnapshotPanelOpen(false);
    } catch (e) {
      alert("Failed to parse snapshot: " + e.message);
    }
  };

  const viewCells = [];
  for (let ph = 0; ph < side; ph++)
    for (let pw = 0; pw < side; pw++) {
      const { orig_r, orig_c } = viewToOrig(ph, pw, safeChannel, zoom);
      viewCells.push({ ph, pw, orig_r, orig_c });
    }

  const centerView = origToView(selOrig.r, selOrig.c, zoom);
  const candVisible = candOn && !showOriginal && safeChannel === centerView.channel;
  // "show original" view: the grid renders the retained step-0 snapshot (positions before any routing),
  // at the current zoom — a pure display, orthogonal to the zoom selector (so L0..L3 stay usable).
  const viewPts = showOriginal ? rawSST.pts : pts;
  const viewSim = useMemo(() => showOriginal ? simulate(rawSST.pts, rawSST.tgt, rectify, futFilterFixed, bFaithful) : simNow,
    [showOriginal, rawSST, rectify, futFilterFixed, bFaithful, simNow]);
  const viewSimFixed = useMemo(() => showOriginal ? simulate(rawSST.pts, rawSST.tgt, rectify, true, true) : simFixed,
    [showOriginal, rawSST, rectify, simFixed]);
  const sliceAt = (ph, pw) => {
    if (!candVisible) return -1;
    const dr = ph - centerView.ph, dc = pw - centerView.pw;
    if (dr < -1 || dr > 1 || dc < -1 || dc > 1) return -1;
    return (dr + 1) * 3 + (dc + 1);
  };
  const clickCell = (ph, pw, orig_r, orig_c) => {
    if (!candOn) {
      if (orig_r === selOrig.r && orig_c === selOrig.c) { setCandSlice(4); setCandOn(true); }
      else setSelOrig({ r: orig_r, c: orig_c });
      return;
    }
    const s = sliceAt(ph, pw);
    if (s === -1) { setSelOrig({ r: orig_r, c: orig_c }); setCandSlice(4); setCandOn(true); return; }
    if (s === 4) { if (candSlice !== 4) setCandSlice(4); else setCandOn(false); return; }
    setCandSlice(s);
  };

  const selP = viewPts[selOrig.r][selOrig.c];
  const selMeta = viewSim.meta[selOrig.r][selOrig.c];
  const selRoute = viewSim.route[selOrig.r][selOrig.c];
  const selInfo = META[selMeta];
  const selRouteInfo = META_ROUTE[selRoute];
  const nextMetaSel = showOriginal ? selMeta : nextSim.meta[selOrig.r][selOrig.c];
  const nextRouteSel = showOriginal ? selRoute : nextSim.route[selOrig.r][selOrig.c];
  const gN = viewSim.grad[selOrig.r][selOrig.c];
  const gF = viewSimFixed.grad[selOrig.r][selOrig.c];
  const selCand = candOn ? candInfo(selP, selOrig.r, selOrig.c, candSlice, zoom, showOriginal ? rawSST.tgt : tgt) : null;
  const comm = commentary(selOrig, viewPts, showOriginal ? rawSST.tgt : tgt, zoom, safeChannel, rectify, viewSim, viewSimFixed, nextMetaSel, nextRouteSel, schema);
  const cellPx = Math.floor(360 / side);

  const bugRows = [
    { id: "futFilter", label: "FUT convergence filter", applies: rectify === "FUT", fixed: futFilterFixed,
      toggle: () => setFutFilterFixed(v => !v), severity: "CORRECTNESS", where: "rectify.py: DEP L294 vs FUT (absent L405-425)",
      effect: rectify === "FUT" ? (futFilterFixed ? "FIX (if applied): S1 absorbing (g_m=0)." : "SOURCE (buggy): S1 with y_flow==1 not a strict fixed point (g_m≠0). Tap reinforcing (+α₁, m→1) ⇒ Φ does NOT rise — confirmed via real-source backward (t_a2.py). Latent: training uses DEP.") : "N/A — only when Rectify = FUT." },
    { id: "kSched", label: "k_const schedule", applies: true, fixed: kSchedFixed,
      toggle: () => setKSchedFixed(v => !v), severity: "AUDITABILITY", where: "notebook L435 discards ramp L432; smap.py L57 finest: higher k=more warming (⚠ L62 intermediate path inverts the sense — fix must reconcile both)",
      effect: kSchedFixed ? `FIX (if applied): apply intended 0→1 schedule (notebook L432; now k=${kEff.toFixed(2)}).` : `SOURCE: schedule discarded → k_const pinned constant (real source = 1.0 = MAX warming; app current = ${kBaseEff.toFixed(2)}). At k≥suggested occluded targets ARE covered (no starvation); deviation = the missing 0→1 schedule, not a low value. Precision/rate effect pending source execution.` },
    { id: "bFaithful", label: "B = source-faithful (per-Rectify)", applies: true, fixed: bFaithful,
      toggle: () => setBFaithful(v => !v), severity: "AUDITABILITY", where: "rectify.py L25/L326 (dim=2) vs L203 (dim=1, DEP)",
      effect: bFaithful ? "FIX (if applied): DEP B → cross-channel→self (source dim=1). DEF/FUT already 3×3 (dim=2) in both modes — this toggle affects DEP only." : "SOURCE (app-approx): DEP uses uniform 3×3 B (wrong; source dim=1). DEF/FUT 3×3 is already faithful." },
    { id: "a1a4", label: "A1/A4 occlusion-uncover", applies: true, fixed: a1a4Fixed,
      toggle: () => setA1a4Fixed(v => !v), severity: "PROOF", where: "smap.py L176-180 (z-buffer fallback) · math_model Lemma 3 / Theorem′",
      effect: a1a4Fixed
        ? "RESOLVED (PROOF, not a source-toggle): Lemma 3 — an inactive z-buffer winner is rejected to the cell's OWN center value (smap.py L180), so a matched, self-projecting target is NEVER uncovered by occlusion ⇒ certificate routing-region is UNCONDITIONAL. Confirmed by t_a1a4.py (cover≡1 over 40 steps). This standard is genuinely closed (proof), unlike the assume-solved bug toggles."
        : "OPEN framing (pre-Lemma-3): assume occlusion COULD uncover a matched S1 (vacate ⇒ Φ↑) ⇒ certificate routing-region only CONDITIONALLY valid. Toggle ON to apply the Lemma-3 fix-framing." },
  ];

  // App self-audit health: a standard's `met` is now DERIVED from the matching assume-solved toggle, so
  // toggling a bug-framing on the page updates the badge live and shows the fix-framing.
  //   a1a4 ← a1a4Fixed (proof toggle, default RESOLVED) · l62 & bugB ← kSchedFixed (the Bug-B schedule fix).
  const stdMet = { a1a4: a1a4Fixed, l62: kSchedFixed, bugB: kSchedFixed };
  const auditStds = AUDIT_STANDARDS.map(s => ({ ...s, met: s.id in stdMet ? stdMet[s.id] : s.met }));
  const openStds = auditStds.filter(s => !s.met).sort((a, b) => stdPriority(b) - stdPriority(a));
  const topStds = openStds.slice(0, 2);
  // Honesty guard: l62/bugB close ONLY by ASSUMING the (unapplied) Bug-B source fix — in audit phase the
  // source is unpatched, so such "met" is a HYPOTHESIS, not a real change. a1a4 is proof-resolved (Lemma 3),
  // not assumed. Never let a green badge read as "source fixed / instrument faithful" when it is assumption-driven.
  const assumedMet = auditStds.filter(s => s.met && (s.id === "l62" || s.id === "bugB")).length;
  // Same guard for the Φ certificate: a convergence-relevant assume-solved FIX toggle being ON makes
  // "certificate valid" a HYPOTHESIS (the assumed-fixed algorithm), not a claim about the unpatched source.
  const assumedFixActive = (rectify === "FUT" && futFilterFixed) || kSchedFixed;

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(180deg,#020617,#0f172a)", color: "#e2e8f0", fontFamily: "'JetBrains Mono',ui-monospace,monospace", padding: 12 }}>
      <div style={{ maxWidth: 1000, margin: "0 auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", borderBottom: "1px solid #1e293b", paddingBottom: 8, marginBottom: 10 }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 800, color: "#f8fafc", letterSpacing: -0.5 }}>SMap Simulator — Meta-states, Φ Convergence & Rectify Mechanics</div>
            <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>anchored to math_model.md · two meta-state layers (routing S_A–S_D · rectify S1–S7) · Φ = #mismatch → 0 · bugs gated per setting</div>
          </div>
          <div style={{ fontSize: 10, color: "#64748b", textAlign: "right" }}>
            <div>Schema: <b style={{ color: "#38bdf8" }}>{schema === "CAM" ? "CAM (m-fixed)" : "Mask-Opt"}</b></div>
            <div>Step <b style={{ color: "#f8fafc" }}>{step}</b> · seed <b style={{ color: "#475569" }}>{SEED.toFixed(0)}</b></div>
            <div>Φ = <b style={{ color: ps.phi ? "#fca5a5" : "#86efac" }}>{ps.phi}</b> / Φ₀ {phi0}</div>
            <div>{cert.valid
              ? <span style={{ color: (assumedFixActive || cert.sev === "NON_MONOTONE") ? "#fcd34d" : "#86efac" }} title={assumedFixActive ? "valid under the assume-solved FIX toggle(s) you set — source UNPATCHED (audit phase); hypothetical, not the actual source" : "valid for the current source-faithful config"}>✓ certificate valid{assumedFixActive ? " (assumed fix)" : ""}{cert.sev === "NON_MONOTONE" ? " (non-monotone)" : ""}</span>
              : <span style={{ color: "#fca5a5" }}>⚠ certificate void</span>}</div>
            <div>{openStds.length
              ? <span style={{ color: "#fbbf24", fontWeight: 700 }} title={`${openStds.length} app-audit standard(s) not yet met`}>⚠ {openStds.length} app-standards</span>
              : <span style={{ color: assumedMet ? "#fcd34d" : "#86efac" }} title={assumedMet ? `${assumedMet} standard(s) cleared by ASSUME-solved toggle(s); source is UNPATCHED (audit phase) — hypothesis, not a real fix` : "all tracked standards genuinely resolved"}>{assumedMet ? `✓ standards met · ${assumedMet} assumed` : "✓ app-standards met"}</span>}</div>
          </div>
        </div>

        {/* APP SELF-AUDIT STANDARDS — process-level health, DISTINCT from the Φ certificate below */}
        <div style={{ marginBottom: 10, padding: "5px 9px", borderRadius: 6, fontSize: 9.5, lineHeight: 1.5,
          background: openStds.length ? "#f59e0b12" : "#10b98112", border: `1px solid ${openStds.length ? "#f59e0b55" : "#10b98155"}`,
          display: "flex", gap: 8, alignItems: "baseline", flexWrap: "wrap" }}>
          <span style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5 }}>App standards</span>
          {openStds.length === 0
            ? <b style={{ color: assumedMet ? "#fcd34d" : "#86efac" }}>{assumedMet
                ? `✓ standards cleared — but ${assumedMet} via ASSUME-solved toggle(s) (hypothesis); SOURCE UNPATCHED (audit phase). Instrument-faithfulness is verified separately (validation + mechanism-specs), NOT implied by these toggles.`
                : "✓ all app-standards genuinely resolved — instrument faithful & self-consistent"}</b>
            : <>
                <b style={{ color: "#fca5a5" }}>⚠ {openStds.length}</b>
                <span style={{ color: "#94a3b8" }}>open · top (Effort×Impact):</span>
                {topStds.map((s, i) => (
                  <span key={s.id} title={s.note} style={{ color: "#fde68a", cursor: "help" }}>
                     {i ? "· " : ""}{s.label}
                     <span style={{ fontSize: 8, marginLeft: 4, padding: "0 4px", borderRadius: 3, background: stdQuadrant(s) === "quick-win" ? "#10b98133" : stdQuadrant(s) === "major" ? "#dc262633" : "#33415588", color: "#cbd5e1" }}>{stdQuadrant(s)}</span>
                  </span>
                ))}
                {openStds.length > topStds.length && <span style={{ color: "#64748b" }}>· +{openStds.length - topStds.length} more</span>}
              </>}
        </div>

        {/* CONVERGENCE CERTIFICATE (math_model.md §5) */}
        <div style={{ marginBottom: 10, padding: 8, background: cert.valid ? "#10b98112" : "#dc262612", border: `1px solid ${cert.valid ? "#10b98155" : "#dc262655"}`, borderRadius: 6 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
            <span style={{ fontSize: 10, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 0.5 }}>Convergence certificate</span>
            <span style={{ fontSize: 10, color: "#cbd5e1" }}>
              targets covered <b style={{ color: ps.covered === ps.totT ? "#86efac" : "#fca5a5" }}>{ps.covered}/{ps.totT}</b> · actives on-target <b style={{ color: ps.clean === ps.totA ? "#86efac" : "#fca5a5" }}>{ps.clean}/{ps.totA}</b>
            </span>
          </div>
          <div style={{ fontSize: 11, color: cert.valid ? "#a7f3d0" : "#fecaca", lineHeight: 1.5 }}>
            <b>{cert.valid ? "✓" : "⚠ " + cert.sev}</b> {cert.valid && assumedFixActive ? <span style={{ color: "#fcd34d" }}>[under assume-solved FIX toggle(s) · source UNPATCHED] </span> : null}{cert.cause}
          </div>
          {phiHist.length > 1 && (
            <div style={{ display: "flex", alignItems: "flex-end", gap: 1, marginTop: 6, height: 22 }}>
              {phiHist.slice(-40).map((v, i) => (
                <div key={i} title={`Φ=${v}`} style={{ width: 5, height: `${phi0 ? Math.max(2, (v / Math.max(1, phi0)) * 22) : 2}px`, background: i > 0 && phiHist.slice(-40)[i - 1] < v ? "#ef4444" : "#10b981", opacity: 0.8 }} />
              ))}
              <span style={{ fontSize: 8, color: "#64748b", marginLeft: 6 }}>Φ trace (≤ Φ₀ crossings to 0; red = rose ⇒ non-monotone)</span>
            </div>
          )}
        </div>

        {/* SCHEMA SELECTOR */}
        <div style={{ display: "flex", gap: 6, marginBottom: 8, alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "#64748b", width: 60 }}>Schema</span>
          {[
            { id: "MASK_OPT", name: "Mask-Opt" },
            { id: "CAM", name: "CAM (m-fixed)" }
          ].map(s => (
            <button key={s.id} onClick={() => handleSchemaChange(s.id)} style={tabBtn(schema === s.id, "#0f766e")}>
              {s.name}
            </button>
          ))}
          <span style={{ fontSize: 9, color: "#94a3b8", marginLeft: 6 }}>
            {schema === "MASK_OPT"
              ? "Mask m is optimized (learnable), k_const defaults to 1.0 (maximal warming)."
              : "Mask m is fixed (ground-truth), rectify set to CAM, k_const defaults to 0.5."}
          </span>
        </div>

        {/* RECTIFY TYPE */}
        <div style={{ display: "flex", gap: 6, marginBottom: 8, alignItems: "center", opacity: schema === "CAM" ? 0.6 : 1 }}>
          <span style={{ fontSize: 10, color: "#64748b", width: 60 }}>Rectify</span>
          {["DEF", "CAM", "DEP", "FUT"].map(rt => (
            <button key={rt} disabled={schema === "CAM" && rt !== "CAM"} onClick={() => setRectify(rt)} style={tabBtn(rectify === rt, rt === "FUT" && !futFilterFixed ? "#dc2626" : "#2563eb")}>
              {rt}{rt === "FUT" && !futFilterFixed ? " ⚠" : ""}
            </button>
          ))}
          <span style={{ fontSize: 9, color: "#64748b", marginLeft: 6 }}>{rectify === "FUT" ? "FUT: convergence-filter bug (Bug A)" : rectify === "DEP" ? "DEP: B is cross-channel (Bug C)" : "filter present → S1 absorbing"}</span>
        </div>

        {/* DEVIATION TOGGLES */}
        <div style={{ marginBottom: 10, padding: 8, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6 }}>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>Deviation toggles (OFF = source/buggy · ON = fix; multiple may apply)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {bugRows.map(b => (
              <div key={b.id} style={{ display: "flex", alignItems: "center", gap: 8, opacity: b.applies ? 1 : 0.4 }}>
                <button onClick={b.toggle} disabled={!b.applies}
                  style={{ padding: "3px 9px", fontSize: 10, borderRadius: 3, border: "1px solid #334155", cursor: b.applies ? "pointer" : "not-allowed",
                    background: !b.applies ? "#1e293b" : b.fixed ? "#10b981" : "#dc2626", color: b.fixed || !b.applies ? "#e2e8f0" : "#fff", fontWeight: 700, minWidth: 64 }}>
                  {b.fixed ? "FIXED" : "SOURCE"}
                </button>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: "#e2e8f0" }}>
                    <b>{b.label}</b>
                    <span style={{ fontSize: 8, marginLeft: 6, padding: "1px 5px", borderRadius: 3, background: b.severity === "CORRECTNESS" ? "#dc262633" : "#f59e0b33", color: b.severity === "CORRECTNESS" ? "#fca5a5" : "#fde68a" }}>{b.severity}</span>
                    <span style={{ fontSize: 9, color: "#64748b", marginLeft: 6 }}>{b.where}</span>
                  </div>
                  <div style={{ fontSize: 9, color: "#94a3b8" }}>{b.effect}</div>
                </div>
              </div>
            ))}
          </div>
          {/* k_const hyperparameter controls */}
          <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8, paddingTop: 8, borderTop: "1px solid #1e293b", flexWrap: "wrap" }}>
            <span style={{ fontSize: 10, color: "#64748b" }}>k_const (hyperparameter)</span>
            <input type="number" min={0} max={1} step={0.05} value={kBaseEff.toFixed(2)} disabled={kMode === "auto"}
              onChange={e => { setKMode("manual"); setKBase(Math.max(0, Math.min(1, +e.target.value || 0))); }}
              style={{ width: 56, fontSize: 10, padding: "3px 4px", borderRadius: 3, background: kMode === "auto" ? "#1e293b" : "#020617", color: "#cbd5e1", border: "1px solid #334155" }} />
            <button onClick={() => setKMode("manual")} style={mini(kMode === "manual")}>Manual</button>
            <button onClick={() => setKMode("auto")} style={mini(kMode === "auto")}>Auto</button>
            <span style={{ fontSize: 9, color: "#a78bfa" }}>aud. heuristic k (non-source): <b>{kSuggest.toFixed(2)}</b> {kSuggest < 0.35 ? "(near-coplanar)" : kSuggest > 0.7 ? "(varied depth)" : "(mixed)"}</span>
            <span style={{ fontSize: 9, color: "#64748b" }}>effective now: <b style={{ color: "#cbd5e1" }}>{kEff.toFixed(2)}</b>{kSchedFixed ? " (ramping)" : " (pinned)"} · source fix = schedule, not value</span>
          </div>
        </div>

        {/* CONTROLS */}
        <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap", alignItems: "center" }}>
          <button onClick={undo} disabled={!hist.length} style={ctl(false)}>◀ Undo</button>
          <button onClick={forward} style={ctl(true, "#0891b2")} title="Forward: apply routing transport only (no gradient) — relocates the point to its current projection. Selection follows the particle by identity.">⤳ Forward (route)</button>
          <button onClick={stepMarkov} style={ctl(true, "#2563eb")}>▶ Markov step</button>
          <button onClick={reset} style={ctl(false)}>Reset</button>
          <button onClick={() => {
            setSnapshotText(getSnapshotJSON());
            setSnapshotPanelOpen(v => !v);
          }} style={ctl(false)}>📷 Snapshot</button>
          <span style={sep} />
          <select value={preset} onChange={e => setPreset(e.target.value)} style={selStyle}>
            <option value="default">Default (response.txt)</option>
            <option value="far">Far target (routing traversal)</option>
            <option value="occlusion">Occlusion (k_const / Bug B)</option>
            <option value="random">Random</option>
            <option value="convergence">Convergence test</option>
          </select>
          <span style={sep} />
          <span style={{ fontSize: 10, color: "#64748b" }}>LR</span>
          <button onClick={() => setLrMode("manual")} style={mini(lrMode === "manual")}>Man</button>
          <button onClick={() => setLrMode("auto")} style={mini(lrMode === "auto")}>Auto</button>
          {lrMode === "manual" && <input type="range" min={-3} max={0} step={0.1} value={Math.log10(lrManual)} onChange={e => setLrManual(Math.pow(10, +e.target.value))} style={{ width: 70 }} />}
          <span style={{ fontSize: 10, color: "#cbd5e1", minWidth: 44 }}>{lr.toFixed(4)}</span>
        </div>

        {/* SNAPSHOT PANEL */}
        {snapshotPanelOpen && (
          <div style={{ marginBottom: 10, padding: 10, background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 11 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#f8fafc", marginBottom: 6, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>📷 State Snapshot (Save/Load)</span>
              <button onClick={() => setSnapshotPanelOpen(false)} style={{ background: "none", border: "none", color: "#64748b", cursor: "pointer", fontSize: 12 }}>✕</button>
            </div>
            <div style={{ fontSize: 9.5, color: "#94a3b8", marginBottom: 6 }}>
              Copy the text below to save/share your simulator state, or paste a snapshot to load it.
            </div>
            <textarea
              value={snapshotText}
              onChange={e => setSnapshotText(e.target.value)}
              style={{ width: "100%", height: 80, fontSize: 9, fontFamily: "monospace", background: "#020617", color: "#34d399", border: "1px solid #475569", borderRadius: 4, padding: 6, resize: "vertical" }}
            />
            <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
              <button onClick={() => {
                navigator.clipboard.writeText(snapshotText);
                alert("Snapshot copied to clipboard!");
              }} style={ctl(true, "#10b981")}>Copy to Clipboard</button>
              <button onClick={async () => {
                try {
                  const res = await fetch("/api/save-snapshot", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: snapshotText
                  });
                  const data = await res.json();
                  if (data.success) {
                    await navigator.clipboard.writeText(data.filename);
                    alert(`Snapshot saved to file: ${data.filename}\n(Filename copied to clipboard!)`);
                  } else {
                    alert("Error saving snapshot: " + data.error);
                  }
                } catch (e) {
                  alert("Network error saving snapshot: " + e.message);
                }
              }} style={ctl(true, "#d97706")}>Save to File</button>
              <button onClick={() => loadSnapshot(snapshotText)} style={ctl(true, "#2563eb")}>Load State</button>
              <button onClick={() => setSnapshotText(getSnapshotJSON())} style={ctl(false)}>Refresh State</button>
            </div>
          </div>
        )}

        {/* ZOOM + CHANNEL */}
        <div style={{ display: "flex", gap: 12, marginBottom: 10, flexWrap: "wrap", alignItems: "center", padding: 8, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 10, color: "#64748b" }}>Zoom (C_zoom)</span>
            {Array.from({ length: L_MAX + 1 }, (_, l) => (
              <button key={l} onClick={() => { setZoom(l); setChannel(cc => Math.min(cc, levelDims(l).chans - 1)); }} style={mini(zoom === l)}>L{l}</button>
            ))}
            <span style={{ fontSize: 10, color: "#94a3b8" }}>{side}×{side} · {chans} chan · 1 step={per}px</span>
            <span style={sep} />
            <button onClick={toggleShowOriginal} style={mini(showOriginal)}
              title="VIEW (display only): render the step-0 ORIGINAL positions — physical points at their raw location before ANY routing — at the current zoom. Toggle on/off to compare positions BEFORE vs AFTER movement. Orthogonal to L0..L3, so you can view the raw state at any size. Does NOT change routing/dynamics.">
              👁 show original{showOriginal ? " · ON" : ""}
            </button>
            <button onClick={() => setFreezeFinest(v => !v)} style={mini(freezeFinest)}
              title="DYNAMICS (changes the step): skip the finest (L0) routing pass on Forward/Markov — runs only coarse passes (L_MAX..1), source 'for i in range(n − zoom)' at zoom=1 — to inspect the pre-finest-routing state. DISTINCT from 'show original' (a pure view of step-0). Kept primarily for the upcoming L53–63 routing-depth debugging arc; agents should treat this as a routing-DEPTH control, not a view.">
              ❄ freeze L0 route{freezeFinest ? " · ON" : ""}
            </button>
          </div>
          {chans > 1 && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, flex: 1, minWidth: 200 }}>
              <span style={{ fontSize: 10, color: "#64748b" }}>Channel</span>
              <input type="range" min={0} max={chans - 1} step={1} value={safeChannel} onChange={e => setChannel(+e.target.value)} style={{ flex: 1 }} />
              <span style={{ fontSize: 10, color: "#cbd5e1", minWidth: 56 }}>{safeChannel} / {chans - 1}</span>
            </div>
          )}
        </div>

        {/* GRID + INSPECTOR */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 330px", gap: 12, marginBottom: 10 }}>
          <div>
            {/* fixed-height placeholder so toggling 'show original' does NOT shift the grid (keeps positions comparable) */}
            <div style={{ fontSize: 10, color: showOriginal ? "#fbbf24" : "transparent", marginBottom: 4, height: 46, overflow: "hidden", lineHeight: 1.35 }} aria-hidden={!showOriginal}>
              {showOriginal ? <>👁 VIEWING ORIGINAL (step 0, pre-routing) at L{zoom} — positions before any movement; toggle off to compare with the live state. (Live state, routing &amp; inspector unaffected.)</> : " "}
            </div>
            {!selVisible && <div style={{ fontSize: 10, color: "#fbbf24", marginBottom: 4 }}>◆ Selected ({selOrig.r},{selOrig.c}) is in channel {selView.channel} at L{zoom} — slide channel.</div>}
            {!candOn && selVisible && <div style={{ fontSize: 10, color: "#94a3b8", marginBottom: 4 }}>● ({selOrig.r},{selOrig.c}) selected — click again to reveal candidate ("ô khả dĩ") cells.</div>}
            {candVisible && <div style={{ fontSize: 10, color: "#fbbf24", marginBottom: 4 }}>◈ Candidate view: 3×3 around ({selOrig.r},{selOrig.c}) = its 9 "ô khả dĩ" (slice 0=↖…4=●…8=↘), all carrying its xyzm. Click a slice; click center to exit.</div>}
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${side}, ${cellPx}px)`, gap: 2, padding: 8, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, justifyContent: "center" }}>
              {viewCells.map(({ ph, pw, orig_r, orig_c }) => {
                const s = sliceAt(ph, pw);
                if (s >= 0) {
                  const ci = candInfo(selP, selOrig.r, selOrig.c, s, zoom, tgt);
                  const isCenter = s === 4, isSelCand = s === candSlice;
                  const col = isCenter ? "#fbbf24" : ci.confirmed ? "#10b981" : "#64748b";
                  return (
                    <div key={ph + "-" + pw} onClick={() => clickCell(ph, pw, orig_r, orig_c)}
                      style={{ width: cellPx, height: cellPx, position: "relative", cursor: "pointer", background: `${col}1f`,
                        border: isCenter ? "2.5px dashed #fbbf24" : isSelCand ? "2.5px solid #f43f5e" : `1px solid ${col}66`,
                        borderRadius: 3, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                        boxShadow: isSelCand ? "0 0 14px #f43f5e66" : "none", overflow: "hidden" }}>
                      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: `${Math.max(0, Math.min(1, selP.m)) * 100}%`, background: `${col}22` }} />
                      <div style={{ position: "relative", fontSize: Math.min(15, cellPx * 0.42), color: col, opacity: 0.9 }}>{ci.shift}</div>
                      {cellPx > 26 && <div style={{ position: "relative", fontSize: 7, color: "#94a3b8" }}>s{s}</div>}
                      {ci.confirmed && <div style={{ position: "absolute", top: 1, right: 2, fontSize: 8, color: "#fbbf24" }}>⊕</div>}
                    </div>
                  );
                }
                const p = viewPts[orig_r][orig_c];
                const meta = viewSim.meta[orig_r][orig_c];
                const info = META[meta];
                const g = viewSim.grad[orig_r][orig_c];
                const isCenterPt = orig_r === selOrig.r && orig_c === selOrig.c;
                const isSel = !candOn && selVisible && isCenterPt;
                const t = tgt[orig_r][orig_c];
                const hard = !!p.stale;
                const rAng = Math.atan2(g.arrowDy, g.arrowDx) * 180 / Math.PI;
                const showArrow = (Math.abs(g.arrowDx) > 0.05 || Math.abs(g.arrowDy) > 0.05) && p.m > OFF_THRESH && !hard;
                return (
                  <div key={ph + "-" + pw} onClick={() => clickCell(ph, pw, orig_r, orig_c)}
                    style={{ width: cellPx, height: cellPx, position: "relative", cursor: "pointer",
                      background: hard ? "#1e293b" : `${info.color}1f`,
                      border: isSel ? "2.5px solid #f43f5e" : t ? "2px dashed #fbbf24" : hard ? "1px dashed #475569" : `1px solid ${info.color}40`,
                      borderRadius: 3, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", boxShadow: isSel ? "0 0 14px #f43f5e66" : "none", overflow: "hidden" }}>
                    <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: `${Math.max(0, Math.min(1, p.m)) * 100}%`, background: hard ? "#47556918" : `${info.color}26` }} />
                    {cellPx > 28 && <div style={{ position: "relative", fontSize: 8, fontWeight: 700, color: hard ? "#64748b" : info.color }}>{hard ? "fb" : meta}</div>}
                    {cellPx > 22 && <div style={{ position: "relative", fontSize: 7, color: hard ? "#64748b" : "#cbd5e1" }}>{p.m.toFixed(2)}</div>}
                    {t === 1 && <div style={{ position: "absolute", top: 1, right: 2, fontSize: 8, color: "#fbbf24" }}>⊕</div>}
                    {showArrow && <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: Math.min(16, cellPx * 0.4), color: info.color, opacity: isCenterPt ? 0.9 : 0.5, transform: `rotate(${rAng}deg)` }}>→</div>}
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: 9, color: "#64748b", marginTop: 6, lineHeight: 1.5 }}>
              <b>Markov step:</b> one forward pass (smap.py L224-238) — routes all zoom levels coarse→fine (≤1 neighbor = 2^L finest px/level), then applies the finest-3×3 rectification gradient (is_last, L244-251). Solid <span style={{ color: "#7dd3fc" }}>→</span> = <b>projection/routing direction</b> (NOT target; target vector is the dashed compass one). <span style={{ color: "#64748b" }}>fb</span> = vacated cell (value routed out → m=0 per smap.py L165-170; no duplication). <b style={{ color: "#22d3ee" }}>⤳ Forward (route)</b> = routing transport ONLY (no gradient); it repositions a point but does NOT count as a training step (Step counter unchanged).
            </div>
          </div>

          {/* INSPECTOR */}
          <div style={{ padding: 10, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, fontSize: 11 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#f8fafc", marginBottom: 6 }}>
              {candOn && candSlice !== 4
                ? <>Candidate s{candSlice} {SLICE_GLYPH[candSlice]} of ({selOrig.r},{selOrig.c}) · {rectify}</>
                : <>Point ({selOrig.r},{selOrig.c}) · {rectify}</>}
            </div>
            {lastMove && (lastMove.occluded
              ? <div style={{ fontSize: 9, color: "#fb7185", marginBottom: 6, lineHeight: 1.4 }}>⚠ Particle was occluded/overwritten last step (a nearer point won the z-buffer) — no longer on the grid.</div>
              : <div style={{ fontSize: 9, color: "#7dd3fc", marginBottom: 6, lineHeight: 1.4 }}>↪ Particle routed ({lastMove.from.r},{lastMove.from.c}) → ({lastMove.to.r},{lastMove.to.c}); selection follows the point (gradient identity preserved via x_z_value, smap.py L83). Origin left as fb.</div>)}
            {/* Two meta-state layers + next state */}
            <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
              <div style={{ flex: 1, padding: 6, background: selInfo.color + "12", borderLeft: `2px solid ${selInfo.color}`, borderRadius: 3 }}>
                <div style={{ fontSize: 9, color: "#64748b" }}>RECTIFY layer</div>
                <div style={{ fontSize: 12, fontWeight: 700, color: selInfo.color }}>{selMeta} <span style={{ fontSize: 9, color: "#64748b" }}>→ {nextMetaSel}</span></div>
                <div style={{ fontSize: 8.5, color: selMeta === "S1" && Math.abs(gN.total) > 1e-6 ? "#fca5a5" : "#94a3b8" }}>{selInfo.name}{selMeta === "S1" && Math.abs(gN.total) > 1e-6 ? " ⚠ NOT absorbing here (g_m≠0)" : ""}</div>
              </div>
              <div style={{ flex: 1, padding: 6, background: selRouteInfo.color + "12", borderLeft: `2px solid ${selRouteInfo.color}`, borderRadius: 3 }}>
                <div style={{ fontSize: 9, color: "#64748b" }}>ROUTING layer</div>
                <div style={{ fontSize: 12, fontWeight: 700, color: selRouteInfo.color }}>{selRoute} <span style={{ fontSize: 9, color: "#64748b" }}>→ {nextRouteSel}</span></div>
                <div style={{ fontSize: 8.5, color: "#94a3b8" }}>{selRouteInfo.name}</div>
              </div>
            </div>
            <div style={{ fontSize: 9, color: "#cbd5e1", marginBottom: 8 }}>{selMeta === "S1" && Math.abs(gN.total) > 1e-6 ? `⚠ S1 is the absorbing class BY DESIGN, but in THIS config its mask-grad = ${gN.total.toFixed(3)} ≠ 0 (FUT convergence-filter SOURCE / Bug A) — NOT absorbing here; the static "ABSORBING" label is the corrected-algorithm target, not the current behaviour.` : selInfo.note}</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 10, marginBottom: 8 }}>
              <div>x = <b style={{ color: "#7dd3fc" }}>{selP.x.toFixed(2)}</b></div>
              <div>y = <b style={{ color: "#7dd3fc" }}>{selP.y.toFixed(2)}</b></div>
              <div>z = <b style={{ color: "#a78bfa" }}>{selP.z.toFixed(2)}</b></div>
              <div>m = <b style={{ color: "#22d3ee" }}>{selP.m.toFixed(3)}</b>{schema === "CAM" && <span style={{ fontSize: 8, color: "#fca5a5", marginLeft: 4 }}>(fixed)</span>}</div>
              <div>t = <b style={{ color: "#fbbf24" }}>{tgt[selOrig.r][selOrig.c]}</b></div>
              <div>B = <b style={{ color: sim.B[selOrig.r][selOrig.c] ? "#fb7185" : "#86efac" }}>{sim.B[selOrig.r][selOrig.c] ? 1 : 0}</b></div>
              <div>y_flow# = <b style={{ color: "#c4b5fd" }}>{gN.yc}</b></div>
              <div>coord = <b style={{ color: "#34d399" }}>{gN.coordFlow}</b></div>
            </div>
            <div style={{ fontSize: 9, color: "#64748b", marginBottom: 6 }}>proj=(col x/z, row y/z)=({(selP.x / selP.z).toFixed(1)},{(selP.y / selP.z).toFixed(1)}) ⇒ lands in cell (row {(selP.y / selP.z).toFixed(0)}, col {(selP.x / selP.z).toFixed(0)}) · camera=identity (utils.py to_3d3x3)</div>
            {selP.m <= OFF_THRESH && selP.z > 0 && (() => {
              const nest = inactiveNest(selOrig.r, selOrig.c, zoom, kEff, step + 1);
              return (
                <div style={{ fontSize: 8.5, marginBottom: 6, lineHeight: 1.45, padding: 5, borderRadius: 3,
                  color: nest.desync ? "#fda4af" : "#94a3b8", border: `1px solid ${nest.desync ? "#fb718544" : "#1e293b"}`, background: nest.desync ? "#fb71850a" : "transparent" }}>
                  <b>L53–63 nest</b> (next step · L{zoom}): mask-gate <b>weights_m</b> P={nest.pThresh.toFixed(2)} ⇒ <b>{nest.maskRoutes ? "mask ROUTES" : "mask stays"}</b>; <b>weights_x</b> via global np_rand={nest.passRand.toFixed(2)} (L54 gate {nest.globalSuppress ? "FIRED ⇒ xyz→center" : "off"}) ⇒ <b>{nest.xyzRoutes ? "xyz ROUTES" : "xyz stays"}</b>. {nest.desync
                    ? "⚠ MASK/XYZ DESYNC (G2): mask & geometry sourced from different neighbours — suspected 'dirty-output' mechanism (Spec 5; full split-transport = real-data arc)."
                    : "mask & xyz agree."}{zoom > 0 ? " · G3: coarse-level participation inverted (1−k)." : ""}
                </div>
              );
            })()}
            <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 3 }}>m-gradient — mask channel = F1+F2 ({rectify}{rectify === "FUT" ? (futFilterFixed ? ", fixed" : ", source") : ""})</div>
            <Bar label="align(F1)" v={gN.t1} />
            <Bar label="activ(F2)" v={gN.t2} />
            <Bar label="m TOTAL" v={gN.total} max={0.6} />
            <div style={{ fontSize: 8.5, color: "#64748b", marginTop: 2, lineHeight: 1.45 }}>
              Shown = <b>sign × scale</b>. Magnitudes respect the source hierarchy α₃(cleanup) &gt; α₂(activate) &gt; α₁(align) (rectify.py 1e1 ≫ 3e0 ≫ 3e-1); the load-bearing property is the <b>sign of the sum</b> &amp; zero/non-zero (S1 absorbing ⇔ TOTAL=0), exact. <b>Force 3 is NOT here</b> — in source it writes x/y/z, not the mask. {schema === "CAM" && <b>Note: In CAM schema, these mask forces do not update m during steps.</b>}
            </div>
            <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, margin: "6px 0 3px" }}>Force 3 (coordinate) — APPLIED to x,y (key_query · sign); z held fixed</div>
            <Bar label="gx" v={gN.gx} max={2} />
            <Bar label="gy" v={gN.gy} max={2} />
            <div style={{ fontSize: 8.5, color: "#64748b", marginTop: 2, lineHeight: 1.45 }}>
              <b>gx/gy ARE the applied Force-3</b> (coordinate) gradient — an active cell adjacent to a target moves its geometry toward it (shown <b>sign × scale</b>; magnitude k_t=k_d/pre_z-conditioned, not shown; CAM/DEP/FUT not Default; k_t&gt;0 ⇒ sign/zero unchanged). In source `key_query_grdf` (rectify.py L305) backprops to x/y/z, never the mask leaf — moves geometry, not `[m&gt;τ]`. <b>z is held fixed (gz≡0)</b> — confirmed faithful to source by `_audit_sandbox/t_force3.py` (z-channel gradient identically zero). The <b>coord</b> field above is the `coord_flow` <i>gate</i> indicator (inactive-target spread candidacy), not a separately-applied force.
            </div>
            {/* compass */}
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
              <svg width="60" height="60" viewBox="-32 -32 64 64" style={{ flexShrink: 0, background: "#020617", borderRadius: 4 }}>
                <line x1="-28" y1="0" x2="28" y2="0" stroke="#1e293b" strokeWidth="1" />
                <line x1="0" y1="-28" x2="0" y2="28" stroke="#1e293b" strokeWidth="1" />
                {(() => {
                  const nrm = (dx, dy) => { const m = Math.hypot(dx, dy) || 1; return [dx / m * 22, dy / m * 22]; };
                  const [rx, ry] = nrm(gN.arrowDx, gN.arrowDy);
                  const [gxn, gyn] = nrm(gN.gx, gN.gy);
                  const routeOn = Math.abs(gN.arrowDx) > 0.05 || Math.abs(gN.arrowDy) > 0.05;
                  const gradOn = Math.abs(gN.gx) > 1e-3 || Math.abs(gN.gy) > 1e-3;
                  return (<>
                    {routeOn && <line x1="0" y1="0" x2={rx} y2={ry} stroke="#7dd3fc" strokeWidth="2.5" markerEnd="url(#arR)" />}
                    {gradOn && <line x1="0" y1="0" x2={gxn} y2={gyn} stroke="#a78bfa" strokeWidth="2" strokeDasharray="3 2" markerEnd="url(#arG)" />}
                  </>);
                })()}
                <defs>
                  <marker id="arR" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto"><path d="M0,0 L5,3 L0,6 Z" fill="#7dd3fc" /></marker>
                  <marker id="arG" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto"><path d="M0,0 L5,3 L0,6 Z" fill="#a78bfa" /></marker>
                </defs>
              </svg>
              <div style={{ fontSize: 9, color: "#94a3b8", lineHeight: 1.5 }}>
                <div><span style={{ color: "#7dd3fc" }}>──▶ routing</span> = proj−cell ({gN.arrowDx.toFixed(1)},{gN.arrowDy.toFixed(1)})</div>
                <div><span style={{ color: "#a78bfa" }}>– –▶ gradient</span> = (gx,gy) ({gN.gx.toFixed(1)},{gN.gy.toFixed(1)}) → target</div>
                <div style={{ color: "#64748b" }}>Two different vectors: grid arrow = routing, not target.</div>
              </div>
            </div>
            {candOn && candSlice !== 4 && selCand && (
              <div style={{ marginTop: 8, padding: 6, background: "#1e293b", borderRadius: 4, fontSize: 10 }}>
                <div style={{ fontWeight: 700, color: "#f8fafc", marginBottom: 3 }}>Candidate slice {candSlice} {selCand.shift}</div>
                <div style={{ color: "#94a3b8" }}>Carries ({selOrig.r},{selOrig.c})'s xyzm at shifted ({selCand.pr},{selCand.pc}).</div>
                <div>target here: <b style={{ color: selCand.confirmed ? "#fbbf24" : "#64748b" }}>{selCand.confirmed ? "yes ⊕ → pulls" : "no → no pull"}</b></div>
                <div>key_query: <b style={{ color: "#c4b5fd" }}>{selCand.kq.toFixed(2)}</b> · gx=<b>{selCand.gx.toFixed(2)}</b>, gy=<b>{selCand.gy.toFixed(2)}</b>{zoom > 0 && <span style={{ color: "#64748b" }}> (0 at L{zoom})</span>}</div>
              </div>
            )}
            <div style={{ borderTop: "1px solid #1e293b", marginTop: 8, paddingTop: 8 }}>
              <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>Edit point</div>
              <Edit label="m" min={0} max={1} step={0.01} val={selP.m} on={v => editSel("m", v)} />
              <NumEdit label="x" min={0} max={FINE * 4} step={0.1} val={selP.x} on={v => editSel("x", v)} />
              <NumEdit label="y" min={0} max={FINE * 4} step={0.1} val={selP.y} on={v => editSel("y", v)} />
              <NumEdit label="z" min={0.01} max={25} step={0.1} val={selP.z} on={v => editSel("z", v)} />
              <div style={{ fontSize: 8, color: "#64748b", marginTop: 1 }}>x,y,z: type exact values (e.g. project to cell c ⇒ x = c·z). m: slider (bounded [0,1]).</div>
              <div style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 10, marginTop: 4 }}>
                <span style={{ width: 14 }}>t</span>
                <button onClick={toggleT} style={{ padding: "3px 8px", fontSize: 10, borderRadius: 2, border: "1px solid #334155", background: (showOriginal ? rawSST.tgt : tgt)[selOrig.r][selOrig.c] ? "#fbbf24" : "#1e293b", color: (showOriginal ? rawSST.tgt : tgt)[selOrig.r][selOrig.c] ? "#0f172a" : "#cbd5e1", cursor: "pointer", fontWeight: 700 }}>{(showOriginal ? rawSST.tgt : tgt)[selOrig.r][selOrig.c] ? "ON" : "OFF"}</button>
              </div>
            </div>
          </div>
        </div>

        {/* COMMENTARY */}
        <div style={{ padding: 10, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, marginBottom: 10 }}>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>Commentary — point ({selOrig.r},{selOrig.c})</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {comm.map((ln, i) => {
              const col = { rule: ["#334155", "#cbd5e1"], view: ["#1e3a5f40", "#bae6fd"], proj: ["#3730a340", "#c7d2fe"], ok: ["#10b98120", "#86efac"], err: ["#dc262640", "#fca5a5"], bug: ["#dc262640", "#fecaca"], fix: ["#10b98130", "#a7f3d0"], grad: ["#0e749040", "#a5f3fc"], pred: ["#fbbf2420", "#fde68a"] }[ln.k];
              return <div key={i} style={{ padding: "5px 8px", borderRadius: 3, fontSize: 10, lineHeight: 1.5, background: col[0], color: col[1], borderLeft: `2px solid ${col[1]}` }}><b style={{ fontSize: 8, opacity: 0.7, marginRight: 6, textTransform: "uppercase" }}>{ln.k}</b>{ln.t}</div>;
            })}
          </div>
        </div>

        {/* LEGENDS */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <div style={{ padding: 8, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, fontSize: 9 }}>
            <div style={{ fontSize: 9, color: "#64748b", marginBottom: 4 }}>RECTIFY layer (m,t,B) — Φ / gradient sign</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 4 }}>
              {Object.entries(META).map(([k, v]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: v.color }} /><span style={{ color: "#94a3b8" }}><b style={{ color: v.color }}>{k}</b> {v.name}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ padding: 8, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, fontSize: 9 }}>
            <div style={{ fontSize: 9, color: "#64748b", marginBottom: 4 }}>ROUTING layer (m,z) — participation / occlusion</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 4 }}>
              {Object.entries(META_ROUTE).map(([k, v]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: v.color }} /><span style={{ color: "#94a3b8" }}><b style={{ color: v.color }}>{k}</b> {v.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


function Bar({ label, v, max = 0.5 }) {
  const w = Math.min(Math.abs(v) / max * 50, 50);
  const pos = v >= 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
      <span style={{ width: 56, color: "#94a3b8", textAlign: "right" }}>{label}</span>
      <div style={{ flex: 1, height: 11, background: "#020617", borderRadius: 2, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#475569" }} />
        <div style={{ position: "absolute", left: pos ? "50%" : `${50 - w}%`, width: `${w}%`, height: "100%", background: pos ? "#10b981" : "#ef4444", opacity: 0.85 }} />
      </div>
      <span style={{ width: 52, color: "#e2e8f0" }}>{v >= 0 ? "+" : ""}{v.toFixed(3)}</span>
    </div>
  );
}
function Edit({ label, min, max, step, val, on }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, marginBottom: 3 }}>
      <span style={{ width: 14 }}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={val} onChange={e => on(+e.target.value)} style={{ flex: 1 }} />
      <span style={{ width: 40, color: "#cbd5e1" }}>{(+val).toFixed(2)}</span>
    </div>
  );
}
// Editable numeric textbox (for x/y/z — precise typing). Local text state lets you type freely
// (incl. partial like "1."); commits valid numbers clamped to [min,max]; resyncs when val changes
// externally (e.g. selecting a different cell).
function NumEdit({ label, min, max, step, val, on }) {
  const [txt, setTxt] = useState(() => String(+val));
  const last = useRef(+val);
  useEffect(() => { if (+val !== last.current) { last.current = +val; setTxt(String(+val)); } }, [val]);
  const commit = (s) => {
    const v = parseFloat(s);
    if (!isFinite(v)) return;
    const c = Math.max(min, Math.min(max, v));
    last.current = c;
    on(c);
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, marginBottom: 3 }}>
      <span style={{ width: 14 }}>{label}</span>
      <input type="number" min={min} max={max} step={step} value={txt}
        onChange={e => { setTxt(e.target.value); commit(e.target.value); }}
        onBlur={e => { const v = parseFloat(e.target.value); const c = isFinite(v) ? Math.max(min, Math.min(max, v)) : last.current; last.current = c; setTxt(String(c)); on(c); }}
        style={{ flex: 1, fontSize: 10, padding: "3px 6px", borderRadius: 3, background: "#020617", color: "#cbd5e1", border: "1px solid #334155" }} />
    </div>
  );
}
const tabBtn = (active, color) => ({ padding: "6px 12px", border: "none", borderRadius: 4, cursor: "pointer", fontSize: 11, fontWeight: 700, background: active ? color : "#1e293b", color: active ? "#fff" : "#94a3b8" });
const ctl = (primary, color) => ({ padding: "5px 12px", fontSize: 11, borderRadius: 3, border: primary ? "none" : "1px solid #334155", background: primary ? (color || "#2563eb") : "#1e293b", color: primary ? "#fff" : "#cbd5e1", cursor: "pointer", fontWeight: primary ? 700 : 400 });
const mini = (active) => ({ padding: "3px 8px", fontSize: 10, borderRadius: 3, border: "1px solid #334155", background: active ? "#2563eb" : "#1e293b", color: active ? "#fff" : "#94a3b8", cursor: "pointer" });
const sep = { width: 1, height: 18, background: "#334155", margin: "0 2px", display: "inline-block" };
const selStyle = { fontSize: 10, padding: "4px 6px", borderRadius: 3, background: "#1e293b", color: "#cbd5e1", border: "1px solid #334155" };
