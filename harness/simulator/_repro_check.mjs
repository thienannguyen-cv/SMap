// THROWAWAY reproducibility harness — ports the PURE functions of smap_simulator.jsx
// verbatim (lines 26-431) and iterates them headless, to independently confirm the
// app's convergence claims WITHOUT trusting the prior "self-verified (node+live)" note.
// Run: node _repro_check.mjs

// ── constants (jsx L26-29) ──
const OFF_THRESH = 0.5;
const FINE = 8;
const L_MAX = Math.log2(FINE);
const RAMP = 16;

// SEED is settable here so runs are reproducible (app re-seeds each open).
let SEED = 123.456;
function setSeed(s) { SEED = s; }

function pseudo(r, c) {
  const s = Math.sin((r + SEED) * 12.9898 + (c + SEED * 0.37) * 78.233) * 43758.5453;
  return s - Math.floor(s);
}
function rand01(a, b, salt) {
  const s = Math.sin((a + SEED) * 12.9898 + b * 78.233 + salt * 37.719) * 43758.5453;
  return s - Math.floor(s);
}

function classify(m, t, B) {
  const mm = m > OFF_THRESH ? 1 : 0;
  if (mm && t && B) return "S1";
  if (mm && !t && B) return "S2";
  if (mm && !t && !B) return "S3";
  if (!mm && !t && !B) return "S4";
  if (!mm && !t && B) return "S5";
  if (!mm && t && B) return "S6";
  return "S7";
}
function classifyRoute(m, z) {
  const mm = m > OFF_THRESH;
  if (mm && z > 0) return "SA";
  if (mm && z <= 0) return "SB";
  if (!mm && z > 0) return "SC";
  return "SD";
}
function projPixel(p) {
  if (Math.abs(p.z) < 1e-6) return { pr: 0, pc: 0 };
  return { pr: p.y / p.z, pc: p.x / p.z };
}

// ── presets (jsx L87-167) ──
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
  pts[2][5] = { x: 5 * 2.0, y: 2 * 2.0, z: 2.0, m: 0.82 };
  pts[3][4] = { x: 5 * 2.0, y: 3 * 2.0, z: 2.0, m: 0.85 };
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
  const pts = [], tgt = [];
  for (let r = 0; r < FINE; r++) {
    pts.push([]); tgt.push([]);
    for (let c = 0; c < FINE; c++) {
      pts[r].push({ x: c * 2.0, y: r * 2.0, z: 2.0, m: 0.05 + 0.08 * pseudo(r, c) });
      tgt[r].push(0);
    }
  }
  tgt[3][3] = 1;
  pts[3][3] = { x: 3 * 0.8, y: 3 * 0.8, z: 0.8, m: 0.1 };
  for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
    const r = 3 + dr, c = 3 + dc;
    pts[r][c] = { x: c * 0.5, y: r * 0.5, z: 0.5, m: 0.9 };
  }
  return { pts, tgt };
}

// ── routing (jsx L204-255) ──
function routeStep(pts, tgt, l, kEff, salt) {
  const per = 1 << l, side = FINE >> l;
  const next = pts.map(row => row.map(p => ({ ...p })));
  const moves = [];
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      const p = pts[r][c];
      if (p.stale) continue;
      const active = p.m > OFF_THRESH;
      // G3 inversion mirror (jsx inactiveNest / smap.py L57 finest vs L62-63 intermediate):
      // inactive (S_C) mask-route prob = kEff at finest (l=0), (1−kEff) at coarser levels.
      const pThresh = (l === 0) ? kEff : (1 - kEff);
      const mayRoute = (active && p.z > 0) || (!active && p.z > 0 && rand01(r * 7 + c, l * 13 + 1, salt) < pThresh);
      if (!mayRoute) continue;
      const { pr, pc } = projPixel(p);
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
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      const p = pts[r][c];
      if (!p.stale && p.m <= OFF_THRESH && p.z > 0 && tgt[r][c] === 1 && rand01(r * 5 + c, l * 11 + 3, salt) < kEff)
        taken[r + "_" + c] = true;
    }
  moves.sort((a, b) => a.z - b.z);
  for (const mv of moves) {
    const key = mv.nr + "_" + mv.nc;
    if (taken[key]) continue;
    if (pts[mv.nr][mv.nc].m > OFF_THRESH && !(mv.nr === mv.r && mv.nc === mv.c)) continue;
    taken[key] = true;
    next[mv.nr][mv.nc] = { ...pts[mv.r][mv.c], stale: false };
    if (!(mv.nr === mv.r && mv.nc === mv.c)) vacated[mv.r + "_" + mv.c] = true;
  }
  for (const k in vacated) {
    if (taken[k]) continue;
    const [r, c] = k.split("_").map(Number);
    next[r][c] = { ...pts[r][c], m: 0, stale: true };
  }
  return next;
}

// ── B (jsx L261-278) ──
function computeB(pts, tgt, rectify, bFaithful) {
  const B = pts.map(row => row.map(() => false));
  for (let r = 0; r < FINE; r++)
    for (let c = 0; c < FINE; c++) {
      if (bFaithful && rectify === "DEP") {
        B[r][c] = pts[r][c].m > OFF_THRESH && tgt[r][c] === 1;
      } else {
        let cnt = 0;
        for (let dr = -1; dr <= 1; dr++)
          for (let dc = -1; dc <= 1; dc++) {
            const nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < FINE && nc >= 0 && nc < FINE && pts[nr][nc].m > OFF_THRESH && tgt[nr][nc] === 1) cnt++;
          }
        B[r][c] = cnt > 0;
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

// ── gradient (jsx L291-340) ──
function gradAt(r, c, pts, tgt, B, rectify, futFilterFixed) {
  const p = pts[r][c];
  const t = tgt[r][c];
  const mm = p.m > OFF_THRESH ? 1 : 0;
  const caseVal = mm === t ? 1 : 0;
  const yc = yflowCount(r, c, pts, tgt);
  const filterApplies = rectify === "DEF" || rectify === "CAM" || rectify === "DEP" || (rectify === "FUT" && futFilterFixed);
  const yflowGate = filterApplies ? (caseVal === 1 ? 0 : (yc === 1 ? 1 : 0)) : (yc === 1 ? 1 : 0);
  const sign = p.m >= OFF_THRESH ? 1 : -1;
  const weightPos = Math.abs(p.m) > 0.01;
  const t1 = (yflowGate === 1 && weightPos) ? sign * 0.10 : 0;
  let t2 = 0;
  if (caseVal === 0) {
    if (t === 1 && mm === 0) t2 = 0.30;
    if (t === 0 && mm === 1) t2 = -0.50;
  }
  let coordFlow = 0;
  if (t === 1 && B[r][c] && mm === 0) coordFlow = 1;
  const t3 = coordFlow * 0.10;   // Force 3 = coordinate force (key_query→xyz), NOT mask
  const gm = t1 + t2;            // mask = Force1+Force2 only
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

// ── one step (jsx L351-365) ──
function oneStep(pts, tgt, rectify, futFilterFixed, kEff, bFaithful, lr, salt, schema = "MASK_OPT") {
  let cur = pts.map(row => row.map(p => ({ ...p, stale: false })));
  for (let l = L_MAX; l >= 0; l--) cur = routeStep(cur, tgt, l, kEff, salt);
  const s = simulate(cur, tgt, rectify, futFilterFixed, bFaithful);
  return cur.map((row, r) => row.map((p, c) => {
    if (p.stale) return { ...p };
    const g = s.grad[r][c];
    let gm = g.gm;
    if (gm > 0 && p.m <= OFF_THRESH && tgt[r][c] === 1 && p.z > 0 && rand01(r * 5 + c, salt * 3 + 9, salt) >= kEff) gm = 0;
    const nextM = schema === "CAM" ? p.m : Math.max(0, Math.min(1, p.m + lr * gm));
    return { ...p, x: p.x + lr * g.gx, y: p.y + lr * g.gy, z: Math.max(0.01, p.z + lr * g.gz), m: nextM };
  }));
}
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
function kEffective(kBase, kSchedFixed, step) {
  return kSchedFixed ? Math.min(1, 0.1 + 0.9 * step / RAMP) : kBase;
}
function suggestK(pts) {
  let zmin = 1e9, zmax = -1e9;
  for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++) { const z = pts[r][c].z; if (z > 0) { zmin = Math.min(zmin, z); zmax = Math.max(zmax, z); } }
  if (zmax <= 0) return 0.3;
  const spread = (zmax - zmin) / Math.max(zmax, 1e-6);
  return Math.round(Math.max(0.2, Math.min(0.95, 0.2 + 0.8 * spread)) * 100) / 100;
}

// ── driver: iterate the Markov chain exactly as stepMarkov does (lr from auto/manual) ──
function lrAuto(sim) {
  let mx = 0;
  for (let r = 0; r < FINE; r++) for (let c = 0; c < FINE; c++) mx = Math.max(mx, Math.abs(sim.grad[r][c].total));
  return mx > 1e-6 ? 0.05 / mx : 0.1;
}

function run(label, init, opts) {
  const { rectify = "FUT", futFilterFixed, kSchedFixed, bFaithful, kMode = "auto", kBaseManual = 1.0, lrMode = "auto", maxSteps = 80, seed = 123.456, schema = "MASK_OPT" } = opts;
  setSeed(seed);
  let { pts, tgt } = init();
  const kSuggest = suggestK(pts);
  const trace = [];
  let ps = phiStats(pts, tgt);
  trace.push(ps.phi);
  let rose = false;
  let maskChangedTotal = false;
  for (let step = 0; step < maxSteps; step++) {
    const kBaseEff = kMode === "auto" ? kSuggest : kBaseManual;
    const kEff = kEffective(kBaseEff, kSchedFixed, step);
    const sim = simulate(pts, tgt, rectify, futFilterFixed, bFaithful);
    const lr = lrMode === "auto" ? lrAuto(sim) : 0.1;
    const next = oneStep(pts, tgt, rectify, futFilterFixed, kEff, bFaithful, lr, step + 1, schema);
    
    // Check if mask changed
    const maskChanged = next.some((row, r) => row.some((p, c) => Math.abs(p.m - pts[r][c].m) > 1e-12));
    if (maskChanged) maskChangedTotal = true;
    
    pts = next;
    const np = phiStats(pts, tgt);
    if (np.phi > ps.phi) rose = true;
    trace.push(np.phi);
    ps = np;
    if (np.phi === 0) break;
  }
  const final = phiStats(pts, tgt);
  const conv = final.phi === 0;
  console.log(`${label.padEnd(46)} Φ₀=${trace[0]} → Φ=${final.phi}  steps=${trace.length - 1}  ${conv ? "CONVERGED" : "stuck"}  ${rose ? "[Φ ROSE]" : ""}  maskFixed=${!maskChangedTotal}  cov ${final.covered}/${final.totT} clean ${final.clean}/${final.totA}  kSug=${kSuggest}`);
  return { trace, conv, rose, final, maskFixed: !maskChangedTotal };
}

console.log("=== SMap reproducibility re-port (headless) ===\n");

console.log("-- Convergence preset (4 simple target/point pairs) --");
run("conv, FIXED (A+B fixed, FUT)", initConvergence, { rectify: "FUT", futFilterFixed: true, kSchedFixed: true, bFaithful: true });
run("conv, SOURCE (A buggy, FUT)", initConvergence, { rectify: "FUT", futFilterFixed: false, kSchedFixed: false, bFaithful: false, kMode: "auto" });
run("conv, DEP FIXED", initConvergence, { rectify: "DEP", futFilterFixed: true, kSchedFixed: true, bFaithful: true });

console.log("\n-- Default preset --");
run("default, FIXED (FUT)", initDefault, { rectify: "FUT", futFilterFixed: true, kSchedFixed: true, bFaithful: true });
run("default, SOURCE (FUT buggy)", initDefault, { rectify: "FUT", futFilterFixed: false, kSchedFixed: false, bFaithful: false });

console.log("\n-- Far preset (routing across distance) --");
run("far, FIXED (FUT)", initFar, { rectify: "FUT", futFilterFixed: true, kSchedFixed: true, bFaithful: true });
// Headless repro of the interactive-validation lead: at pinned k_const=1.0 the dark target always
// claims itself (warming taken prob = kEff = 1) ⇒ blocks the incoming active point. Does the target
// still get covered (via in-place warming over many steps) or stay uncovered? Decides stall vs slow.
run("far, SOURCE k=1.0 pinned (FUT)", initFar, { rectify: "FUT", futFilterFixed: false, kSchedFixed: false, bFaithful: false, kMode: "manual", kBaseManual: 1.0 });

console.log("\n-- Occlusion preset (Bug B) --");
run("occ, k_const FIXED schedule", initOcclusion, { rectify: "FUT", futFilterFixed: true, kSchedFixed: true, bFaithful: true });
run("occ, k_const=0 pinned (starved)", initOcclusion, { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "manual", kBaseManual: 0.0 });
run("occ, k_const auto (suggested)", initOcclusion, { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "auto" });

console.log("\n-- Bug B direction probe: occlusion at k pinned LOW vs MAX(source default 1.0) --");
run("occ, k=0.0 pinned (min warming)", initOcclusion, { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "manual", kBaseManual: 0.0 });
run("occ, k=0.3 pinned (suggested-ish)", initOcclusion, { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "manual", kBaseManual: 0.3 });
run("occ, k=1.0 pinned (SOURCE DEFAULT)", initOcclusion, { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "manual", kBaseManual: 1.0 });

console.log("\n-- Bug A isolation: FUT source vs fixed on convergence preset, multiple seeds --");
for (const seed of [11.1, 222.2, 777.7]) {
  run(`  seed ${seed} FUT SOURCE`, initConvergence, { rectify: "FUT", futFilterFixed: false, kSchedFixed: true, bFaithful: true, seed });
  run(`  seed ${seed} FUT FIXED`, initConvergence, { rectify: "FUT", futFilterFixed: true, kSchedFixed: true, bFaithful: true, seed });
}

console.log("\n-- CAM Schema (m-fixed) tests --");
run("conv, CAM (rectify=CAM, k=0.5)", initConvergence, { rectify: "CAM", schema: "CAM", kMode: "manual", kBaseManual: 0.5 });
run("default, CAM (rectify=CAM, k=0.5)", initDefault, { rectify: "CAM", schema: "CAM", kMode: "manual", kBaseManual: 0.5 });
run("far, CAM (rectify=CAM, k=0.5)", initFar, { rectify: "CAM", schema: "CAM", kMode: "manual", kBaseManual: 0.5 });
run("occ, CAM (rectify=CAM, k=0.5)", initOcclusion, { rectify: "CAM", schema: "CAM", kMode: "manual", kBaseManual: 0.5 });

// ─────────────────────────────────────────────────────────────────────────────
// CI GATE (added for .github/workflows/harness-check.yml).
// The driver above only PRINTS results; on its own `node _repro_check.mjs` always
// exits 0 and so cannot gate CI against a regression. The block below re-runs a
// few source-grounded invariants and exits non-zero if any regresses. It is purely
// additive — simulate()/routeStep()/oneStep() (the dynamics) are untouched; it only
// inspects run()'s returned {conv, rose}. Seeds use the run() default (123.456 ⇒
// deterministic).
console.log("\n-- CI gate (source-grounded invariants) --");
const _gate = [
  ["conv FIXED (FUT) converges, no Φ-rise",
   run("  [gate] conv FIXED (FUT)", initConvergence,
       { rectify: "FUT", futFilterFixed: true, kSchedFixed: true, bFaithful: true }),
   r => r.conv && !r.rose],
  ["conv DEP FIXED converges, no Φ-rise",
   run("  [gate] conv DEP FIXED", initConvergence,
       { rectify: "DEP", futFilterFixed: true, kSchedFixed: true, bFaithful: true }),
   r => r.conv && !r.rose],
  ["occ k=1.0 (SOURCE DEFAULT) converges — no starvation at source default (Lemma 2 corollary)",
   run("  [gate] occ k=1.0 SOURCE DEFAULT", initOcclusion,
       { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "manual", kBaseManual: 1.0 }),
   r => r.conv],
  ["occ k=0.0 stalls — starvation at k=0 is the expected Bug-B mechanism (guards against a false 'fixed')",
   run("  [gate] occ k=0.0 pinned", initOcclusion,
       { rectify: "FUT", futFilterFixed: true, kSchedFixed: false, bFaithful: true, kMode: "manual", kBaseManual: 0.0 }),
   r => !r.conv],
];
let _failed = 0;
for (const [name, res, pred] of _gate) {
  const ok = pred(res);
  console.log(`   ${ok ? "PASS" : "FAIL"}  ${name}`);
  if (!ok) _failed++;
}
if (_failed > 0) {
  console.error(`\n✗ CI gate: ${_failed} invariant(s) regressed — see FAIL line(s) above.`);
  process.exit(1);
}
console.log("\n✓ CI gate: all 4 invariants hold.");

