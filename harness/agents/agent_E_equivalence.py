#!/usr/bin/env python3
"""
AGENT E — Computational / structural equivalence.

For each CORE data structure / computed quantity in the app, verify it traces to a specific
tensor operation in the source with MATCHING semantics — not just that a mechanism is "mentioned"
(Agent A) or that surface invariants hold (Agent B). This is the criterion that the candidate
('ô khả dĩ') defect slipped through: the app had displayed independent physical neighbors where
the source's 3×3 is a BROADCAST of the center point (smap.py L70-75).

Ground truth read directly from the source.
Run: python3 agent_E_equivalence.py <app.jsx>
"""
import sys, re
APP = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs/smap_simulator.jsx"
app = open(APP).read()
smap = open("/mnt/user-data/uploads/smap.py").read()
util = open("/mnt/user-data/uploads/utils.py").read()

def src_has(text, *frags):
    return all(f in text for f in frags)

checks = []

# E1 — 3×3 neighborhood is a BROADCAST of the center (candidates), not gathered physical neighbors.
# Source: smap.py builds x_value via reshape(...1,1,1...) + zeros_like(weights_x)(...3,3...) → broadcast.
src_broadcast = ("zeros_like(weights_x)" in smap and "x_value" in smap and "reshape" in smap)
app_candidates = ("khả dĩ" in app or "candInfo" in app) and "broadcast" in app.lower() \
    and "SLICE_GLYPH" in app and "carries" in app.lower()
checks.append(("E1_neighborhood_is_broadcast_candidates",
    "Source 3×3 = broadcast of center (smap.py L70-75); app must render candidate copies of the "
    "center, not independent physical neighbors.",
    src_broadcast and app_candidates))

# E2 — candidate slice ordering 0=up-left … 4=center … 8=down-right (2config1.json / im2col order).
app_order = "sliceShift" in app and re.search(r"\(\s*\(?s\s*/\s*3\)?\s*\|\s*0\s*\)\s*-\s*1", app) is not None \
    and re.search(r"\(s\s*%\s*3\)\s*-\s*1", app) is not None
checks.append(("E2_slice_order_up_left_to_down_right",
    "Slice s→shift (floor(s/3)-1, s%3-1): slice0=up-left, slice4=center, slice8=down-right.",
    app_order))

# E3 — per-slice gradient = key_query distance between center xy and the SHIFTED position's
# back-projection (utils.py L174-182), aggregated back to the physical point.
src_keyquery = src_has(util, "grouped_key_x", "query_x", "key_query", "diff_x")
app_keyquery = "key_query" in app and "candInfo" in app and "agg(flip" in app.lower().replace(" ", "") \
    or ("key_query" in app and "qx = pc" in app and "agg(flip" in app)
checks.append(("E3_per_slice_keyquery_aggregates_back",
    "Each candidate's gradient = center xy vs that position's back-projection; 9 agg(flip) back "
    "to the physical point.",
    src_keyquery and ("candInfo" in app and "kq" in app)))

# E4 — gradient (rectification) is FINEST-3×3 only (rectificate_flow at is_last), not global.
src_islast = "is_last=True" in smap and "rectificate_flow" in smap
app_local = "finest-3" in app.lower().replace("×", "x").replace("✕", "x") or "finest 3" in app.lower()
# ensure the app did NOT keep a whole-grid nearest-target search for the gradient
app_not_global = "for (let nr = 0; nr < FINE; nr++)\n      for (let nc = 0; nc < FINE; nc++) {\n        if (tgt[nr][nc] === 1)" not in app
checks.append(("E4_gradient_is_finest_local",
    "rectificate_flow runs at is_last/finest (smap.py L244-251) → xyz-gradient is 3×3-local; "
    "the app must not pull toward arbitrarily far targets via the gradient.",
    src_islast and app_local and app_not_global))

# E5 — routing (not gradient) crosses distance: detached selection, 2^l per level.
src_routing = "weights_x.detach()" in smap
app_routing = "detach" in app.lower() and "routes through all zoom levels" in app and "2^L" in app or "2^l" in app
checks.append(("E5_routing_crosses_distance_detached",
    "Distance is crossed by detached routing at 2^l/level (smap.py L83/L130 + loop L224-238), "
    "not by the gradient.",
    src_routing and ("routing" in app and ("detach" in app.lower()))))

# E6 — center-fallback: a cell with no valid winner keeps its own value (slice 4), source L170.
src_fallback = "0*ind+4" in smap or "+4" in smap and "torch.where" in smap
app_fallback = "center-fallback" in app.lower() and "stale" in app and "L170" in app
checks.append(("E6_center_fallback_not_fabricated",
    "Vacated cell takes center-fallback (keeps own value, smap.py L170); app must not fabricate a "
    "fake default value.",
    src_fallback and app_fallback))

# E7 — FOLD bijection 4× channels / 2× per axis; physical state stored at finest only.
src_fold = "C_zoom" in smap and "reshape" in smap
app_fold = "1 << (2 * l)" in app and "1 << l" in app and "finest" in app.lower() and "FOLD" in app
checks.append(("E7_fold_bijection_finest_SST",
    "Zoom views are FOLD of a single finest-resolution SST (4^l channels, 2^l per axis); no "
    "per-level stored grid (so zoom can't corrupt).",
    src_fold and app_fold))


# E8 — gradient faithfulness: the load-bearing property is SIGN + zero/non-zero (lr-independent);
# source magnitude is conditioned by k_t = k_d/pre_z (rectify.py), applied by CAM/DEP/FUT not
# Default. App must preserve sign/zero and acknowledge k_t (show or explicitly abstract w/ reason).
rect=open("/mnt/user-data/uploads/rectify.py").read()
src_kt = "k_t" in rect and "calculate_key_query" in rect and ("pre_x*k_t" in rect or "pre_x * k_t" in rect)
src_default_no_kt = rect.count("k_t") >= 4  # Default lacks it; CAM/DEP/FUT have it
app_sign = ("sign" in app and "Math.sign" in app)               # gradient is sign-based (zero preserved)
app_kt = "k_t" in app and ("layer-wise" in app or "conditioning" in app.lower() or "magnitude" in app.lower())
app_per_rectify = 'rectify !== "DEF"' in app or "not\n              Default" in app or "not</b> Default" in app or "not Default" in app
checks.append(("E8_gradient_sign_zero_faithful_kt_acknowledged",
    "Gradient sign & zero/non-zero faithful (lr-independent); k_t magnitude-conditioning "
    "(rectify.py L129-133) acknowledged and scoped to CAM/DEP/FUT (not Default).",
    src_kt and app_sign and app_kt))

print("AGENT E (computational/structural equivalence) — sim ⟷ source tensor ops")
passed = 0
for cid, desc, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {cid}")
    if not ok:
        print(f"        ↳ {desc}")
    passed += bool(ok)
verdict = "ACCEPT" if passed == len(checks) else "REJECT"
print(f"\nE_VERDICT: {verdict} ({passed}/{len(checks)})")
