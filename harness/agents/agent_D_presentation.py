#!/usr/bin/env python3
"""
AGENT D — Presentation faithfulness / no-astonishment.

A reasoning-style criterion implemented as a checklist over EVERY visual signal the app renders.
For each signal: {what it encodes} vs {what a viewer naturally reads}. If they can diverge, the
app must DISAMBIGUATE (label / split / legend). The checker verifies the disambiguation artifacts
are present in the source. This is the criterion that the arrow defect (a routing-lag arrow read
as a target-direction arrow) had previously slipped through.

Run: python3 agent_D_presentation.py <app.jsx>
"""
import sys, re
APP = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs/smap_simulator.jsx"
app = open(APP).read()

# Each check: (id, description, predicate over `app`). Predicate True = disambiguated/faithful.
checks = [
    ("P1_arrow_is_labeled_projection_not_target",
     "Grid arrow encodes projection/routing direction; viewer may read it as target direction. "
     "App must state it is projection/routing and NOT target.",
     ("projection / routing direction" in app or "projection/routing" in app)
     and "NOT" in app and "target" in app),

    ("P2_two_vectors_separated",
     "Routing vector and gradient-toward-target vector differ. App must show BOTH, distinctly "
     "(compass with solid routing + dashed gradient).",
     ("routing" in app and "gradient" in app)
     and "strokeDasharray" in app                       # dashed gradient vector exists
     and "arrowDx" in app and ("gx" in app and "gy" in app)),

    ("P3_meta_color_has_legend",
     "Cell color encodes meta-state; must have a legend mapping color→state.",
     "META" in app and "legend" in app.lower() or all(k in app for k in ["S1", "S2", "S7"]) and "name" in app),

    ("P4_candidate_vs_physical_distinguished",
     "Candidate ('ô khả dĩ') cells vs physical cells must be visually distinct and labeled, so a "
     "viewer is not misled that surrounding cells are independent physical points.",
     ("khả dĩ" in app or "candidate" in app.lower())
     and "SLICE_GLYPH" in app                            # candidates show shift glyphs, not meta
     and ("dashed" in app or "dash" in app)),            # center marked distinctly

    ("P5_hard_cell_distinct",
     "Center-fallback ('hard') cells (point routed away) must be colored/labeled differently from "
     "live physical points, and not fabricated with a fake default value.",
     "stale" in app and "center-fallback" in app.lower()
     and "fb" in app),                                   # distinct label

    ("P6_fill_height_is_mask",
     "Cell fill height encodes m (mask). Should be derived from p.m, not an unrelated quantity.",
     re.search(r"height:\s*`?\$\{[^}]*p\.m", app) is not None
     or re.search(r"Min\(1,\s*p\.m\)|p\.m\)\s*\*\s*100", app) is not None),

    ("P7_target_marker",
     "Target cells must carry an unambiguous marker (⊕) distinct from selection borders.",
     "⊕" in app),

    ("P9_structural_value_labeled",
     "Any displayed numeric that is a structural approximation (sign/scaled, not the true "
     "magnitude) must be labeled as such, so a viewer does not read it as the true value. "
     "Applies to the xyz-gradient (shown as sign×scale, true magnitude k_t-conditioned).",
     ("sign" in app and ("Absolute magnitude" in app or "not shown" in app.lower() or "NOT shown" in app))
     and "k_t" in app),
    ("P8_selection_border_unique",
     "Selected/center markers (red solid, yellow dashed) must be distinct from target dashed and "
     "ordinary borders.",
     "#f43f5e" in app and "dashed #fbbf24" in app),
]

print("AGENT D (presentation faithfulness) — viewer-reading vs actual semantics")
passed = 0
for cid, desc, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {cid}")
    if not ok:
        print(f"        ↳ {desc}")
    passed += bool(ok)
verdict = "ACCEPT" if passed == len(checks) else "REJECT"
print(f"\nD_VERDICT: {verdict} ({passed}/{len(checks)})")
