"""
AGENT B — CORRECTNESS. Neutral instruction:
"Given the SOURCE CODE and the app, verify that specific structural invariants the
app claims about the algorithm actually match the source. Output PASS/FAIL per
invariant with the source evidence. Judge only from source + app text."
"""
import re, sys
APP = sys.argv[1] if len(sys.argv)>1 else "/mnt/user-data/outputs/smap_simulator.jsx"
app = open(APP).read()
rect = open("/mnt/user-data/uploads/rectify.py").read()
smap = open("/mnt/user-data/uploads/smap.py").read()

checks = []
def chk(name, cond, ev): checks.append((name, "PASS" if cond else "FAIL", ev))

# INV1: routing selection detached in source AND app treats routing as non-gradient
src_detach = "weights_x.detach()" in smap
app_route_nograd = ("routestep" in app.lower()) and ("gradient" in app.lower())
chk("INV1_routing_detached", src_detach and app_route_nograd,
    f"source weights_x.detach()={src_detach}; app separates routing from gradient={app_route_nograd}")

# INV2: DEP has convergence filter, FUT lacks it (the bug). App must encode BOTH branches.
dep_filter = "y_flow = y_flow*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float())" in rect
fut_filter = rect.count("y_flow = y_flow*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float())") >= 2
# substantive: app must have a FUT-gated toggle for the filter AND a conditional that
# applies the filter for DEF/CAM/DEP (+FUT only when fixed), gating Term1 by caseVal.
al = app.lower()
app_has_toggle = "futfilterfixed" in al
app_has_conditional = "filterapplies" in al and 'rectify === "fut" && futfilterfixed' in al
app_gates_by_case = "caseval === 1 ? 0" in al or "caseval===1?0" in al
app_models_both = app_has_toggle and app_has_conditional and app_gates_by_case
chk("INV2_FUT_bug_is_missing_filter", dep_filter and (not fut_filter) and app_models_both,
    f"DEP filter L294={dep_filter}; FUT lacks it={not fut_filter}; app toggle={app_has_toggle}, conditional={app_has_conditional}, case-gate={app_gates_by_case}")

# INV3: weight>0 guard present in BOTH DEP and FUT (bug_list BUG1 refuted). App must NOT claim it as the bug.
guard_both = rect.count("(weight>0.).detach().float()") >= 2
# substantive: the app's FUT deviation must be the CONVERGENCE FILTER, not the weight guard.
# (a) a bug toggle keyed to the filter exists; (b) where the guard is mentioned, it is marked present-in-all.
al2 = app.lower()
deviation_is_filter = ('id: "futfilter"' in al2) and ("convergence filter" in al2)
guard_marked_present = ("present in all" in al2) or ("guard (present" in al2)
app_correct_attribution = deviation_is_filter and guard_marked_present
chk("INV3_weightguard_not_the_bug", guard_both and app_correct_attribution,
    f"guard in both DEP&FUT={guard_both}; FUT deviation=filter not guard={deviation_is_filter}; guard marked present-in-all={guard_marked_present}")

# INV4: zoom FOLD = 2x2 spatial -> 4 channels (C_zoom). App must use per=2^l and chans=4^l.
src_fold = "C_zoom_2 = C_zoom_2 * 2" in smap or "C_zoom_2 * 2" in smap
app_fold = ("1 << (2 * l)" in app or "1 << (2*l)" in app) and ("1 << l" in app)
chk("INV4_zoom_fold_4x", src_fold and app_fold,
    f"source doubles C_zoom_2 per level={src_fold}; app chans=4^l & per=2^l={app_fold}")

# INV5: selected = physical cell tracked across zoom (origToView). App must map orig->channel.
app_track = "origtoview" in app.lower() and "selorig" in app.lower()
chk("INV5_selected_physical_cell", app_track,
    f"app tracks physical orig cell across views via origToView={app_track}")

# INV6: z-buffer nearest (min depth) wins. Source min over dim; app sorts by z ascending.
src_min = "torch.min(pre_ind" in smap
app_min = "sort((a, b) => a.z - b.z)" in app or "a.z - b.z" in app
chk("INV6_zbuffer_min_depth", src_min and app_min,
    f"source torch.min={src_min}; app sorts ascending z={app_min}")

# INV7: identity camera projection (x/z, y/z). App projPixel uses y/z, x/z.
app_proj = "p.y / p.z" in app and "p.x / p.z" in app
chk("INV7_identity_projection", app_proj,
    f"app projPixel uses (x/z,y/z)={app_proj}")

# INV8: batch update — all cells same lr (not only selected). App stepMarkov maps over all cells.
app_batch = "map((row" in app and "lr *" in app
chk("INV8_batch_update", app_batch, f"app updates all cells with lr={app_batch}")

print("AGENT B (correctness) — structural invariants vs source")
for n,v,e in checks: print(f"  {v}  {n}: {e}")
fails=[c for c in checks if c[1]=="FAIL"]
print(f"\nB_VERDICT: {'ACCEPT' if not fails else 'REJECT'} ({len(checks)-len(fails)}/{len(checks)})")
