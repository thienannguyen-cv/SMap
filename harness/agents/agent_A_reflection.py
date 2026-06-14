"""
AGENT A — REFLECTION. Neutral instruction:
"Given the mechanism inventory and the app source, determine for EACH mechanism
whether the app provides an observable scenario (a UI control, a function, or an
explicit hook the user can exercise). Output PASS/FAIL per mechanism. No access to
the author's reasoning; judge only from app text + inventory."
"""
import json, re, sys
APP = sys.argv[1] if len(sys.argv)>1 else "/mnt/user-data/outputs/smap_simulator.jsx"
inv = json.load(open("/home/claude/verify/mechanisms.json"))
app = open(APP).read().lower()

# Each mechanism maps to keyword evidence the app must contain to be "observable".
EVIDENCE = {
 "M1_routing_key_query": ["route", "projpixel", "neighbor"],
 "M2_zbuffer_seed_noise": ["sort", ".z", "z-buffer|zbuffer|nearest"],
 "M3_kconst_stochastic": ["k_const|kconst|stochastic"],
 "M4_weights_m_x_final": ["m > off_thresh|p.m >|active"],
 "M5_weights_intermediate": ["level|zoom"],
 "M6_broadcast_3x3": ["3|neighbor|fold"],
 "M7_einsum_select": ["detach|routing|gradient"],
 "M8_is_last_agg_flip": ["classify|meta"],
 "M9_calc_weights_zmax": ["z", "depth"],
 "M10_calc_weights_ind": ["depth", "min|nearest|winner"],
 "M11_calc_weights_select": ["winner|select|occlu"],
 "M12_zoom_fold": ["fold", "channel", "c_zoom|chans"],
 "M13_sign_prep": ["sign|z"],
 "M14_forward_loop_unfold": ["markov|forward|level", "per|2^"],
 "M15_final_rectify_dispatch": ["target", "gradient|render"],
 "M0_go_entry": ["fold|pad|channel|projpixel"],
 "M16_module_init_rectify_dispatch": ["rectify", "def|cam|dep|fut"],
 "R1_default_pre_allow": ["computeb|b[", "meta"],
 "R2_default_mask_flow": ["flow"],
 "R3_default_coord_flow": ["coord"],
 "R4_default_weight_constr": ["def|default|term"],
 "R5_cam_weight_constr": ["cam"],
 "R6_dep_pre_allow": ["dep"],
 "R7_dep_mask_flow": ["flow"],
 "R8_dep_coord_flow": ["coord"],
 "R9_dep_weight_constr_FILTER": ["case", "filter|1 - case|1-case"],
 "R10_fut_pre_allow": ["fut"],
 "R11_fut_mask_flow": ["flow"],
 "R12_fut_coord_flow": ["coord|yflow|y_flow"],
 "R13_fut_weight_constr_BUG": ["fut", "bug|filter|current|correct"],
}
def has(eg):
    for term in eg:
        opts = term.split("|")
        if not any(re.search(re.escape(o) if "[" not in o and "^" not in o else o, app) for o in opts):
            return False
    return True

results = {}
for f in inv:
    if f == "note": continue
    for mech in inv[f]:
        eg = EVIDENCE.get(mech, [])
        results[mech] = "PASS" if (eg and has(eg)) else "FAIL"
fails = [k for k,v in results.items() if v=="FAIL"]
print("AGENT A (reflection) — observable scenario per mechanism")
for k,v in results.items(): print(f"  {v}  {k}")
print(f"\nA_VERDICT: {'ACCEPT' if not fails else 'REJECT'} ({len(results)-len(fails)}/{len(results)} observable)")
if fails: print("  FAILED:", ", ".join(fails))
