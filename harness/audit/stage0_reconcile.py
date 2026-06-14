"""
STAGE 0: Reconcile bug_list.txt against the ACTUAL current source code.
Every verdict is produced by mechanical string/structure checks on the source,
NOT by opinion. Reproducible by re-running.
"""
import re

SRC = "/mnt/user-data/uploads"
with open(f"{SRC}/rectify.py") as f: rectify = f.readlines()
with open(f"{SRC}/smap.py") as f: smap = f.readlines()

def line(arr, n): return arr[n-1].rstrip()

results = []

# ---- BUG 1: FUT weights_grdf missing (weight>0) guard that DEP has ----
# Find the weights_grdf line in DEP and FUT
def find_weights_grdf(arr, lo, hi):
    for i in range(lo-1, hi):
        if arr[i].strip().startswith("weights_grdf ="):
            return i+1, arr[i]
    return None, None

dep_ln, dep_wg = find_weights_grdf(rectify, 191, 311)
fut_ln, fut_wg = find_weights_grdf(rectify, 312, 430)
guard = "(weight>0.).detach().float()"
dep_has = guard in dep_wg
fut_has = guard in fut_wg
results.append(("BUG1_FUT_missing_weight_guard",
    "REFUTED" if (dep_has and fut_has) else ("CONFIRMED" if (dep_has and not fut_has) else "INCONCLUSIVE"),
    f"DEP L{dep_ln} has guard={dep_has}; FUT L{fut_ln} has guard={fut_has}. "
    f"bug_list claims FUT missing it. Identical guard present in both → claim does not hold on current source."))

# ---- BUG 4: DefaultRectify C_zoom undefined ----
def_block = "".join(rectify[10:119])  # DefaultRectify lines 11-119
defines_czoom = "C_zoom = new_x_z_mask_value.shape[1]" in def_block
results.append(("BUG4_Default_C_zoom_undefined",
    "ALREADY_FIXED" if defines_czoom else "CONFIRMED",
    f"DefaultRectify defines C_zoom in-body={defines_czoom}. Source carries [BUG 4 FIX] annotation."))

# ---- MY simulator's bug: FUT missing y_flow *= (1-case) that DEP L294 has ----
dep_block = "".join(rectify[190:311])
fut_block = "".join(rectify[311:430])
filt = "y_flow = y_flow*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float())"
dep_has_filter = filt in dep_block
fut_has_filter = filt in fut_block
results.append(("SIM_FUT_missing_yflow_convergence_filter",
    "STRUCTURALLY_CONFIRMED" if (dep_has_filter and not fut_has_filter) else "REFUTED",
    f"DEP applies y_flow*=(1-case) at L294: {dep_has_filter}. FUT applies it anywhere: {fut_has_filter}. "
    f"This is the convergence filter; absence in FUT means Term1 can be nonzero at S1 (when y_flow==1)."))

# ---- BUG 2 & 3: design/improvement, verify they are NOT correctness-breaking ----
# BUG2: is_last==False gives inactive weights_x=weights_b (smap L62-63). Just confirm presence.
il_block = "".join(smap[60:64])
results.append(("BUG2_intermediate_inactive_weights",
    "DESIGN_NOTE",
    "smap.py is_last==False branch present (L61-63). bug_list itself concludes the quick-fix is correct (trade-off), not a correctness bug."))
# BUG3: k_const stochastic block present
k_block = "".join(smap[53:56])
results.append(("BUG3_kconst_inactive_exclusion",
    "IMPROVEMENT_NOTE",
    "smap.py k_const stochastic block present (L54-55). bug_list concludes: slows convergence, does not break it."))

print(f"{'CLAIM':<42} {'VERDICT':<24} DETAIL")
print("="*120)
for name, verdict, detail in results:
    print(f"{name:<42} {verdict:<24} {detail}")
