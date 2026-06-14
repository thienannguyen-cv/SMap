import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import torch.nn as nn
# disable backward-hook registration (testbot hooks do file I/O and crash backward)
nn.Module.register_backward_hook = nn.Module.register_full_backward_hook = lambda self, h: None
from smap.smap import SMap
from smap import rectify, specials

# INDEPENDENT probe: matched-target mask-channel gradient.
# A "matched-target cell" = grid cell that is BOTH active (mask > OFF_THRESH) AND a target cell (tgt=1).
# We measure the gradient that the rectify algorithm puts on that cell's MASK channel (input chan 3),
# for DEP vs FUT, and how the FUT "convergence filter" (y_flow==1 gating) changes it.

torch.manual_seed(0); np.random.seed(0)
dev = "cpu"; K = np.eye(3).astype(np.float64); H = W = 8
NLEVEL = 1

def mk(rtype):
    rt = {"DEP": rectify.types.DEP, "FUT": rectify.types.FUT}[rtype]
    model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=rt, device=dev, n=NLEVEL)
    model.smap3x3.k_const = 1.0   # deterministic (no random dropout of weights_b)
    model.smap3x3.vtestcase.testbot_input  = lambda *a, **k: None
    model.smap3x3.vtestcase.testbot_target = lambda *a, **k: None
    model.smap3x3.vtestcase.testbot_in     = lambda x, *a, **k: x
    model.smap3x3.vtestcase.testbot_out    = lambda x, *a, **k: x
    for m in model.modules():
        if getattr(m, "_backward_hooks", None): m._backward_hooks.clear()
    return model

def build_scene(matched_rc, extra_active=None):
    # identity-cam field: z>0, x=col*z, y=row*z. mask OFF everywhere (0.1) except active cells (0.9).
    B = 1
    xx = torch.zeros(B, 4, H, W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0,0,r,c] = float(c)*2.0
            xx[0,1,r,c] = float(r)*2.0
            xx[0,2,r,c] = 2.0
            xx[0,3,r,c] = 0.1
    mr, mc = matched_rc
    xx[0,3,mr,mc] = 0.9            # active at matched cell
    tgt = torch.zeros(B,1,H,W, dtype=torch.float64)
    tgt[0,0,mr,mc] = 1.0          # target at matched cell -> MATCHED (active & target)
    if extra_active is not None:
        for (r,c) in extra_active:
            xx[0,3,r,c] = 0.9
    return xx, tgt

def isolate_mask_grad(rtype, matched_rc, extra_active, label):
    """Backprop a unit upstream gradient through ONLY the matched cell's output, read input mask grad there."""
    model = mk(rtype)
    xx, tgt = build_scene(matched_rc, extra_active)
    x = xx.clone().requires_grad_(True)
    out = model(x, target=tgt, zoom=0)        # rectified mask map, shape [B, C, h, w]
    mr, mc = matched_rc
    out2 = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
    # sum over any channel dim at the matched spatial cell -> isolates that cell's own output force
    sel = out2[0, :, mr, mc].sum()
    sel.backward()
    g = x.grad[0]
    gm = g[3, mr, mc].item()
    print(f"  [{rtype}] {label}: out@cell={out2[0,:,mr,mc].detach().cpu().numpy()}  "
          f"input-grad@matched cell  x={g[0,mr,mc]:+.3e} y={g[1,mr,mc]:+.3e} "
          f"z={g[2,mr,mc]:+.3e}  MASK={gm:+.4e}")
    return gm

print("="*78)
print("MATCHED-TARGET MASK-CHANNEL GRADIENT PROBE (DEP vs FUT)")
print("OFF_THRESH =", specials.OFF_THRESH)
print("="*78)

# The FUT 'convergence filter' is the y_flow==1 gating. y_flow comes from pre_allow over the 3x3
# neighborhood of target cells. We toggle it by varying the local target/active configuration so the
# matched cell either IS counted as converged (y_flow==1) or is NOT.
print("\n-- DEP rectify (matched cell at (3,3)) --")
dep_iso = isolate_mask_grad("DEP", (3,3), None, "isolated single matched cell")
dep_iso2 = isolate_mask_grad("DEP", (3,3), [(3,4),(4,3)], "matched + adjacent actives")

print("\n-- FUT rectify (matched cell at (3,3)) --")
fut_a = isolate_mask_grad("FUT", (3,3), None, "isolated single matched cell (filter case A)")
fut_b = isolate_mask_grad("FUT", (3,3), [(3,4),(4,3),(2,3),(3,2)], "matched + 4-neighbors active (filter case B)")
fut_c = isolate_mask_grad("FUT", (4,4), None, "isolated single matched cell elsewhere")

# Also directly probe y_flow / the gating by sweeping many isolated configs to see if FUT ever gives nonzero.
print("\n-- FUT sweep over isolated matched cells to surface filter-ON vs filter-OFF --")
fut_vals = []
for rc in [(2,2),(2,3),(3,2),(3,3),(3,4),(4,3),(4,4),(5,5)]:
    v = isolate_mask_grad("FUT", rc, None, f"matched@{rc}")
    fut_vals.append((rc, v))

print("\n" + "="*78)
print("SUMMARY")
print(f"  DEP matched mask-grad (isolated)         = {dep_iso:+.4e}")
print(f"  DEP matched mask-grad (with neighbors)   = {dep_iso2:+.4e}")
print(f"  FUT matched mask-grad (isolated A)       = {fut_a:+.4e}")
print(f"  FUT matched mask-grad (4-neighbor B)     = {fut_b:+.4e}")
print(f"  FUT sweep: {[(rc, round(v,4)) for rc,v in fut_vals]}")
print("="*78)
