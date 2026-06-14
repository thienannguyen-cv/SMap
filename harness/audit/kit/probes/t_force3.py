import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import torch.nn as _nn
_nn.Module.register_backward_hook = lambda self, hook: None
_nn.Module.register_full_backward_hook = lambda self, hook: None
from smap.smap import SMap
from smap import rectify, specials

# Force-3 / gz faithfulness probe.
# QUESTION: in the real source, does the COORDINATE force put gradient on the x/y/z input channels,
# and in particular is the z-channel (index 2) gradient identically zero? The app hardcodes gz=0 and
# only computes gx,gy for an active cell adjacent to a target; we check the real autograd.

torch.manual_seed(0); np.random.seed(0)
dev = "cpu"; K = np.eye(3).astype(np.float64); H = W = 8
NLEVEL = 1

# l1_error (real training loss) — verbatim from depth-estimation.ipynb cell-17
def l1_error(pred, target):
    shapes = pred.size()
    m_batch_size, height, width = target.size(0), shapes[-2], shapes[-1]
    pred_m = (torch.max((pred > specials.OFF_THRESH).reshape(m_batch_size, -1, height*width), dim=1, keepdim=True).values).float()
    abs_pred = torch.abs(pred)
    pred = ((1.-2e-3)*abs_pred.detach()+1e-3+(2.*((pred) > 0.).float()-1.)*(pred-pred.detach())).reshape(m_batch_size, -1, height*width)
    target_m = target.reshape(m_batch_size, 1, height*width)
    loss = torch.abs((pred.reshape(m_batch_size, -1, height, width)).reshape(m_batch_size, -1, height*width) - target_m)
    loss = loss.reshape(m_batch_size, -1, height*width)
    loss_m = torch.abs(pred_m - target_m)
    denom = loss.reshape(m_batch_size, -1).sum(dim=1)
    denom = torch.where(denom > 3e2, denom, 0.*denom+1e2)
    loss = loss.reshape(m_batch_size, -1).sum(dim=1)/denom.detach()
    loss_m = loss_m.reshape(m_batch_size, -1).sum(dim=1)
    loss = torch.mean(loss_m.detach()+1e5*(loss-loss.detach()))
    return loss

def build_scene():
    # base identity-cam field; an ACTIVE non-target cell at (3,3) ADJACENT to a VACANT TARGET at (3,4).
    # This is exactly the app's gx/gy eliciting state (active + adjacent target).
    B = 1
    xx = torch.zeros(B, 4, H, W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0,0,r,c] = float(c)*2.0; xx[0,1,r,c] = float(r)*2.0; xx[0,2,r,c] = 2.0; xx[0,3,r,c] = 0.1
    xx[0,3,3,3] = 0.9            # active point at (3,3), NOT a target
    tgt = torch.zeros(B,1,H,W, dtype=torch.float64)
    tgt[0,0,3,4] = 1.0          # vacant target at (3,4) (adjacent, different column)
    return xx, tgt

def build_scene_S1():
    # also test a matched S1 (active target) at (3,3) — does its own coordinate get any grad?
    B = 1
    xx = torch.zeros(B, 4, H, W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0,0,r,c] = float(c)*2.0; xx[0,1,r,c] = float(r)*2.0; xx[0,2,r,c] = 2.0; xx[0,3,r,c] = 0.1
    xx[0,3,3,3] = 0.9
    tgt = torch.zeros(B,1,H,W, dtype=torch.float64); tgt[0,0,3,3] = 1.0
    return xx, tgt

def mk(rtype):
    rt = {"DEP": rectify.types.DEP, "FUT": rectify.types.FUT, "CAM": rectify.types.CAM}[rtype]
    model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=rt, device=dev, n=NLEVEL)
    model.smap3x3.k_const = 1.0
    model.smap3x3.vtestcase.testbot_input = lambda *a, **k: None
    model.smap3x3.vtestcase.testbot_target = lambda *a, **k: None
    for m in model.modules():
        if getattr(m, "_backward_hooks", None): m._backward_hooks.clear()
    return model

def chan_report(grad, name):
    # grad: [4,H,W]; channels 0=x,1=y,2=z,3=mask
    for ci, cn in [(0,"x"),(1,"y"),(2,"z"),(3,"mask")]:
        g = grad[ci]
        nz = int((g.abs() > 1e-12).sum().item())
        mx = g.abs().max().item()
        print(f"    {name} chan {cn}: nonzero cells={nz:3d}  max|grad|={mx:.4e}")

def run(rtype, scene_fn, scene_name, loss_kind):
    model = mk(rtype)
    xx, tgt = scene_fn()
    x = xx.clone().requires_grad_(True)
    out = model(x, target=tgt, zoom=0)
    if loss_kind == "self":   # isolate the (3,3) cell's own coordinate force via center slice
        loss = out[0,4,3,3]
    else:                      # real training loss over the whole field
        loss = l1_error(out, tgt)
    loss.backward()
    g = x.grad[0]
    print(f"\n=== {rtype} | scene={scene_name} | loss={loss_kind} | loss={loss.item():.6g} ===")
    chan_report(g, "field")
    print(f"    @ active(3,3): x={g[0,3,3].item():+.4e} y={g[1,3,3].item():+.4e} z={g[2,3,3].item():+.4e} m={g[3,3,3].item():+.4e}")
    return g

print("FORCE-3 / gz PROBE — does the source coordinate force write x/y/z, and is z-grad zero?")
# 1) self coordinate force at an active cell adjacent to a target
for rt in ["DEP","CAM","FUT"]:
    run(rt, build_scene, "active(3,3)+adjTarget(3,4)", "self")
# 2) real training loss, same scene
gA = run("DEP", build_scene, "active(3,3)+adjTarget(3,4)", "l1")
# 3) real training loss, matched S1 scene
gS = run("DEP", build_scene_S1, "matchedS1(3,3)", "l1")

print("\n================ FORCE-3 VERDICT ================")
zmaxA = gA[2].abs().max().item(); zmaxS = gS[2].abs().max().item()
xymaxA = max(gA[0].abs().max().item(), gA[1].abs().max().item())
print(f"adjTarget scene (l1): max|x,y grad|={xymaxA:.3e}  max|z grad|={zmaxA:.3e}")
print(f"matchedS1 scene (l1): max|z grad|={zmaxS:.3e}")
if max(zmaxA, zmaxS) < 1e-12:
    print("=> z-channel gradient is IDENTICALLY ZERO in source on these scenes => app gz=0 is FAITHFUL here.")
else:
    print("=> z-channel gradient is NONZERO in source => app gz=0 is a GAP (z does rectify).")
