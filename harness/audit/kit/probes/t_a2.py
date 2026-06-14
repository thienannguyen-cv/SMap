import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import torch.nn as _nn
# disable all backward-hook registration (test-harness logging only; not part of the gradient)
_nn.Module.register_backward_hook = lambda self, hook: None
_nn.Module.register_full_backward_hook = lambda self, hook: None
from smap.smap import SMap
from smap import rectify, specials

torch.manual_seed(0); np.random.seed(0)
dev = "cpu"
K = np.eye(3).astype(np.float64)
H = W = 8

# --- exact l1_error from depth-estimation.ipynb cell-17 ---
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

def build_input():
    # input shape [B, 4, H, W], channels = (x, y, z, mask)
    B = 1
    xx = torch.zeros(B, 4, H, W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0, 0, r, c] = float(c) * 2.0   # x = c*z
            xx[0, 1, r, c] = float(r) * 2.0   # y = r*z
            xx[0, 2, r, c] = 2.0              # z>0
            xx[0, 3, r, c] = 0.1              # mask low
    # S1 cell: matched target (active AND target) at (3,3)
    xx[0, 3, 3, 3] = 0.9
    tgt = torch.zeros(B, 1, H, W, dtype=torch.float64)
    tgt[0, 0, 3, 3] = 1.0
    return xx, tgt

NLEVEL = 1  # n>=1 required (fold loop must run); H=W=8 divisible by 2^n

def run(rtype, label):
    rt = {"DEP": rectify.types.DEP, "FUT": rectify.types.FUT}[rtype]
    model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=rt, device=dev, n=NLEVEL)
    model.smap3x3.k_const = 1.0
    # silence test-harness side effects (forward logging + backward hooks; pure logging, not part of grad)
    model.smap3x3.vtestcase.testbot_input = lambda *a, **k: None
    model.smap3x3.vtestcase.testbot_target = lambda *a, **k: None
    for m in model.modules():
        if getattr(m, "_backward_hooks", None): m._backward_hooks.clear()
    xx, tgt = build_input()
    x = xx.clone().requires_grad_(True)
    out = model(x, target=tgt, zoom=0)
    # Isolate the cell's OWN self-force: loss = center-slice (idx 4) output at the matched cell.
    # d(out[center,3,3])/d m(3,3) is exactly the S1 self mask-force (no broadcast/neighbor sum).
    loss = out[0, 4, 3, 3]
    loss.backward()
    g = x.grad[0, 3]  # mask-channel gradient, [H,W]
    s1 = (3, 3)
    gm = g[s1].item()
    # loss = rectified center-slice weight at the cell, so grad = d(rectified weight)/dm = the
    # ATTRACTIVE self-force direction. >0 reinforces activation (m up); <0 would deactivate (Phi up).
    print(f"\n=== {label} ({rtype}) ===")
    print(f"  out.shape={tuple(out.shape)}  rectified center weight @ S1 = {loss.item():.6g}")
    print(f"  self mask-force d(rectified)/dm @ target S1{s1} = {gm:+.6e}")
    print(f"  => {'ZERO (absorbing, strict fixed point)' if abs(gm)<1e-12 else ('+ REINFORCING (m->1; Phi does NOT rise)' if gm>0 else '- DEACTIVATING (m below tau; Phi CAN rise)')}")
    print(f"  total nonzero self mask-force cells = {int((g.abs() > 1e-12).sum().item())}")
    return g

print("A2 — realized autograd mask-gradient at a matched-target S1 cell (real source backward)")
gd = run("DEP", "DEP (filter present, rectify.py L294)")
gf = run("FUT", "FUT (filter absent, Bug A)")
s1 = (3, 3)
gd1, gf1 = gd[s1].item(), gf[s1].item()
diff = (gf - gd)
print("\n================ A2 VERDICT ================")
print(f"DEP mask-grad @ target S1 = {gd1:+.3e}  (expect ~0 = absorbing)")
print(f"FUT mask-grad @ target S1 = {gf1:+.3e}  (Bug A: expect nonzero)")
print(f"max |FUT-DEP| over field   = {diff.abs().max().item():.3e}  at cells {[(int(r),int(c)) for r,c in zip(*np.where(diff.abs().numpy()>1e-9))][:8]}")
if abs(gd1) < 1e-9 and abs(gf1) > 1e-9:
    print("=> CONFIRMS: DEP target cell ABSORBING (self-force=0); FUT NOT absorbing (Bug A real in source).")
    print(f"=> FUT realized self mask-force = {'+ REINFORCING (m->1; Phi does NOT numerically rise) — matches the corrected app/math_model' if gf1>0 else '- DEACTIVATING (drives m below tau => Phi CAN RISE)'}")
elif diff.abs().max().item() > 1e-9:
    fr, fc = [(int(r),int(c)) for r,c in zip(*np.where(diff.abs().numpy()>1e-9))][0]
    print(f"=> DEP/FUT DIFFER (filter effect real). At target {s1} both ~0; first differing cell {(fr,fc)}: DEP={gd[fr,fc].item():+.3e} FUT={gf[fr,fc].item():+.3e}")
    print(f"=> FUT extra force (-(FUT-DEP)) @ that cell = {-(gf[fr,fc].item()-gd[fr,fc].item()):+.3e} ({'reinforcing/up' if -(gf[fr,fc].item()-gd[fr,fc].item())>0 else 'down'})")
else:
    print("=> DEP and FUT identical on this scene; need a scene exciting y_flow==1 at a matched cell.")
