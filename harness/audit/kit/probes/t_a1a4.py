import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import torch.nn as _nn
_nn.Module.register_backward_hook = lambda self, hook: None
_nn.Module.register_full_backward_hook = lambda self, hook: None
from smap.smap import SMap
from smap import rectify, specials

# A1/A4 probe — does the ROUTING / z-buffer layer ever raise Phi?
#   A1: does the z-buffer flip a matched target's case (uncover it)?
#   A4: does neighbour coupling induce a Phi-increasing transition over multi-step dynamics?
# Decided EMPIRICALLY by executing the real source: (1) a single-forward overwrite test, and
# (2) a multi-step real-source dynamics trajectory tracking Phi.

torch.manual_seed(0); np.random.seed(0)
dev="cpu"; K=np.eye(3).astype(np.float64); H=W=8; NLEVEL=1
OFF = specials.OFF_THRESH

def l1_error(pred, target):
    shapes = pred.size(); m_bs, height, width = target.size(0), shapes[-2], shapes[-1]
    pred_m = (torch.max((pred > OFF).reshape(m_bs, -1, height*width), dim=1, keepdim=True).values).float()
    abs_pred = torch.abs(pred)
    pred = ((1.-2e-3)*abs_pred.detach()+1e-3+(2.*((pred) > 0.).float()-1.)*(pred-pred.detach())).reshape(m_bs, -1, height*width)
    target_m = target.reshape(m_bs, 1, height*width)
    loss = torch.abs((pred.reshape(m_bs, -1, height, width)).reshape(m_bs, -1, height*width) - target_m).reshape(m_bs, -1, height*width)
    loss_m = torch.abs(pred_m - target_m)
    denom = loss.reshape(m_bs, -1).sum(dim=1); denom = torch.where(denom > 3e2, denom, 0.*denom+1e2)
    loss = loss.reshape(m_bs, -1).sum(dim=1)/denom.detach()
    loss_m = loss_m.reshape(m_bs, -1).sum(dim=1)
    return torch.mean(loss_m.detach()+1e5*(loss-loss.detach()))

def mk(rtype, k):
    rt = {"DEP": rectify.types.DEP, "FUT": rectify.types.FUT}[rtype]
    model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=rt, device=dev, n=NLEVEL)
    model.smap3x3.k_const = float(k)
    model.smap3x3.vtestcase.testbot_input = lambda *a, **k: None
    model.smap3x3.vtestcase.testbot_target = lambda *a, **k: None
    for m in model.modules():
        if getattr(m, "_backward_hooks", None): m._backward_hooks.clear()
    return model

def predm(out):
    B = out.size(0)
    return (out > OFF).reshape(B, -1, H*W).max(dim=1).values.float().reshape(B, H, W)

def phi(out, tgt):
    return int((predm(out)[0] != tgt[0,0]).sum().item())

def base_field():
    xx = torch.zeros(1,4,H,W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0,0,r,c]=c*2.0; xx[0,1,r,c]=r*2.0; xx[0,2,r,c]=2.0; xx[0,3,r,c]=0.1
    return xx

# ---- Scene OCC-B: matched S1 at (3,3) + a NEARER INACTIVE intruder projecting onto it ----
# Decisive A1 test: in source the z-buffer takes the MIN-depth winner regardless of activity, so a
# nearer inactive point could overwrite an active matched target => uncover it => Phi+1.
def scene_overwrite():
    xx = base_field()
    xx[0,3,3,3] = 0.9                              # matched S1 (active target), z=2.0, projects to (3,3)
    # nearer INACTIVE intruder at (3,2) whose projection = (3,3): x=3*0.5, y=3*0.5, z=0.5
    xx[0,0,3,2]=3*0.5; xx[0,1,3,2]=3*0.5; xx[0,2,3,2]=0.5; xx[0,3,3,2]=0.1
    tgt = torch.zeros(1,1,H,W, dtype=torch.float64); tgt[0,0,3,3]=1.0
    return xx, tgt

# ---- Scene OCC-A: dark target warming (app initOcclusion) ----
def scene_warming():
    xx = base_field()
    xx[0,0,3,3]=3*0.8; xx[0,1,3,3]=3*0.8; xx[0,2,3,3]=0.8; xx[0,3,3,3]=0.1   # dark target, projects to self
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r,c=3+dr,3+dc
        xx[0,0,r,c]=c*0.5; xx[0,1,r,c]=r*0.5; xx[0,2,r,c]=0.5; xx[0,3,r,c]=0.9  # nearer bright occluders, project to self
    tgt = torch.zeros(1,1,H,W, dtype=torch.float64); tgt[0,0,3,3]=1.0
    return xx, tgt

def dynamics(rtype, k, scene_fn, name, steps=40):
    # small ANNEALED step (Lemma-0 'eta_0 small' regime); bang-bang fixed steps create spurious
    # limit cycles, so the mask step decays over time to expose the true monotone behaviour.
    model = mk(rtype, k)
    xx, tgt = scene_fn()
    x = xx.clone()
    phis, cover, rises, uncover = [], [], [], []
    for t in range(steps):
        x = x.detach().requires_grad_(True)
        out = model(x, target=tgt, zoom=0)
        ph = phi(out, tgt); pm = predm(out)[0]
        phis.append(ph); cover.append(int(pm[3,3].item() > 0.5))
        loss = l1_error(out, tgt); loss.backward()
        g = x.grad[0]
        step = 0.20 * (0.85 ** t)                              # small, annealed -> 0
        with torch.no_grad():
            nx = x.clone()
            nx[0,3] = (x[0,3] - step*torch.sign(g[3])).clamp(0,1)
            nx[0,0] = x[0,0] - 1e-3*g[0]; nx[0,1] = x[0,1] - 1e-3*g[1]
            nx[0,2] = x[0,2].clamp(min=0.01)
        x = nx
    for i in range(1,len(phis)):
        if phis[i] > phis[i-1]: rises.append((i-1,i,phis[i-1],phis[i]))
        if cover[i-1] == 1 and cover[i] == 0: uncover.append((i-1,i))   # matched-target UNCOVERED (A1/A4 risk)
    tail = phis[-8:]
    print(f"\n=== dynamics {rtype} k={k} | {name} ===")
    print(f"  Phi trajectory : {phis}")
    print(f"  tail (last 8)  : {tail}  -> {'CONVERGED' if max(tail)==min(tail) else 'still moving/oscillating'}")
    print(f"  target covered : {cover}")
    print(f"  target UNCOVER events (1->0, the A1/A4 risk) : {uncover if uncover else 'NONE'}")
    print(f"  Phi increases (any)                          : {rises if rises else 'NONE'}")
    return phis, rises, uncover

print("A1/A4 — real-source occlusion dynamics & overwrite test\n" + "="*52)

# (1) Decisive single-forward A1 overwrite test
for rt in ["DEP","FUT"]:
    model = mk(rt, 1.0)
    xx, tgt = scene_overwrite()
    x = xx.clone().requires_grad_(True)
    out = model(x, target=tgt, zoom=0)
    pm = predm(out)[0]
    print(f"\n[A1 overwrite test | {rt}] matched S1 at (3,3) + nearer INACTIVE intruder routing onto it")
    print(f"  pred_m @ target (3,3) after forward = {int(pm[3,3].item()>0.5)}  (1=still covered, 0=UNCOVERED => Phi+1)")
    print(f"  Phi @ step0 = {phi(out, tgt)}  => {'A1 HOLDS (z-buffer did NOT uncover the matched target)' if pm[3,3].item()>0.5 else 'A1 VIOLATED (z-buffer uncovered a matched target => case (b))'}")

# (2) Multi-step dynamics: warming scene at source-default k=1.0 vs starved k=0.0
dynamics("DEP", 1.0, scene_warming, "warming (source default k=1.0)")
dynamics("DEP", 0.0, scene_warming, "warming (k=0 -> starvation contrast)")
# and the overwrite scene as dynamics (does Phi ever rise over steps?)
dynamics("DEP", 1.0, scene_overwrite, "overwrite scene (k=1.0)")

print("\n================ A1/A4 VERDICT ================")
print("Read above: if no 'Phi INCREASES' across all DEP k=1.0 dynamics AND A1 overwrite test HOLDS,")
print("evidence supports case (a) (proof gap, same algorithm; Bug B precision-only).")
print("Any Phi rise on a matched-target uncover supports case (b) (routing must change; Bug B real).")
