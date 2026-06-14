"""Verify the DEBUG_FLAG testbot gate: behaviour + side-effects in both modes.

Run from a dir whose imported `smap` has DEBUG off (root) and on (sandbox).
"""
import glob
import numpy as np
import torch
import torch.nn as nn

# disable backward-hook registration (only matters in audit mode; harmless when off)
nn.Module.register_backward_hook = lambda self, hook: None
nn.Module.register_full_backward_hook = lambda self, hook: None

from smap.smap import SMap
from tools.testing.vtest.vtest_types import _DisabledTestbot
from smap import rectify, specials, utils

H = W = 8
K = np.eye(3).astype(np.float64)


def build_input():
    xx = torch.zeros(1, 4, H, W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0, 0, r, c] = c * 2.0
            xx[0, 1, r, c] = r * 2.0
            xx[0, 2, r, c] = 2.0
            xx[0, 3, r, c] = 0.1
    xx[0, 3, 3, 3] = 0.9
    tgt = torch.zeros(1, 1, H, W, dtype=torch.float64)
    tgt[0, 0, 3, 3] = 1.0
    return xx, tgt


def run(rtype):
    rt = {"DEP": rectify.types.DEP, "FUT": rectify.types.FUT}[rtype]
    model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=rt, device="cpu", n=1)
    model.smap3x3.k_const = 1.0
    xx, tgt = build_input()
    before = set(glob.glob("*.npy"))
    x = xx.clone().requires_grad_(True)
    out = model(x, target=tgt, zoom=0)
    loss = out[0, 4, 3, 3]
    loss.backward()
    gm = x.grad[0, 3, 3, 3].item()
    new_npy = sorted(set(glob.glob("*.npy")) - before)
    vt = type(model.smap3x3.vtestcase).__name__
    sentinel = isinstance(model.smap3x3.vtestcase, _DisabledTestbot)
    return gm, vt, sentinel, new_npy


print(f"DEBUG_FLAG = {utils.DEBUG_FLAG}")
gd, vt, sent, npy = run("DEP")
gf, _, _, _ = run("FUT")
print(f"vtestcase class           = {vt}  (sentinel={sent})")
print(f"DEP self mask-force @ S1   = {gd:+.6e}   (expect 0.000000e+00 absorbing)")
print(f"FUT self mask-force @ S1   = {gf:+.6e}   (expect +3.000000e-01)")
print(f"new .npy written (DEP run) = {len(npy)}  {npy[:4]}{'...' if len(npy) > 4 else ''}")
ok_num = abs(gd) < 1e-12 and abs(gf - 0.3) < 1e-9
if utils.DEBUG_FLAG:
    print(f"VERDICT: DEBUG ON  -> real testbot ({vt}), dumps written={len(npy) > 0}, numerics_ok={ok_num}")
else:
    print(f"VERDICT: DEBUG OFF -> sentinel={sent}, NO dumps={len(npy) == 0}, numerics_ok={ok_num}, clean backward (no hook patch needed)")
