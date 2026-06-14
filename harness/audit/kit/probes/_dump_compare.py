"""Deterministic HDVO-dump generator for before/after refactor byte-comparison.

Seeds torch + numpy so np_rand/temp_rand are fixed, builds the real SMap with the
testbot ACTIVE (DEBUG mode), runs one forward, and lets the testbot write its
SMap_Z_10_*.npy dumps into the cwd. md5sum those dumps before vs after a refactor
to prove behaviour is byte-identical.
"""
import glob
import numpy as np
import torch
import torch.nn as nn

nn.Module.register_backward_hook = lambda self, hook: None
nn.Module.register_full_backward_hook = lambda self, hook: None

from smap.smap import SMap
from smap import rectify, utils

H = W = 8
K = np.eye(3).astype(np.float64)

torch.manual_seed(0)
np.random.seed(0)

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

model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=rectify.types.DEP, device="cpu", n=1)
model.smap3x3.k_const = 1.0
out = model(xx.clone().requires_grad_(True), target=tgt, zoom=0)
dumps = sorted(glob.glob("SMap_Z_10_*.npy"))
print(f"DEBUG_FLAG={utils.DEBUG_FLAG}  out.shape={tuple(out.shape)}  dumps={len(dumps)}")
