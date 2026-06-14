import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
from smap.smap import SMap
import smap.utils as utils
utils.DEBUG_FLAG = False
torch.manual_seed(0); np.random.seed(0)
K = np.eye(3).astype(np.float64)
H=W=8   # try 8 so that after the conv/pad geometry the inner shapes line up
model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=None, device="cpu", n=0)
model.smap3x3.k_const = 1.0
B=1
xx = torch.zeros(B,1,4,H,W, dtype=torch.float64)
for r in range(H):
    for c in range(W):
        xx[0,0,0,r,c]=float(c); xx[0,0,1,r,c]=float(r); xx[0,0,2,r,c]=2.0; xx[0,0,3,r,c]=0.1
xx[0,0,3,2,3]=0.8
tgt = torch.zeros(B,1,H,W, dtype=torch.float64); tgt[0,0,2,3]=1.0
x = xx.clone().requires_grad_(True)
try:
    out = model(x, target=tgt, zoom=0)
    print("FORWARD OK shape", tuple(out.shape), "req_grad", out.requires_grad)
    out.sum().backward()
    g=x.grad
    print("BACKWARD OK; mask-grad nonzero:", int((g[0,0,3]!=0).sum()))
    print("grad mask @ target (2,3):", g[0,0,3,2,3].item())
    print("grad mask @ neighbor(3,3):", g[0,0,3,3,3].item())
except Exception as e:
    import traceback; traceback.print_exc()
