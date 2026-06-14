import sys, numpy as np, torch
sys.argv=['x']
import matplotlib
matplotlib.use("Agg")  # no display
from smap.smap import SMap
import smap.utils as utils
utils.DEBUG_FLAG = False  # silence prints

torch.manual_seed(0); np.random.seed(0)
dev="cpu"
# identity-ish camera
K = np.eye(3).astype(np.float64)
# small image: choose H=W so that with n=0 there is exactly one go(is_last=True)
H=W=6
model = SMap(window_h=H, window_w=W, camera_matrix=K, rectify_type=None, device=dev, n=0)
model.smap3x3.k_const = 1.0

# input x: [B, C=1, 4, H, W]  channels = (x,y,z,mask)
B=1
xx = torch.zeros(B,1,4,H,W, dtype=torch.float64)
# put a couple of active points
# coordinates: choose z>0
for r in range(H):
    for c in range(W):
        xx[0,0,0,r,c] = float(c)   # x
        xx[0,0,1,r,c] = float(r)   # y
        xx[0,0,2,r,c] = 2.0        # z
        xx[0,0,3,r,c] = 0.1        # mask
xx[0,0,3,2,3] = 0.8  # one active
xx[0,0,2,2,3] = 2.0

# target silhouette
tgt = torch.zeros(B,1,H,W, dtype=torch.float64)
tgt[0,0,2,3] = 1.0

x = xx.clone().requires_grad_(True)
try:
    out = model(x, target=tgt, zoom=0)
    print("FORWARD OK. out.shape =", tuple(out.shape), "dtype", out.dtype)
    print("out requires_grad:", out.requires_grad)
    loss = out.sum()
    loss.backward()
    print("BACKWARD OK. x.grad is None:", x.grad is None)
    if x.grad is not None:
        g = x.grad
        print("grad shape", tuple(g.shape))
        # mask channel gradient at the target cell and a neighbor
        print("grad mask @ target (2,3) =", g[0,0,3,2,3].item())
        print("grad mask @ (3,3)        =", g[0,0,3,3,3].item())
        print("nonzero mask-grad count  =", int((g[0,0,3]!=0).sum().item()))
except Exception as e:
    import traceback; traceback.print_exc()
