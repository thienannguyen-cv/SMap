import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import smap.utils as utils
utils.DEBUG_FLAG=False
from smap import smap as smapmod
from smap.smap import SMap

# monkeypatch to trace shapes inside forward of the rectify call
orig = smapmod.rectify.DefaultRectify.rectificate_flow
def traced(self, new_x_z_mask_value, pre_x, pre_y, pre_z, pre_mask, panels, target, original_size):
    print("RECT new_x_z_mask_value", tuple(new_x_z_mask_value.shape))
    print("RECT pre_mask", tuple(pre_mask.shape), "target", tuple(target.shape), "orig", original_size)
    return orig(self, new_x_z_mask_value, pre_x, pre_y, pre_z, pre_mask, panels, target, original_size)
smapmod.rectify.DefaultRectify.rectificate_flow = traced

torch.manual_seed(0); np.random.seed(0)
K=np.eye(3).astype(np.float64); H=W=8
model=SMap(window_h=H,window_w=W,camera_matrix=K,rectify_type=None,device="cpu",n=0)
model.smap3x3.k_const=1.0
xx=torch.zeros(1,1,4,H,W,dtype=torch.float64)
for r in range(H):
  for c in range(W):
    xx[0,0,0,r,c]=c; xx[0,0,1,r,c]=r; xx[0,0,2,r,c]=2.0; xx[0,0,3,r,c]=0.1
xx[0,0,3,2,3]=0.8
tgt=torch.zeros(1,1,H,W,dtype=torch.float64); tgt[0,0,2,3]=1.0
x=xx.clone().requires_grad_(True)
import traceback
try:
    out=model(x,target=tgt,zoom=0)
except Exception:
    traceback.print_exc()
