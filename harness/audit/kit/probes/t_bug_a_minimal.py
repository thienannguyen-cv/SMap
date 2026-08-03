import sys, os
import subprocess
import torch
import torch.nn as _nn
import numpy as np

# Suppress debug output
_nn.Module.register_backward_hook = lambda self, hook: None
_nn.Module.register_full_backward_hook = lambda self, hook: None

from smap.smap import SMap
from smap import rectify, specials

# Record version/environment context
def get_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def check_bug_a():
    torch.manual_seed(1024)
    np.random.seed(1024)
    
    B, H, W = 1, 8, 8
    K = np.eye(3).astype(np.float64)
    
    # Minimal deterministic scene
    # We want a target matched cell at (2,2)
    tgt = torch.zeros(B, 1, H, W, dtype=torch.float64)
    tgt[0, 0, 2, 2] = 1.0
    
    # Input
    xx = torch.zeros(B, 4, H, W, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            xx[0, 0, r, c] = float(c) * 3.0
            xx[0, 1, r, c] = float(r) * 3.0
            xx[0, 2, r, c] = 3.0
            xx[0, 3, r, c] = 0.1 # inactive background
    
    # Matched active cell
    xx[0, 3, 2, 2] = 0.95
    
    def run_mode(rectify_type, mode_name):
        model = SMap(window_h=H, window_w=W, camera_matrix=K, 
                     rectify_type=rectify_type, device="cpu", n=1)
        model.smap3x3.k_const = 1.0
        
        # Disable testbot side-effects
        model.smap3x3.vtestcase.testbot_input = lambda *a, **k: None
        model.smap3x3.vtestcase.testbot_target = lambda *a, **k: None
        for m in model.modules():
            if getattr(m, "_backward_hooks", None): m._backward_hooks.clear()
            
        x = xx.clone().requires_grad_(True)
        out = model(x, target=tgt, zoom=0)
        
        # Self-force center slice
        loss = out[0, 4, 2, 2]
        loss.backward()
        
        # Gradients
        g_m = x.grad[0, 3, 2, 2].item()
        g_all = x.grad.clone()
        return g_m, g_all
        
    g_m_dep, g_all_dep = run_mode(rectify.types.DEP, "DEP")
    g_m_fut, g_all_fut = run_mode(rectify.types.FUT, "FUT")
    
    print("=== SMap Bug A Minimal Validation ===")
    print(f"Commit: {get_commit()}")
    print(f"Seed: 1024")
    print(f"Command: python {sys.argv[0]}")
    print(f"DEP g_m @ (2,2): {g_m_dep:+.8e}")
    print(f"FUT g_m @ (2,2): {g_m_fut:+.8e}")
    
    diff_mask = torch.abs(g_all_fut[0, 3] - g_all_dep[0, 3]).max().item()
    diff_spatial = torch.abs(g_all_fut[0, :3] - g_all_dep[0, :3]).max().item()
    print(f"Max mask grad diff (FUT vs DEP): {diff_mask:.8e}")
    print(f"Max spatial grad diff (FUT vs DEP): {diff_spatial:.8e}")
    
    return g_m_dep, g_m_fut, g_all_dep, g_all_fut

if __name__ == "__main__":
    check_bug_a()
