#!/usr/bin/env python
"""HDVO data generator — ONE camera-calibration training step (audit mode).

Faithfully replays a single parameter-update iteration of the
`applications/Camera Calibration/camera-calibration.ipynb` training loop on ONE
non-trivial Sky303 sample (batch_size=1), at zoom = n-1, with the testbot
instrumentation ON (utils.DEBUG_FLAG=True). The forward pass therefore makes the
SMap renderer dump its internal L62-72 tensors (weights_b/weights_m/weights_x,
np_rand, temp_rand, z-buffer, ...) as SMap_Z_*.npy — the behavioural ground truth
HDVO works from.

WHY this configuration:
  * m-fixed regime: the camera-calibration model optimises only the camera pose
    (phi/gamma/rho rotation + theta translation); the mask `m` is FIXED
    (mask = .999 * r_mask). Only geometry (xyz) moves -> this is exactly the
    m-fixed camera-calibration regime the proof must be re-derived for.
  * zoom = n-1: at i=0 (zoom=n, coarsest) target ~= output, so the comparison
    yields almost no gradient. One level finer (zoom=n-1) is the first level that
    carries real optimisation information, while the grid is still small
    (H/2^(n-1) x W/2^(n-1)) and analysable for the L62-72 nest.
  * DEBUG set BEFORE building the model: our gate (smap/smap.py) builds the
    testbot only if utils.DEBUG_FLAG is True AT CONSTRUCTION. The notebook flips
    it later, which would now skip the testbot; here we set it first so dumps
    are produced.

NOT the HDVO procedure itself — it produces the data the user then guides HDVO on.

Run:  python audit/hdvo/run_calib_step.py
Dumps land in audit/hdvo/artifacts/SMap_Z_5_*.npy  (k_const=0.5 -> "_5"); these are
the artifact data the HDVO session (audit/hdvo/instrument/) references via exec.py.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- AUDIT MODE: must be set BEFORE the SMap model is constructed (gate timing) ----
from smap import utils
utils.DEBUG_FLAG = True

from smap import SMap, rectify, specials  # noqa: E402

# ----------------------------- constants (from the notebook) -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "applications" / "Camera Calibration" / "Sky-dataset" / "Sky303"
N_CONFIGURATIONS = 4262
IMG_SHAPE = (1355, 3384, 3)
H, W = 128, 256
CAMERA = np.array([[2304.5479, 0, 1686.2379],
                   [0, 2305.8757, -0.0151],
                   [0, 0, 1.]], dtype=np.float32)
N = 6                 # SMap levels (notebook: n = 6)
LR = 1e-3             # notebook lr
K_CONST = 0.5         # notebook pins smap3x3.k_const = 0.5 during training
ZOOM = N - 2          # standard HDVO tensor size (=4 -> 8x16 grid, >=8 each dim -> hotspot-able)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- data (from the notebook dataset) -----------------------------
def load_sample(idx):
    """Return (input_bin_mask[1,H,W], input_2Drepr[3,H,W], target_2Drepr[1,H,W])."""
    indice = np.load(DATA_DIR / f"input_indices_{idx}.npy")
    data = np.load(DATA_DIR / f"input_data_{idx}.npy")
    target = np.load(DATA_DIR / f"output_indices_{idx}.npy")

    bin_mask = np.zeros([H, W, 1], dtype="float16")
    bin_mask[indice[:, 0], indice[:, 1], :] = 1.0
    repr2d = np.zeros([H, W, 3], dtype="float16")
    repr2d[indice[:, 0], indice[:, 1], :] = data
    tgt2d = np.zeros([H, W, 1], dtype="float16")
    tgt2d[target[:, 0], target[:, 1], :] = 1.0

    to_t = lambda a: torch.from_numpy(np.rollaxis(a, 2, 0)).float()
    return to_t(bin_mask), to_t(repr2d), to_t(tgt2d), indice, target


def pick_nontrivial_idx():
    """First sample (in a fixed shuffled order) that is genuinely informative:
    enough active + target points, both spread over >1 row and >1 col, and the
    input/target are not near-identical (otherwise the comparison is trivial)."""
    order = np.random.RandomState(SEED).permutation(N_CONFIGURATIONS)
    for idx in order:
        try:
            ind = np.load(DATA_DIR / f"input_indices_{idx}.npy")
            tgt = np.load(DATA_DIR / f"output_indices_{idx}.npy")
        except FileNotFoundError:
            continue
        if len(ind) < 40 or len(tgt) < 40:
            continue
        spread = (ind[:, 0].ptp() > 8 and ind[:, 1].ptp() > 8 and
                  tgt[:, 0].ptp() > 8 and tgt[:, 1].ptp() > 8)
        # require input and target to differ meaningfully (non-degenerate gradient)
        differ = len(ind) != len(tgt) or not np.array_equal(np.sort(ind, 0), np.sort(tgt, 0))
        if spread and differ:
            return int(idx), len(ind), len(tgt)
    raise RuntimeError("no non-trivial sample found")


# ----------------------------- model (verbatim from the notebook) -----------------------------
class Model(nn.Module):
    def __init__(self, camera_matrix, n=0):
        super(Model, self).__init__()
        self.n = n
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)

        self.phi = nn.Parameter(torch.from_numpy(np.array([.0])).float().reshape(1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.from_numpy(np.array([.0])).float().reshape(1, 1), requires_grad=True)
        self.rho = nn.Parameter(torch.from_numpy(np.array([.0])).float().reshape(1, 1), requires_grad=True)
        self.theta = nn.Parameter(torch.from_numpy(np.array([0., -0., 0.])).float().reshape(1, 3, 1, 1), requires_grad=True)
        self.smap_layer = SMap(IMG_SHAPE[0], IMG_SHAPE[1], camera_matrix, rectify_type=rectify.types.CAM, n=n, device=device)

        self.I00 = nn.Parameter(torch.from_numpy(np.array([[1., .0, .0], [.0, 0., .0], [.0, .0, 0.]])).float().reshape(3, 3), requires_grad=False)
        self.I01 = nn.Parameter(torch.from_numpy(np.array([[0., 1., .0], [.0, 0., .0], [.0, .0, 0.]])).float().reshape(3, 3), requires_grad=False)
        self.I10 = nn.Parameter(torch.from_numpy(np.array([[0., .0, .0], [1., 0., .0], [.0, .0, 0.]])).float().reshape(3, 3), requires_grad=False)
        self.I11 = nn.Parameter(torch.from_numpy(np.array([[0., .0, .0], [.0, 1., .0], [.0, .0, 0.]])).float().reshape(3, 3), requires_grad=False)
        self.I22 = nn.Parameter(torch.from_numpy(np.array([[0., .0, .0], [.0, 0., .0], [.0, .0, 1.]])).float().reshape(3, 3), requires_grad=False)

        self.net = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, r_x, r_mask, target_2Dr=None, zoom=0):
        from smap import utils
        mask = .999 * r_mask  # mask is FIXED (m-fixed regime); the net is intentionally unused

        x = 1. * (r_x[:, -1:, :, :])
        height, width = x.size(-2), x.size(-1)
        panels = list(np.where(np.ones([height, width])))
        panels[0] = panels[0] + .5
        panels[1] = panels[1] + .5
        y_im = torch.from_numpy(panels[0]).float().reshape(1, H, W).to(device) + torch.zeros_like(x[:, 0, :, :])
        x_im = torch.from_numpy(panels[1]).float().reshape(1, H, W).to(device) + torch.zeros_like(x[:, 0, :, :])
        x = (utils.to_3d(x.reshape(-1, 1, height, width), H, W, y_im, x_im, (H, W), IMG_SHAPE, self.camera_matrix_inv, device)[:, 0, :, :, :])

        phi = torch.tanh(3e-2 * self.phi) * np.pi
        gamma = torch.tanh(3e-2 * self.gamma) * np.pi
        rho = torch.tanh(3e-2 * self.rho) * np.pi
        theta = self.theta.detach() + (self.theta - self.theta.detach())
        r = torch.cos(phi) * self.I00 - torch.sin(phi) * self.I01 + torch.sin(phi) * self.I10 + torch.cos(phi) * self.I11 + self.I22
        y = torch.cos(gamma) * self.I00 - torch.sin(gamma) * self.I01 + torch.sin(gamma) * self.I10 + torch.cos(gamma) * self.I11 + self.I22
        p = torch.cos(rho) * self.I00 - torch.sin(rho) * self.I01 + torch.sin(rho) * self.I10 + torch.cos(rho) * self.I11 + self.I22

        x_r = torch.einsum('bdef,cd->bcef', x, r)
        x_r = torch.cat([x_r[:, 1:, :, :], x_r[:, :1, :, :]], dim=1)
        x_r = torch.einsum('bdef,cd->bcef', x_r, y)
        x_r = torch.cat([x_r[:, 1:, :, :], x_r[:, :1, :, :]], dim=1)
        x_r = torch.einsum('bdef,cd->bcef', x_r, p)
        x_r = torch.cat([x_r[:, 1:, :, :], x_r[:, :1, :, :]], dim=1)

        x_t = x_r + (mask > .5).float() * theta
        x_z = torch.where(torch.abs(x_t[:, -1:, :, :]) > .5, 0. * (x_t[:, -1:, :, :]), 0. * (x_t[:, -1:, :, :]) + (2. * ((x_t[:, -1:, :, :]) > 0.).float() - 1.) * .5)
        temp = torch.cat([torch.zeros_like(x_t[:, :-1, :, :]), x_z], dim=1)
        x_t = x_t + temp.detach()
        x_t = torch.where((x[:, -1:, :, :]) > 0., x_t, x_t.detach())
        x_t = torch.where((mask > .5), x_t, temp)

        weights = self.smap_layer(torch.cat([x_t, (mask)], dim=1), target_2Dr, zoom)
        return phi, gamma, rho, mask, weights


# ----------------------------- loss (verbatim from the notebook) -----------------------------
def l1_error(pred, target):
    shapes = target.size()
    m_batch_size, height, width = shapes[0], shapes[-2], shapes[-1]
    pred_m = (torch.max((pred > specials.OFF_THRESH).reshape(m_batch_size, -1, height * width), dim=1, keepdim=True).values).float()
    abs_pred = torch.abs(pred)
    pred = ((1. - 2e-3) * abs_pred.detach() + 1e-3 + (2. * ((pred) > 0.).float() - 1.) * (pred - pred.detach())).reshape(m_batch_size, -1, height * width)
    target_m = target.reshape(m_batch_size, 1, -1)
    loss = torch.abs(pred - target_m)
    loss_m = torch.abs(pred_m - target_m)
    denom = loss.reshape(m_batch_size, -1).sum(dim=1, keepdim=True)
    denom = torch.where(denom > 3e3, 3e-1 * denom, 3e-1 * denom + 1e3)
    loss = (loss.reshape(m_batch_size, -1) / denom.detach()).sum(dim=1)
    loss_m = loss_m.reshape(m_batch_size, -1).sum(dim=1)
    loss = torch.mean(loss_m.detach() + 3e3 * (loss - loss.detach()))
    return loss


def params_snapshot(model):
    return {
        "theta_x": float(model.theta[0, 0, 0, 0]),
        "theta_y": float(model.theta[0, 1, 0, 0]),
        "theta_z": float(model.theta[0, 2, 0, 0]),
        "phi": float(torch.tanh(3e-2 * model.phi)[0, 0] * np.pi),
        "gamma": float(torch.tanh(3e-2 * model.gamma)[0, 0] * np.pi),
        "rho": float(torch.tanh(3e-2 * model.rho)[0, 0] * np.pi),
    }


def main():
    assert DATA_DIR.exists(), f"Sky303 data not found at {DATA_DIR}"
    idx, n_in, n_tgt = pick_nontrivial_idx()
    bin_mask, repr2d, tgt2d, ind, tgt = load_sample(idx)
    print(f"[hdvo] sample idx={idx}  active_pts={n_in}  target_pts={n_tgt}  (batch_size=1)")
    print(f"[hdvo] n={N}  zoom={ZOOM}  grid={H // 2**ZOOM}x{W // 2**ZOOM}  k_const={K_CONST}  DEBUG_FLAG={utils.DEBUG_FLAG}")

    # batch_size = 1
    r_mask_b = bin_mask.unsqueeze(0).to(device)
    input_2Dr_b = repr2d.unsqueeze(0).to(device)
    target_2Dr_b = tgt2d.unsqueeze(0).to(device)
    m_bs = 1

    model = Model(CAMERA, N).to(device)
    model.train()
    model.smap_layer.smap3x3.k_const = K_CONST
    optimizer = optim.SGD(list(model.parameters()), lr=LR)

    # coarsen the target to the chosen zoom level (notebook's reshape/max)
    z = ZOOM
    target_2Dr = target_2Dr_b.reshape(m_bs, 1, H // (2 ** z), (2 ** z), W // (2 ** z), (2 ** z)).permute(0, 3, 5, 1, 2, 4).reshape(m_bs, (2 ** (z + z)), 1, H // (2 ** z), W // (2 ** z))
    target_2Dr = torch.max(target_2Dr, dim=1, keepdim=False).values

    # dumps land in cwd -> chdir into the artifact-data folder (consumed by exec.py)
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.npy"):
        f.unlink()
    os.chdir(out_dir)

    before = params_snapshot(model)
    optimizer.zero_grad()
    phi, gamma, rho, mask, weights = model(input_2Dr_b, r_mask_b, target_2Dr, z)
    loss = 1. * l1_error(weights, target_2Dr)
    loss.backward()
    optimizer.step()
    after = params_snapshot(model)

    dumps = sorted(p.name for p in out_dir.glob("*.npy"))
    print(f"[hdvo] forward out.shape={tuple(weights.shape)}  loss={float(loss):.6f}")
    print(f"[hdvo] ONE optimizer step applied (param deltas, before -> after):")
    for k in before:
        print(f"         {k:8s} {before[k]:+.6e} -> {after[k]:+.6e}   (d={after[k]-before[k]:+.3e})")
    print(f"[hdvo] HDVO dumps written: {len(dumps)} -> {out_dir}")
    print(f"        {dumps}")


if __name__ == "__main__":
    main()
