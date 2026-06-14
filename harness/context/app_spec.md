# SMap Visualization App — Specification

## Purpose
Interactive visualization demonstrating the SMap Markov chain mechanism for 3D reconstruction. Allows stepping through optimization, inspecting per-point gradients, meta-states, and transitions across zoom levels.

This specification assumes familiarity with `math_model.md`. Terms (S_A, y_flow, n_flow, Force 1/2/3, etc.) are defined there.

## Core Simplifications (for real-time performance)

The app uses a simplified SMap that preserves mathematical structure but removes batching and complex camera projection:
- Grid size: up to 16×32 (512 points maximum)
- Multi-scale levels: up to 3 (so up to 64 channels at coarsest)
- Camera: identity projection (2D distance = 3D projected distance)
- Convergence damper k_t: constant 0.1 (depth-dominant regime)
- Direct learnable state per point (no neural network `dnet`)
- Batch size: 1

## Data Model

**Per-point state** at original resolution (h, w) ∈ [0, H) × [0, W):
- `x, y`: coordinates in 3D (initially random small values)
- `z`: depth (initially random in (0, z_max_init])
- `m`: mask/activation (initially uniform in [-ε, ε])

**Derived at each zoom level l** (l ∈ {0, 1, ..., L}):
- Channel count: C_l = 4^l
- Spatial size: (H/2^l, W/2^l)
- Channel-to-original mapping:
  ```
  sub_r = channel_idx // 2^l
  sub_c = channel_idx % 2^l
  orig_r = position_h * 2^l + sub_r
  orig_c = position_w * 2^l + sub_c
  ```

**Target** T(h, w) ∈ {0, 1} at original resolution. At zoom level l:
- T_l(h, w) = max over block of 2^l × 2^l sub-pixels (max-pooling)

## Meta-State Classification (per point, per zoom view)

| State | Condition | Display color | Behavior |
|-------|-----------|---------------|----------|
| S_A | m > τ ∧ z > 0 | Green | Real 3D point — full neighbor participation |
| S_B | m > τ ∧ z ≤ 0 | Yellow | Active without depth — self-reference only |
| S_C | m ≤ τ ∧ z > 0 | Cyan | Inactive with depth — stochastic participation |
| S_D | m ≤ τ ∧ z ≤ 0 | Gray | No information — excluded |

τ = OFF_THRESH ≈ 0.01

## Per-Step Computation (all points updated in parallel)

For each optimization step:

**1. Multi-scale forward (zoom level L down to 1):**
- Route neighbors: each point classifies state, computes 3×3 projected distances, selects best neighbor (routing only, values unchanged at intermediate levels)
- Z-buffer: per-channel winner = argmin{z > 0} within 3×3 neighborhood
- Merge: 4 channels → 2×2 spatial (inverse quadtree)

**2. Final level (zoom=0):**
- Route with actual value propagation (gather from selected neighbors)
- Compute reachability: `pre_allow(p, n) = 𝟙(weight(n) > τ) × T(proj(n))`
- Compute flows: y_flow, n_flow, coord_flow (see math_model §4)
- Construct synthetic gradient:
  - `G_align` for positions with y_flow=1 and weight>0
  - `G_act` for positions with y_flow≠1 and mismatch
  - `G_spread` for positions with T=1, modulated by local density

**3. Update:**
- `point.state += lr × total_gradient`

**4. Learning rate auto-scaling** (for selected point):
- Compute lr such that |lr × G_selected| ≈ 1 grid cell
- `lr = cell_size / max(|G_selected|, ε)`
- User can override with manual lr

## Layout

```
┌────────────────────────────────────────────┐
│ Target Editor │ k_const ▼ │ n_levels ▼     │
├────────────────────────────────────────────┤
│                                            │
│           Main Canvas (zoom view)          │
│                                            │
│  Grid of points at current zoom level      │
│  Selected point: magenta thick border      │
│  Co-located channels: orange dashed border │
│  Gradient arrows on selected point         │
│                                            │
├────────────────────────────────────────────┤
│ Zoom: [L] ... [1] [0]  │  Channel: ▼       │
├────────────────────────────────────────────┤
│ Step: [◀] [142] [▶]    │  LR: [auto]       │
├────────────────────────────────────────────┤
│ Selected point inspector:                  │
│   Original pixel: (5, 12)                  │
│   State: x=3.2 y=-1.1 z=4.5 m=0.82         │
│   Meta-state: S_A                          │
│   y_flow=1  n_flow=3.2  coord_flow=(↗)     │
│   G_align=(0.02, 0.01)                     │
│   G_act=(0, 0)                             │
│   G_spread=(0.005, 0.01)                   │
│   Total G=(0.025, 0.02)                    │
├────────────────────────────────────────────┤
│ [Export State]  [Import State]             │
└────────────────────────────────────────────┘
```

## Interactions

**Point selection:** Click any point → becomes selected. Inspector shows all gradient components and flow values. Selection persists across steps (track even if point moves to adjacent cell).

**Co-located points:** At current (zoom level, position), all channels at that position are visible. Selected = magenta solid. Others = orange dashed. All map to same 2^l × 2^l block in original view.

**Step forward (+1):**
1. Save snapshot of all point states
2. Run one optimization step on all points
3. Update display
4. Re-compute lr for selected point if auto mode

**Step backward (-1):**
1. Restore previous snapshot
2. Allow user to adjust lr before re-stepping

**Zoom navigation:**
- Zoom level buttons: switch between L (coarsest), L-1, ..., 0 (finest)
- Channel dropdown: select specific channel at current zoom
- Main canvas shows grid at selected (zoom, channel)
- Selected point's original pixel is always highlighted regardless of zoom

## Visual Indicators

**Force arrows (on selected point):**
- Blue = Force 1 (Alignment): reinforces selection
- Red = Force 2 (Activation): toggles active/inactive (shown as ± sign)
- Green = Force 3 (Spreading): redistributes toward gaps

Arrow length ∝ gradient magnitude. Arrows combine to show total gradient.

**Flow signals (annotations on each point):**
- y_flow=1: green border (confirmed)
- y_flow>1: orange border (competing)
- y_flow=0: red border (isolated)
- n_flow: point opacity or size
- coord_flow: thin gray arrow pointing toward gap

## Export Format

When user wants to debug convergence, they export state as JSON:

```json
{
  "step": 142,
  "n_levels": 3,
  "k_const": 0.1,
  "grid_size": [8, 16],
  "lr": 0.001,
  "target": [[0,0,1,1,...], ...],
  "points": [
    {"orig_r": 3, "orig_c": 5, "x": 1.2, "y": -0.3, "z": 2.1, "m": 0.5}
  ],
  "selected": {"orig_r": 3, "orig_c": 5},
  "history": [
    {"step": 0, "selected_state": [...], "selected_gradient": [...]}
  ]
}
```

Exported state can be sent to an AI with the request: "This point hasn't converged after N steps. Is there a bug, or when will it converge?" The AI analyzes gradient history, checks meta-state transitions, and predicts convergence or identifies issue.

## Implementation Notes

**Stochastic aspects to seed for reproducibility:**
- Neighbor selection random draws (np.random.rand in k_const checks)
- Z-buffer jitter (torch.rand_like for breaking ties)
- Initial point state

**Performance:**
- All per-point computation vectorized
- Multi-scale forward: O(L × H × W) operations
- Not batched (single image)

**Known behavioral patterns** (to help users interpret visualization):
- At coarse zoom (large k): y_flow > 1 is common → mostly Force 2 (activation). Few points move, many toggle.
- At fine zoom (small k): y_flow = 1 more achievable → Force 1 (alignment) fires → points stabilize at correct positions.
- S_C points at k_const=0.1: participate ~10% — slow z warming expected (accumulates over epochs).
- Points in dense regions should flow toward sparse regions visibly (density coefficient enforces this direction).