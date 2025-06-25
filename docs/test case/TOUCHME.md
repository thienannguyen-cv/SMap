# TOUCHME.md

## Table of Contents

- [Test Case Overview](#test-case-overview)
- [File: `utest_smap.py`](#file-utest_smappy)
  - [test_prepare_flows_for_coord](#test-case-test_prepare_flows_for_coord)
  - [test_prepare_flows_for_mask](#test-case-test_prepare_flows_for_mask)
  - [test_SMap_forward](#test-case-test_smap_forward)
- [File: `utest_smap3x3.py`](#file-utest_smap3x3py)
  - [test_to_3d3x3](#test-case-test_to_3d3x3)
  - [test_agg_factor_only](#test-case-test_agg_factor_only)
  - [test_agg_ind](#test-case-test_agg_ind)
- [File: `vtest_smap3x3.py`](#file-vtest_smap3x3py)
  - [test_in_x_1st_stage](#test-case-test_in_x_1st_stage)
  - [test_in_x_2st_stage](#test-case-test_in_x_2st_stage)
  - [test_in_y_1st_stage](#test-case-test_in_y_1st_stage)
  - [test_in_y_2st_stage](#test-case-test_in_y_2st_stage)
  - [test_in_r_1st_stage](#test-case-test_in_r_1st_stage)
  - [test_in_r_2st_stage](#test-case-test_in_r_2st_stage)
- [Summary](#summary)
- [Contribution Guidelines](#contribution-guidelines)

---

## Test Case Overview

The `${{ github.workspace }}/tests` directory is essential for ensuring the correctness, stability, and reliability of the SMap library. This document describes the meaning, detailed logic, and expected outputs for each test case, with illustrative examples and diagrams to help contributors and users understand the test coverage.

Tests are executed via:
```bash
python -m unittest discover -s tests -p "[vu]test*.py"
```

Main test files:
- `utest_smap.py`: Unit tests for the SMap class.
- `utest_smap3x3.py`: Unit tests for SMap3x3 and utility functions.
- `vtest_smap3x3.py`: Integration and gradient tests for SMap3x3 with real data.

---

## File: `utest_smap.py`

### Test Case: `test_prepare_flows_for_coord`

**Purpose:**  
Verifies that the `prepare_flows_for_coord` method correctly identifies valid and blocked flows based on the "allow matrix" and target.

**Typical Example & Expected Output:**

- **Scenario:**  
  - Image: 5x5, active pixel at (2,2).
  - Only the offset (1,1) is activated (center in 3x3).
- **Input:**
  - `weights`: 3x3x7x7 tensor, only `weights[1,1,3,3]=1`
  - `target`: 5x5, `target[2,2]=1`
- **Expected:**  
  - For offset matching the active point (offset 4), sum is 0 (blocked).
  - For other offsets: sum is 1 if within boundary, else 0.

**Diagram:**

```
Input (target):
[0 0 0 0 0]
[0 0 0 0 0]
[0 0 1 0 0]
[0 0 0 0 0]
[0 0 0 0 0]

Weights (offset 1,1 activated):
[0 0 0]
[0 1 0]
[0 0 0]

Output "actual" sum for each offset:
offset 4 (center): 0
other offsets: 1 (if not on edge), 0 otherwise
```

---

### Test Case: `test_prepare_flows_for_mask`

**Purpose:**  
Validates flow generation for a mask with multiple activated points and tests both matching and non-matching offset cases.

**Example & Expected Output:**

- **Scenario:**  
  - Image: 5x5, mask has two active points: (2,2) and (3,3).
  - Offset tested: (1,1).
- **Expected:**  
  - If offset == target_offset: number of valid flows = 8 (since one is blocked).
  - If offset != target_offset: number of valid flows = 9.

**Diagram:**

```
Mask (active points):
[0 0 0 0 0]
[0 0 0 0 0]
[0 0 1 0 0]
[0 0 0 1 0]
[0 0 0 0 0]

Valid flows: 8 if offset matches, 9 otherwise.
```

---

### Test Case: `test_SMap_forward`

**Purpose:**  
Ensures that the forward pass of SMap maps the activated mask position to the correct output after a geometric transformation.

**Example & Expected Output:**

- **Input:**  
  - mask: activated at (10,20)
- **Expected Output:**  
  - Only the projected position (computed via camera transform) is 1; all others are 0.

**Diagram:**

```
Input mask:
[... 0 ...]
... 
[... 1 ...]  (row 10, col 20)
... 
[... 0 ...]

Output:
[... 0 ...]
... 
[... 1 ...] (at projected location)
... 
[... 0 ...]
```

---

## File: `utest_smap3x3.py`

### Test Case: `test_to_3d3x3`

**Purpose:**  
Checks that `to_3d3x3` correctly converts a depth map to 3D coordinates for a 3x3 neighborhood.

**Example & Expected Output:**

- **Input:**  
  - depth_map: all zeros except `depth_map[2,3]=1000`
  - offsetx=1, offsety=1
- **Expected:**  
  - Output is the 3D coordinate for (2,3), calculated as `inv(camera) x [3,2,1].T x 1000`.

**Diagram:**

```
depth_map:
[0 0 0 ...]
[0 0 0 ...]
[0 0 0 1e3 ...]
[...]

3D output:
inv(camera) @ [3,2,1] * 1000
```

---

### Test Case: `test_agg_factor_only`

**Purpose:**  
Checks aggregation logic using only a factor; ensures that only activated points overwrite the default value.

**Example & Expected Output:**

- **Input:**  
  - `unfolded_depth_map[1,1,2,3]=100`
  - `factor=999`
- **Expected Output:**  
  - `expected[1,1,3,4]=100` (after offset adjustment)
  - All other positions = 999

**Diagram:**

```
Unfolded depth (only one active point):
offset (1,1) active at (2,3)
expected[1,1,3,4]=100
other positions=999
```

---

### Test Case: `test_agg_ind`

**Purpose:**  
Checks indexed aggregation for multi-channel tensors, ensuring correct overwrite by active indices.

**Example & Expected Output:**

- **Input:**  
  - `unfolded_depth_map[0,0,:,2,3]=[100,200,300,400]` (vector)
  - `ind`: one-hot at (2,3)
- **Expected:**  
  - `expected[:,2,3]=[100,200,300,400]`, other positions = factor.

**Diagram:**

```
Unfolded depth:
only [0,0,:,2,3] is active (multi-channel)

ind: one-hot for (2,3)

expected:
[:,2,3]=[100,200,300,400], rest=factor
```

---

## File: `vtest_smap3x3.py`

### Test Case: `test_in_x_1st_stage`

**Purpose:**  
Tests backward gradient propagation when shifting mask along x direction, ensuring gradients only at valid positions.

**Example & Expected Output:**

- **Input:**  
  - mask: active at (2,2)
  - target: shift up → (1,2)
- **Expected Output:**  
  - Gradient nonzero at (2,2), zero elsewhere.

**Diagram:**

```
Input mask:        Target:           Gradient:
[0 0 0]            [0 0 0]           [0 0 0]
[0 0 0]            [0 0 0]           [0 0 0]
[0 1 0]            [0 1 0]           [0 1 0]
```

---

### Test Case: `test_in_x_2st_stage`

**Purpose:**  
Tests gradients for two-stage shifts along x, ensuring final gradient is at the initial position if the chain of movements is valid.

**Example & Expected Output:**

- **Input:**  
  - mask: (2,2)
  - shift up → (1,2), then right → (1,3)
  - target: (1,3)
- **Expected Output:**  
  - Gradient at (2,2) if both shifts are valid.

**Diagram:**

```
Initial:        Stage 1:         Stage 2:
[0 0 0]         [0 0 0]          [0 0 0]
[0 0 0]         [0 1 0]          [0 0 1]
[0 1 0]         [0 0 0]          [0 0 0]

Gradient:
[0 0 0]
[0 0 0]
[0 1 0]
```

---

### Test Case: `test_in_y_1st_stage`

**Purpose:**  
Tests backward gradient propagation when shifting mask along y direction, ensuring gradients only at valid positions (e.g. moving left/right).

**Example & Expected Output:**

- **Input:**  
  - mask: active at (2,2)
  - target: shift right → (2,3)
- **Expected Output:**  
  - Gradient nonzero at (2,2), zero elsewhere.

**Diagram:**

```
Input mask:        Target:           Gradient:
[0 0 0]            [0 0 0]           [0 0 0]
[0 0 0]            [0 0 0]           [0 0 0]
[0 1 0]            [0 0 1]           [0 1 0]
```

---

### Test Case: `test_in_y_2st_stage`

**Purpose:**  
Tests gradients for two-stage shifts along y direction (e.g. left then right), ensuring correct gradient at the initial position.

**Example & Expected Output:**

- **Input:**  
  - mask: (2,2)
  - shift right → (2,3), then up → (1,3)
  - target: (1,3)
- **Expected Output:**  
  - Gradient at (2,2) if both shifts are valid.

**Diagram:**

```
Initial:        Stage 1:         Stage 2:
[0 0 0]         [0 0 0]          [0 0 0]
[0 0 0]         [0 0 0]          [0 1 0]
[0 1 0]         [0 0 1]          [0 0 0]

Gradient:
[0 0 0]
[0 0 0]
[0 1 0]
```

---

### Test Case: `test_in_r_1st_stage`

**Purpose:**  
Tests backward gradient propagation for mask (r), ensuring gradients are only at valid positions, including the "still" (no movement) case.

**Example & Expected Output:**

- **Input:**  
  - mask: (2,2)
  - test_type: "still"
- **Expected Output:**  
  - Gradient all zero, since no movement.

**Diagram:**

```
Input:           Target:           Gradient:
[0 0 0]          [0 0 0]           [0 0 0]
[0 0 0]          [0 0 0]           [0 0 0]
[0 1 0]          [0 1 0]           [0 0 0]
```

---

### Test Case: `test_in_r_2st_stage`

**Purpose:**  
Tests backward gradient propagation for two-stage mask (r) transformations, ensuring correct gradient at the initial position depending on movement sequence.

**Example & Expected Output:**

- **Input:**  
  - mask: (2,2)
  - first move: right → (2,3), second move: up → (1,3)
  - target: (1,3)
- **Expected Output:**  
  - Gradient at (2,2) if both moves are valid; all zero if second move is "still".

**Diagram:**

```
Initial:        Stage 1:         Stage 2:
[0 0 0]         [0 0 0]          [0 0 0]
[0 0 0]         [0 0 0]          [0 1 0]
[0 1 0]         [0 0 1]          [0 0 0]

Gradient (if both moves valid):
[0 0 0]
[0 0 0]
[0 1 0]

Gradient (if second move is "still"):
[0 0 0]
[0 0 0]
[0 0 0]
```

---

## Summary: 

### Effectiveness and Completeness

The current test suite offers thorough coverage for the main components and behaviors of the SMap library:

- **Core Functionality:**  
  The unit tests in `utest_smap.py` and `utest_smap3x3.py` validate the correctness of fundamental operations, including geometric transformations, aggregation, and mask/flow preparation. These are the backbone of the spatial mapping logic, ensuring the mathematical and logical integrity of each step.

- **Boundary and Edge Cases:**  
  The tests systematically address both typical and edge cases, such as offsets at boundaries, multiple mask activations, and "blocked" scenarios. This helps catch subtle bugs that could arise in real-world usage.

- **Gradient and Integration Tests:**  
  The `vtest_smap3x3.py` suite includes integration and gradient propagation checks, crucial for validating the library’s use in learning-based or differentiable settings (e.g., deep learning pipelines). These tests confirm that gradients are properly routed through the spatial transformations, which is essential for any trainable model built on top of SMap.

- **Compositionality:**  
  The two-stage movement tests (`*_2st_stage`) and various direction-based tests ensure that the system behaves correctly under sequential operations and multi-step manipulations, reflecting realistic spatial mapping tasks.

---

## Contribution Guidelines

- When adding new tests, include a concrete example, expected output, and a simple diagram.
- Please comment or document the test logic for future maintainers.
- For questions or issues, contact the maintainer or open a GitHub issue.

---