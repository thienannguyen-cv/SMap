# SMap Knowledge Graph

This knowledge graph is designed to help readers understand the functional structure of `smap.py`, the relationships between its core components, and how the unit-tests relate to the module's correctness. The graph demonstrates how data and computation flow through the system, highlighting both the architectural logic and the verification points via test cases.

------------------------------------------------------------------------

## 1. Key Components and Functions in `smap.py`

**Nodes:** - **SMap (nn.Module)** - Main entry point for spatial mapping operations. - **SMap3x3 (nn.Module)** - Core for 3x3 neighborhood processing. - **flip (function)** - Utility for tensor dimension flipping. - **compute_allow_matrix (method, DefaultRectify)** - Computes allowed spatial flows. - **prepare_flows_for_mask (method,DefaultRectify)** - Prepares flows for mask-based activation. - **prepare_flows_for_coord (method, DefaultRectify)** - Prepares flows for coordinate-based activation. - **calculate_weights (method, SMap)** - Aggregates and selects weights for output. - **rectificate_flow (method,** DefaultRectify**)** - Applies rectification logic and computes final flows and gradients. - **forward (method, SMap and SMap3x3)** - Performs the main forward/inference pass. - **calculate_key_query (method, SMap3x3)** - Calculates matching between keys and queries for 3D mapping. - **go (method, SMap3x3)** - Adds padding and prepares batch for processing. - **utils, specials** - Utility and special constant modules.

------------------------------------------------------------------------

## 2. Knowledge Graph Diagram (Textual Form)

```{mermaid}
graph TD
    subgraph "SMap Main Execution"
        direction TB
        SMap_forward["SMap.forward (Entry Point)"]
    end

    subgraph "Step 1: Core Projection (SMap3x3)"
        direction TB
        SMap3x3_go["SMap3x3.go()"] --> SMap3x3_forward["SMap3x3.forward()"]
        SMap3x3_forward --> SMap3x3_calc_kq["calculate_key_query()"]
        SMap3x3_calc_kq --> utils_to_3d3x3["utils.to_3d3x3"]
    end

    subgraph "Step 2: Flow Rectification (rectify.py)"
        direction TB
        rectify_module_flow["rectify_module.rectificate_flow (DefaultRectify/CAMRectify)"]
        rectify_module_flow --> compute_allow["compute_allow_matrix()"]
        rectify_module_flow --> prepare_mask["prepare_flows_for_mask()"]
        rectify_module_flow --> prepare_coord["prepare_flows_for_coord()"]
    end

    subgraph "Step 3: Weight Calculation (SMap)"
        direction TB
        SMap_calc_weights["SMap.calculate_weights()"]
    end

    subgraph "Utilities"
        utils_agg["utils.agg"]
        utils_flip["utils.flip"]
    end

    %% Main sequential flow
    SMap_forward --> SMap3x3_go
    SMap_forward --> rectify_module_flow
    SMap_forward --> SMap_calc_weights

    %% Utility dependencies
    compute_allow -- uses --> utils_agg & utils_flip
    prepare_mask -- uses --> utils_agg
    prepare_coord -- uses --> utils_agg
    SMap_calc_weights -- uses --> utils_agg

    %% Styling
    style SMap_forward fill:#c9f,stroke:#333,stroke-width:2px;
    style SMap3x3_go fill:#bbf,stroke:#333,stroke-width:2px;
    style rectify_module_flow fill:#bfa,stroke:#333,stroke-width:2px;
    style SMap_calc_weights fill:#f9d,stroke:#333,stroke-width:2px;
    classDef utility fill:#f8f4a6,stroke:#333,stroke-width:1px;
    class utils_to_3d3x3,utils_agg,utils_flip utility;
```

### Explanation

-   **SMap** is the main orchestrator, calling into SMap3x3 for local (3x3) spatial reasoning.
-   **SMap3x3.forward** depends on `calculate_key_query` for key-query matching, which itself uses `utils.to_3d3x3`.
-   The **rectification** and **flow preparation** logic are modular, with `compute_allow_matrix`, `prepare_flows_for_mask`, and `prepare_flows_for_coord` all relying on shared utility functions.
-   The **weight calculation** uses aggregation and selection logic, and is crucial for producing the final output map.

------------------------------------------------------------------------

## 3. Test Case Mapping on the Graph

-   **test_prepare_flows_for_coord** (utest_smap.py)
    -   Validates: `SMap.prepare_flows_for_coord`, indirectly `utils.agg`
    -   Ensures correct flow field for given coordinate activations.
-   **test_prepare_flows_for_mask** (utest_smap.py)
    -   Validates: `SMap.prepare_flows_for_mask`, indirectly `utils.agg`
    -   Ensures correct flow field when activating via mask.
-   **test_SMap_forward** (utest_smap.py)
    -   Validates: `SMap.forward` (main logic), `SMap3x3.go`, `SMap3x3.forward`, `calculate_weights`
    -   Ensures spatial mapping from input mask to output is geometrically correct.
-   **test_to_3d3x3** (utest_smap3x3.py)
    -   Validates: `utils.to_3d3x3`, used in `SMap3x3.calculate_key_query`
    -   Ensures correct 3D coordinate transformation.
-   **test_agg_factor_only**, **test_agg_ind** (utest_smap3x3.py)
    -   Validate: `utils.agg`
    -   Ensure aggregation logic for weights and indexing works as intended.
-   \*\*test_in\_\*\_stage\*\* (vtest_smap3x3.py)
    -   Validate: Gradient propagation and correctness in `SMap.rectificate_flow`, `SMap3x3.forward`
    -   These tests check that not only the forward but also the backward (gradient) logic is correct under various spatial scenarios (single and two-stage moves, x/y/r directions).

------------------------------------------------------------------------

## 4. How the Tests Validate Implementation

-   **Direct Coverage:** Every computational path in `smap.py` that affects the core "spatial mapping" and "flow field" logic is directly tested by at least one unit test.
-   **Backward Passes:** The vtest_smap3x3.py tests simulate learning scenarios, checking that gradients (required for optimization and learning-based applications) are routed properly.
-   **Edge Cases:** Tests specifically target "blocked", "random", "still", multi-stage, and out-of-bounds scenarios.
-   **Geometric Consistency:** By exercising different input masks and transformations, the tests ensure that both the geometric and logical expectations are met.

------------------------------------------------------------------------

## 5. Summary Table

| Component/Function | Directly Tested By | Test Purpose |
|------------------|-------------------------|-----------------------------|
| SMap.forward | test_SMap_forward, vtest_smap3x3.py | Global correctness of spatial mapping |
| SMap3x3.forward | test_SMap_forward, vtest_smap3x3.py | 3x3 local mapping logic |
| SMap3x3.calculate_key_query | test_to_3d3x3, indirectly via forward | Key-query computation, 3D logic |
| DefaultRectify.compute_allow_matrix | test_prepare_flows_for_coord/mask | Correct flow mask logic |
| DefaultRectify.prepare_flows_for_coord | test_prepare_flows_for_coord | Flow logic for coordinates |
| DefaultRectify.prepare_flows_for_mask | test_prepare_flows_for_mask | Flow logic for mask activations |
| SMap.calculate_weights | test_SMap_forward, test_agg\_\* | Aggregation and selection logic |
| rectificate_flow | vtest_smap3x3.py | Gradient, learning consistency |
| utils.agg | test_agg_factor_only, test_agg_ind | Core aggregation utility logic |
| utils.to_3d3x3 | test_to_3d3x3, SMap3x3.calculate_key_query | 3D transformation utility |

------------------------------------------------------------------------

## 6. Visual Overview

```{mermaid}
graph TD
  A[SMap.forward] --> B[SMap3x3.go & .forward]
  B --> C[calculate_key_query]
  B --> D[utils.to_3d3x3]
  A --> E[calculate_weights]
  E --> F[utils.agg]
  A --> G[DefaultRectify.rectificate_flow]
  G --> H[DefaultRectify.compute_allow_matrix]
  G --> I[DefaultRectify.prepare_flows_for_mask]
  G --> J[DefaultRectify.prepare_flows_for_coord]
  H --> F
  I --> F
  J --> F
  F --> K[utils / specials]
```

**Test coverage (overlay):** - \[utest_smap.py\] covers: A, H, I, J, E - \[utest_smap3x3.py\] covers: D, F, C - \[vtest_smap3x3.py\] covers: A, B, G, gradient flows

------------------------------------------------------------------------

# Principles

Essentially, the optimization process of SMap can be modeled as a Markov chain of two states, one **Recurrent State** and one **Steady State**. We can determine if the current status of the optimization process is in **Recurrent State** by checking the values of matrices returned by the functions `prepare_flows_for_mask` and `prepare_flows_for_coord`. If the current status of the optimization process is not in **Recurrent State**, it is in **Steady State**. Thus, we have the following principles: "The optimization process of SMap is only in **Steady State** if and only if for each active point in the target, there is only one active point in the output of SMap at the same location (i.e., one-to-one mapping). Otherwise, the optimization process is in **Recurrent State**."  

## DefaultRectify

### Steady State
- **Mask:** Only one active point across for each active point in the target. 
- **Coordinate:** The proper 2D representation of each active point in the target is exactly the same as the coordinate of the corresponding active point in the output.

### Recurrent State
- **Mask:** Zero or more than one active point across for each active point in target. And, there no non-steady active point in the target around an non-active point in the ouput. 
- **Coordinate:** The rest of cases that are not in Steady State. And, the case of a "non-steady" active point in the target around an non-active point in the ouput.

------------------------------------------------------------------------
