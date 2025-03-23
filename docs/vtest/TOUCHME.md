# Technical Documentation: GradientFlowInteractive Tool

## Overview

**GradientFlowInteractive** is an interactive visualization and debugging tool for analyzing gradient flows in neural networks (specifically the SMap model). The tool supports two display modes:
- **IPyWidgets mode:** Designed for interactive usage in Jupyter Notebook.
- **WebView mode:** A standalone web interface built with Dash that automatically opens in your default browser.

The tool is built using a modular design following an MVC (Model–View–Controller) architecture, with an additional Facade layer that bundles together the components into a single, easy-to-use interface.

---

## Architecture

### 1. Model – `GradientModel`

**Responsibilities:**
- **Data Loading:** Loads gradient data, input representation, and target representation from disk.
- **Data Processing:** Flattens gradient matrices, builds layer and node mappings (including global indices, node positions, and color maps).
- **Flow Matrix Computation:** Computes inter-layer gradient flow matrices and sets dynamic threshold limits.

**Key Methods:**
- `load_data()`: Loads all required data files.
- `process_gradients()`: Processes activation gradients and computes global node indices.
- `compute_flow_matrices()`: Computes and returns flow matrices based on gradient flows.
- `set_flow_threshold_limits()`: Determines dynamic minimum and maximum thresholds for flow visualization.

---

### 2. View – `Visualizer`

**Responsibilities:**
- **Visualization Creation:** Generates interactive Plotly figures for:
  - Sankey diagram (representing gradient flows between nodes/layers).
  - Multiple heatmaps (showing gradients for selected layer, output layer, target representation, and editable input representation).

**Key Methods:**
- `create_sankey(current_layer, current_node_index, threshold_value, region_settings, selected_all_layers)`: Generates a Sankey diagram based on current selections.
- `create_all_heatmaps(current_node_index, current_layer, threshold_value, offset_params, region_settings)`: Returns a tuple of heatmaps for the different representations.
- Private helper methods:
  - `_create_heatmap_conv1()`: Generates the heatmap for the selected (input/conv1) layer.
  - `_create_heatmap_conv2()`: Generates the heatmap for the output (connected) layer.
  - `_create_heatmap_target()`: Generates the heatmap for the target representation.
  - `_create_heatmap_input()`: Generates the editable heatmap for the input representation.

---

### 3. Controller – `UIController`

**Responsibilities:**
- **UI Management:** Builds and arranges interactive ipywidgets (dropdowns, sliders, buttons, etc.).
- **Event Handling:** Sets up callbacks to update visualizations when the user changes settings (e.g., selected layer, node index, threshold, region selection, offsets).
- **Interaction:** Bridges user input and the visualization view, updating figures on interaction events.

**Key Methods:**
- `build_widgets()`: Instantiates and configures all UI components.
- `setup_callbacks()`: Registers all widget callbacks.
- `update_visualizations()`: Updates the Sankey diagram and heatmaps based on current user settings.
- `render_ui()`: Arranges the UI layout and displays it in Jupyter Notebook.

---

### 4. Facade – `GradientFlowInteractive`

**Responsibilities:**
- **Unified Interface:** Bundles together the model, visualizer, and controller into one single class.
- **Rendering Modes:** Provides separate methods to render the tool using ipywidgets (for Jupyter) or via a standalone Dash WebView.
- **Flexibility:** Allows users to select the preferred rendering mode without modifying the underlying code.

**Key Methods:**
- `render_ipywidgets()`: Renders the interactive UI using ipywidgets (suitable for Jupyter).
- `render_webview(port=8050)`: Launches a Dash web server to display the tool in a browser window, printing the local URL so users can manually open it if needed.

---

## Deployment & Usage

### Installation
1. Ensure you have the required Python libraries:
   - numpy, pickle, torchvision, plotly, ipywidgets, matplotlib, dash.
2. Install via pip if needed:
   ```bash
   pip install numpy torchvision plotly ipywidgets matplotlib dash
   ```

### Running the Tool

#### From Jupyter Notebook:
```python
from gradient_flow_interactive import GradientFlowInteractive

# Initialize the tool with the debug folder path
tool = GradientFlowInteractive("../../../tutorial/")

# Render the UI using ipywidgets (for Jupyter)
tool.render_ipywidgets()
```

#### Standalone WebView Mode:
```python
from gradient_flow_interactive import GradientFlowInteractive

# Initialize the tool with the debug folder path
tool = GradientFlowInteractive("../../../tutorial/")

# Render the tool in a standalone WebView using Dash
tool.render_webview(port=8050)
```
- When the WebView mode is invoked, the tool starts a Dash server on the specified port (default 8050), prints the local URL to the console, and attempts to automatically open the URL in your default web browser.

---

## Design Patterns Utilized

1. **Model–View–Controller (MVC):**
   - **Model:** `GradientModel` manages data loading and processing.
   - **View:** `Visualizer` handles visualization creation.
   - **Controller:** `UIController` manages user interactions and widget events.

2. **Facade Pattern:**
   - `GradientFlowInteractive` provides a simplified interface for initializing and rendering the tool.

3. **Observer Pattern:**
   - UI widget callbacks in `UIController` implement an observer-like pattern to update visualizations when user interactions occur.

---

## Future Enhancements

- **Extend Debugging Capabilities:**
  Allow editing of heatmap values beyond binary (0/1) and integrate a more advanced neural network for gradient debugging.

- **Enhanced Customization:**
  Support additional configuration options for visual styles and layout.

- **Modularization:**
  Consider splitting the module into separate files (e.g., model.py, view.py, controller.py) if the project grows further.

- **Testing and CI:**
  Add unit tests for individual components and set up continuous integration to ensure robustness during future changes.

---

This documentation outlines the internal structure, usage, and design philosophy behind **GradientFlowInteractive**. It serves as a reference for developers who maintain or extend the tool in the future.
