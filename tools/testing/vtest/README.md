# Gradient Flow Visualization with Interactive Debugging Mode

## 1. Overview

This project extends the gradient flow visualization tool for the **SMap model** by adding an **Interactive Debugging Mode**. In addition to displaying gradient flows between layers using a Sankey-style diagram, the tool now allows users to directly modify the heatmap values (limited to 0 or 1) for visual debugging of gradients.

## 2. Main Features

### 2.1. Gradient Flow Visualization
- **Display Gradient Flow**:  
  - Uses a Sankey diagram to represent gradient flows between layers.
- **Gradient Data Processing**:  
  - Extracts, flattens, and normalizes gradients stored in the `flow_info.pkl` file.
  
### 2.2. Interactive Debugging
- **Input-Editing Mode**:  
  - Users can click on individual heatmap cells to toggle values between 0 and 1, which serves as a visual debugging tool for gradients.
- **Save Input**:  
  - A **Save** button (visible only in Input-Editing mode) allows users to persist their modified input values.
- **Gradient-Debugging Mode**:  
  - A **Gradient Debug** button triggers the execution of a neural network (designed to return files of gradient-flow information according to a predefined standard) that take the saved input/target values as the input/target for gradient computing.  
  - The network generates new gradients, which are then visualized in a similar manner to the original gradient flow.
  - During the debugging process, the Input-Editing mode is temporarily disabled to prevent conflicts.
- **State Preservation**:  
  - After debugging, the previously saved heatmap values remain visible, allowing further adjustments and iterative debugging.

## 3. System Architecture

### 3.1. Data Processing
- **Gradient Extraction**:  
  - Computes gradients for each layer and saves them in the `flow_info.pkl` file.
- **Data Transformation**:  
  - Flattens 2D gradients into 1D vectors.

### 3.2. Visualization
- **Sankey Diagram**:  
  - Each node represents a pixel or neuron, and each link represents the gradient flow between layers.
  - Utilizes Plotly for rendering with fixed properties like positions, colors, and dynamic thresholds for gradient filtering.
- **Heatmaps**:  
  - Displays gradient values for selected layers.
  - In Input Editting mode, the heatmap allows direct interaction to edit the underlying input values.

### 3.3. User Interface (UI)
- **ipywidgets**:  
  - Provides interactive components such as dropdowns (for layer selection), sliders (for node selection and threshold adjustment), and buttons (for Save and Debug).
- **Event Handling**:  
  - Any change in the UI (layer, node, threshold, or heatmap edits) triggers an update of the Sankey diagram and heatmaps.

### 3.4. Debugging Module
- **Interactive Debugging Mode**:
  - **Input Editing Mode**:  
    - Allows toggling heatmap values between 0 and 1.
  - **Save Button**:  
    - Saves the modified heatmap values to maintain state.
  - **Gradient Debug Button**:  
    - Executes a standardized neural network model that takes the saved heatmap values as input and generates new gradients.
    - Updates the gradient flow visualization accordingly.
  - **Mode Management**:  
    - During debugging, the editing functionality is disabled to ensure data consistency.

## 4. Implementation Details

### 4.1. Language & Libraries
- **Programming Language**: Python
- **Key Libraries**:
  - **PyTorch**: For computing and extracting gradients.
  - **Plotly**: For visualizing the Sankey diagram and heatmaps.
  - **ipywidgets**: For building the interactive user interface.
  - **NumPy, torchvision, matplotlib**: For data processing and color mapping.

### 4.2. Code Structure
- **Gradient Extraction & Processing**:  
  - A script to load and process data from `flow_info.pkl`.
  - Calculation of flow matrices with dynamic thresholding.
- **Visualization Module**:  
  - Functions to generate the Sankey diagram and heatmaps.
- **Interactive Debugging Module**:  
  - UI components for editing heatmaps, saving state, and triggering debugging.
  - A standardized neural network interface that accepts edited heatmap values and produces new gradients.
  - State management to ensure the heatmap values persist across debugging sessions.

## 5. Usage Instructions

1. **Environment Setup**:  
   - Install the necessary libraries:  
     ```
     pip install torch torchvision plotly ipywidgets numpy matplotlib
     ```

2. **Run the Application**:  
   - Launch a Jupyter Notebook or execute the provided Python script.

3. **Using Gradient Flow Visualization**:  
   - Use the dropdown to select a layer and the sliders to choose a node and adjust the threshold.
   - Observe the updated Sankey diagram and heatmap reflecting the original gradient data.

4. **Using Interactive Debugging Mode**:
   - Activate the Interactive Debugging mode (e.g., via a toggle or dedicated button).
   - Modify the heatmap values by clicking on individual cells to toggle between 0 and 1.
   - Click the **Save** button to store the modified heatmap values.
   - Click the **Gradient Debug** button to run the standardized neural network with the saved heatmap values and update the gradient visualization.
   - Note: During the debugging process, the editing mode will be disabled to ensure consistency.
   - After debugging, the saved heatmap values remain visible for further adjustments.

## 6. Future Development

- **Extended Debugging Capabilities**:  
  - Support for modifying heatmap values beyond binary (0/1) to fine-tune sensitivity.
  - Integration of real-time feedback during model training to provide deeper insights into gradient behavior.
  
- **Standardized Neural Network Interface**:  
  - Define a clear interface for neural network debugging, facilitating the integration and replacement of models without altering the overall architecture.

## 7. Contribution & Maintenance

- **Contribution**:  
  - Feedback, improvements, and bug reports are welcome.
  - Please submit pull requests or open issues on the repository.
- **Maintenance**:  
  - The modular design facilitates easy maintenance, extension, and debugging of the tool.
