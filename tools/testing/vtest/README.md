# Gradient Flow Visualization with Interactive Debugging Mode

## 1. Overview
A tool for visually debugging gradient flows between convolutional layers of neural networks. 

## 2. Main Features
What are supported? 

### 2.1. Gradient Flow Visualization
- **Display Gradient Flow**:  
  - Specify a z value in the "Z:" textbox.
  - Uses a Sankey diagram to represent gradient flows between layers.
- **Interactive UI for visual gradient debugging**:  
  - Use heatmaps for displaying gradient values. 
  - Highlights points that have gradients to a selected node in the convolutional layer (input). 
  - Support padding correction via the Offset Adjustments feature. 
  
### 2.2. Interactive Debugging
- **Input-Editing Mode**:  
  - Users can click on individual heatmap cells to toggle values between 0 and z, which serves as a visual debugging tool for gradients.
- **Save Input**:  
  - The **Save** button allows users to store the modified input values in `flow_info.pkl`, which will later be processed by [Adapters](https://refactoring.guru/design-patterns/adapter) to generate gradient flow information.
- **Gradient-Debugging Workflow**:  
  - Users run an external neural network (with the Adapter attached) in the `DEBUG_FOLDER` to compute new gradient flows using the saved input.
  - Clicking the **Render** button reloads the updated gradient information, reflecting the new computed gradients.
- **State Preservation**:  
  - After debugging, the previously saved heatmap values remain visible, allowing further adjustments and iterative debugging.

## 3. Architecture

### 3.1. Data Processing
- **Gradient Extraction**:  
  - Computes gradients for each layer and stores them in a Python dict, which is eventually saved to a pickle file with the (same) name `flow_info.pkl`, of the following format: 
   
   ```python
   { 'activation_gradients': { 'conv2': array([[15.879998, 15.899999], [15.899999, 15.929999]], dtype=float32), 'conv3': array([[4.]], dtype=float32), 'conv1': array([[3.97, 3.98, 3.97, 3.97], [3.98, 4. , 3.98, 3.98], [3.97, 3.98, 3.97, 3.97], [3.97, 3.98, 3.97, 3.97]], dtype=float32) }, 'gradient_flows': { ('conv2', 'conv3'): array([[1.], [1.], [1.], [1.]], dtype=float32), ('conv1', 'conv2'): array([ [1. , 0.99, 0.99, 0.99], [1. , 1. , 0.99, 0.99], [0.99, 1. , 0.99, 0.99], [0.99, 1. , 0.99, 0.99], [1. , 0.99, 1. , 0.99], [1. , 1. , 1. , 1. ], [0.99, 1. , 0.99, 1. ], [0.99, 1. , 0.99, 1. ], [0.99, 0.99, 1. , 0.99], [0.99, 0.99, 1. , 1. ], [0.99, 0.99, 0.99, 1. ], [0.99, 0.99, 0.99, 1. ], [0.99, 0.99, 1. , 0.99], [0.99, 0.99, 1. , 1. ], [0.99, 0.99, 0.99, 1. ], [0.99, 0.99, 0.99, 1. ] ], dtype=float32) } }
   ```

   Accordingly,

   `flow_info['activation_gradients']`: A Python `dict` object where each key ("conv1", "conv2", ...) is the name you give to the neural network's layers and its content is a 2d tensor of the gradients at that layer (results of [user-defined backward hook](https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution) procedures or through accessing `.grad` property of pytorch tensors).

   `flow_info['gradient_flows']`: A `dict` object where each key is a tuple, respectively, of the names of input and output layers representing a set of gradient flows, for node pairs connecting two consecutive layers, and its content is a 2-dimensional tensor whose first dimension is the number of nodes in the input layer and the second dimension is the number of nodes in the output layer.
- **Flow Matrix Computation**:  
  - Uses values from `gradient_flows`.  
  - If a flow is undefined (set to `None`), it is interpolated using the available gradient data from `activation_gradients` to maintain consistency with chain rule calculations.

### 3.2. Visualization
- **Sankey Diagram**:  
  - Each node represents a pixel or neuron, and each link represents the gradient flow between layers.
  - Utilizes Plotly for rendering with fixed properties like positions, colors, and dynamic thresholds for gradient filtering.
- **Heatmaps**:  
  - Displays gradient values for selected layers.
  - In Input Editing mode, the heatmap allows direct interaction to edit the underlying input values.

### 3.3. User Interface (UI)
- **ipywidgets**:  
  - Provides interactive components such as dropdowns (for layer selection), sliders (for node selection and threshold adjustment), and buttons (for Save and Render).
- **Event Handling**:  
  - Any change in the UI (layer, node, threshold, or heatmap edits) triggers an update of the Sankey diagram and heatmaps.

### 3.4. Adapter Mechanism
- **Standardized Adapter Interface**:  
  - Users must implement an Adapter to load the saved input from `flow_info.pkl`, process it using their neural network, and store the computed gradient flow back into `flow_info.pkl`.
  - The Adapter ensures the debugging tool remains independent of specific network architectures.

#### Example Adapter Implementation:

```python
import torch

class DebugAdapter:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def compute_gradients(self, input_path, output_path):
        # Load input from file
        input_data = torch.load(input_path).to(self.device)
        
        # Ensure gradients are tracked
        input_data.requires_grad = True
        
        # Forward pass
        output = self.model(input_data)
        
        # Compute gradients
        output.backward(torch.ones_like(output))
        
        # Extract and save gradients
        gradients = input_data.grad.cpu().detach().numpy()
        torch.save(gradients, output_path)
```

## 4. Usage Workflow with the Adapter

1. **Edit and Save Input:**  
   Use the debug tool to modify the input heatmap and click **Save** to store the current input representation into the `flow_info.pkl` file.

2. **External Neural Network Execution:**  
   In a separate notebook or process within the `DEBUG_FOLDER`, run your neural network (with the Adapter attached in a similar way to setting traditional breakpoints as you can see in *vinittest* files, definied below, in the `${{ github.workspace }}/tests` folder) so that it processes the saved input, computes updated gradient flows, and writes the new data to `flow_info.pkl`.

3. **Render Updated Data:**  
   In the debug tool, click the **Render** button to reload the updated gradient information from the `DEBUG_FOLDER`. The visualizations (Sankey Diagram and Heatmaps) will then refresh to display the new data.

---

## 5. Usage Instructions

### Environment Setup

1. **Install Dependencies:**
   ```bash
   pip install numpy==1.19.5 plotly==5.12.0 ipywidgets==7.6.5 matplotlib==3.3.4 --force-reinstall
   ```
2. **Configure DEBUG_FOLDER:**  
   Set the `DEBUG_FOLDER` variable (e.g., `"../../../tests/output"`) to point to your data directory.
3. **Prepare Data Files:**  
   Ensure that `flow_info.pkl`, `input_representation.npy` and `target_representation.npy` are located in the DEBUG_FOLDER.

### Running the Tool

1. **Launch the Debug Notebook:**  
   The UI displays:
   - **Sankey Diagram**
   - **Control Widgets:** Layer Dropdown, Node Slider, and Threshold Slider.
   - **Heatmaps:** For Conv1, Conv2, and Target.
   - **Input Section:** Input Heatmap along with Render and Save buttons.

2. **Modify the Input:**  
   Click on the Input Heatmap to toggle cell values (0 ↔ 1). The Save button becomes enabled upon modification.

3. **Save Input Changes:**  
   Click the **Save** button to save the modified input representation to `flow_info.pkl`. (This action only updates the file; it does not trigger gradient computation.)

4. **Run the Neural Network Externally:**  
   In a separate notebook or process within the DEBUG_FOLDER, run your neural network (with the Adapter attached) so that it processes the saved input and writes updated gradient information to file.

5. **Render Updated Data:**  
   Click the **Render** button in the debug tool to reload the updated gradient information from the DEBUG_FOLDER. The Sankey Diagram and Heatmaps will refresh accordingly. After rendering, the Save button is disabled until new input modifications occur.

6. **Interact and Inspect:**  
   Use the control widgets to select different layers, nodes, and thresholds. The visualizations update automatically.

### Creating a Gradient Test Case (vnittests)

---

## 6. Future Development

- **Extended Debugging Capabilities:**  
  - Integration of real-time gradient feedback during training.
  
- **Enhanced Adapter Interface:**  
  - Further standardize the Adapter to support various network architectures seamlessly.

---

## 7. Contribution & Maintenance

- **Contribution:**  
  Contributions, feedback, and bug reports are welcome. Please submit pull requests or open issues on the repository.
  
- **Maintenance:**  
  The modular design facilitates easy updates and extensions for future debugging needs.
