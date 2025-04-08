# Gradient Flow Visualization with Interactive Debugging Mode

## Table of Contents
- [1. Overview](#1-overview)
- [2. Main Features](#2-main-features)
  - [2.1. Gradient Flow Visualization](#21-gradient-flow-visualization)
  - [2.2. Interactive Debugging](#22-interactive-debugging)
- [3. Architecture](#3-architecture)
  - [3.1. Data Processing](#31-data-processing)
  - [3.2. Visualization](#32-visualization)
  - [3.3. User Interface (UI)](#33-user-interface-ui)
  - [3.4. Adapter Mechanism](#34-adapter-mechanism)
- [4. Workflow with Adapter](#4-workflow-with-adapter)
- [5. Usage Instructions](#5-usage-instructions)
  - [Environment Setup](#environment-setup)
  - [Running the Tool](#running-the-tool)
  - [Creating a Vnit-Test Case with vtest tool](#creating-a-vnit-test-case-with-vtest-tool)
- [6. Future Development](#6-future-development)
- [7. Contribution & Maintenance](#7-contribution--maintenance)


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
  - Support padding correction via the **Offset Adjustments** feature. 
  
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
  - Users must implement an Adapter to load the saved input, gradient information from `flow_info.pkl`, process it using their neural network, and store the computed gradient flow back into `flow_info.pkl`.
  - The Adapter ensures that the debugging tool is independent of any particular network architecture.

#### Example Adapter Implementation:
An example of Adapter from the [SMap](https://github.com/thienannguyen-cv/SMap) project. 

```python
import torch
from torch import nn
import numpy as np

class TestBot_In(nn.Module):
    def __init__(self, module=None, offset_in=0, name="in", connet2name="out"):
        super(TestBot_In, self).__init__()
        def get_activation_grad(name, connet2name="out"):
            def hook(module, grad_inputs, grad_outputs):
                if name is not None:
                    # Lấy grad_out: shape [N, C_out, H_out, W_out]
                    grad_out, grad_in = None, None
                    for grad in grad_inputs:
                        if grad is not None:
                            grad_in = grad
                    H_in, W_in = (grad_in.shape[-2]), (grad_in.shape[-1])
                    grad_in = (grad_in[:,:,offset_in,:,:]).reshape(H_in, W_in)
                    
                    self.testcase.activation_gradients[name] = grad_in.cpu().numpy()
                    
                    self.testcase.gradient_flows[(name, connet2name)] = None
                    import pickle
                    flow_info = {"activation_gradients": self.testcase.activation_gradients, 
                                 "gradient_flows": self.testcase.gradient_flows}
                    np.save(self.testcase.out_path+self.testcase.name+"_input_representation.npy", self.input_representation.reshape(H_in, W_in))
                    with open(self.testcase.out_path+self.testcase.name+"_flow_info.pkl", 'wb') as f:
                        pickle.dump(flow_info, f)
            return hook
        self.testcase = None
        self.module = NoneBot()
        if module is not None:
            self.module = module
        self.name = name
        self.connet2name = connet2name
        self.input_representation = None
        self.module.register_backward_hook(get_activation_grad(self.name, self.connet2name))
        
    def forward(self, x, mask):
        h, w = mask.shape[-2], mask.shape[-1]
        self.input_representation = mask.detach().cpu().numpy().reshape(h, w)
        return self.module(x)
```

*Inside an unit test function.*

```python
from tools.testing.vtest.vtest_types import *

# Initialize testing environment and a SMap3x3 instance
# ...

# Construct a test case with built-in testbot Adapters
testcase = TestCase(name=f"test_{test_type}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())

# Cover the input of the SMap3x3 instance with the testbot Adapter for hooking gradient information
input_repr_x = self.vtestcase.testbot_in(input_repr_x, input_mask)

# Run the neural instance to generate the data of the vnittest tool
weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])

# ...
```

## 4. Workflow with Adapter

1. **Edit and Save Input:**  
   Use the debug tool to modify the input heatmap and click **Save** to store the current input representation into the `%DEBUG_FOLDER%input_representation.npy` file (the destination can be changed with the **Save to:** textbox in the UI). 

2. **External Neural Network Execution:**  
   In a separate notebook or process within the `DEBUG_FOLDER`, run your neural network (with the Adapter attached in a similar way to setting traditional breakpoints as you can see in *vinittest* files, definied below, in the `${{ github.workspace }}/tests` folder) so that it processes the saved input, computes updated gradient flows, and writes the new data to `flow_info.pkl`.

3. **Render Updated Data:**  
   In the debug tool, click the **Render** button to reload the updated gradient information from the `DEBUG_FOLDER`. The visualizations (Sankey Diagram and Heatmaps) will then refresh to display the new data.

---

## 5. Usage Instructions

### Environment Setup

1. **Install from PyPI**
   
   You can install **vnittest** directly from PyPI using pip:

   ```bash
   pip install vnittest
   ```
   
2. **Configure DEBUG_FOLDER:**  
   Set the `DEBUG_FOLDER` variable (e.g., `"../tests/test_data/test_"`) to point to your data directory.
3. **Prepare Data Files:**  
   Ensure that `%DEBUG_FOLDER%flow_info.pkl`, `%DEBUG_FOLDER%input_representation.npy` and `%DEBUG_FOLDER%target_representation.npy` are located in the `DEBUG_FOLDER`.

### Running the Tool

1. **Launch the Debug Notebook:**  
   Windows:
   ```bash
   SET DEBUG_FOLDER="../tests/test_data/test_"
   vnittest %DEBUG_FOLDER%
   ```
      
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

### Creating a Vnit-Test Case with vtest tool

1. **Extract Gradient Information** 
   - Prepare your pathology input and a debug folder `%DEBUG_FOLDER%`
   - Construct Adapters to "hook" gradient information of a pathology input without detroying the original structure of debug-related components. 
   - Convert the input tensor to numpy array and save it as `%DEBUG_FOLDER%input_representation.npy`.
   - Convert the target tensor to numpy array and save it as `%DEBUG_FOLDER%target_representation.npy`.
   - Save the gradient information as `%DEBUG_FOLDER%flow_info.pkl` in the format mentioned in [Section 3.1](#31-data-processing).

2. **Launch the Debug Notebook:**  
   Windows:
   ```bash
   vnittest %DEBUG_FOLDER%
   ```

3. **Visual Calibration**  
   - Fix bugs related to getting gradient information if present, as well as adjusting offsets related to padding. Make sure the tool's display reflects the bug's behavior.
   - Select cells with incorrect gradient values ​​to move the inner circle (red) to the position of this cell as a reference point.
   
4. **Generate Test Case's Output**  
   - Change the value in heatmap input to create the correct gradient values
   - Save gradient values to suitable tensors. 
5. **Create A Test Case**  
   Create a new test case in the `${{ github.workspace }}/tests` folder.
   
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
