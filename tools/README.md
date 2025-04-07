# Tools
Available tools and user guide overview. 

## Vtest
A tool (the first of its kind at the time of its release) for visually debugging gradient flows between layers of neural networks (especially for convolutional layers). 

### Workflow
1. **Compute Gradients**: 
   - Given an input image, compute the gradients through the network.
   - The gradients/gradient flows are stored in the following format in a Python dict, which is eventually saved to a pickle file with the (same) name `flow_info.pkl`: 
   
   ```python
   { 'activation_gradients': { 'conv2': array([[15.879998, 15.899999], [15.899999, 15.929999]], dtype=float32), 'conv3': array([[4.]], dtype=float32), 'conv1': array([[3.97, 3.98, 3.97, 3.97], [3.98, 4. , 3.98, 3.98], [3.97, 3.98, 3.97, 3.97], [3.97, 3.98, 3.97, 3.97]], dtype=float32) }, 'gradient_flows': { ('conv2', 'conv3'): array([[1.], [1.], [1.], [1.]], dtype=float32), ('conv1', 'conv2'): array([ [1. , 0.99, 0.99, 0.99], [1. , 1. , 0.99, 0.99], [0.99, 1. , 0.99, 0.99], [0.99, 1. , 0.99, 0.99], [1. , 0.99, 1. , 0.99], [1. , 1. , 1. , 1. ], [0.99, 1. , 0.99, 1. ], [0.99, 1. , 0.99, 1. ], [0.99, 0.99, 1. , 0.99], [0.99, 0.99, 1. , 1. ], [0.99, 0.99, 0.99, 1. ], [0.99, 0.99, 0.99, 1. ], [0.99, 0.99, 1. , 0.99], [0.99, 0.99, 1. , 1. ], [0.99, 0.99, 0.99, 1. ], [0.99, 0.99, 0.99, 1. ] ], dtype=float32) } }
   ```

   Accordingly,

   `flow_info['activation_gradients']`: A dictionary where each key is the name you give to the layers and its content is a 2d tensor of the gradients at that layer (results of [user-defined backward hook](https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution) procedures or through accessing the `.grad` property of the tensors).

   `flow_info['gradient_flows']`: A dictionary where each key is a tuple, respectively, of the names of two input and output layers representing a set of gradient flows connecting two consecutive layers and its content is a 2-dimensional tensor whose first dimension is the number of nodes in the input and the second dimension is the number of nodes in the output.
   
3. **Flattening & Projection**:
   - Convert 2D gradient maps into 1D representations.
   - Map these 1D values to appropriate nodes in a Sankey diagram.

4. **Constructing the Sankey Diagram and Heatmaps**:
   - The first layer consists of individual input pixels.
   - Each subsequent layer contains neurons (nodes), with connections representing the gradient flow intensity.
   - The gradients at the nodes are projected onto two heatmaps named **Selected-Layer Gradient** and **Output-Layer Gradient**, corresponding to the input and output layers, respectively.
   - Data from two numpy arrays from files named `input_representation.npy` and `target_representation.npy` will be displayed on heatmaps named **Input Representation** and **Target Representation** respectively. These heatmaps will play a role in evaluating the gradients while generating gradient test cases.

5. **Rendering the Visualization**:
   - Use a Sankey diagram framework to generate the final visualization.
   - Each node represents a neuron or input pixel, and each edge represents the flow of gradients.
   - Displays gradient values for selected layers.
   - In Input Editing mode, the heatmap allows direct interaction to edit the underlying input values.
   - A highlight mechanism on selected cells is integrated to distinguish gradient flows between two points on the heatmaps.

### UI
The interface of vtest tool:

![Vtest UI](https://raw.githubusercontent.com/thienannguyen-cv/SMap/75b05ba93fc8fbf5f8effe5099a76bfa3d14b571/media/images/tools_vtest_interface.png)

### Implementation Details
- **Source**: ./testing/vtest/vtest_tutorial.ipynb
- **Language**: Python
- **Libraries**: PyTorch for gradient computation, Matplotlib + Plotly (4.14.3) for visualization
