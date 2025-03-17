# Tools
A technical description of available tools. 

## Gradient Flow Visualization in SMap Model
The methodology for visualizing gradient flows in the SMap model, which processes a 2D image. The visualization employs a Sankey-style diagram to represent gradient flows between layers of the neural network.

### Workflow
1. **Compute Gradients**: 
   - Given an input image, compute the gradients through the network.
   - Gradients are stored as 3D matrices per layer.

2. **Flattening & Projection**:
   - Convert 2D gradient maps into 1D representations.
   - Map these 1D values to appropriate nodes in a Sankey diagram.

3. **Constructing the Sankey Diagram**:
   - The first layer consists of individual input pixels.
   - Each subsequent layer contains neurons (nodes), with connections representing the gradient flow intensity.

4. **Rendering the Visualization**:
   - Use a Sankey diagram framework to generate the final visualization.
   - Each node represents a neuron or input pixel, and each edge represents the flow of gradients.

### Diagram Illustration
Below is a conceptual illustration of the gradient flow visualization:

![Gradient Flow Sankey Diagram](https://raw.githubusercontent.com/thienannguyen-cv/SMap/75b05ba93fc8fbf5f8effe5099a76bfa3d14b571/media/images/A_Sankey_diagram_illustrating_gradient_flow_in_a_n.png)

### Implementation Details
- **Language**: Python
- **Libraries**: PyTorch for gradient computation, Matplotlib/Plotly for visualization

#### Code Snippet for Gradient Extraction
```python
import torch
import torch.nn.functional as F

# Example Model
def model(x):
    W = torch.tensor([[0.5, -0.2], [0.3, 0.8]], requires_grad=True)
    return x @ W

# Compute Gradients
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y = model(x)
y.backward(torch.ones_like(y))
print(x.grad)  # Gradient of input
```
