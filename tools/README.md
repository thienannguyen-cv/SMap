# Tools
A technical description of available tools. 

## Gradient Flow Visualization in SMap Model
The methodology for visualizing gradient flows in the SMap model, which processes a 2D image. The visualization employs a Sankey-style diagram to represent gradient flows between layers of the neural network.

### Workflow
1. **Compute Gradients**: 
   - Given an input image, compute the gradients through the network.
   - Gradients are stored  as 3D matrices per layer into a pickle file named `flow_info.pkl`.

2. **Flattening & Projection**:
   - Convert 2D gradient maps into 1D representations.
   - Map these 1D values to appropriate nodes in a Sankey diagram.

3. **Constructing the Sankey Diagram**:
   - The first layer consists of individual input pixels.
   - Each subsequent layer contains neurons (nodes), with connections representing the gradient flow intensity.

4. **Rendering the Visualization**:
   - Use a Sankey diagram framework to generate the final visualization.
   - Each node represents a neuron or input pixel, and each edge represents the flow of gradients.

### Input Data


### Diagram Illustration
Below is a conceptual illustration of the gradient flow visualization:

![Gradient Flow Sankey Diagram](https://raw.githubusercontent.com/thienannguyen-cv/SMap/75b05ba93fc8fbf5f8effe5099a76bfa3d14b571/media/images/A_Sankey_diagram_illustrating_gradient_flow_in_a_n.png)

### Implementation Details
- **Language**: Python
- **Libraries**: PyTorch for gradient computation, Matplotlib + Plotly (4.14.3) for visualization

#### Code Snippet for Gradient Extraction
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#########################################
# 1. Define a simple CNN with 3 conv layers
#########################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # conv1: from 4x4 input → output remains 4x4
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # conv2: from 4x4 → 2x2 (stride=2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # conv3: from 2x2 → 1x1 (stride=2)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fcn = nn.Linear(16, 10)  # not used for interaction

    def forward(self, x):
        a1 = torch.tanh(self.conv1(x))    # shape: (1,1,4,4)
        a2 = torch.tanh(self.conv2(a1))     # shape: (1,1,2,2)
        a3 = torch.tanh(self.conv3(a2))     # shape: (1,1,1,1)
        return self.fcn(a3.reshape(1, -1))

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


#########################################
# 2. Register hooks to capture gradients
#########################################
activation_gradients = {}
gradient_flows = {}
def get_activation_grad(name, connet2name=None):
    def hook(module, grad_input, grad_output):
        # TODO: Your gradient computing code. 
    return hook

model.conv1.register_backward_hook(get_activation_grad(None, "conv1"))
model.conv2.register_backward_hook(get_activation_grad("conv1", "conv2"))
model.conv3.register_backward_hook(get_activation_grad("conv2", "conv3"))

#########################################
# 3. Run forward/backward on a random input
#########################################
input_tensor = torch.randn(1, 1, 4, 4)
label = torch.tensor([1])
optimizer.zero_grad()
a3 = model(input_tensor)
loss = criterion(a3.reshape(1, -1), label)  # use conv3 output for loss
loss.backward()

#########################################
# 4. Save gradient flow information for Viz
#########################################
import pickle
flow_info = {"activation_gradients": activation_gradients, 
             "gradient_flows": gradient_flows}
with open('flow_info.pkl', 'wb') as f:
    pickle.dump(flow_info, f)
```
