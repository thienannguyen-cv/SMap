import torch
from torch import nn
import numpy as np

class NoneBot(nn.Module):
    def __init__(self, module=None):
        super(NoneBot, self).__init__()
        self.module = module
        
    def forward(self, x):
        if self.module is not None:
            return self.module(x)
        return x
    
class TestBot_Out(nn.Module):
    def __init__(self, module=None, name="out"):
        super(TestBot_Out, self).__init__()
        def get_activation_grad(name='out'):
            def hook(module, grad_input, grad_output):
                if name is not None:
                    # Lấy grad_out: shape [N, C_out, H_out, W_out]
                    grad_out = (grad_output[0]).detach()
                    N, C_out, H_out, W_out = grad_out.shape
                    
                    grad_out = (grad_out[0,0,:,:])
                    idx_map = torch.arange(H_out*W_out, device=grad_out.device)+1
                    gradient_flow = torch.zeros(H_out*W_out+1, H_out*W_out, device=grad_out.device)
                    gradient_flow.scatter_add_(0, idx_map.reshape(1,-1).long(), grad_out.reshape(1,-1))
                    gradient_flow = (gradient_flow[1:,:])
                    
                    gradient_flow = torch.abs(gradient_flow)
                    gradient_flow[gradient_flow>1e-5] = 1.
                    gradient_flow[gradient_flow<=1e-5] = .99

                    # Lưu kết quả vào activation_gradients
                    self.target_representation = gradient_flow.sum(axis=-1,keepdims=False).reshape(H_out, W_out).cpu().numpy()
                    
                    np.save("target_representation.npy", self.target_representation)
            return hook
        self.module = NoneBot()
        if module is not None:
            self.module = module
        self.target_representation = None
        self.name = name
        self.module.register_backward_hook(get_activation_grad(self.name))
        
    def forward(self, x):
        return self.module(x)
    
class TestBot(nn.Module):
    def __init__(self, module=None, name="in", connet2name="out"):
        super(TestBot, self).__init__()
        def get_activation_grad(name, connet2name=None):
            def hook(module, grad_input, grad_output):
                if name is not None:
                    # Lấy grad_out: shape [N, C_out, H_out, W_out]
                    grad_out = (grad_output[0]).detach()
                    N, C_out, H_out, W_out = grad_out.shape
                    
                    # Lấy grad_in: shape [N, C_out, C_in, H_in, W_in]
                    grad_in = (grad_input[0]).detach()
                    N, C_in, H_in, W_in = grad_in.shape
                    grad_out = (grad_out[0,0,:,:])
                    grad_in = (grad_in[0,0,:,:])
                    assert C_out == C_out, "Mismatch in output channels."
                    idx_map = torch.arange(H_out*W_out, device=grad_out.device)+1
                    gradient_flow = torch.zeros(H_in*W_in+1, H_out*W_out, device=grad_out.device)
                    gradient_flow.scatter_add_(0, idx_map.reshape(1,-1).long(), grad_out.reshape(1,-1))
                    gradient_flow = (gradient_flow[1:,:])
                    assert np.abs(gradient_flow.cpu().numpy().sum(axis=-1,keepdims=False)-(grad_in).cpu().numpy().reshape(-1)).sum() < 1e-7, "Mismatch in gradient values."
                    gradient_flow = torch.abs(gradient_flow)
                    gradient_flow[gradient_flow>1e-5] = 1.
                    gradient_flow[gradient_flow<=1e-5] = .99

                    # Lưu kết quả vào activation_gradients
                    self.gradient_flows[(name, connet2name)] = gradient_flow.cpu().numpy()  # kích thước: (Len_in, Len_out)

                    self.activation_gradients[name] = (self.gradient_flows[(name, connet2name)]).sum(axis=-1,keepdims=False).reshape(H_in, W_in)
                    self.activation_gradients[connet2name] = (self.gradient_flows[(name, connet2name)]).sum(axis=0,keepdims=False).reshape(H_out, W_out)
                    
                    import pickle
                    np.save("input_representation.npy", self.input_representation)
                    flow_info = {"activation_gradients": self.activation_gradients, 
                                 "gradient_flows": self.gradient_flows}
                    with open('flow_info.pkl', 'wb') as f:
                        pickle.dump(flow_info, f)
            return hook
        self.module = NoneBot()
        if module is not None:
            self.module = module
        self.input_representation = None
        self.activation_gradients = {}
        self.gradient_flows = {}
        self.name = name
        self.connet2name = connet2name
        self.module.register_backward_hook(get_activation_grad(self.name, self.connet2name))
        
    def forward(self, x):
        self.input_representation = (x[0,0,:,:]).detach().cpu().numpy()
        return self.module(x)