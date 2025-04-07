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
        return x+0
    
class TestBot_Target(nn.Module):
    def __init__(self, module=None, name="out"):
        super(TestBot_Target, self).__init__()
        def get_activation_grad(name='out'):
            def hook(module, grad_inputs, grad_outputs):
                pass
            return hook
        self.testcase = None
        self.module = NoneBot()
        if module is not None:
            self.module = module
        self.target_representation = None
        self.name = name
        self.module.register_backward_hook(get_activation_grad(self.name))
        
    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        self.target_representation = x.detach().cpu().numpy().reshape(h, w)
        if self.testcase is None:
            np.save(self.testcase.out_path+"target_representation.npy", self.target_representation)
        else:
            np.save(self.testcase.out_path+self.testcase.name+"_target_representation.npy", self.target_representation)
        return self.module(x)
    
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
                    if self.testcase is None:
                        np.save(self.testcase.out_path+"input_representation.npy", self.input_representation.reshape(H_in, W_in))
                        with open(self.testcase.out_path+'flow_info.pkl', 'wb') as f:
                            pickle.dump(flow_info, f)
                    else:
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
    
class TestBot_Out(nn.Module):
    def __init__(self, module=None, offset_out=4, name="in", connet2name="out"):
        super(TestBot_Out, self).__init__()
        def get_activation_grad(name, connet2name="out"):
            def hook(module, grad_inputs, grad_outputs):
                if name is not None:
                    # Lấy grad_out: shape [N, C_out, H_out, W_out]
                    grad_out, grad_in = None, None
                    for grad in grad_outputs:
                        if grad is not None:
                            grad_out = grad
                    H_out, W_out = (grad_out.shape[-2]), (grad_out.shape[-1])
                    
                    grad_out = (grad_out[:,offset_out,:,:]).reshape(H_out, W_out)
                    
                    self.testcase.activation_gradients[connet2name] = grad_out.cpu().numpy()
                    
                    self.testcase.gradient_flows[(name, connet2name)] = None
            return hook
        self.testcase = None
        self.module = NoneBot()
        if module is not None:
            self.module = module
        self.name = name
        self.connet2name = connet2name
        self.module.register_backward_hook(get_activation_grad(self.name, self.connet2name))
        
    def forward(self, x):
        return self.module(x)

class TestCase():
    def __init__(self, name="", testbot_in=None, testbot_out=None, testbot_target=None):
        self.out_path = "./tests/vtest_data/output/"
        self.name = name
        self.activation_gradients = {}
        self.gradient_flows = {}
        self.testbot_in = testbot_in
        self.testbot_out = testbot_out
        self.testbot_target = testbot_target
        self.testbot_in.testcase = self
        self.testbot_out.testcase = self
        self.testbot_target.testcase = self
        
    def get_testbot_in(self):
        return self.testbot_in
    
    def get_testbot_out(self):
        return self.testbot_out
    
    def get_testbot_target(self):
        return self.testbot_target
