import torch
from torch import nn
import numpy as np
from smap import utils

class NoneBot(nn.Module):
    def __init__(self, module=None):
        super(NoneBot, self).__init__()
        self.module = module
        
    def forward(self, x):
        if self.module is not None:
            return self.module(x)
        return x+0
    
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
    
class TestBot_Input_3_3(nn.Module):
    def __init__(self, module=None, name="out"):
        super(TestBot_Input_3_3, self).__init__()
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
        C_zoom, h, w = ((x.shape[1])//(3*3)), (x.shape[-2]), (x.shape[-1])
        if utils.DEBUG_FLAG:
            H_orig, W_orig = self.testcase.orig_shape
            H_diff, W_diff = ((h-H_orig)//2), ((w-W_orig)//2)
            print(x.shape)
            print(f"input[4,0,0,0] = {(x[0,:,H_diff:(h-H_diff),W_diff:(w-W_diff)]).reshape(C_zoom,(3*3),h,w).permute(1,0,2,3)[4,0,0,0]}")
        C_zoom_2 = int(np.sqrt(C_zoom))
        current_zoom = int(np.log2(C_zoom_2))
        x = (x[0,:,:,:]).reshape(C_zoom,3*3,h,w).permute(1,0,2,3)
        for i in range(current_zoom):
            C_zoom_2 = (C_zoom_2//2)
            x = x.reshape(-1,C_zoom_2,2,C_zoom_2,2,h,w).permute(0,1,3,5,2,6,4)
            h, w = h*2, w*2
            x = x.reshape(-1,C_zoom_2,C_zoom_2,h,w)
        x = x.reshape(-1,h,w)
        self.input_representation = x.detach().cpu().numpy()
        
        if self.testcase is None:
            np.save(self.testcase.out_path+"input_representation.npy", self.input_representation)
        else:
            np.save(self.testcase.out_path+self.testcase.name+"_input_representation.npy", self.input_representation)
        
        return self.module(x)
    
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
        C_zoom, h, w = x.shape[1], x.shape[-2], x.shape[-1]
        if utils.DEBUG_FLAG:
            print(x.shape)
            print(f"target[0,0,0,0] = {(x[0,:,:,:]).reshape(C_zoom,1,h,w).permute(1,0,2,3)[0,0,0,0]}")
        C_zoom_2 = int(np.sqrt(C_zoom))
        current_zoom = int(np.log2(C_zoom_2))
        x = (x[0,:,:,:])
        for i in range(current_zoom):
            C_zoom_2 = (C_zoom_2//2)
            x = x.reshape(-1,C_zoom_2,2,C_zoom_2,2,h,w).permute(0,1,3,5,2,6,4)
            h, w = h*2, w*2
            x = x.reshape(-1,C_zoom_2,C_zoom_2,h,w)
        x = x.reshape(-1,h,w)
        self.target_representation = x.detach().cpu().numpy()
        if self.testcase is None:
            np.save(self.testcase.out_path+"target_representation.npy", self.target_representation)
        else:
            np.save(self.testcase.out_path+self.testcase.name+"_target_representation.npy", self.target_representation)
        return self.module(x)
    
class TestBot_In_3_3(nn.Module):
    def __init__(self, module=None, offset_in=0, name="in", connet2name="out"):
        super(TestBot_In_3_3, self).__init__()
        def get_activation_grad(name, connet2name="out"):
            def hook(module, grad_inputs, grad_outputs):
                if name is not None:
                    # Lấy grad_out: shape [N, C_out, H_out, W_out]
                    grad_out, grad_in = None, None
                    for grad in grad_inputs:
                        if grad is not None:
                            grad_in = grad
                    target_representation_shape = self.testcase.testbot_target.target_representation.shape
                    H_target, W_target = (target_representation_shape[-2]), (target_representation_shape[-1])
                    C_zoom, H_out, W_out = ((grad_in.shape[1])//(3*3)), (grad_in.shape[-2]), (grad_in.shape[-1])
                    
                    C_zoom_2 = int(np.sqrt(C_zoom))
                    current_zoom = int(np.log2(C_zoom_2))
                    grad_in = (grad_in[0,:,:,:]).reshape(C_zoom,3*3,H_out,W_out).permute(1,0,2,3)
                    if utils.DEBUG_FLAG:
                        print(grad_in.shape)
                        H_orig, W_orig = self.testcase.orig_shape
                        H_diff, W_diff = ((H_out-H_orig)//2), ((W_out-W_orig)//2)
                        print(f"in[4,0,0,0] = {(grad_in[:,:,H_diff:(H_out-H_diff),W_diff:(W_out-W_diff)])[4,0,0,0]}")
                    for i in range(current_zoom):
                        C_zoom_2 = (C_zoom_2//2)
                        grad_in = grad_in.reshape(-1,C_zoom_2,2,C_zoom_2,2,H_out,W_out).permute(0,1,3,5,2,6,4)
                        H_out, W_out = H_out*2, W_out*2
                        grad_in = grad_in.reshape(-1,C_zoom_2,C_zoom_2,H_out,W_out)
                    H_diff, W_diff = ((H_out-H_target)//2), ((W_out-W_target)//2)
                    grad_in = (grad_in.reshape(-1,H_out, W_out)[:,H_diff:(-H_diff),W_diff:(-W_diff)]).reshape(-1,H_target, W_target)
                    
                    self.testcase.activation_gradients[name] = grad_in.cpu().numpy()
                    
                    self.testcase.gradient_flows[(name, connet2name)] = None
                    import pickle
                    flow_info = {"activation_gradients": self.testcase.activation_gradients, 
                                 "gradient_flows": self.testcase.gradient_flows}
                    if self.testcase is None:
                        with open(self.testcase.out_path+'flow_info.pkl', 'wb') as f:
                            pickle.dump(flow_info, f)
                    else:
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
        
    def forward(self, x, mask=None):
        self.input_representation = None
        if mask is not None:
            h, w = mask.shape[-2], mask.shape[-1]
            self.input_representation = mask.detach().cpu().numpy().reshape(h, w)
        return self.module(x)
    
class TestBot_Out_3_3(nn.Module):
    def __init__(self, module=None, offset_out=4, name="in", connet2name="out"):
        super(TestBot_Out_3_3, self).__init__()
        def get_activation_grad(name, connet2name="out"):
            def hook(module, grad_inputs, grad_outputs):
                if name is not None:
                    # Lấy grad_out: shape [N, C_out, H_out, W_out]
                    grad_out, grad_in = None, None
                    for grad in grad_outputs:
                        if grad is not None:
                            grad_out = grad
                    C_zoom, H_out, W_out = ((grad_out.shape[1])//(3*3)), (grad_out.shape[-2]), (grad_out.shape[-1])
                    
                    C_zoom_2 = int(np.sqrt(C_zoom))
                    current_zoom = int(np.log2(C_zoom_2))
                    grad_out = (grad_out[0,:,:,:]).reshape(C_zoom,3*3,H_out,W_out).permute(1,0,2,3)
                    if utils.DEBUG_FLAG:
                        H_orig, W_orig = self.testcase.orig_shape
                        H_diff, W_diff = ((H_out-H_orig)//2), ((W_out-W_orig)//2)
                        print(grad_out.shape)
                        print(f"out[4,0,0,0] = {(grad_out[:,:,H_diff:(H_out-H_diff),W_diff:(W_out-W_diff)])[4,0,0,0]}")
                    for i in range(current_zoom):
                        C_zoom_2 = (C_zoom_2//2)
                        grad_out = grad_out.reshape(-1,C_zoom_2,2,C_zoom_2,2,H_out,W_out).permute(0,1,3,5,2,6,4)
                        H_out, W_out = H_out*2, W_out*2
                        grad_out = grad_out.reshape(-1,C_zoom_2,C_zoom_2,H_out,W_out)
                    grad_out = grad_out.reshape(-1,H_out, W_out)
                    
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
    def __init__(self, orig_shape=None, name="", testbot_in=None, testbot_out=None, testbot_input=None, testbot_target=None, out_path=None):
        self.orig_shape = orig_shape
        if out_path is None:
            self.out_path = "./tests/vtest_data/output/"
        else:
            self.out_path = out_path
        self.name = name
        self.activation_gradients = {}
        self.gradient_flows = {}
        if testbot_in is not None:
            self.testbot_in = testbot_in
            self.testbot_in.testcase = self
        if testbot_out is not None:
            self.testbot_out = testbot_out
            self.testbot_out.testcase = self
        if testbot_input is not None:
            self.testbot_input = testbot_input
            self.testbot_input.testcase = self
        if testbot_target is not None:
            self.testbot_target = testbot_target
            self.testbot_target.testcase = self
        
    def get_testbot_in(self):
        return self.testbot_in
    
    def get_testbot_out(self):
        return self.testbot_out
    
    def get_testbot_input(self):
        return self.testbot_input
    
    def get_testbot_target(self):
        return self.testbot_target
