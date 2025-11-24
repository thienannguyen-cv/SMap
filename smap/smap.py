import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smap import specials, utils, rectify
import matplotlib.pyplot as plt

class SMap3x3(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, device):
        super(SMap3x3,self).__init__()
        self.window_h = window_h
        self.window_w = window_w
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)
        self.device = device

        self.sm = nn.Softmax(dim=2)
    
    def go(self, x, original_size, is_last=False):
        pre_x, pre_y, pre_z, pre_mask = x[:,:,:1,:,:], x[:,:,1:2,:,:], x[:,:,2:3,:,:], x[:,:,3:4,:,:]
        
        pre_x, pre_y, pre_z, pre_mask, panels = utils.add_pad(pre_x, pre_y, pre_z, pre_mask, original_size)
        return pre_x, pre_y, pre_z, pre_mask, panels, self(pre_x, pre_y, pre_z, pre_mask, panels, original_size, is_last=is_last)
    
    def forward(self, x_value, y_value, z_value, r_mask, panels, original_size, is_last=False):
        
        shapes = x_value.size()
        BATCH_SIZE, height, width = shapes[0], shapes[-2], shapes[-1]
        
        z_values = z_value.reshape(BATCH_SIZE,-1,1,height, width)
        r_mask = r_mask.reshape(BATCH_SIZE,-1,1,height, width)
        x_z_value = (torch.cat([x_value, y_value, z_value], dim=2)).reshape(BATCH_SIZE,-1,3,height, width)
        
        key_query = utils.calculate_key_query(x_value, y_value, z_value, panels, original_size, (self.window_h, self.window_w), self.camera_matrix_inv, self.device)
        
        
        # 4. Setting proper tensor, named `weights_b', for differentiable rendering
        ind = torch.max(-key_query,dim=2,keepdim=True).indices
        ind_mask = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,height, width,3*3).permute(0,1,4,2,3).reshape(BATCH_SIZE,-1,3*3,height, width)
        weights_b = torch.where(ind_mask>0., torch.ones_like(key_query), torch.zeros_like(key_query))
        temp = torch.zeros_like(weights_b)
        temp[:,:,4,:,:] = 1.

        weights_x = torch.where(((z_values>0.).detach().float()+torch.zeros_like(weights_b))>.5, weights_b, torch.zeros_like(weights_b))
        weights_x = torch.where(((r_mask>specials.OFF_THRESH).float()*(~(z_values>0.)).detach().float()+torch.zeros_like(weights_b))>.5, temp, weights_x)
        weights_m = torch.where((r_mask+torch.zeros_like(weights_b))>specials.OFF_THRESH, torch.where(~(z_values>0.), temp, weights_b), temp)
        weights_m = torch.where(weights_m>specials.OFF_THRESH, torch.ones_like(weights_b), .01*torch.ones_like(weights_b))
        if is_last==True:
            weights_x = torch.where(((r_mask>specials.OFF_THRESH).float()*(z_values>0.).detach().float()+torch.zeros_like(weights_b))>.5, weights_b, torch.zeros_like(weights_b))
            weights_x = torch.where(((r_mask>specials.OFF_THRESH).float()*(~(z_values>0.)).detach().float()+torch.zeros_like(weights_b))>.5, temp, weights_x)
            weights_m = torch.where((((r_mask>specials.OFF_THRESH).float()*(z_values>0.).detach().float())+torch.zeros_like(weights_x))>.5, weights_x, temp)
            

        new_x_z_value = torch.einsum('bcsthw,bczhw->bcstzhw', weights_x.detach().reshape(BATCH_SIZE,-1,3,3,height, width), x_z_value)

        new_r_mask = (weights_m.detach()*r_mask).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        new_x_z_mask_value = torch.cat([new_x_z_value, new_r_mask], dim=4)
        #######################
        
        return new_x_z_mask_value

class SMap(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, rectify_type=None, device="cpu", n=0):
        super(SMap,self).__init__()
        self.window_h = window_h
        self.window_w = window_w
        self.n = n
        self.device = device
        self.smap3x3 = SMap3x3(window_h, window_w, camera_matrix, device)
        self.rectify_module = rectify.DefaultRectify(self.smap3x3)
        if rectify_type==rectify.types.CAM:
            self.rectify_module = rectify.CAMRectify(self.smap3x3)
        if rectify_type==rectify.types.FUT:
            self.rectify_module = rectify.FUTRectify(self.smap3x3)
    
    def calculate_weights(self, new_x_z_mask_value, original_size=None, zoom=0):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        pre_ind = utils.agg(torch.where((new_x_z_mask_value[:,:,:,:,-1:,:,:])>specials.OFF_THRESH, new_x_z_mask_value[:,:,:,:,-2:-1,:,:], torch.max(new_x_z_mask_value[:,:,:,:,-2:-1,:,:])+1e-7*torch.sign(torch.max(new_x_z_mask_value[:,:,:,:,-2:-1,:,:]))+(1. if (self.training) else 0.)*(1e-5*torch.sign(torch.max(new_x_z_mask_value[:,:,:,:,-2:-1,:,:]))*(1.-(new_x_z_mask_value[:,:,:,:,-1:,:,:])))).detach(), factor=specials.INF)
        
        val, ind = torch.min(pre_ind,dim=2,keepdim=True)
        ind = torch.where((val>0.)&(val<specials.INF), ind, 0*ind+4)
        ind = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1,height, width,3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        ind = (ind>.0)
        weights = utils.agg(new_x_z_mask_value, ind=ind).reshape(-1,4,height, width)
        
        
        if original_size is not None:
            weights = (weights[:,-1,:,:]).reshape(BATCH_SIZE,-1, height, width)
            weights = utils.recover_size(weights, self.n, zoom)
            weights = (weights[:,:,0,((weights.size(-2)-(original_size[0]))//2):((weights.size(-2)+(original_size[0]))//2),((weights.size(-1)-(original_size[1]))//2):((weights.size(-1)+(original_size[1]))//2)]).reshape(BATCH_SIZE,-1,original_size[0], original_size[1])
        
        return weights
        
    def forward(self, x, target=None, zoom=0):
        shapes = x.size()
        BATCH_SIZE, height, width = shapes[0], shapes[2], shapes[3]
        C_zoom = 2**(self.n+self.n)
        C_zoom_2 = 1
        height_zoom = height
        width_zoom = width
        
        for i in range(self.n):
            height_zoom = height_zoom // 2
            width_zoom = width_zoom // 2
            x = x.reshape(BATCH_SIZE,C_zoom_2,C_zoom_2,4,height_zoom,2, width_zoom,2).permute(0,1,5,2,7,3,4,6).contiguous().reshape(BATCH_SIZE,-1,4,height_zoom, width_zoom)
            
            C_zoom_2 = C_zoom_2 * 2
        
        pre_x, pre_y, pre_z, pre_mask, panels = None, None, None, None, None
        for i in range(self.n-zoom):
            x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
            pre_x, pre_y, pre_z, pre_mask, panels, x = self.smap3x3.go(x, (height, width))
            h_out, w_out = x.size(-2), x.size(-1)
            
            C_zoom = C_zoom//4
            C_zoom_2 = C_zoom_2//2
            height_zoom = height_zoom*2
            width_zoom = width_zoom*2
            
            x = self.calculate_weights(x)
            x = x.reshape(BATCH_SIZE,C_zoom_2,2,C_zoom_2,2,4,h_out, w_out).permute(0,1,3,5,6,2,7,4).contiguous().reshape(BATCH_SIZE,C_zoom,4,h_out*2, w_out*2)
            
            h_out = h_out*2
            w_out = w_out*2
            
            
        # x.shape
        # >>> torch.Size([16, 256, 4, 8, 16])
        x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
        pre_x, pre_y, pre_z, pre_mask, panels, x = self.smap3x3.go(x, (height, width), is_last=True)
        pre_mask = (pre_mask)
        h_out, w_out = x.size(-2), x.size(-1)
        
        
        if target is not None:
            pre_mask = (pre_mask)
            weights = self.rectify_module.rectificate_flow(x, pre_x, pre_y, pre_z, pre_mask, panels, target.reshape(BATCH_SIZE,1, height_zoom, width_zoom), (height, width))
            return weights
        
        return self.calculate_weights(x, (height, width), zoom)
