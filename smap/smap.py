import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smap import specials, utils, rectify

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]
    
class SMap3x3(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, device):
        super(SMap3x3,self).__init__()
        self.window_h = window_h
        self.window_w = window_w
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)
        self.device = device

        self.sm = nn.Softmax(dim=2)
    
    def forward(self, x_value, y_value, z_value, r_mask, panels, original_size):
        shapes = x_value.size()
        BATCH_SIZE, height, width = shapes[0], shapes[-2], shapes[-1]
        
        z_values = z_value.reshape(BATCH_SIZE,-1,1,height, width)
        r_mask = r_mask.reshape(BATCH_SIZE,-1,1,height, width)
        x_z_value = (torch.cat([x_value, y_value, z_value], dim=2)).reshape(BATCH_SIZE,-1,3,height, width)
        
        key_query = utils.calculate_key_query(x_value, y_value, z_value, panels, (self.window_h, self.window_w), original_size, self.camera_matrix_inv, device=self.device)
        
        # 4. Setting proper tensor, named `weights_b', for differentiable rendering
        weights_b = torch.zeros_like(key_query)
        ind = torch.max(-key_query,dim=2,keepdim=True).indices
        ind_mask = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,height, width,3*3).permute(0,1,4,2,3).reshape(BATCH_SIZE,-1,3*3,height, width)
        weights_b[ind_mask>.5] = 1.
        weights_b = torch.where(((r_mask>specials.OFF_THRESH).float()*(z_values>0.).float()+key_query*0.)>.5, weights_b, torch.zeros_like(weights_b))
        temp = torch.zeros_like(weights_b)
        temp[:,:,4,:,:] = 1.
        weights_b = torch.where((0.*weights_b+(1.-(r_mask>specials.OFF_THRESH).float()*(~(z_values>0.)).float()))>.5, weights_b, temp)
        new_x_z_value = torch.einsum('bcsthw,bczhw->bcstzhw', (1.*weights_b).detach().reshape(BATCH_SIZE,-1,3,3,height, width), x_z_value)
        
        weights_b = torch.where((0.*weights_b+((r_mask>specials.OFF_THRESH).float()*(z_values>0.).float()))>.5, weights_b, temp)
        new_r_mask = ((1.*weights_b).detach()*r_mask).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        new_x_z_mask_value = torch.cat([new_x_z_value, new_r_mask], dim=4)
        #######################
        
        return new_x_z_mask_value
    
    def go(self, x, original_size):
        pre_x, pre_y, pre_z, pre_mask = x[:,:,:1,:,:], x[:,:,1:2,:,:], x[:,:,2:3,:,:], x[:,:,3:4,:,:]
        
        pre_x, pre_y, pre_z, pre_mask, panels = utils.add_pad(pre_x, pre_y, pre_z, pre_mask, original_size)
        return pre_x, pre_y, pre_z, pre_mask, panels, self(pre_x, pre_y, pre_z, pre_mask, panels, original_size)

class SMap(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, rectify_type=None, n=0, device="cpu"):
        super(SMap,self).__init__()
        self.n = n
        self.smap3x3 = SMap3x3(window_h, window_w, camera_matrix, device)
        self.rectify_module = rectify.DefaultRectify(self.smap3x3)
        if rectify_type==rectify.types.CAM:
            self.rectify_module = rectify.CAMRectify(self.smap3x3)
    
    def calculate_weights(self, new_x_z_mask_value, original_size=None):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        ind = utils.agg((new_x_z_mask_value[:,:,:,:,-2:-1,:,:]).detach(), factor=specials.INF)
        val, ind = torch.min(ind,dim=2,keepdim=True)
        ind = torch.where((val>0.)&(val<specials.INF), ind, 0*ind+4)
        ind = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1,height, width,3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        ind = (ind>.5)
        weights = utils.agg(new_x_z_mask_value, ind=ind).reshape(-1,4,height, width)
        
        if original_size is not None:
            weights = (weights[:,:,((height-(original_size[0]))//2):((height+(original_size[0]))//2),((width-(original_size[1]))//2):((width+(original_size[1]))//2)]).reshape(BATCH_SIZE,-1,original_size[0], original_size[1])
        
        return weights
    
    def forward(self, x, target=None, zoom=0):
        shapes = x.size()
        BATCH_SIZE, height, width = shapes[0], shapes[2], shapes[3]
        C_zoom = 2**(self.n+self.n)
        C_zoom_2 = 1
        height_zoom = height
        width_zoom = width
        
        target_2Dr, target_rpr = target, None
        for i in range(self.n):
            height_zoom = height_zoom // 2
            width_zoom = width_zoom // 2
            x = x.reshape(BATCH_SIZE,C_zoom_2,C_zoom_2,4,height_zoom,2, width_zoom,2).permute(0,1,5,2,7,3,4,6).contiguous().reshape(BATCH_SIZE,-1,4,height_zoom, width_zoom)
            if target is not None:
                target_2Dr = (1.*target_2Dr).reshape(BATCH_SIZE,C_zoom_2,C_zoom_2,1,height_zoom,2, width_zoom,2).permute(0,1,5,2,7,3,4,6).contiguous().reshape(BATCH_SIZE,-1,1,height_zoom, width_zoom)
            C_zoom_2 = C_zoom_2 * 2
        
        pre_x, pre_y, pre_z, pre_mask, panels = None, None, None, None, None
        for i in range(self.n-zoom):
            x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
            pre_x, pre_y, pre_z, pre_mask, panels, x = self.smap3x3.go(x, (height_zoom, width_zoom))
            h_out, w_out = x.size(-2), x.size(-1)
            
            C_zoom = C_zoom//4
            C_zoom_2 = C_zoom_2//2
            height_zoom = height_zoom*2
            width_zoom = width_zoom*2
            
            if target is not None:
                target_2Dr = (1.*target_2Dr).reshape(BATCH_SIZE,C_zoom_2,2,C_zoom_2,2,1,height_zoom//2, width_zoom//2).permute(0,1,3,5,6,2,7,4).contiguous().reshape(BATCH_SIZE,C_zoom,height_zoom, width_zoom)
            
            x = self.calculate_weights(x)
            x = x.reshape(BATCH_SIZE,C_zoom_2,2,C_zoom_2,2,4,h_out, w_out).permute(0,1,3,5,6,2,7,4).contiguous().reshape(BATCH_SIZE,C_zoom,4,h_out*2, w_out*2)
            
            h_out = h_out*2
            w_out = w_out*2
            
            
        # x.shape
        # >>> torch.Size([16, 256, 4, 8, 16])
        x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
        pre_x, pre_y, pre_z, pre_mask, panels, x = self.smap3x3.go(x, (height_zoom, width_zoom))
        h_out, w_out = x.size(-2), x.size(-1)
        
        
        if target is not None:
            
            weights = self.rectify_module.rectificate_flow(x, pre_x, pre_y, pre_z, pre_mask, panels, target_2Dr, (height_zoom, width_zoom))
            return weights
        
        return self.calculate_weights(x, (height_zoom, width_zoom))
