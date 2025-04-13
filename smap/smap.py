import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smap import specials, utils

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]
    
class SMap3x3(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, device):
        super(SMap3x3,self).__init__()
        self.vtest_service = None
        self.window_h = window_h
        self.window_w = window_w
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)
        self.device = device
        self.grad_grouped_key = None
        self.sm = nn.Softmax(dim=2)
    
    def go(self, x, original_size):
        x_, y, z, mask = x[:,:,:1,:,:], x[:,:,1:2,:,:], x[:,:,2:3,:,:], x[:,:,3:4,:,:]
        return self(x_, y, z, mask, original_size)
    
    def forward(self, x_value, y_value, z_value, r_mask, original_size):
        from smap import utils
        
        shapes = x_value.size()
        BATCH_SIZE, height, width = shapes[0], shapes[-2], shapes[-1]
        
        # 1. Prepare configuration for to_3d unit
        height = height + 2**0 + 2**0
        width = width + 2**0 + 2**0
        
        panels = list(np.where(np.ones([height, width])))
        offset_codes = ((height-original_size[0]), (width-original_size[1]))
        panels[0] = panels[0] - (offset_codes[0]//2) + .5
        panels[1] = panels[1] - (offset_codes[1]//2) + .5
        #######################
        
        
        # 2. Prepare input tensors
        x_value = torch.cat([torch.zeros_like(x_value[:,:,:,:,:(2**0)]), x_value, torch.zeros_like(x_value[:,:,:,:,:(2**0)])], dim=-1)
        x_value = torch.cat([torch.zeros_like(x_value[:,:,:,:(2**0),:]), x_value, torch.zeros_like(x_value[:,:,:,:(2**0),:])], dim=-2)
        
        y_value = torch.cat([torch.zeros_like(y_value[:,:,:,:,:(2**0)]), y_value, torch.zeros_like(y_value[:,:,:,:,:(2**0)])], dim=-1)
        y_value = torch.cat([torch.zeros_like(y_value[:,:,:,:(2**0),:]), y_value, torch.zeros_like(y_value[:,:,:,:(2**0),:])], dim=-2)
        
        z_value = torch.cat([torch.zeros_like(z_value[:,:,:,:,:(2**0)]), z_value, torch.zeros_like(z_value[:,:,:,:,:(2**0)])], dim=-1)
        z_value = torch.cat([torch.zeros_like(z_value[:,:,:,:(2**0),:]), z_value, torch.zeros_like(z_value[:,:,:,:(2**0),:])], dim=-2)
        
        r_mask = torch.cat([torch.zeros_like(r_mask[:,:,:,:,:(2**0)]), r_mask, torch.zeros_like(r_mask[:,:,:,:,:(2**0)])], dim=-1)
        r_mask = torch.cat([torch.zeros_like(r_mask[:,:,:,:(2**0),:]), r_mask, torch.zeros_like(r_mask[:,:,:,:(2**0),:])], dim=-2)
        
        z_values = z_value.reshape(BATCH_SIZE,-1,1,height, width)
        r_mask = r_mask.reshape(BATCH_SIZE,-1,1,height, width)
        x_z_value = (torch.cat([x_value, y_value, z_value], dim=2)).reshape(BATCH_SIZE,-1,3,height, width)
        grouped_key_x = x_value.reshape(BATCH_SIZE,-1,1,1,height*width)
        grouped_key_y = y_value.reshape(BATCH_SIZE,-1,1,1,height*width)
        #######################
        
        
        # 3. Prepare spatial placeholders for recifying gradients
        updated_key_z = utils.to_3d3x3(z_values.reshape(BATCH_SIZE,-1,height, width), height, width, panels, original_size, (self.window_h, self.window_w), self.camera_matrix_inv, self.device).permute(0,1,3,2).reshape(BATCH_SIZE,-1,3,3*3,height*width)
        query_x = (updated_key_z[:,:,:1,:,:]).detach()
        query_y = (updated_key_z[:,:,1:2,:,:]).detach()
        
        key_query = torch.sum(torch.abs(grouped_key_x-query_x)+torch.abs(grouped_key_y-query_y),dim=2).reshape(BATCH_SIZE,-1,3*3,height, width)
        #######################
        
        
        # 4. Setting proper tensor, named `weights_b', for differentiable rendering
        new_r_mask = torch.zeros_like(key_query)
        ind = torch.max(-key_query,dim=2,keepdim=True).indices
        ind_mask = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,height, width,3*3).permute(0,1,4,2,3).reshape(BATCH_SIZE,-1,3*3,height, width)
        new_r_mask[ind_mask>.5] = 1.
        new_r_mask = torch.where((r_mask+(query_x[:,:,0,:,:]).reshape(BATCH_SIZE,-1,3*3,height, width)*0.)>specials.OFF_THRESH, new_r_mask, torch.zeros_like(new_r_mask))
        weights_b = (1.*new_r_mask).reshape(BATCH_SIZE,-1,3,3,height, width)
        new_r_mask[:,:,4,:,:] = torch.where(r_mask.reshape(BATCH_SIZE,-1,height, width)>specials.OFF_THRESH, new_r_mask[:,:,4,:,:], torch.ones_like(new_r_mask[:,:,4,:,:]))
        new_x_z_value = torch.einsum('bcsthw,bczhw->bcstzhw', weights_b.detach(), x_z_value)
        new_r_mask = (new_r_mask*r_mask).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        new_x_z_mask_value = torch.cat([new_x_z_value, new_r_mask], dim=4)
        #######################
        
        return new_x_z_mask_value

class SMap(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, device, n=0):
        super(SMap,self).__init__()
        self.device = device
        self.n = n
        self.window_h = window_h
        self.window_w = window_w
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)
        self.smap3x3 = SMap3x3(window_h, window_w, camera_matrix, device)
        
    def compute_allow_matrix(self, ws, tgt_repr):
        BATCH_SIZE, C_zoom ,height, width, w_zoom, h_zoom = ws.shape[0], ws.shape[1], ws.shape[-2], ws.shape[-1], tgt_repr.shape[-1], tgt_repr.shape[-2]
        
        ws = utils.agg(ws).reshape(BATCH_SIZE,C_zoom,3*3,1,height, width)
        
        allow = 2.*torch.ones_like(ws)
        wsxtgt_repr = ((ws[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])==tgt_repr)
        # Dùng để đếm các điểm trên ws đã kích hoạt trùng khớp với tgt_repr. 
        wsxtgt_repr = wsxtgt_repr*(ws[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])
        # Nếu các điểm ứng với điểm trên tgt_rept đã được kích hoạt và số trùng khớp là khác 1. 
        wsxtgt_repr = (ws[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])*(torch.sum(wsxtgt_repr,dim=2,keepdim=True)==1.).float()
        
        allow[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)] = wsxtgt_repr
        allow = utils.agg(flip(allow,2).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)).reshape(BATCH_SIZE,C_zoom,3*3,1,1,height, width)
        allow = torch.max(allow,dim=2,keepdim=True).values
        allow = 1.-allow
        return allow
    
    def prepare_flows_for_mask(self, allow, ws, tgt_repr):
        BATCH_SIZE, C_zoom ,height, width, w_zoom, h_zoom = ws.shape[0], ws.shape[1], ws.shape[-2], ws.shape[-1], tgt_repr.shape[-1], tgt_repr.shape[-2]
        
        allow = torch.cat([allow, allow, allow],dim=2)
        allow = torch.cat([allow, allow, allow],dim=3).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        allow = (utils.agg(allow).reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        return allow
    
    def prepare_flows_for_coord(self, allow, ws, tgt_repr):
        allow = (allow>=0).float()
            
        BATCH_SIZE, C_zoom ,height, width, w_zoom, h_zoom = ws.shape[0], ws.shape[1], ws.shape[-2], ws.shape[-1], tgt_repr.shape[-1], tgt_repr.shape[-2]
        
        allow = self.prepare_flows_for_mask(allow, ws, tgt_repr)
        return allow*tgt_repr.reshape(BATCH_SIZE,1,h_zoom, w_zoom)
        
    def calculate_weights(self, new_x_z_mask_value, original_size=None):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        ind = utils.agg((new_x_z_mask_value[:,:,:,:,-2:-1,:,:]).detach(), factor=1e7)
        ind = torch.min(ind,dim=2,keepdim=True).indices
        ind = torch.where(torch.sum(utils.agg(new_x_z_mask_value[:,:,:,:,-1:,:,:], factor=0.),dim=2,keepdim=True)>.5, ind, 0*ind+4)
        ind = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1,height, width,3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        ind = (ind>.5)
        weights = utils.agg(new_x_z_mask_value, ind=ind).reshape(-1,4,height, width)
        
        if original_size is not None:
            weights = (weights[:,:,((height-(original_size[0]))//2):((height+(original_size[0]))//2),((width-(original_size[1]))//2):((width+(original_size[1]))//2)]).reshape(BATCH_SIZE,-1,original_size[0], original_size[1])
        
        return weights
        
    def calculate_key_query(self, weights, original_size):
        BATCH_SIZE, height, width = weights.shape[0], weights.shape[-2], weights.shape[-1]
        
        
        grouped_key_x = (weights[:,:1,:,:]).reshape(BATCH_SIZE,-1,1,1,height*width)
        grouped_key_y = (weights[:,1:2,:,:]).reshape(BATCH_SIZE,-1,1,1,height*width)
        z_values = (weights[:,2:3,:,:]).reshape(BATCH_SIZE,-1,height, width)
        
        panels = list(np.where(np.ones([height, width])))
        offset_codes = ((height-original_size[0]), (width-original_size[1]))
        panels[0] = panels[0] - (offset_codes[0]//2) + .5
        panels[1] = panels[1] - (offset_codes[1]//2) + .5
        
        updated_key_z = utils.to_3d3x3(z_values, height, width, panels, original_size, (self.window_h, self.window_w), self.camera_matrix_inv, self.device).permute(0,1,3,2).reshape(BATCH_SIZE,-1,3,3*3,height*width)
        query_x = (updated_key_z[:,:,:1,:,:]).detach()
        query_y = (updated_key_z[:,:,1:2,:,:]).detach()
        
        key_query = torch.sum(torch.abs(grouped_key_x-query_x)+torch.abs(grouped_key_y-query_y),dim=2).reshape(BATCH_SIZE,-1,3*3,height, width)
        
        return key_query
        
    def rectificate_flow(self, new_x_z_mask_value, target, original_size):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        weights = self.calculate_weights(new_x_z_mask_value)
        key_query = self.calculate_key_query(weights, original_size)
        pre_mask = (weights[:,-1:,:,:]).reshape(BATCH_SIZE,-1,1,1,1,height, width)
        new_r_mask = (new_x_z_mask_value[:,:,:,:,-1:,:,:]).detach()
        weights_b = new_r_mask.detach()
        
        
        # 7. Triggering gradient at the origins of the image rectification
        shapes = target.size()
        BATCH_SIZE, C_zoom, h_zoom, w_zoom = shapes[0], shapes[1], shapes[-2], shapes[-1]
        
        target_2Dr = torch.max(target.reshape(BATCH_SIZE,-1,1,1,h_zoom, w_zoom),dim=1,keepdim=True).values
        
        pre_allow = self.compute_allow_matrix(weights_b.detach(), target_2Dr).detach()
        allow = torch.abs(pre_allow)
        
        a_weights = new_r_mask*(new_r_mask>specials.OFF_THRESH).float()*(1.-allow)
        a_weights = a_weights.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        a_weights = utils.agg(a_weights).reshape(BATCH_SIZE,-1,3*3,height, width) # 2nd diff
        n_weights = torch.zeros_like(new_r_mask)+.75*(pre_mask.detach() - (2.*(pre_mask>0.).float()-1.)*(pre_mask-pre_mask.detach()))*allow
        n_weights = n_weights.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        n_weights = utils.agg(n_weights).reshape(BATCH_SIZE,-1,3*3,height, width)
        n_weights = n_weights.detach()+torch.max(1.-(a_weights>specials.OFF_THRESH).float(),dim=2,keepdim=True).values*(n_weights-n_weights.detach())
        weights = (a_weights-n_weights).reshape(BATCH_SIZE,-1,height, width)
        new_r_mask = new_r_mask.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        new_r_mask = utils.agg(new_r_mask).reshape(BATCH_SIZE,-1,height, width) # 2nd diff
        
        key_query_grdf = -(key_query-key_query.detach()).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        key_query_grdf = utils.agg(key_query_grdf).reshape(BATCH_SIZE,-1,height, width)
        
        mask_flow = self.prepare_flows_for_mask(allow, weights_b, target_2Dr)
        coord_flow = self.prepare_flows_for_coord(allow, weights_b, target_2Dr)
        
        
        weights = (weights[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        key_query_grdf = (key_query_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        new_r_mask = (new_r_mask[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        
        target_2Dr = target_2Dr.reshape(BATCH_SIZE,1,h_zoom, w_zoom)
        weights_grdf = (weights-weights.detach())
        
        case = ((torch.abs(weights)>specials.OFF_THRESH).long()==target_2Dr.long()).float()
        n_case = ((new_r_mask>specials.OFF_THRESH).float()==target_2Dr).float()
        factor = torch.where(torch.abs(weights)>specials.OFF_THRESH, 1e-1*(1.-case), (1.-case))
        weights = weights.detach() + (2.*(weights>0.).float()-1.)*((1.-case)*factor*weights_grdf + key_query_grdf) # apply attractive rectification for this implementation
        #######################
        
        return weights
    
    def forward(self, x, target_2Dr=None, zoom=0):
        shapes = x.size()
        BATCH_SIZE, height, width = shapes[0], shapes[2], shapes[3]
        C_zoom = 2**(self.n+self.n)
        C_zoom_2 = 1
        height_zoom = height
        width_zoom = width
        for i in range(self.n):
            height_zoom = height_zoom // 2
            width_zoom = width_zoom // 2
            x = x.reshape(BATCH_SIZE,C_zoom_2,C_zoom_2,4,height_zoom,2, width_zoom,2).permute(0,5,1,7,2,3,4,6).reshape(BATCH_SIZE,-1,4,height_zoom, width_zoom)
            C_zoom_2 = C_zoom_2 * 2
            
        target = None
        if target_2Dr is not None:
            target_2Dr = target_2Dr.reshape(BATCH_SIZE,1,height_zoom,C_zoom_2, width_zoom,C_zoom_2).permute(0,3,5,1,2,4).reshape(BATCH_SIZE,C_zoom,1,height_zoom, width_zoom)
            if self.n==zoom:
                target = target_2Dr.reshape(BATCH_SIZE,-1,height_zoom, width_zoom)
        
        # x.shape
        # >>> torch.Size([16, 256, 4, 8, 16])
        x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
        x = self.smap3x3.go(x, (height_zoom, width_zoom))
        h_out, w_out = x.size(-2), x.size(-1)
        
        for i in range(self.n-zoom):
            x = self.calculate_weights(x)
            x = x.reshape(BATCH_SIZE,2,C_zoom_2//2,2,C_zoom_2//2,4,h_out, w_out).permute(0,2,4,5,6,1,7,3).reshape(BATCH_SIZE,C_zoom//4,4,h_out*2, w_out*2)
            
            C_zoom = C_zoom//4
            C_zoom_2 = C_zoom_2//2
            height_zoom = height_zoom*2
            width_zoom = width_zoom*2
            h_out = h_out*2
            w_out = w_out*2
            
            target = None
            if target_2Dr is not None:
                target_2Dr = target_2Dr.reshape(BATCH_SIZE,2,C_zoom_2,2,C_zoom_2,1,height_zoom//2, width_zoom//2).permute(0,2,4,5,6,1,7,3).reshape(BATCH_SIZE,C_zoom,height_zoom, width_zoom)
                if i==(self.n-zoom-1):
                    target = target_2Dr.reshape(BATCH_SIZE,-1,height_zoom, width_zoom)
            
            x = self.smap3x3.go(x, (height_zoom, width_zoom))
            h_out, w_out = x.size(-2), x.size(-1)
        
        if target is not None:
            return self.rectificate_flow(x, target, (height_zoom, width_zoom))
        
        return x