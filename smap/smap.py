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
        self.window_h = window_h
        self.window_w = window_w
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)
        self.device = device

        self.sm = nn.Softmax(dim=2)
        
    def calculate_key_query(self, x_value, y_value, z_value, panels, original_size):
        shapes = x_value.size()
        BATCH_SIZE, height, width = shapes[0], shapes[-2], shapes[-1]
        
        grouped_key_x = x_value.reshape(BATCH_SIZE,-1,1,1,height*width)
        grouped_key_y = y_value.reshape(BATCH_SIZE,-1,1,1,height*width)
        #######################
        
        
        # 3. Prepare spatial placeholders for recifying gradients
        updated_key_z = utils.to_3d3x3(z_value.reshape(BATCH_SIZE,-1,height, width), height, width, panels, original_size, (self.window_h, self.window_w), self.camera_matrix_inv, self.device).permute(0,1,3,2).reshape(BATCH_SIZE,-1,3,3*3,height*width)
        query_x = (updated_key_z[:,:,:1,:,:]).detach()
        query_y = (updated_key_z[:,:,1:2,:,:]).detach()
        
        key_query = torch.sum(torch.abs(grouped_key_x-query_x)+torch.abs(grouped_key_y-query_y),dim=2).reshape(BATCH_SIZE,-1,3*3,height, width)
        #######################
        
        return key_query
    
    def go(self, x, original_size):
        pre_x, pre_y, pre_z, pre_mask = x[:,:,:1,:,:], x[:,:,1:2,:,:], x[:,:,2:3,:,:], x[:,:,3:4,:,:]
        
        pre_x, pre_y, pre_z, pre_mask, panels = utils.add_pad(pre_x, pre_y, pre_z, pre_mask, original_size)
        return pre_x, pre_y, pre_z, pre_mask, panels, self(pre_x, pre_y, pre_z, pre_mask, panels, original_size)
    
    def forward(self, x_value, y_value, z_value, r_mask, panels, original_size):
        shapes = x_value.size()
        BATCH_SIZE, height, width = shapes[0], shapes[-2], shapes[-1]
        
        z_values = z_value.reshape(BATCH_SIZE,-1,1,height, width)
        r_mask = r_mask.reshape(BATCH_SIZE,-1,1,height, width)
        x_z_value = (torch.cat([x_value, y_value, z_value], dim=2)).reshape(BATCH_SIZE,-1,3,height, width)
        
        key_query = self.calculate_key_query(x_value, y_value, z_value, panels, original_size)

        # 4. Setting proper tensor, named `weights_b', for differentiable rendering
        new_r_mask = torch.zeros_like(key_query)
        ind = torch.max(-key_query,dim=2,keepdim=True).indices
        ind_mask = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,height, width,3*3).permute(0,1,4,2,3).reshape(BATCH_SIZE,-1,3*3,height, width)
        new_r_mask[ind_mask>.5] = 1.
        new_r_mask = torch.where((r_mask+key_query*0.)>specials.OFF_THRESH, new_r_mask, torch.zeros_like(new_r_mask))
        weights_b = (1.*new_r_mask).reshape(BATCH_SIZE,-1,3,3,height, width)
        new_r_mask[:,:,4,:,:] = torch.where(r_mask.reshape(BATCH_SIZE,-1,height, width)>specials.OFF_THRESH, new_r_mask[:,:,4,:,:], torch.ones_like(new_r_mask[:,:,4,:,:]))
        new_x_z_value = torch.einsum('bcsthw,bczhw->bcstzhw', weights_b.detach(), x_z_value)
        new_r_mask = (new_r_mask.detach()*r_mask).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        new_x_z_mask_value = torch.cat([new_x_z_value, new_r_mask], dim=4)
        #######################
        
        return new_x_z_mask_value

class SMap(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, device, n=0):
        super(SMap,self).__init__()
        self.n = n
        self.smap3x3 = SMap3x3(window_h, window_w, camera_matrix, device)
    
    def compute_allow_matrix(self, ws, tgt_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = ws.shape[0], ws.shape[-2], ws.shape[-1], tgt_repr.shape[-1], tgt_repr.shape[-2]
        
        ws = utils.agg(ws).reshape(BATCH_SIZE,-1,3*3,1,height, width)
        
        allow = 2.*torch.ones_like(ws)
        allow[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)] = (ws[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])*tgt_repr.reshape(BATCH_SIZE,1,1,1,h_zoom, w_zoom)
        allow = utils.agg(flip(allow,2).reshape(BATCH_SIZE,-1,3,3,1,height, width)).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        allow = torch.max(allow,dim=2,keepdim=True)[0]
        allow = 1.-allow
        return allow
    
    def prepare_flows_for_mask(self, allow, tgt_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = allow.shape[0], allow.shape[-2], allow.shape[-1], tgt_repr.shape[-1], tgt_repr.shape[-2]
        
        allow = torch.cat([allow, allow, allow],dim=2)
        allow = torch.cat([allow, allow, allow],dim=3).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        mask_flow = (utils.agg(allow).reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        return mask_flow
    
    def prepare_flows_for_coord(self, allow, tgt_repr):
        allow = (allow>=0).float()
            
        BATCH_SIZE, height, width, w_zoom, h_zoom = allow.shape[0], allow.shape[-2], allow.shape[-1], tgt_repr.shape[-1], tgt_repr.shape[-2]
        
        allow = torch.cat([allow, allow, allow],dim=2)
        allow = torch.cat([allow, allow, allow],dim=3).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        coord_flow = (utils.agg(allow).reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        return coord_flow*tgt_repr.reshape(-1,1,h_zoom, w_zoom)
    
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
            
    def rectificate_flow(self, new_x_z_mask_value, pre_x, pre_y, pre_z, pre_mask, panels, target, original_size):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        pre_key_query = self.smap3x3.calculate_key_query(pre_x, pre_y, pre_z, panels, original_size)
        pre_mask = pre_mask.reshape(BATCH_SIZE,-1,1,1,1,height, width)
        weights_b = (new_x_z_mask_value[:,:,:,:,-1:,:,:]).detach()
        
        
        # 7. Triggering gradient at the origins of the image rectification
        shapes = target.size()
        BATCH_SIZE, C_zoom, h_zoom, w_zoom = shapes[0], shapes[1], shapes[-2], shapes[-1]
        
        target_2Dr = torch.max(target.reshape(BATCH_SIZE,-1,1,1,h_zoom, w_zoom),dim=1,keepdim=True).values
        
        key_query_grdf = -(pre_key_query-pre_key_query.detach())
        new_r_mask = (1.*pre_mask)

        weights = weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width).detach()
        weights = weights.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        weights = utils.agg(weights).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        key_query_grdf = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+key_query_grdf.reshape(BATCH_SIZE,C_zoom,3*3,height, width)
        key_query_grdf = key_query_grdf.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        key_query_grdf = utils.agg(key_query_grdf).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        new_r_mask = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+new_r_mask.reshape(BATCH_SIZE,C_zoom,1,height, width)
        new_r_mask = new_r_mask.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        new_r_mask = utils.agg(new_r_mask).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)


        allow = self.compute_allow_matrix(weights_b.detach().reshape(BATCH_SIZE,-1,3,3,1,height, width), target_2Dr)
        mask_flow = 1.-(self.prepare_flows_for_mask(allow, target_2Dr)==0.).float()
        coord_flow = self.prepare_flows_for_coord(allow, target_2Dr)
        target_2Dr = target_2Dr.reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        
        weights = (weights.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        key_query_grdf = (key_query_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        new_r_mask = (new_r_mask.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        new_r_mask_grdf = new_r_mask-new_r_mask.detach()

        case = ((new_r_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float()
        n_case = (1.-mask_flow.detach())*(weights>specials.OFF_THRESH).float() # ((new_r_mask>specials.OFF_THRESH).float()==target_2Dr).float()
        factor = torch.where(new_r_mask_grdf>specials.OFF_THRESH, 1e-1*(1.-case), (1.-case))
        case = (1.-torch.max(n_case,dim=1,keepdim=True).values)*mask_flow.detach()*(1.-case)
        weights_grdf = (case*factor+n_case).detach()*new_r_mask_grdf + coord_flow.detach()*key_query_grdf

        weights = weights.detach() + weights_grdf # apply attractive rectification for this implementation
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
            
        pre_x, pre_y, pre_z, pre_mask, panels, target = None, None, None, None, None, None
        if target_2Dr is not None:
            target_2Dr = target_2Dr.reshape(BATCH_SIZE,1,height_zoom,C_zoom_2, width_zoom,C_zoom_2).permute(0,3,5,1,2,4).reshape(BATCH_SIZE,C_zoom,1,height_zoom, width_zoom)
            if self.n==zoom:
                target = target_2Dr.reshape(BATCH_SIZE,-1,height_zoom, width_zoom)
        
        # x.shape
        # >>> torch.Size([16, 256, 4, 8, 16])
        x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
        pre_x, pre_y, pre_z, pre_mask, panels, x = self.smap3x3.go(x, (height_zoom, width_zoom))
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
            
            pre_x, pre_y, pre_z, pre_mask, panels, target = None, None, None, None, None, None
            if target_2Dr is not None:
                target_2Dr = target_2Dr.reshape(BATCH_SIZE,2,C_zoom_2,2,C_zoom_2,1,height_zoom//2, width_zoom//2).permute(0,2,4,5,6,1,7,3).reshape(BATCH_SIZE,C_zoom,height_zoom, width_zoom)
                if i==(self.n-zoom-1):
                    target = target_2Dr.reshape(BATCH_SIZE,-1,height_zoom, width_zoom)
            
            x = x.reshape(x.size(0),-1,x.size(-3),x.size(-2),x.size(-1))
            pre_x, pre_y, pre_z, pre_mask, panels, x = self.smap3x3.go(x, (height_zoom, width_zoom))
            h_out, w_out = x.size(-2), x.size(-1)
        
        if target is not None:
            return self.rectificate_flow(x, pre_x, pre_y, pre_z, pre_mask, panels, target, (height_zoom, width_zoom))
        
        return x