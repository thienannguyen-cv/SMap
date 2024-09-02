import torch
import torch.nn as nn
import numpy as np

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))

def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))

class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)

class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishJitAutoFn.apply(x)
    
class SMap3x3(nn.Module):
    def __init__(self, camera_matrix):
        super(SMap3x3,self).__init__()
        
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)

        self.sm = nn.Softmax(dim=2)
    
    def to_3d(self, z, height, width, panels, original_size):
        y_im, x_im = panels
        y_im, x_im = torch.from_numpy(y_im).reshape(height, width), torch.from_numpy(x_im).reshape(height, width)
        y_im = y_im * IMG_SHAPE[0] / original_size[0]
        x_im = x_im * IMG_SHAPE[1] / original_size[1]
        y_im, x_im = y_im.to(device), x_im.to(device)
        
        imp_co = torch.cat([torch.einsum('hw,bczhw->bczhw', x_im.float(), torch.ones_like(z.unsqueeze(2)).float()), torch.einsum('hw,bczhw->bczhw', y_im.float(), torch.ones_like(z.unsqueeze(2)).float()), torch.ones_like(z.unsqueeze(2))], 2)
        imp_co = F.unfold(imp_co.reshape(1, -1, height, width), kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=(1,1)).reshape(z.size(0),z.size(1),3,3*3,height,width)
        imp_co = torch.einsum('bchw,bczshw->bczshw', z.float(), imp_co.float()).reshape(z.size(0),z.size(1),3,3*3,-1)
        regr_co = torch.einsum('xz,yz->xy', imp_co.reshape(z.size(0),z.size(1),3,-1).permute(0,1,3,2).reshape(-1,3).float(), self.camera_matrix_inv.float())
        regr_co = regr_co.reshape(z.size(0),z.size(1),-1,3).permute(0,1,3,2).reshape(z.size(0),z.size(1),3,3*3,height*width)
        return regr_co
    
    def agg(self, x, ind=None, factor=None):
        fct = 0.
        if factor is not None:
            fct = factor
        x = x + ((x>0.).float()-1.)*(-fct)
        
        sizes = list(x.size())
        sizes[2] = 1
        sizes[3] = 1
        
        y01 = torch.cat([(x[:,:,0,1,:,1:,:]), torch.ones_like(x[:,:,0,1,:,:1,:])*fct],dim=-2).reshape(*sizes)
        y21 = torch.cat([torch.ones_like(x[:,:,2,1,:,-1:,:])*fct, (x[:,:,2,1,:,:-1,:])],dim=-2).reshape(*sizes)
        y10 = torch.cat([(x[:,:,1,0,:,:,1:]), torch.ones_like(x[:,:,1,0,:,:,:1])*fct],dim=-1).reshape(*sizes)
        y12 = torch.cat([torch.ones_like(x[:,:,1,2,:,:,-1:])*fct, (x[:,:,1,2,:,:,:-1])],dim=-1).reshape(*sizes)
        
        y00 = torch.cat([(x[:,:,0,0,:,1:,:]), torch.ones_like(x[:,:,0,0,:,:1,:])*fct],dim=-2).reshape(*sizes)
        y00 = torch.cat([(y00[:,:,0,0,:,:,1:]), torch.ones_like(y00[:,:,0,0,:,:,:1])*fct],dim=-1).reshape(*sizes)
        
        y02 = torch.cat([(x[:,:,0,2,:,1:,:]), torch.ones_like(x[:,:,0,2,:,:1,:])*fct],dim=-2).reshape(*sizes)
        y02 = torch.cat([torch.ones_like(y02[:,:,0,0,:,:,-1:])*fct, (y02[:,:,0,0,:,:,:-1])],dim=-1).reshape(*sizes)
        
        y20 = torch.cat([torch.ones_like(x[:,:,2,0,:,-1:,:])*fct, (x[:,:,2,0,:,:-1,:])],dim=-2).reshape(*sizes)
        y20 = torch.cat([(y20[:,:,0,0,:,:,1:]), torch.ones_like(y20[:,:,0,0,:,:,:1])*fct],dim=-1).reshape(*sizes)
        
        y22 = torch.cat([torch.ones_like(x[:,:,2,2,:,-1:,:])*fct, (x[:,:,2,2,:,:-1,:])],dim=-2).reshape(*sizes)
        y22 = torch.cat([torch.ones_like(y22[:,:,0,0,:,:,-1:])*fct, (y22[:,:,0,0,:,:,:-1])],dim=-1).reshape(*sizes)
        y11 = (x[:,:,1,1,:,:,:]).reshape(*sizes)
        
        sizes[2] = 3*3
        sizes[3] = 1
        
        x = torch.cat([y00,y01,y02,y10,y11,y12,y20,y21,y22],dim=2).reshape(*sizes)

        if ind is None:
            return x
        if sizes[4] == 4:
            sizes[4] = 1
            x0 = (x[:,:,:,:,:1,:,:]).reshape(*sizes)
            x1 = (x[:,:,:,:,1:2,:,:]).reshape(*sizes)
            x2 = (x[:,:,:,:,2:3,:,:]).reshape(*sizes)
            x3 = (x[:,:,:,:,3:,:,:]).reshape(*sizes)
            
            sizes[2] = 1
            sizes[3] = 1
            
            x0 = torch.sum(torch.where(ind,x0,torch.zeros_like(x0)),dim=2,keepdim=True)
            x1 = torch.sum(torch.where(ind,x1,torch.zeros_like(x1)),dim=2,keepdim=True)
            x2 = torch.sum(torch.where(ind,x2,torch.zeros_like(x2)),dim=2,keepdim=True)
            x3 = torch.sum(torch.where(ind,x3,torch.zeros_like(x3)),dim=2,keepdim=True)
            
            return torch.cat([x0.reshape(*sizes), x1.reshape(*sizes), x2.reshape(*sizes), x3.reshape(*sizes)], dim=4)
        
        sizes[4] = 1
        x0 = (x[:,:,:,:,:1,:,:]).reshape(*sizes)

        sizes[2] = 1
        sizes[3] = 1

        x0 = torch.sum(torch.where(ind,x0,torch.zeros_like(x0)),dim=2,keepdim=True)
        return x0.reshape(*sizes)
    
    def forward(self, x, target_2Dr, original_size):
        shapes = x.size()
        BATCH_SIZE, height, width = shapes[0], shapes[-2], shapes[-1]
        
        # 
        height = height + 2**0 + 2**0
        width = width + 2**0 + 2**0
        
        x_z_mask_value = torch.cat([torch.zeros_like(x[:,:,:,:,:(2**0)]), x, torch.zeros_like(x[:,:,:,:,:(2**0)])], dim=-1)
        x_z_mask_value = torch.cat([torch.zeros_like(x_z_mask_value[:,:,:,:(2**0),:]), x_z_mask_value, torch.zeros_like(x_z_mask_value[:,:,:,:(2**0),:])], dim=-2)
        
        panels = list(np.where(np.ones([height, width])))
        offset_codes = ((height-original_size[0]), (width-original_size[1]))
        panels[0] = panels[0] - (offset_codes[0]//2) + .5
        panels[1] = panels[1] - (offset_codes[1]//2) + .5
        
        # 
        z_values = (x_z_mask_value[:,:,2:3,:,:]).reshape(BATCH_SIZE,-1,1,height, width).detach()
        r_mask = (x_z_mask_value[:,:,3:4,:,:]).reshape(BATCH_SIZE,-1,1,height, width)
        x_z_value = x_z_mask_value[:,:,:3,:,:].reshape(BATCH_SIZE,-1,3,height, width)
        grouped_key = x_z_value.reshape(BATCH_SIZE,-1,3,1,height*width)
        
        updated_key_z = self.to_3d(z_values.reshape(BATCH_SIZE,-1,height, width), height, width, panels, original_size)
        query = updated_key_z.reshape(BATCH_SIZE,-1,3,3*3,height*width).detach()
        
        key_query = torch.sum(torch.abs(grouped_key[:,:,:2,:,:]-query[:,:,:2,:,:]),dim=2).reshape(BATCH_SIZE,-1,3*3,height, width)
        new_r_mask = torch.zeros_like(key_query)
        ind = torch.max(-key_query,dim=2,keepdim=True).indices
        ind_mask = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,height, width,3*3).permute(0,1,4,2,3).reshape(BATCH_SIZE,-1,3*3,height, width)
        new_r_mask[ind_mask>.5] = 1.
        new_r_mask = torch.where((r_mask+(query[:,:,0,:,:]).reshape(BATCH_SIZE,-1,3*3,height, width)*0.)>.5, new_r_mask, torch.zeros_like(new_r_mask))
        new_r_mask[:,:,4,:,:] = torch.where(r_mask.reshape(BATCH_SIZE,-1,height, width)>.5, new_r_mask[:,:,4,:,:], torch.ones_like(new_r_mask[:,:,4,:,:]))
        weights_b = (new_r_mask*(r_mask>.5).float()).reshape(BATCH_SIZE,-1,3,3,height, width)
        new_x_z_value = torch.einsum('bcsthw,bczhw->bcstzhw', weights_b.detach(), x_z_value)
        new_z_values = torch.einsum('bcsthw,bczhw->bcstzhw', weights_b, z_values).detach()
        new_r_mask = (new_r_mask*r_mask).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        new_x_z_mask_value = torch.cat([new_x_z_value, new_r_mask], dim=4)
        
        new_x_z_value = None
        weights = None
        
        ind = self.agg(new_z_values, factor=1e7)
        ind = torch.min(ind,dim=2,keepdim=True).indices
        ind = torch.where(torch.sum(self.agg(new_r_mask, factor=0.),dim=2,keepdim=True)>.5, ind, 0*ind+4)
        ind = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1,height, width,3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        ind = (ind>.5)
        new_x_z_mask_value = self.agg(new_x_z_mask_value, ind=ind).reshape(-1,4,height, width)
        weights = new_x_z_mask_value[:,-1:,:,:]
        
        if target_2Dr is not None:
            shapes = target_2Dr.size()
            BATCH_SIZE, C_zoom, h_zoom, w_zoom = shapes[0], shapes[1], shapes[-2], shapes[-1]
            
            target_2Dr, _ = torch.max(target_2Dr.reshape(batch_size,-1,1,h_zoom, w_zoom),dim=1,keepdim=False)
            
            key_query_grdf = -(key_query-key_query.detach())
            new_r_mask_grdf = (r_mask-r_mask.detach())

            weights = weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width).detach()
            weights = weights.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            weights = self.agg(weights).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            key_query_grdf = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+key_query_grdf.reshape(BATCH_SIZE,C_zoom,3*3,height, width)
            key_query_grdf = key_query_grdf.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            key_query_grdf = self.agg(key_query_grdf).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            new_r_mask_grdf = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+new_r_mask_grdf.reshape(BATCH_SIZE,C_zoom,1,height, width)
            new_r_mask_grdf = new_r_mask_grdf.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            new_r_mask_grdf = self.agg(new_r_mask_grdf).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            
            allow = torch.ones_like(weights).reshape(BATCH_SIZE,C_zoom,3*3,1,height, width)
            allow[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)] = (weights.reshape(BATCH_SIZE,C_zoom,3*3,1,height, width)[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])*target_2Dr.reshape(BATCH_SIZE,1,1,1,h_zoom, w_zoom)
            allow = self.agg(flip(allow,2).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)).reshape(BATCH_SIZE,C_zoom,3*3,1,1,height, width)
            allow = torch.max(allow,dim=2,keepdim=True)[0]
            allow = 1.-allow
            allow = torch.cat([allow, allow, allow],dim=2)
            allow = torch.cat([allow, allow, allow],dim=3).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
            allow = (self.agg(allow).reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
            
            new_x_z_mask_value = None
            weights = (weights.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
            key_query_grdf = (key_query_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
            new_r_mask_grdf = (new_r_mask_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
            
            weights_grdf = weights.detach()+new_r_mask_grdf+key_query_grdf*target_2Dr.reshape(BATCH_SIZE,1,h_zoom, w_zoom)
            
            weights = weights.detach() + weights_grdf*allow.detach()
            
        return new_x_z_mask_value, weights

class SMap(nn.Module):
    def __init__(self, n, camera_matrix):
        super(SMap,self).__init__()
        self.n = n
        self.smap3x3 = SMap3x3(camera_matrix)
    
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
            target_2Dr = target_2Dr.reshape(batch_size,1,height_zoom,C_zoom_2, width_zoom,C_zoom_2).permute(0,3,5,1,2,4).reshape(batch_size,C_zoom,1,height_zoom, width_zoom)
            if self.n==zoom:
                target = target_2Dr.reshape(batch_size,-1,height_zoom, width_zoom)
        x, weights = self.smap3x3(x, target, (height_zoom, width_zoom))
        _, _, h_out, w_out = weights.size()
        
        for i in range(self.n-zoom):
            C_zoom = C_zoom//4
            C_zoom_2 = C_zoom_2//2
            height_zoom = height_zoom*2
            width_zoom = width_zoom*2
            h_out = h_out*2
            w_out = w_out*2
            x = x.reshape(batch_size,2,C_zoom_2,2,C_zoom_2,4,h_out//2, w_out//2).permute(0,2,4,5,6,1,7,3).reshape(batch_size,C_zoom,4,h_out, w_out)
            if target_2Dr is not None:
                target_2Dr = target_2Dr.reshape(batch_size,2,C_zoom_2,2,C_zoom_2,1,height_zoom//2, width_zoom//2).permute(0,2,4,5,6,1,7,3).reshape(batch_size,C_zoom,height_zoom, width_zoom)
                if i==(self.n-zoom-1):
                    target = target_2Dr.reshape(batch_size,-1,height_zoom, width_zoom)

            x, weights = self.smap3x3(x, target, (height_zoom, width_zoom))
            _, _, h_out, w_out = weights.size()
        
        return weights