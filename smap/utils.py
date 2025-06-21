import numpy as np
import torch
import torch.nn.functional as F
from smap import specials

DEBUG_FLAG = False

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

def to_3d3x3(z, height, width, panels, original_size, window_size, camera_matrix_inv, device):
    y_im, x_im = panels
    y_im, x_im = torch.from_numpy(y_im).reshape(height, width), torch.from_numpy(x_im).reshape(height, width)
    y_im = y_im * window_size[0] / original_size[0]
    x_im = x_im * window_size[1] / original_size[1]
    y_im, x_im = y_im.to(device), x_im.to(device)

    imp_co = torch.cat([torch.einsum('hw,bczhw->bczhw', x_im.float(), torch.ones_like(z.unsqueeze(2)).float()), torch.einsum('hw,bczhw->bczhw', y_im.float(), torch.ones_like(z.unsqueeze(2)).float()), torch.ones_like(z.unsqueeze(2))], 2)
    imp_co = F.unfold(imp_co.reshape(1, -1, height, width), kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=(1,1)).reshape(z.size(0),z.size(1),3,3*3,height,width)
    
    imp_co = torch.einsum('bchw,bczshw->bczshw', z.float(), imp_co.float()).reshape(z.size(0),z.size(1),3,3*3,-1)
    
    regr_co = torch.einsum('xz,yz->xy', imp_co.reshape(z.size(0),z.size(1),3,-1).permute(0,1,3,2).reshape(-1,3).float(), camera_matrix_inv.float())
    regr_co = regr_co.reshape(z.size(0),z.size(1),-1,3)
    return regr_co

def to_3d(z, height, width, panels, original_size, window_size, camera_matrix_inv, device):
    regr_co = to_3d3x3(z, height, width, panels, original_size, window_size, camera_matrix_inv, device).reshape(z.size(0),z.size(1),3,3,-1,3)
    return (regr_co[:,:,1,1,:,:]).permute(0,1,3,2).reshape(-1,3, height, width)
    
def agg(x, ind=None, factor=None):
    fct = 0.
    if factor is not None:
        fct = factor
    x = x + (x==0.).float()*fct
    
    sizes = list(x.size())
    sizes[2] = 3
    sizes[3] = 3
    
    x = x.reshape(*sizes)
    
    sizes[2] = 1
    sizes[3] = 1
    
    def abs_alignment(x, relx, rely, fct):
        x = 1.*(x[:,:,relx,rely,:,:,:])
        if relx<1:
            x = torch.cat([(x[:,:,:,1:,:]), torch.ones_like(x[:,:,:,:1,:])*fct],dim=-2)
        if relx>1:
            x = torch.cat([torch.ones_like(x[:,:,:,-1:,:])*fct, (x[:,:,:,:-1,:])],dim=-2)
        if rely<1:
            x = torch.cat([(x[:,:,:,:,1:]), torch.ones_like(x[:,:,:,:,:1])*fct],dim=-1)
        if rely>1:
            x = torch.cat([torch.ones_like(x[:,:,:,:,-1:])*fct, (x[:,:,:,:,:-1])],dim=-1)
        return x
    
    ys = []
    for i in range(3):
        for j in range(3):
            ys.append(abs_alignment(x, i, j, fct).reshape(*sizes))
    
    sizes[2] = 3*3
    sizes[3] = 1
    
    x = torch.cat(ys,dim=2).reshape(*sizes) # [y00,y01,y02,y10,y11,y12,y20,y21,y22]

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

def add_pad(x_value, y_value, z_value, r_mask, original_size):
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
    
    return x_value, y_value, z_value, r_mask, panels


        
def calculate_key_query(x_value, y_value, z_value, panels, window_size, original_size, camera_matrix_inv, device="cpu"):
    shapes = x_value.size()
    BATCH_SIZE, C_zoom, height, width = shapes[0], shapes[1], shapes[-2], shapes[-1]
    C_zoom_2 = int(np.sqrt(C_zoom))
    zoom = int(np.log2(C_zoom_2))

    grouped_key_x = x_value.reshape(BATCH_SIZE,-1,1,1,height*width)
    grouped_key_y = y_value.reshape(BATCH_SIZE,-1,1,1,height*width)
    #######################


    # 3. Prepare spatial placeholders for recifying gradients
    updated_key_z = to_3d3x3(z_value.reshape(BATCH_SIZE,-1,height, width), height, width, panels, original_size, window_size, camera_matrix_inv, device).permute(0,1,3,2).contiguous().reshape(BATCH_SIZE,-1,3,3*3,height*width)

    query_x = (updated_key_z[:,:,:1,:,:]).detach()
    query_y = (updated_key_z[:,:,1:2,:,:]).detach()

    diff_x = torch.sign(grouped_key_x-query_x).detach()*(grouped_key_x-query_x)
    diff_y = torch.sign(grouped_key_y-query_y).detach()*(grouped_key_y-query_y)
    key_query = torch.sum(diff_x+diff_y,dim=2)
    #######################

    return key_query.reshape(BATCH_SIZE,-1,3*3,height, width)

def recover_size(x, n, zoom=0):
    BATCH_SIZE, C_zoom, h_out, w_out = x.size()
    C_zoom_2 = int(np.sqrt(C_zoom))
    x = (1.*x).reshape(BATCH_SIZE,C_zoom,-1,h_out, w_out)
    for i in range(n-zoom,n):
        C_zoom = C_zoom//4
        C_zoom_2 = C_zoom_2//2
        h_out = h_out*2
        w_out = w_out*2
        x = x.reshape(BATCH_SIZE,C_zoom_2,2,C_zoom_2,2,-1,h_out//2, w_out//2).permute(0,1,3,5,6,2,7,4).reshape(BATCH_SIZE,C_zoom,-1,h_out, w_out)
    return x

def save_for_vtest(path,activation_gradients, gradient_flows, input_representation, target_representation):
    import pickle
    
    flow_info = {"activation_gradients": activation_gradients, 
                 "gradient_flows": gradient_flows}
    with open(f'{path}/flow_info.pkl', 'wb') as f:
                    pickle.dump(flow_info, f)
    np.save(f"{path}/input_representation.npy", input_representation)
    np.save(f"{path}/target_representation.npy", target_representation)
    