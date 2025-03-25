import numpy as np
import torch
import torch.nn.functional as F
from smap import specials


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

def recover_size(x, n, zoom):
    BATCH_SIZE, C_zoom, h_out, w_out = x.size()
    C_zoom = C_zoom//(3*3)
    C_zoom_2 = int(np.sqrt(C_zoom))
    x = (1.*x).reshape(BATCH_SIZE,-1,3*3,h_out, w_out)
    for i in range(zoom):
        C_zoom = C_zoom//4
        C_zoom_2 = C_zoom_2//2
        h_out = h_out*2
        w_out = w_out*2
        x = x.reshape(BATCH_SIZE,2,C_zoom_2,2,C_zoom_2,3*3,h_out//2, w_out//2).permute(0,2,4,5,6,1,7,3).reshape(BATCH_SIZE,C_zoom,3*3,h_out, w_out)
    return torch.max((x>specials.OFF_THRESH).float(),dim=2,keepdim=False).values.reshape(BATCH_SIZE,1,h_out, w_out)

def save_for_vtest(path,activation_gradients, gradient_flows, input_representation, target_representation):
    import pickle
    
    flow_info = {"activation_gradients": activation_gradients, 
                 "gradient_flows": gradient_flows}
    with open(f'{path}/flow_info.pkl', 'wb') as f:
                    pickle.dump(flow_info, f)
    np.save(f"{path}/input_representation.npy", input_representation)
    np.save(f"{path}/target_representation.npy", target_representation)
    