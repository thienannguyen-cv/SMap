import numpy as np
import torch

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
    return torch.max(x,dim=2,keepdim=False).values.reshape(BATCH_SIZE,1,h_out, w_out)

def save_for_vtest(path,activation_gradients, gradient_flows, input_representation, target_representation):
    import pickle
    
    flow_info = {"activation_gradients": activation_gradients, 
                 "gradient_flows": gradient_flows}
    with open(f'{path}/flow_info.pkl', 'wb') as f:
                    pickle.dump(flow_info, f)
    np.save(f"{path}/input_representation.npy", input_representation)
    np.save(f"{path}/target_representation.npy", target_representation)