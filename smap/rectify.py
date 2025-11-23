import torch
from smap import specials, utils
from enum import Enum, auto

class types(Enum):
    DEF = auto()
    CAM = auto()
    DEP = auto()

class DefaultRectify(torch.nn.Module):
    def __init__(self, smap3x3):
        super(DefaultRectify,self).__init__()
        self.smap3x3 = smap3x3
    
    def compute_pre_allow_matrices(self, weight, target_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = weight.shape[0], weight.shape[-2], weight.shape[-1], target_repr.shape[-1], target_repr.shape[-2]
        
        weight = weight.reshape(BATCH_SIZE,-1,3*3,1,height, width)

        pre_allow = specials.INF*torch.ones_like(weight)
        pre_allow[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)] = (weight[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])*target_repr.reshape(BATCH_SIZE,-1,1,1,h_zoom, w_zoom)
        pre_allow_s = torch.where(pre_allow<specials.INF,torch.zeros_like(weight)+torch.sum((pre_allow==1.).float(),dim=2,keepdim=True),pre_allow)
        pre_allow = utils.agg(utils.flip(pre_allow_s,2).reshape(BATCH_SIZE,-1,3,3,1,height, width)).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        return pre_allow.detach()
    
    def prepare_flows_for_mask(self, pre_allow, weights_b, target_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = pre_allow.shape[0], pre_allow.shape[-2], pre_allow.shape[-1], target_repr.shape[-1], target_repr.shape[-2]
        
        allow = torch.max(torch.where(pre_allow<specials.INF,(pre_allow>0.).detach().float(),pre_allow),dim=2,keepdim=True).values
        allow = 1.-allow
        
        
        allow = 1.-(allow==0.).float()
        allow = torch.cat([allow, allow, allow],dim=2)
        allow = torch.cat([allow, allow, allow],dim=3).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        mask_flow = utils.agg(allow)
        
        mask_flow = (mask_flow.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        
        return mask_flow.detach()
    
    def prepare_flows_for_coord(self, pre_allow, weights_b, target_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = pre_allow.shape[0], pre_allow.shape[-2], pre_allow.shape[-1], target_repr.shape[-1], target_repr.shape[-2]
        
        allow = torch.max(torch.where(pre_allow<specials.INF,(pre_allow>0.).detach().float(),pre_allow),dim=2,keepdim=True).values
        allow = 1.-allow
        
        
        allow = (allow>.9).float()
        
        allow = torch.cat([allow, allow, allow],dim=2)
        allow = torch.cat([allow, allow, allow],dim=3).reshape(BATCH_SIZE,-1,3,3,1,height, width)
        coord_flow = (utils.agg(allow).reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        return coord_flow.detach()*target_repr.reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
    
    def forward(self, weights, weights_grdf, key_query_grdf, pre_mask_grdf=None):
        if pre_mask_grdf is not None:
            return weights.detach()+weights_grdf+key_query_grdf+pre_mask_grdf
        return weights.detach()+weights_grdf+key_query_grdf
        
    def rectificate_flow(self, new_x_z_mask_value, pre_x, pre_y, pre_z, pre_mask, panels, target, original_size):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        pre_key_query = utils.calculate_key_query(pre_x, pre_y, pre_z, panels, original_size, (self.smap3x3.window_h, self.smap3x3.window_w), self.smap3x3.camera_matrix_inv, device=self.smap3x3.device)
        pre_mask = pre_mask.reshape(BATCH_SIZE,-1,1,1,1,height, width)
        weights_b = (new_x_z_mask_value[:,:,:,:,-1:,:,:]).detach()
        
        
        # 7. Triggering gradient at the origins of the image rectification
        shapes = target.size()
        h_zoom, w_zoom = shapes[-2], shapes[-1]

        target_2Dr = target.reshape(BATCH_SIZE,1,1,1,h_zoom, w_zoom)
        
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


        pre_allow = self.compute_pre_allow_matrices((pre_weights>specials.OFF_THRESH).detach().float(), target_2Dr)
        mask_flow = 1.-(self.prepare_flows_for_mask(pre_allow, (weights_b>specials.OFF_THRESH).detach().float(), target_2Dr)==0.).float()
        coord_flow = self.prepare_flows_for_coord(pre_allow, (weights_b>specials.OFF_THRESH).detach().float(), target_2Dr)
        target_2Dr = target_2Dr.reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        
        weights = (weights.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        key_query_grdf = (key_query_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        new_r_mask = (new_r_mask.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        new_r_mask_grdf = new_r_mask-new_r_mask.detach()

        case = ((new_r_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float()
        n_case = (1.-mask_flow.detach())*(weights>specials.OFF_THRESH).float() # ((new_r_mask>specials.OFF_THRESH).float()==target_2Dr).float()
        factor = torch.where(new_r_mask_grdf>specials.OFF_THRESH, 1e-1*(1.-case), (1.-case))
        case = (1.-torch.max(n_case,dim=1,keepdim=True).values)*mask_flow.detach()*(1.-case)
        key_query_grdf = coord_flow.detach()*key_query_grdf
        weights_grdf = (case*factor+n_case).detach()*new_r_mask_grdf

        weights = self(weights, weights_grdf, key_query_grdf) # apply attractive rectification for this implementation
        #######################
        
        return weights

class CAMRectify(DefaultRectify):
    def __init__(self, smap3x3):
        super(CAMRectify, self).__init__(smap3x3)
        self.smap3x3 = smap3x3
    
    def rectificate_flow(self, new_x_z_mask_value, pre_x, pre_y, pre_z, pre_mask, panels, target, original_size):
        BATCH_SIZE, C_zoom, height, width = new_x_z_mask_value.shape[0], pre_mask.shape[1], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        k_d = torch.abs(pre_z)
        k_t = torch.max(torch.abs(torch.cat([pre_x, pre_y],dim=2)),dim=2,keepdim=True).values
        k_d = torch.where((k_d/(k_t+1.))>1., 1e-1/((k_d+1e-7)**(1-1e-5)/(k_d+1e-7)**(-1e-5))*(k_d**2), 3e-1*(k_d+1.)**(4)/((k_d+1.)**(6)+1.)*k_d)
        k_t = k_d.detach()/(pre_z+torch.exp(-1e7*pre_z.detach()))
        
        pre_key_query = utils.calculate_key_query((pre_x*k_t), (pre_y*k_t), pre_z*k_t, panels, original_size, (self.smap3x3.window_h, self.smap3x3.window_w), self.smap3x3.camera_matrix_inv, device=self.smap3x3.device)
        pre_key_query = pre_key_query - pre_key_query.min().detach() + 1.
        pre_mask = pre_mask.reshape(BATCH_SIZE,-1,1,1,1,height, width)
        
        weights_b = (new_x_z_mask_value[:,:,:,:,-1:,:,:]).detach()
        
        
        # 7. Triggering gradient at the origins of the image rectification
        shapes = target.size()
        h_zoom, w_zoom = shapes[-2], shapes[-1]

        target_2Dr = target.reshape(BATCH_SIZE,1,1,1,h_zoom, w_zoom)
        
        pre_weights = (new_x_z_mask_value[:,:,:,:,-1:,:,:]).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pre_weights = utils.agg(pre_weights).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pre_key_query = pre_key_query.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pre_key_query = utils.agg(pre_key_query).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pre_mask = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+pre_mask.reshape(BATCH_SIZE,C_zoom,1,height, width)
        pre_mask = pre_mask.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pre_mask = utils.agg(pre_mask).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)


        pre_allow = self.compute_pre_allow_matrices((pre_weights>specials.OFF_THRESH).detach().float(), target_2Dr)
        
        pre_weights = (pre_weights.reshape(BATCH_SIZE,C_zoom*3*3,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,C_zoom*3*3,h_zoom, w_zoom)
        pre_key_query = (pre_key_query.reshape(BATCH_SIZE,C_zoom*3*3,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,C_zoom*3*3,h_zoom, w_zoom)
        pre_mask = (pre_mask.reshape(BATCH_SIZE,C_zoom*3*3,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,C_zoom*3*3,h_zoom, w_zoom)
        
        target_2Dr = target_2Dr.reshape(BATCH_SIZE,1,h_zoom, w_zoom)
        mask_flow = self.prepare_flows_for_mask(pre_allow, (weights_b>specials.OFF_THRESH).detach().float(), target_2Dr)
        coord_flow = self.prepare_flows_for_coord(pre_allow, (weights_b>specials.OFF_THRESH).detach().float(), target_2Dr)
        mask_flow = mask_flow*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float())
        
        
        weights_grdf = pre_weights-pre_weights.detach()
        key_query_grdf = (pre_key_query.detach()-pre_key_query)
        pre_mask_grdf = pre_mask-pre_mask.detach()
        
        
        weights = pre_weights - (1.-(pre_weights>specials.OFF_THRESH).float().detach())*(.5*pre_mask+pre_weights)
        
        weights_grdf = weights_grdf*(pre_weights>specials.OFF_THRESH).float().detach()
        key_query_grdf = (2.*(weights>0.).float().detach()-1.)*key_query_grdf*coord_flow.detach()
        pre_mask_grdf = torch.where(pre_mask_grdf>specials.OFF_THRESH, -1e-1*pre_mask_grdf*mask_flow.detach()*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float()), pre_mask_grdf*(1e-3*(1.-mask_flow.detach())-1e-1*mask_flow.detach()*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float())))
        
        weights = self(weights, weights_grdf, key_query_grdf, pre_mask_grdf) # apply attractive rectification for this implementation
        #######################
        
        return weights
        
class DEPRectify(DefaultRectify):
    def __init__(self, smap3x3):
        super(DEPRectify, self).__init__(smap3x3)
        self.smap3x3 = smap3x3
        from tools.testing.vtest.vtest_types import TestBot_In_3_3, TestBot_Out_3_3, TestBot_Input_3_3, TestBot_Target, TestCase
        self.vtestcase = TestCase(name=f"SMap_Z", testbot_in=TestBot_In_3_3(), testbot_out=TestBot_Out_3_3(), testbot_input=TestBot_Input_3_3(), testbot_target=TestBot_Target(), out_path="./")

    def compute_pre_allow_matrices(self, weight, target_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = weight.shape[0], weight.shape[-2], weight.shape[-1], target_repr.shape[-1], target_repr.shape[-2]
        
        weight = weight.reshape(BATCH_SIZE,-1,3*3,1,height, width)

        pre_allow = specials.INF*torch.ones_like(weight)
        pre_allow[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)] = (weight[:,:,:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)])*target_repr.reshape(BATCH_SIZE,-1,1,1,h_zoom, w_zoom)
        s = torch.sum(torch.sum((pre_allow==1.).float(),dim=1,keepdim=True),dim=2,keepdim=True)
        pre_allow_s = torch.zeros_like(weight)+s
        pre_allow = utils.agg(utils.flip(pre_allow_s,2).reshape(BATCH_SIZE,-1,3,3,1,height, width)).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        return pre_allow.detach()
    
    def prepare_flows_for_mask(self, pre_allow, weights_b, target_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = pre_allow.shape[0], pre_allow.shape[-2], pre_allow.shape[-1], target_repr.shape[-1], target_repr.shape[-2]
        
        allow = torch.max(pre_allow,dim=2,keepdim=True).values

        n_flow = pre_allow*torch.max((weights_b).reshape(BATCH_SIZE,-1,3*3,1,1,height, width),dim=2,keepdim=True).values
        n_flow = utils.agg(n_flow.reshape(BATCH_SIZE,-1,3,3,1,height, width))
        
        y_flow = allow*torch.ones_like(weights_b).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        y_flow = utils.agg(y_flow.reshape(BATCH_SIZE,-1,3,3,1,height, width))
        
        n_flow = n_flow*torch.ones_like(n_flow)
        y_flow = y_flow*torch.ones_like(y_flow)
        
        n_flow = (n_flow.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        y_flow = (y_flow.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        
        return torch.where(n_flow==0.,-torch.ones_like(n_flow),n_flow), y_flow.detach()
    
    def prepare_flows_for_coord(self, pre_allow, weights_b, target_repr):
        BATCH_SIZE, height, width, w_zoom, h_zoom = pre_allow.shape[0], pre_allow.shape[-2], pre_allow.shape[-1], target_repr.shape[-1], target_repr.shape[-2]
        
        allow = torch.max(pre_allow,dim=2,keepdim=True).values
        allow = (allow<specials.INF).detach().float()*(~(allow==1.)).detach().float()*torch.max((weights_b).reshape(BATCH_SIZE,-1,3*3,1,1,height, width),dim=2,keepdim=True).values*torch.ones_like(weights_b).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)

        coord_flow = (utils.agg(allow).reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        return coord_flow.detach()
    
    def rectificate_flow(self, new_x_z_mask_value, pre_x, pre_y, pre_z, pre_mask, panels, target, original_size):
        BATCH_SIZE, C_zoom, height, width = new_x_z_mask_value.shape[0], pre_mask.shape[1], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        
        k_d = torch.abs(pre_z)
        k_t = torch.max(torch.abs(torch.cat([pre_x, pre_y],dim=2)),dim=2,keepdim=True).values
        k_d = torch.where((k_d/(k_t+1.))>1., 1e-1/((k_d+1e-7)**(1-1e-5)/(k_d+1e-7)**(-1e-5))*(k_d**2), 3e-1*(k_d+1.)**(4)/((k_d+1.)**(6)+1.)*k_d)
        k_t = k_d.detach()/(pre_z+torch.exp(-1e7*pre_z.detach()))
        
        neg_pre_key_query = utils.calculate_key_query(pre_x*k_t.detach(), pre_y*k_t.detach(), (pre_z*k_t).detach(), panels, original_size, (self.smap3x3.window_h, self.smap3x3.window_w), self.smap3x3.camera_matrix_inv, device=self.smap3x3.device)
        pos_pre_key_query = utils.calculate_key_query((pre_x*k_t), (pre_y*k_t), pre_z*k_t, panels, original_size, (self.smap3x3.window_h, self.smap3x3.window_w), self.smap3x3.camera_matrix_inv, device=self.smap3x3.device)
        
        pre_mask = pre_mask.reshape(BATCH_SIZE,-1,1,1,1,height, width)
        
        weights_b = (new_x_z_mask_value[:,:,:,:,-1:,:,:]).detach()
        
        
        # 7. Triggering gradient at the origins of the image rectification
        shapes = target.size()
        h_zoom, w_zoom = shapes[-2], shapes[-1]

        target_2Dr = target.reshape(BATCH_SIZE,1,1,1,h_zoom, w_zoom)
        
        # testing/target
        self.vtestcase.orig_shape = (h_zoom, w_zoom)
        self.vtestcase.testbot_target(target_2Dr.reshape(BATCH_SIZE, 1, h_zoom, w_zoom)+torch.zeros_like(pre_mask.reshape(BATCH_SIZE, -1,height, width)[:1,:,:1,:1]))

        
        neg_key_query_grdf = -(neg_pre_key_query-neg_pre_key_query.detach())+1.
        pos_key_query_grdf = -(pos_pre_key_query-pos_pre_key_query.detach())+1.

        neg_key_query_grdf = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+neg_key_query_grdf.reshape(BATCH_SIZE,C_zoom,3*3,height, width)
        neg_key_query_grdf = neg_key_query_grdf.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        neg_key_query_grdf = utils.agg(neg_key_query_grdf).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        neg_key_query_grdf = neg_key_query_grdf-1.
        
        pos_key_query_grdf = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+pos_key_query_grdf.reshape(BATCH_SIZE,C_zoom,3*3,height, width)
        # testing/in
        pos_key_query_grdf = self.vtestcase.testbot_in(pos_key_query_grdf.reshape(BATCH_SIZE,C_zoom*3*3,height, width))
        pos_key_query_grdf = pos_key_query_grdf.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pos_key_query_grdf = utils.agg(pos_key_query_grdf).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pos_key_query_grdf = pos_key_query_grdf-1.
        
        self.vtestcase.testbot_input(pre_mask.reshape(BATCH_SIZE,C_zoom,1,height, width))
        pre_mask = torch.zeros_like(weights_b.reshape(BATCH_SIZE,C_zoom,3*3,height, width))+(pre_mask).reshape(BATCH_SIZE,C_zoom,1,height, width)
        pre_mask = (pre_mask).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        pre_mask = utils.agg(pre_mask).reshape(BATCH_SIZE,C_zoom,3,3,1,height, width)
        weight = utils.agg(weights_b.reshape(BATCH_SIZE,C_zoom,3,3,1,height, width))
        
        pre_allow = self.compute_pre_allow_matrices((weight>specials.OFF_THRESH).detach().float(), target_2Dr)
        
        coord_flow = self.prepare_flows_for_coord(pre_allow, (weights_b>specials.OFF_THRESH).detach().float(), target_2Dr)
        n_flow, y_flow = self.prepare_flows_for_mask(pre_allow, (weights_b>specials.OFF_THRESH).detach().float(), target_2Dr)
        
        
        neg_key_query_grdf = (neg_key_query_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        pos_key_query_grdf = (pos_key_query_grdf.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        # testing/out
        pos_key_query_grdf = self.vtestcase.testbot_out(pos_key_query_grdf)
        pre_mask = (pre_mask.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        weight = (weight.reshape(BATCH_SIZE,-1,height, width)[:,:,((height-h_zoom)//2):((height+h_zoom)//2),((width-w_zoom)//2):((width+w_zoom)//2)]).reshape(BATCH_SIZE,-1,h_zoom, w_zoom)
        # testing/input
        self.vtestcase.testbot_input(n_flow.reshape(BATCH_SIZE,C_zoom,3*3,h_zoom, w_zoom), filename="_x_flow.npy")
        self.vtestcase.testbot_input(y_flow.reshape(BATCH_SIZE,C_zoom,3*3,h_zoom, w_zoom), filename="_y_flow.npy")
        
        target_2Dr = target_2Dr.reshape(BATCH_SIZE,1,h_zoom, w_zoom)
        
        weight_grdf = (weight)-weight.detach()
        pre_mask_grdf = (pre_mask)-pre_mask.detach()
        
        
        weights = torch.where(pre_mask>specials.OFF_THRESH, torch.where(weight>specials.OFF_THRESH,pre_mask,.5*pre_mask), -.5*pre_mask)
        weights_grdf = (1.-2.*(weights<0.).float().detach())*weight_grdf*3e-1*(y_flow==1.).detach().float()*(weight>0.).detach().float()+torch.where(pre_mask>specials.OFF_THRESH, pre_mask_grdf*3e0*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float()), -pre_mask_grdf*1e1*(1.-((pre_mask>specials.OFF_THRESH).long()==target_2Dr.long()).float()))*(1.-(y_flow==1.).detach().float()) + (1.-2.*(weights<0.).float().detach())*torch.where(n_flow.detach()>1.,neg_key_query_grdf*3e-1*(1.-n_flow/torch.max(n_flow)).detach(),pos_key_query_grdf*coord_flow.detach())*(target_2Dr>specials.OFF_THRESH).detach().float()
        weights = weights.detach() + weights_grdf # apply attractive rectification for this implementation
        #######################
        self.vtestcase.testbot_input(weights.reshape(BATCH_SIZE,C_zoom,3*3,h_zoom, w_zoom), filename="_output.npy")
        
        return weights
    
