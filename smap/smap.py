import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smap import specials, utils, rectify

class SMap3x3(nn.Module):
    def __init__(self, window_h, window_w, camera_matrix, device):
        super(SMap3x3,self).__init__()
        self.k_const = 1.
        self.window_h = window_h
        self.window_w = window_w
        self.camera_matrix = nn.Parameter(torch.from_numpy(camera_matrix), requires_grad=False)
        self.camera_matrix_inv = nn.Parameter(torch.from_numpy(np.linalg.inv(camera_matrix)), requires_grad=False)
        self.device = device

        self.sm = nn.Softmax(dim=2)
        # Testbot (HDVO instrumentation) is built ONLY in audit mode (utils.DEBUG_FLAG).
        # When DEBUG is off, no vtest object is constructed and no backward hooks are
        # registered; sentinels keep the data path identical (identity passthroughs).
        if utils.DEBUG_FLAG:
            from tools.testing.vtest.vtest_types import TestBot_In_3_3, TestBot_Out_3_3, TestBot_Input_3_3, TestBot_Target, TestCase
            self.vtestcase = TestCase(name=f"SMap_Z", testbot_in=TestBot_In_3_3(), testbot_out=TestBot_Out_3_3(), testbot_input=TestBot_Input_3_3(), testbot_target=TestBot_Target(), out_path="./")
            self.testbot_out_z = TestBot_Out_3_3(name="None",connet2name="z_out")
            self.testbot_out_z.testcase = self.vtestcase
            self.testbot_out_r = TestBot_Out_3_3(name="None",connet2name="r_out")
            self.testbot_out_r.testcase = self.vtestcase
        else:
            from tools.testing.vtest.vtest_types import _DisabledTestbot
            self.vtestcase = _DisabledTestbot()
            self.testbot_out_z = _DisabledTestbot()
            self.testbot_out_r = _DisabledTestbot()

    def go(self, x, original_size, is_last=False):
        pre_x, pre_y, pre_z, pre_mask = x[:,:,:1,:,:], x[:,:,1:2,:,:], x[:,:,2:3,:,:], x[:,:,3:4,:,:]

        pre_x, pre_y, pre_z, pre_mask, panels = utils.add_pad(pre_x, pre_y, pre_z, pre_mask, original_size)
        return self(pre_x, pre_y, pre_z, pre_mask, panels, original_size, is_last=is_last)

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
        temp_rand = torch.rand_like(temp[:,:,4:5,:,:])+torch.zeros_like(weights_b)
        np_rand = (np.random.rand(1)[0])

        weights_b = torch.where(((z_values>0.).detach().float()+torch.zeros_like(weights_b))>.5, weights_b, temp)
        if (self.k_const>np_rand):
            weights_b = torch.where(((r_mask>specials.OFF_THRESH).float()*(z_values>0.).detach().float()+torch.zeros_like(weights_b))>.5, weights_b, torch.where(((z_values>0.).detach().float()+torch.zeros_like(weights_b))>.5, temp, torch.zeros_like(weights_b)))

        weights_m = torch.where((r_mask+torch.zeros_like(weights_b))>specials.OFF_THRESH, torch.where(~(z_values>0.), temp, weights_b), torch.where(temp_rand>self.k_const,temp,weights_b))
        weights_x = torch.where((r_mask+torch.zeros_like(weights_b))>specials.OFF_THRESH, torch.where(~(z_values>0.), torch.zeros_like(weights_b), weights_m), weights_b)


        if is_last==False:
            weights_m = torch.where((((r_mask>specials.OFF_THRESH).float()*(z_values>0.).detach().float())+torch.zeros_like(weights_x))>.5, weights_b, torch.where(temp_rand>self.k_const,weights_b,temp))
            weights_x = torch.where(((r_mask>specials.OFF_THRESH).float()*(z_values>0.).detach().float()+torch.zeros_like(weights_b))>.5, weights_m, torch.where(temp_rand>self.k_const,weights_m,torch.zeros_like(weights_b)))

            self.vtestcase.testbot_input((r_mask.reshape(BATCH_SIZE,-1,weights_x.size(-2),weights_x.size(-1))), filename="_input_representationf.npy")
            self.vtestcase.testbot_input(temp_rand.reshape(BATCH_SIZE,-1,3*3,weights_b.size(-2),weights_b.size(-1)), filename="_tmpf.npy", dim=4)
            self.vtestcase.testbot_input((torch.zeros_like(temp_rand)+np_rand).reshape(BATCH_SIZE,-1,3*3,weights_b.size(-2),weights_b.size(-1)), filename="_rnpf.npy", dim=4)
            self.vtestcase.testbot_input(weights_b.reshape(BATCH_SIZE,-1,3*3,weights_b.size(-2),weights_b.size(-1)), filename="_wbf.npy", dim=4)

            x_value = x_value.reshape(BATCH_SIZE,-1,1,1,1,height, width)+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,height, width)
            y_value = y_value.reshape(BATCH_SIZE,-1,1,1,1,height, width)+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,height, width)
            z_value = z_value.reshape(BATCH_SIZE,-1,1,1,1,height, width)+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,height, width)
            r_mask = r_mask.reshape(BATCH_SIZE,-1,1,1,1,height, width)+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,height, width)
            x_z_value = torch.cat([x_value, y_value, z_value], dim=-3).reshape(BATCH_SIZE,-1,3,3,3,weights_x.size(-2),weights_x.size(-1))
            r_mask = r_mask.reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1))


            self.vtestcase.testbot_input(z_value.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_zf.npy", dim=4)
            self.vtestcase.testbot_input(r_mask.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_rf.npy", dim=4)
            self.vtestcase.testbot_input(weights_x.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_xf.npy", dim=4)
            self.vtestcase.testbot_input(weights_m.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_mf.npy", dim=4)

            new_x_z_value = torch.einsum('bcsthw,bcstzhw->bcstzhw', weights_x.detach().reshape(BATCH_SIZE,-1,3,3,height, width), x_z_value)

            self.vtestcase.testbot_input((new_x_z_value[:,:,:,:,:1,:,:]).reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_xvf.npy", dim=4)
            self.vtestcase.testbot_input((new_x_z_value[:,:,:,:,1:2,:,:]).reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_yvf.npy", dim=4)
            self.vtestcase.testbot_input((new_x_z_value[:,:,:,:,2:,:,:]).reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_zvf.npy", dim=4)

            new_r_mask = torch.einsum('bcsthw,bcstzhw->bcstzhw', weights_m.detach().reshape(BATCH_SIZE,-1,3,3,height, width), r_mask)
            self.vtestcase.testbot_input(new_r_mask.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_rmf.npy", dim=4)

        else:
            self.vtestcase.testbot_input(temp_rand.reshape(BATCH_SIZE,-1,3*3,weights_b.size(-2),weights_b.size(-1)), filename="_tmpl.npy", dim=4)
            self.vtestcase.testbot_input((torch.zeros_like(temp_rand)+np_rand).reshape(BATCH_SIZE,-1,3*3,weights_b.size(-2),weights_b.size(-1)), filename="_rnpl.npy", dim=4)
            self.vtestcase.testbot_input(weights_b.reshape(BATCH_SIZE,-1,3*3,weights_b.size(-2),weights_b.size(-1)), filename="_wbl.npy", dim=4)

            self.vtestcase.testbot_input((r_mask.reshape(BATCH_SIZE,-1,weights_x.size(-2),weights_x.size(-1))))

            x_value = utils.agg(x_value.reshape(BATCH_SIZE,-1,1,1,1,weights_x.size(-2),weights_x.size(-1))+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            y_value = utils.agg(y_value.reshape(BATCH_SIZE,-1,1,1,1,weights_x.size(-2),weights_x.size(-1))+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            z_value = utils.agg(z_value.reshape(BATCH_SIZE,-1,1,1,1,weights_x.size(-2),weights_x.size(-1))+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            r_mask = utils.agg(r_mask.reshape(BATCH_SIZE,-1,1,1,1,weights_x.size(-2),weights_x.size(-1))+torch.zeros_like(weights_x).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            # testing/out
            z_value = self.testbot_out_z(z_value.reshape(BATCH_SIZE,-1,weights_x.size(-2),weights_x.size(-1)))
            r_mask = self.testbot_out_r(r_mask.reshape(BATCH_SIZE,-1,weights_x.size(-2),weights_x.size(-1)))
            self.vtestcase.testbot_input(z_value.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_zb.npy", dim=4)

            x_value = utils.agg(utils.flip(x_value.reshape(BATCH_SIZE,-1,3*3,1,1,weights_x.size(-2),weights_x.size(-1)),2).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            y_value = utils.agg(utils.flip(y_value.reshape(BATCH_SIZE,-1,3*3,1,1,weights_x.size(-2),weights_x.size(-1)),2).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            z_value = utils.agg(utils.flip(z_value.reshape(BATCH_SIZE,-1,3*3,1,1,weights_x.size(-2),weights_x.size(-1)),2).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            r_mask = utils.agg(utils.flip(r_mask.reshape(BATCH_SIZE,-1,3*3,1,1,weights_x.size(-2),weights_x.size(-1)),2).reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1)))
            x_z_value = torch.cat([x_value, y_value, z_value], dim=-3).reshape(BATCH_SIZE,-1,3,3,3,weights_x.size(-2),weights_x.size(-1))
            r_mask = r_mask.reshape(BATCH_SIZE,-1,3,3,1,weights_x.size(-2),weights_x.size(-1))


            self.vtestcase.testbot_input(z_value.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_zl.npy", dim=4)
            self.vtestcase.testbot_input(r_mask.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_rl.npy", dim=4)
            self.vtestcase.testbot_input(weights_x.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_xl.npy", dim=4)
            self.vtestcase.testbot_input(weights_m.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_ml.npy", dim=4)

            new_x_z_value = torch.einsum('bcsthw,bcstzhw->bcstzhw', weights_x.detach().reshape(BATCH_SIZE,-1,3,3,height, width), x_z_value)

            self.vtestcase.testbot_input((new_x_z_value[:,:,:,:,:1,:,:]).reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_xvl.npy", dim=4)
            self.vtestcase.testbot_input((new_x_z_value[:,:,:,:,1:2,:,:]).reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_yvl.npy", dim=4)
            self.vtestcase.testbot_input((new_x_z_value[:,:,:,:,2:,:,:]).reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_zvl.npy", dim=4)

            new_r_mask = torch.einsum('bcsthw,bcstzhw->bcstzhw', weights_m.detach().reshape(BATCH_SIZE,-1,3,3,height, width), r_mask)
            self.vtestcase.testbot_input(new_r_mask.reshape(BATCH_SIZE,-1,3*3,weights_x.size(-2),weights_x.size(-1)), filename="_rml.npy", dim=4)

        new_x_z_value = torch.einsum('bcsthw,bcstzhw->bcstzhw', weights_x.detach().reshape(BATCH_SIZE,-1,3,3,height, width), x_z_value)

        new_r_mask = torch.einsum('bcsthw,bcstzhw->bcstzhw', weights_m.detach().reshape(BATCH_SIZE,-1,3,3,height, width), r_mask)
        new_x_z_mask_value = torch.cat([new_x_z_value, new_r_mask], dim=4)
        #######################

        return x_value, y_value, z_value, r_mask, panels, new_x_z_mask_value

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
        if rectify_type==rectify.types.DEP:
            self.rectify_module = rectify.DEPRectify(self.smap3x3)
        if rectify_type==rectify.types.FUT:
            self.rectify_module = rectify.FUTRectify(self.smap3x3)

    def calculate_weights(self, new_x_z_mask_value, original_size=None, zoom=0):
        BATCH_SIZE, height, width = new_x_z_mask_value.shape[0], new_x_z_mask_value.shape[-2], new_x_z_mask_value.shape[-1]
        new_z_max = torch.max(torch.max(torch.max(torch.max(new_x_z_mask_value[:,:,:,:,-2:-1,:,:],dim=2,keepdim=True).values,dim=3,keepdim=True).values,dim=-2,keepdim=True).values,dim=-1,keepdim=True).values
        new_z_values = (new_x_z_mask_value[:,:,:,:,-2:-1,:,:])
        new_z_values_rand = torch.rand_like(new_z_values)
        new_z_values = new_z_values+1e-5*new_z_values_rand
        # Per-channel z_max: sufficient because z-buffer competition is within-channel (dim=2 min)
        self.smap3x3.vtestcase.testbot_input((new_x_z_mask_value[:,:,:,:,-1:,:,:]).reshape(BATCH_SIZE,-1,3*3,height,width), filename="_cal_i.npy", dim=4)
        self.smap3x3.vtestcase.testbot_input(new_z_max.reshape(BATCH_SIZE,-1,1,1,1)+torch.zeros_like(new_z_values).reshape(BATCH_SIZE,-1,3*3,height,width), filename="_cal_z_max.npy", dim=4)
        self.smap3x3.vtestcase.testbot_input(new_z_values.reshape(BATCH_SIZE,-1,3*3,height,width), filename="_cal_z.npy", dim=4)
        self.smap3x3.vtestcase.testbot_input(new_z_values_rand.reshape(BATCH_SIZE,-1,3*3,height,width), filename="_cal_z_rd.npy", dim=4)
        ind = (torch.where((new_x_z_mask_value[:,:,:,:,-1:,:,:])>specials.OFF_THRESH, new_z_values, torch.where(~((new_x_z_mask_value[:,:,:,:,-2:-1,:,:])>0.),new_z_max+1e-5,(new_x_z_mask_value[:,:,:,:,-2:-1,:,:])+1e-5*(self.smap3x3.k_const)))).detach()
        self.smap3x3.vtestcase.testbot_input(ind.reshape(BATCH_SIZE,-1,3*3,height,width), filename="_cal_ind.npy", dim=4)
        pre_ind = utils.agg(ind, factor=specials.INF)

        val, ind = torch.min(pre_ind,dim=2,keepdim=True)
        ind = torch.where((val>0.)&(val<specials.INF), ind, 0*ind+4)
        ind4 = 0*ind+4
        ind = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1,height, width,3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        ind = (ind>.0)
        ind4 = F.one_hot(ind4, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1,height, width,3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1,height, width)
        ind4 = (ind4>.0)
        weights = utils.agg(new_x_z_mask_value, ind=ind).reshape(-1,4,height, width)
        weights4 = utils.agg(new_x_z_mask_value, ind=ind4).reshape(-1,4,height, width)
        weights = torch.where(((weights[:,-2:-1,:,:]>0.).float()+torch.zeros_like(weights))>.5, weights, weights4)
        weights4 = torch.cat([(weights[:,:-1,:,:]).detach(), weights[:,-1:,:,:]],dim=1)
        weights = torch.where(((weights[:,-1:,:,:])+torch.zeros_like(weights))>specials.OFF_THRESH, weights, weights4)


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

        self.smap3x3.vtestcase.name = f"SMap_Z_{int(self.smap3x3.k_const*10)}"
        # testing/target
        self.smap3x3.vtestcase.orig_shape = (height, width)
        if target is not None:
            self.smap3x3.vtestcase.testbot_target(target)

        for i in range(self.n):
            height_zoom = height_zoom // 2
            width_zoom = width_zoom // 2
            x = x.reshape(BATCH_SIZE,C_zoom_2,C_zoom_2,4,height_zoom,2, width_zoom,2).permute(0,1,5,2,7,3,4,6).contiguous().reshape(BATCH_SIZE,-1,4,height_zoom, width_zoom)

            C_zoom_2 = C_zoom_2 * 2

        pre_x, pre_y, pre_z, pre_mask, panels = None, None, None, None, None
        pre_sign = torch.sign(x[:,:,2:3,:,:]).detach()
        pre_x, pre_y, pre_z, pre_mask = pre_sign*x[:,:,:1,:,:], pre_sign*x[:,:,1:2,:,:], pre_sign*x[:,:,2:3,:,:], x[:,:,3:4,:,:]
        x = (torch.cat([pre_x, pre_y, pre_z, pre_mask], dim=2))
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
