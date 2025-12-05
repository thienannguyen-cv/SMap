import unittest
import os
import numpy as np
import torch
from torch import nn
import pickle
from smap import *
from tools.testing.vtest.vtest_types import *


class SMap3x3VTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.input_mask = torch.from_numpy(np.load("./tests/vtest_data/smap3x3/input.npy"))
        self.camera = np.array([[2304.5479, 0,  1686.2379], 
                                [0, 2305.8757, -0.0151],
                                [0, 0, 1.]], dtype=np.float32)
        self.panel = list(np.where(np.ones([self.input_mask.shape[0], self.input_mask.shape[1]])))
        self.panel[0] = self.panel[0] + .5
        self.panel[1] = self.panel[1] + .5
        self.smap = SMap(self.input_mask.shape[0], self.input_mask.shape[1], self.camera, rectify_type=rectify.types.CAM, device=self.device).to(self.device)
        try:
            if not os.path.exists("./tests/vtest_data/output"):
                os.mkdir("./tests/vtest_data/output")
        except OSError as error:
            print(error)
        
    def test_in_x_1st_stage(self):
        BATCH_SIZE = 1
        
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left"]
        test_id = np.random.choice(range(len(test_types)), size=None)
        test_type = test_types[test_id]
        # testing/testcase
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(BATCH_SIZE,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap.smap3x3.camera_matrix_inv, self.device)
        input_repr_x = nn.Parameter(input_repr.reshape(BATCH_SIZE,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr.reshape(BATCH_SIZE,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr.reshape(BATCH_SIZE,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(BATCH_SIZE,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(BATCH_SIZE,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        # testing/target
        
        
        self.smap.smap3x3.zero_grad()
        # testing/in
        weights = self.smap.smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, self.panel, self.input_mask.shape)
        target_2Dr, weights = self.smap.rectify_module.rectificate_flow(weights, input_repr_x, input_repr_y, input_repr_z, input_mask, self.panel, target_repr, self.input_mask.shape)
        weights = weights.reshape(BATCH_SIZE,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        # testing/out
        weights = torch.abs((weights)+1e-7)
        loss_m = torch.abs(weights-target_2Dr)
    
        loss_m = loss_m.reshape(BATCH_SIZE, -1).sum(dim=1)

        loss = torch.mean(loss_m)
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["in"] != 0).astype(int)
            
        expected = np.load(f"./tests/vtest_data/smap3x3/input.npy").astype(int)
        if test_id==1 or test_id==5:
            expected = 0.*expected
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg=f"Gradient test with {test_type} target failed.")
        except Exception as e:
            print(f"z: {z}")
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            

if __name__ == "__main__":
    unittest.main()