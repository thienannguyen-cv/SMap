import unittest
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
        self.smap3x3 = SMap3x3(self.input_mask.shape[0], self.input_mask.shape[1], self.camera, self.device).to(self.device)
        try:
            if not os.path.exists("./tests/vtest_data/output"):
                os.mkdir("./tests/vtest_data/output")
        except OSError as error:
            print(error)
        
    def test_in_x_1st_stage(self):
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left"]
        test_type = test_types[np.random.choice(range(len(test_types)), size=None)]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = smap3x3.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape).reshape(3,3,3, self.input_mask.shape[0], self.input_mask.shape[1])
        input_repr = nn.Parameter(input_repr[1:2,1:2,:,:,:], requires_grad=True).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_repr = nn.Parameter(torch.cat([input_repr, input_mask], dim=2), requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr)
        
        
        smap3x3.zero_grad()
        input_repr = self.vtestcase.testbot_in(input_repr)
        weights = smap3x3(input_repr, target_repr, self.input_mask.shape)
        weights = self.vtestcase.testbot_out(weights)
        weights_m = torch.max(weights.reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1]),dim=-1,keepdim=True).values
        
        loss = torch.sum(torch.abs(weights-target_repr), dim=1, keepdim=True)
        loss_m = torch.abs(weights_m-target_repr)
        
        loss = torch.mean(torch.sum(loss_m.detach()+(loss-loss.detach()), dim=-1))
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["out"] != 0).astype(int)
        expected = np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy").astype(int)
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg="Gradient test with 'above' target failed.")
        except Exception as e:
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            
    def test_in_y_1st_stage(self):
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left"]
        test_type = test_types[np.random.choice(range(len(test_types)), size=None)]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type}_target", testbot_in=TestBot_In(offset_in=1), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = smap3x3.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape).reshape(3,3,3, self.input_mask.shape[0], self.input_mask.shape[1])
        input_repr = nn.Parameter(input_repr[1:2,1:2,:,:,:], requires_grad=True).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_repr = nn.Parameter(torch.cat([input_repr, input_mask], dim=2), requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr)
        
        
        smap3x3.zero_grad()
        input_repr = self.vtestcase.testbot_in(input_repr)
        weights = smap3x3(input_repr, target_repr, self.input_mask.shape)
        weights = self.vtestcase.testbot_out(weights)
        weights_m = torch.max(weights.reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1]),dim=-1,keepdim=True).values
        
        loss = torch.sum(torch.abs(weights-target_repr), dim=1, keepdim=True)
        loss_m = torch.abs(weights_m-target_repr)
        
        loss = torch.mean(torch.sum(loss_m.detach()+(loss-loss.detach()), dim=-1))
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["out"] != 0).astype(int)
        expected = np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy").astype(int)
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg="Gradient test with 'above' target failed.")
        except Exception as e:
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
    
    def test_in_r_1st_stage(self):
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left"]
        test_type = test_types[np.random.choice(range(len(test_types)), size=None)]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type}_target", testbot_in=TestBot_In(offset_in=3), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = smap3x3.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape).reshape(3,3,3, self.input_mask.shape[0], self.input_mask.shape[1])
        input_repr = nn.Parameter(input_repr[1:2,1:2,:,:,:], requires_grad=True).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_repr = nn.Parameter(torch.cat([input_repr, input_mask], dim=2), requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr)
        
        
        smap3x3.zero_grad()
        input_repr = self.vtestcase.testbot_in(input_repr)
        weights = smap3x3(input_repr, target_repr, self.input_mask.shape)
        weights = self.vtestcase.testbot_out(weights)
        weights_m = torch.max(weights.reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1]),dim=-1,keepdim=True).values
        
        loss = torch.sum(torch.abs(weights-target_repr), dim=1, keepdim=True)
        loss_m = torch.abs(weights_m-target_repr)
        
        loss = torch.mean(torch.sum(loss_m.detach()+(loss-loss.detach()), dim=-1))
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["out"] != 0).astype(int)
        expected = np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy").astype(int)
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg="Gradient test with 'above' target failed.")
        except Exception as e:
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            

if __name__ == "__main__":
    unittest.main()