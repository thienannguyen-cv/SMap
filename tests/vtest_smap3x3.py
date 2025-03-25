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
        self.panel[0] = self.panel[0] + .5
        self.panel[1] = self.panel[1] + .5
        self.smap3x3 = SMap3x3(self.input_mask.shape[0], self.input_mask.shape[1], self.camera, self.device).to(self.device)
        try:
            if not os.path.exists("./tests/vtest_data/output"):
                os.mkdir("./tests/vtest_data/output")
        except OSError as error:
            print(error)
        
    def test_in_x_1st_stage(self):
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left"]
        test_id = np.random.choice(range(len(test_types)), size=None)
        test_type = test_types[test_id]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap3x3.camera_matrix_inv, self.device)
        input_repr_x = nn.Parameter(input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr).detach()
        
        
        smap3x3.zero_grad()
        input_repr_x = self.vtestcase.testbot_in(input_repr_x, input_mask)
        weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        weights = torch.abs(self.vtestcase.testbot_out(weights)+1e-7)
        loss_m = torch.abs(weights-target_repr)
    
        loss_m = loss_m.reshape(1, -1).sum(dim=1)

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
            
    def test_in_x_2st_stage(self):
        test_type_param_dict = {"above_left": [0,0], "above": [0,1], "above_right": [0,2], "right": [1,0], "below_right": [2,0], "below": [2,1], "below_left": [2,2], "left": [1,2], "still": [1,1]}
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left", "still"]
        test_id = [np.random.choice(range(len(test_types)), size=None), np.random.choice(range(len(test_types)), size=None)]
        test_type = [test_types[test_id[0]], test_types[test_id[1]]]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type[0]}_{test_type[1]}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        offsetx, offsety = test_type_param_dict[test_type[0]]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap3x3.camera_matrix_inv, self.device)
        input_repr = input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])
        active_point_img_coords = np.where(self.input_mask!=0)
        temp = (input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]]).cpu().numpy()
        input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]] = 0.*(input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]])
        input_repr[0,0,:,active_point_img_coords[0]+offsetx-1, active_point_img_coords[1]+offsety-1] = 0.+torch.from_numpy(temp)
        input_repr_x = nn.Parameter(input_repr[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        temp = (input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]]).cpu().numpy()
        input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]] = 0*(input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]])
        input_mask[0,0,:,active_point_img_coords[0]+offsetx-1, active_point_img_coords[1]+offsety-1] = 0.+torch.from_numpy(temp)
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type[1]}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr).detach()
        
        smap3x3.zero_grad()
        input_repr_x = self.vtestcase.testbot_in(input_repr_x, input_mask)
        weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, None, self.input_mask.shape)
        weights = smap3x3.go(weights[:,None,:,:,:], target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        weights = torch.abs(self.vtestcase.testbot_out(weights)+1e-7)
        loss_m = torch.abs(weights-target_repr)
    
        loss_m = loss_m.reshape(1, -1).sum(dim=1)

        loss = torch.mean(loss_m)
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type[0]}_{test_type[1]}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["in"] != 0).astype(int)
            
        expected = input_mask.detach().reshape(input_mask.shape[-2], input_mask.shape[-1]).cpu().numpy()
        if test_id[1]==1 or test_id[1]==5 or test_types[test_id[1]]=="still":
            expected = 0.*expected
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg=f"Gradient test with {test_type} target failed.")
        except Exception as e:
            print(f"offsetx, offsety: {(offsetx, offsety)}")
            print(f"z: {z}")
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            
    def test_in_y_1st_stage(self):
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left"]
        test_id = np.random.choice(range(len(test_types)), size=None)
        test_type = test_types[test_id]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap3x3.camera_matrix_inv, self.device)
        input_repr_x = nn.Parameter(input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr).detach()
        
        
        smap3x3.zero_grad()
        input_repr_y = self.vtestcase.testbot_in(input_repr_y, input_mask)
        weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        weights = torch.abs(self.vtestcase.testbot_out(weights)+1e-7)
        loss_m = torch.abs(weights-target_repr)
    
        loss_m = loss_m.reshape(1, -1).sum(dim=1)

        loss = torch.mean(loss_m)
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["in"] != 0).astype(int)
            
        expected = np.load(f"./tests/vtest_data/smap3x3/input.npy").astype(int)
        if test_id==3 or test_id==7:
            expected = 0.*expected
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg=f"Gradient test with {test_type} target failed.")
        except Exception as e:
            print(f"z: {z}")
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            
    def test_in_y_2st_stage(self):
        test_type_param_dict = {"above_left": [0,0], "above": [0,1], "above_right": [0,2], "right": [1,0], "below_right": [2,0], "below": [2,1], "below_left": [2,2], "left": [1,2], "still": [1,1]}
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left", "still"]
        test_id = [np.random.choice(range(len(test_types)), size=None), np.random.choice(range(len(test_types)), size=None)]
        test_type = [test_types[test_id[0]], test_types[test_id[1]]]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"test_{test_type[0]}_{test_type[1]}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        offsetx, offsety = test_type_param_dict[test_type[0]]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap3x3.camera_matrix_inv, self.device)
        input_repr = input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])
        active_point_img_coords = np.where(self.input_mask!=0)
        temp = (input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]]).cpu().numpy()
        input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]] = 0.*(input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]])
        input_repr[0,0,:,active_point_img_coords[0]+offsetx-1, active_point_img_coords[1]+offsety-1] = 0.+torch.from_numpy(temp)
        input_repr_x = nn.Parameter(input_repr[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        temp = (input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]]).cpu().numpy()
        input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]] = 0*(input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]])
        input_mask[0,0,:,active_point_img_coords[0]+offsetx-1, active_point_img_coords[1]+offsety-1] = 0.+torch.from_numpy(temp)
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type[1]}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr).detach()
        
        smap3x3.zero_grad()
        input_repr_y = self.vtestcase.testbot_in(input_repr_y, input_mask)
        weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, None, self.input_mask.shape)
        weights = smap3x3.go(weights[:,None,:,:,:], target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        weights = torch.abs(self.vtestcase.testbot_out(weights)+1e-7)
        loss_m = torch.abs(weights-target_repr)
    
        loss_m = loss_m.reshape(1, -1).sum(dim=1)

        loss = torch.mean(loss_m)
        loss.backward()
        
        with open(f'./tests/vtest_data/output/test_{test_type[0]}_{test_type[1]}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["in"] != 0).astype(int)
            
        expected = input_mask.detach().reshape(input_mask.shape[-2], input_mask.shape[-1]).cpu().numpy()
        if test_id[1]==3 or test_id[1]==7 or test_types[test_id[1]]=="still":
            expected = 0.*expected
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg=f"Gradient test with {test_type} target failed.")
        except Exception as e:
            print(f"offsetx, offsety: {(offsetx, offsety)}")
            print(f"z: {z}")
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
    
    def test_in_r_1st_stage(self):
        testcase_name = "test_in_r_1st_stage"
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left", "still"]
        test_id = np.random.choice(range(len(test_types)), size=None)
        test_type = test_types[test_id]
        
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"{testcase_name}_{test_type}", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap3x3.camera_matrix_inv, self.device)
        input_repr_x = nn.Parameter(input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type}_target.npy"))
        if test_type=="still":
            target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/input.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr)
        
        
        smap3x3.zero_grad()
        input_mask = self.vtestcase.testbot_in(input_mask, input_mask)
        weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        weights = torch.abs(self.vtestcase.testbot_out(weights)+1e-7)
        loss_m = torch.abs(weights-target_repr)
    
        loss_m = loss_m.reshape(1, -1).sum(dim=1)

        loss = torch.mean(loss_m)
        loss.backward()
        
        is_negative_check = np.random.choice([True, False], size=None)
        with open(f'./tests/vtest_data/output/{testcase_name}_{test_type}_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["in"])
        if is_negative_check:
            actual = (actual < 0).astype(int)
            expected = np.load(f"./tests/vtest_data/smap3x3/{testcase_name}_{test_type}_target_expected.npy").astype(int)
        else:
            actual = (actual > 0).astype(int)
            expected = np.load("./tests/vtest_data/smap3x3/input.npy").astype(int)
            if test_type=="still":
                expected = 0.*expected
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg=f"Gradient test with {test_type} target failed.")
        except Exception as e:
            print(f"z: {z}")
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            
    def test_in_r_2st_stage(self):
        testcase_name = "test_in_r_2st_stage"
        test_type_param_dict = {"above_left": [0,0], "above": [0,1], "above_right": [0,2], "right": [1,0], "below_right": [2,0], "below": [2,1], "below_left": [2,2], "left": [1,2], "still": [1,1]}
        test_types = ["above_left", "above", "above_right", "right", "below_right", "below", "below_left", "left", "still"]
        test_id = [np.random.choice(range(len(test_types)), size=None), np.random.choice(range(len(test_types)), size=None)]
        test_type = [test_types[test_id[0]], test_types[test_id[1]]]
        smap3x3 = self.smap3x3.to(self.device)
        self.vtestcase = TestCase(name=f"{testcase_name}_{test_type[0]}_{test_type[1]}_target", testbot_in=TestBot_In(), testbot_out=TestBot_Out(), testbot_target=TestBot_Target())
        
        z = 1e3*np.random.rand(1)[0]
        offsetx, offsety = test_type_param_dict[test_type[0]]
        input_repr = (z*(self.input_mask))
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]), self.input_mask.shape[0], self.input_mask.shape[1], self.panel, self.input_mask.shape, self.input_mask.shape, self.smap3x3.camera_matrix_inv, self.device)
        input_repr = input_repr.reshape(1,1,3, self.input_mask.shape[0], self.input_mask.shape[1])
        active_point_img_coords = np.where(self.input_mask!=0)
        temp = (input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]]).cpu().numpy()
        input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]] = 0.*(input_repr[0,0,:,active_point_img_coords[0], active_point_img_coords[1]])
        input_repr[0,0,:,active_point_img_coords[0]+offsetx-1, active_point_img_coords[1]+offsety-1] = 0.+torch.from_numpy(temp)
        input_repr_x = nn.Parameter(input_repr[:,:,:1,:,:], requires_grad=True).to(self.device)
        input_repr_y = nn.Parameter(input_repr[:,:,1:2,:,:], requires_grad=True).to(self.device)
        input_repr_z = (input_repr[:,:,2:3,:,:]).to(self.device)
        
        input_mask = self.input_mask.reshape(1,1,1, self.input_mask.shape[0], self.input_mask.shape[1])
        temp = (input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]]).cpu().numpy()
        input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]] = 0*(input_mask[0,0,:,active_point_img_coords[0], active_point_img_coords[1]])
        input_mask[0,0,:,active_point_img_coords[0]+offsetx-1, active_point_img_coords[1]+offsety-1] = 0.+torch.from_numpy(temp)
        input_mask = nn.Parameter(input_mask, requires_grad=True).to(self.device)
        
        target_repr = torch.from_numpy(np.load(f"./tests/vtest_data/smap3x3/{test_type[1]}_target.npy"))
        target_repr = (1.*(target_repr)).reshape(1,1, self.input_mask.shape[0], self.input_mask.shape[1]).to(self.device)
        target_repr = self.vtestcase.testbot_target(target_repr).detach()
        
        smap3x3.zero_grad()
        input_mask = self.vtestcase.testbot_in(input_mask, input_mask)
        weights = smap3x3(input_repr_x, input_repr_y, input_repr_z, input_mask, None, self.input_mask.shape)
        weights = smap3x3.go(weights[:,None,:,:,:], target_repr, self.input_mask.shape).reshape(1,-1, self.input_mask.shape[0], self.input_mask.shape[1])
        weights = torch.abs(self.vtestcase.testbot_out(weights)+1e-7)
        loss_m = torch.abs(weights-target_repr)
    
        loss_m = loss_m.reshape(1, -1).sum(dim=1)

        loss = torch.mean(loss_m)
        loss.backward()
        
        is_negative_check = np.random.choice([True, False], size=None)
        with open(f'./tests/vtest_data/output/{testcase_name}_{test_type[0]}_{test_type[1]}_target_flow_info.pkl', 'rb') as file:
            flow_data = pickle.load(file)
            actual = ((flow_data["activation_gradients"])["in"])
            
        expected_factor = (0 if test_type[1]=="still" else 1)
        expected_pos = [active_point_img_coords[0]+(offsetx-1), active_point_img_coords[1]+(offsety-1)]
        if is_negative_check:
            actual = (actual < 0).astype(int)
            expected = np.load(f"./tests/vtest_data/smap3x3/test_in_r_1st_stage_{test_type[1]}_target_expected.npy").astype(int)
            expected[expected_pos[0], expected_pos[1]] = 0
        else:
            actual = (actual > 0).astype(int)
            expected = 0.*np.load("./tests/vtest_data/smap3x3/input.npy").astype(int)
            expected[expected_pos[0], expected_pos[1]] = expected_factor
        
        try:
            np.testing.assert_array_equal(actual, expected,
                                       err_msg=f"Gradient test with {test_type} target failed.")
        except Exception as e:
            print(f"offsetx, offsety: {(offsetx, offsety)}")
            print(f"z: {z}")
            print(f"actual: {actual}")
            print(f"expected: {expected}")
            raise e
            

if __name__ == "__main__":
    unittest.main()