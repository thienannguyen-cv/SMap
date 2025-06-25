import unittest
import numpy as np
import torch
import torch.nn.functional as F
from smap import *

# the test case
class RectifyUTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.n = 2
        self.img_shape = [128, 256]
        self.camera = np.array([[2304.5479, 0,  1686.2379], 
                                [0, 2305.8757, -0.0151],
                                [0, 0, 1.]], dtype=np.float32)
        self.panel = list(np.where(np.ones([self.img_shape[0], self.img_shape[1]])))
        self.panel[0] = self.panel[0] + .5
        self.panel[1] = self.panel[1] + .5
        self.smap = SMap(self.img_shape[0], self.img_shape[1], self.camera, n=self.n, device=self.device).to(self.device)
        self.rectify_module = self.smap.rectify_module
            
    def test_prepare_flows_for_coord(self):
        cases = ["blocked", "random"]
        case_id = np.random.choice(len(cases),size=None)
        
        # With the assumption that the mask at the activated point is 1. 
        height = self.img_shape[0]+2
        width = self.img_shape[1]+2
        
        weights = torch.from_numpy(np.zeros((3,3,height, width))).float().to(self.device)
        active_point_img_coords = [np.random.choice(range(2,self.img_shape[0]-2),size=None), np.random.choice(range(2,self.img_shape[1]-2),size=None)]
        
        case = cases[case_id]
        if case=="blocked":
            offsetx, offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
            target_offsetx, target_offsety = offsetx, offsety
            weights_active_point_img_coords = [active_point_img_coords[0]+1, active_point_img_coords[1]+1]
            weights[offsetx, offsety, weights_active_point_img_coords[0],weights_active_point_img_coords[1]] = 1.
            target_pointx = active_point_img_coords[0]+target_offsetx-1
            target_pointy = active_point_img_coords[1]+target_offsety-1

            target = torch.from_numpy(np.zeros((self.img_shape[0], self.img_shape[1]))).float().to(self.device)
            target[target_pointx, target_pointy] = 1.
            
            allow = self.rectify_module.compute_allow_matrix(weights.reshape(1,1,3,3,1, height, width), target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            actual = self.rectify_module.prepare_flows_for_coord(allow, target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            offset = np.random.choice(range(3*3),size=None)
            referenced_point_x = active_point_img_coords[0]-(offset//3)
            referenced_point_y = active_point_img_coords[1]-(offset%3)
            try:
                if offset==((offsetx*3)+offsety):
                    assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 0, "Flows preparing test failed. "
                else:
                    if (referenced_point_x>0) and (referenced_point_x<(self.img_shape[0]-1)) and (referenced_point_y>0) and (referenced_point_y<(self.img_shape[1]-1)):
                        assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 1, "Flows preparing test failed. "
                    else:
                        assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 0, "Flows preparing test failed. "
            except Exception as e:
                print(f"offset: {offset}")
                print(f"active_point_img_coords: {active_point_img_coords}")
                print(f"offsetx, offsety: {(offsetx, offsety)}")
                print(f"target_offsetx, target_offsety: {(target_offsetx, target_offsety)}")
                print(f"actual: {actual}")
                raise e
        else:
            offsetx, offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
            target_offsetx, target_offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
            weights_active_point_img_coords = [active_point_img_coords[0]+1, active_point_img_coords[1]+1]
            weights[offsetx, offsety, weights_active_point_img_coords[0],weights_active_point_img_coords[1]] = 1.
            target_pointx = active_point_img_coords[0]+target_offsetx-1
            target_pointy = active_point_img_coords[1]+target_offsety-1

            target = torch.from_numpy(np.zeros((self.img_shape[0], self.img_shape[1]))).float().to(self.device)
            target[target_pointx, target_pointy] = 1.
            
            allow = self.rectify_module.compute_allow_matrix(weights.reshape(1,1,3,3,1, height, width), target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            actual = self.rectify_module.prepare_flows_for_coord(allow, target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            offset = np.random.choice(range(3*3),size=None)
            referenced_point_x = active_point_img_coords[0]-(offset//3)
            referenced_point_y = active_point_img_coords[1]-(offset%3)
            try:
                if offsetx!=target_offsetx or offsety!=target_offsety:
                    if (referenced_point_x>0) and (referenced_point_x<(self.img_shape[0]-1)) and (referenced_point_y>0) and (referenced_point_y<(self.img_shape[1]-1)):
                        assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 1, "Flows preparing test failed. "
                    else:
                        assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 0, "Flows preparing test failed. "
                else:
                    if offset==((offsetx*3)+offsety):
                        assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 0, "Flows preparing test failed. "
                    else:
                        assert torch.sum(actual[0,offset,:,:]).reshape(-1).cpu().numpy()[0] == 1, "Flows preparing test failed. "
            except Exception as e:
                print(f"offset: {offset}")
                print(f"active_point_img_coords: {active_point_img_coords}")
                print(f"offsetx, offsety: {(offsetx, offsety)}")
                print(f"target_offsetx, target_offsety: {(target_offsetx, target_offsety)}")
                print(f"actual: {actual}")
                raise e
            
    def test_prepare_flows_for_mask(self):
        cases = ["blocked", "random"]
        case_id = np.random.choice(len(cases),size=None)
        
        # With the assumption that the mask at the activated point is 1. 
        height = self.img_shape[0]+2
        width = self.img_shape[1]+2
        
        weights = torch.from_numpy(np.zeros((3,3,height, width))).float().to(self.device)
        active_point_img_coords = [np.random.choice(range(2,self.img_shape[0]-2),size=None), np.random.choice(range(2,self.img_shape[1]-2),size=None)]
        is_activated = True
            
        case = cases[case_id]
        if case=="blocked":
            offsetx, offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
            target_offsetx, target_offsety = offsetx, offsety
            weights_active_point_img_coords = [active_point_img_coords[0]+1, active_point_img_coords[1]+1]
            if is_activated:
                weights[offsetx, offsety, weights_active_point_img_coords[0],weights_active_point_img_coords[1]] = 1.
            target_pointx = active_point_img_coords[0]+target_offsetx-1
            target_pointy = active_point_img_coords[1]+target_offsety-1

            target = torch.from_numpy(np.zeros((self.img_shape[0], self.img_shape[1]))).float().to(self.device)
            target[target_pointx, target_pointy] = 1.
            
            allow = self.rectify_module.compute_allow_matrix(weights.reshape(1,1,3,3,1, height, width), target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            actual = self.rectify_module.prepare_flows_for_mask(allow, target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            offset = np.random.choice(range(3*3),size=None)
            try:
                assert torch.sum((actual[0,offset,:,:]>0).float()).reshape(-1).cpu().numpy()[0] == (((self.img_shape[0])*(self.img_shape[1])-1) if is_activated else ((self.img_shape[0])*(self.img_shape[1]))), "Flows preparing test failed. "
            except Exception as e:
                print(f"is_activated: {is_activated}")
                print(f"offset: {offset}")
                print(f"active_point_img_coords: {active_point_img_coords}")
                print(f"offsetx, offsety: {(offsetx, offsety)}")
                print(f"target_offsetx, target_offsety: {(target_offsetx, target_offsety)}")
                print(f"actual: {actual}")
                raise e
        else:
            offsetx, offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
            target_offsetx, target_offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
            weights_active_point_img_coords = [active_point_img_coords[0]+1, active_point_img_coords[1]+1]
            if is_activated:
                weights[offsetx, offsety, weights_active_point_img_coords[0],weights_active_point_img_coords[1]] = 1.
            target_pointx = active_point_img_coords[0]+target_offsetx-1
            target_pointy = active_point_img_coords[1]+target_offsety-1

            target = torch.from_numpy(np.zeros((self.img_shape[0], self.img_shape[1]))).float().to(self.device)
            target[target_pointx, target_pointy] = 1.
            
            allow = self.rectify_module.compute_allow_matrix(weights.reshape(1,1,3,3,1, height, width), target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            actual = self.rectify_module.prepare_flows_for_mask(allow, target.reshape(1,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            offset = np.random.choice(range(3*3),size=None)
            try:
                if offsetx!=target_offsetx or offsety!=target_offsety:
                    assert torch.sum((actual[0,offset,:,:]>0).float()).reshape(-1).cpu().numpy()[0] == ((self.img_shape[0])*(self.img_shape[1])), "Flows preparing test failed. "
                else:
                    assert torch.sum((actual[0,offset,:,:]>0).float()).reshape(-1).cpu().numpy()[0] == (((self.img_shape[0])*(self.img_shape[1])-1) if is_activated else ((self.img_shape[0])*(self.img_shape[1]))), "Flows preparing test failed. "
            except Exception as e:
                print(f"is_activated: {is_activated}")
                print(f"offset: {offset}")
                print(f"active_point_img_coords: {active_point_img_coords}")
                print(f"offsetx, offsety: {(offsetx, offsety)}")
                print(f"target_offsetx, target_offsety: {(target_offsetx, target_offsety)}")
                print(f"actual: {actual}")
                raise e

    