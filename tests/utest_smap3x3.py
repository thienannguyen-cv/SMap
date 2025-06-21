import unittest
import numpy as np
import torch
import torch.nn.functional as F
from smap import *

# the test case
class SMap3x3UTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.img_shape = [8, 12]
        self.camera = np.array([[2304.5479, 0,  1686.2379], 
                                [0, 2305.8757, -0.0151],
                                [0, 0, 1.]], dtype=np.float32)
        self.panel = list(np.where(np.ones([self.img_shape[0], self.img_shape[1]])))
        self.smap3x3 = SMap3x3(self.img_shape[0], self.img_shape[1], self.camera, self.device).to(self.device)

    def test_to_3d3x3(self):
        BATCH_SIZE = 1
        
        depth_map = torch.from_numpy(np.zeros((self.img_shape[0], self.img_shape[1]))).float().to(self.device)
        
        active_point_img_coords = [np.random.choice(range(self.img_shape[0]),size=None), np.random.choice(range(self.img_shape[1]),size=None)]
        depth_map[active_point_img_coords[0],active_point_img_coords[1]] = 1e3*(np.random.rand(1)[0])
        offsetx, offsety = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
        
        actual = (utils.to_3d3x3(depth_map.reshape(BATCH_SIZE,1,self.img_shape[0],self.img_shape[1]), self.img_shape[0], self.img_shape[1], self.panel, self.img_shape, self.img_shape, self.smap3x3.camera_matrix_inv, self.device).reshape(3,3,self.img_shape[0],self.img_shape[1],3)[offsetx,offsety,active_point_img_coords[0],active_point_img_coords[1],:]).cpu().numpy()
        pointy = (active_point_img_coords[1])+(offsety-1)
        if pointy<0 or pointy>=(self.img_shape[1]):
            pointy=0
        pointx = (active_point_img_coords[0])+(offsetx-1)
        if pointx<0 or pointx>=(self.img_shape[0]):
            pointx=0
        expected = torch.einsum("x,yx->y", (depth_map[active_point_img_coords[0], active_point_img_coords[1]])*torch.from_numpy(np.array([pointy,pointx,1.])).float(), torch.from_numpy(np.linalg.inv(self.camera))).numpy()
        try:
            np.testing.assert_allclose(actual, expected, atol=1e-3,
                                       err_msg="Transforming from depth to 3D coordinates failed.")
        except Exception as e:
            print(f"active_point_img_coords: {active_point_img_coords}")
            print(f"Activated point of depth_map: {depth_map[active_point_img_coords[0],active_point_img_coords[1]]}")
            print(f"offsetx: {offsetx}")
            print(f"offsety: {offsety}")
            raise e
            
    def test_agg_factor_only(self):
        BATCH_SIZE = 1
        
        active_point_img_coords0 = [np.random.choice(range(self.img_shape[0]),size=None), np.random.choice(range(self.img_shape[1]),size=None)]
        active_point_img_coords1 = [np.random.choice(range(self.img_shape[0]),size=None), np.random.choice(range(self.img_shape[1]),size=None)]
        offsetx0, offsety0 = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
        offsetx1, offsety1 = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
        first_value = 1e3*(np.random.rand(1)[0])
        second_value = 1e3*(np.random.rand(1)[0])
        unfolded_depth_map = torch.from_numpy(np.zeros((3,3, self.img_shape[0], self.img_shape[1]))).float().to(self.device)
        unfolded_depth_map[offsetx0,offsety0,active_point_img_coords0[0],active_point_img_coords0[1]] = first_value
        unfolded_depth_map[offsetx1,offsety1,active_point_img_coords1[0],active_point_img_coords1[1]] = second_value
        
        factor = 1e15*(np.random.rand(1)[0])
        actual = utils.agg(unfolded_depth_map.reshape(BATCH_SIZE,1,3,3,1, self.img_shape[0], self.img_shape[1])
        , factor=factor).cpu().numpy()
        
        expected = torch.from_numpy(factor*np.ones((3,3, self.img_shape[0], self.img_shape[1]))).float().to(self.device).numpy()
        pointx0 = active_point_img_coords0[0]+offsetx0-1
        pointy0 = active_point_img_coords0[1]+offsety0-1
        pointx1 = active_point_img_coords1[0]+offsetx1-1
        pointy1 = active_point_img_coords1[1]+offsety1-1
        if pointx0<(self.img_shape[0]) and pointy0<(self.img_shape[1]):
            expected[offsetx0,offsety0,pointx0,pointy0] = first_value
        if pointx1<(self.img_shape[0]) and pointy1<(self.img_shape[1]):
            expected[offsetx1,offsety1,pointx1,pointy1] = second_value
        expected = expected.reshape(1,1,3*3,1,1, self.img_shape[0], self.img_shape[1])
        
        try:
            np.testing.assert_allclose(actual, expected,
                                       err_msg="Transforming to absolute-alignment representation failed.")
        except Exception as e:
            print(f"factor: {factor}")
            print(f"active_point_img_coords0: {active_point_img_coords0}")
            print(f"active_point_img_coords1: {active_point_img_coords1}")
            print(f"offsets0: {(offsetx0, offsety0)}")
            print(f"offsets1: {(offsetx1, offsety1)}")
            print(f"first_value: {first_value}")
            print(f"second_value: {second_value}")
            raise e
            
    def test_agg_ind(self):
        BATCH_SIZE = 1
        
        active_point_img_coords0 = [np.random.choice(range(self.img_shape[0]),size=None), np.random.choice(range(self.img_shape[1]),size=None)]
        active_point_img_coords1 = [np.random.choice(range(self.img_shape[0]),size=None), np.random.choice(range(self.img_shape[1]),size=None)]
        offsetx0, offsety0 = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
        offsetx1, offsety1 = np.random.choice(range(3),size=None), np.random.choice(range(3),size=None)
        channel_num = np.random.choice([1,4],size=None)
        first_values = np.random.rand(channel_num)
        first_values[:3] = 1e3*(first_values[:3])
        second_values = np.random.rand(channel_num)
        second_values[:3] = 1e3*(second_values[:3])
        unfolded_depth_map = torch.from_numpy(np.zeros((3,3,channel_num, self.img_shape[0], self.img_shape[1]))).float().to(self.device)
        unfolded_depth_map[offsetx0,offsety0,:channel_num,active_point_img_coords0[0],active_point_img_coords0[1]] = torch.from_numpy(first_values)
        unfolded_depth_map[offsetx1,offsety1,:channel_num,active_point_img_coords1[0],active_point_img_coords1[1]] = torch.from_numpy(second_values)
        
        factor = 1e15*(np.random.rand(1)[0])
        pointx0 = active_point_img_coords0[0]+offsetx0-1
        pointy0 = active_point_img_coords0[1]+offsety0-1
        pointx1 = active_point_img_coords1[0]+offsetx1-1
        pointy1 = active_point_img_coords1[1]+offsety1-1
        
        _ind = utils.agg((unfolded_depth_map[:,:,:1,:,:]).reshape(BATCH_SIZE,1,3*3,1,1, self.img_shape[0], self.img_shape[1]), factor=1e7)
        if channel_num==4:
            _ind = utils.agg((unfolded_depth_map[:,:,2:3,:,:]).reshape(BATCH_SIZE,1,3*3,1,1, self.img_shape[0], self.img_shape[1]), factor=1e7)
        _ind = torch.min(_ind,dim=2,keepdim=True).indices
        ind = 4*torch.ones_like(_ind)
        if pointx0>=0 and pointx0<(self.img_shape[0]) and pointy0>=0 and pointy0<(self.img_shape[1]):
            ind[0,0,0,0,0,pointx0,pointy0] = _ind[0,0,0,0,0,pointx0,pointy0]
        if pointx1>=0 and pointx1<(self.img_shape[0]) and pointy1>=0 and pointy1<(self.img_shape[1]):
            ind[0,0,0,0,0,pointx1,pointy1] = _ind[0,0,0,0,0,pointx1,pointy1]
        ind = F.one_hot(ind, num_classes=3*3).reshape(BATCH_SIZE,-1,1,1, self.img_shape[0], self.img_shape[1],3*3).permute(0,1,6,2,3,4,5).reshape(BATCH_SIZE,-1,3*3,1,1, self.img_shape[0], self.img_shape[1])
        ind = (ind>.5)
        actual = utils.agg(unfolded_depth_map.reshape(BATCH_SIZE,1,3,3,channel_num, self.img_shape[0], self.img_shape[1]), ind=ind, factor=factor).cpu().numpy()
        
        expected = torch.from_numpy(factor*np.ones((channel_num, self.img_shape[0], self.img_shape[1]))).float().to(self.device).numpy()
        first_value = (first_values[-1])
        second_value = (second_values[-1])
        if channel_num==4:
            first_value = (first_values[-2])
            second_value = (second_values[-2])
        if pointx1>=0 and pointx1<(self.img_shape[0]) and pointy1>=0 and pointy1<(self.img_shape[1]):
            expected[:,pointx1,pointy1] = torch.from_numpy(second_values)
        if pointx0>=0 and pointx0<(self.img_shape[0]) and pointy0>=0 and pointy0<(self.img_shape[1]):
            if pointx0==pointx1 and pointy0==pointy1:
                expected[:,pointx0,pointy0] = torch.from_numpy(first_values+second_values)
            else:
                expected[:,pointx0,pointy0] = torch.from_numpy(first_values)
        expected = expected.reshape(1,1,1,1,channel_num, self.img_shape[0], self.img_shape[1])
        try:
            np.testing.assert_allclose(actual, expected,
                                       err_msg="Aggregating absolute-alignment representation failed.")
        except Exception as e:
            print(f"factor: {factor}")
            print(f"active_point_img_coords0: {active_point_img_coords0}")
            print(f"active_point_img_coords1: {active_point_img_coords1}")
            print(f"offsets0: {(offsetx0, offsety0)}")
            print(f"offsets1: {(offsetx1, offsety1)}")
            print(f"first_value: {first_value}")
            print(f"second_value: {second_value}")
            raise e
    

if __name__ == "__main__":
    unittest.main()
