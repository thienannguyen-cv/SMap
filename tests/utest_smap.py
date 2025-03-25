import unittest
import numpy as np
import torch
from smap import *

# the test case
class SMap3x3UTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.img_shape = [8, 8]
        self.depth_map = torch.from_numpy(np.zeros((self.img_shape[0], self.img_shape[1]))).float().to(self.device)
        self.active_point_img_coords = [3,4]
        self.depth_map[self.active_point_img_coords[0],self.active_point_img_coords[1]] = 50.
        self.camera = np.array([[2304.5479, 0,  1686.2379], 
                                [0, 2305.8757, -0.0151],
                                [0, 0, 1.]], dtype=np.float32)
        self.panel = list(np.where(np.ones([self.img_shape[0], self.img_shape[1]])))
        self.smap3x3 = SMap3x3(self.img_shape[0], self.img_shape[1], self.camera, self.device).to(self.device)

    def test_to_3d(self):
        import numpy as np
        offsetx, offsety = 1, 1
        actual = (self.smap3x3.to_3d(self.depth_map.reshape(1,1,self.img_shape[0],self.img_shape[1]), self.img_shape[0], self.img_shape[1], self.panel, self.img_shape).reshape(3,3,3,self.img_shape[0],self.img_shape[1])[:,offsetx,offsety,self.active_point_img_coords[0],self.active_point_img_coords[1]]).cpu().numpy()
        expected = torch.einsum("x,yx->y", (self.depth_map[self.active_point_img_coords[0], self.active_point_img_coords[1]])*torch.from_numpy(np.array([(self.active_point_img_coords[1])+(offsety-1),(self.active_point_img_coords[0])+(offsetx-1),1.])).float(), torch.from_numpy(np.linalg.inv(self.camera))).numpy()
        try:
            np.testing.assert_allclose(actual, expected,
                                       err_msg="Transforming from depth to 3D coordinates failed.")
        except Exception as e:
            raise e

if __name__ == "__main__":
    unittest.main()
