import unittest
import numpy as np
import torch
import torch.nn.functional as F
from smap import *

# the test case
class SMapUTestCase(unittest.TestCase):
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
        self.smap = SMap(self.n, self.img_shape[0], self.img_shape[1], self.camera, self.device).to(self.device)
        self.smap3x3 = self.smap.smap3x3

    def test_SMap_forward(self):
        zoom = 0
        activated_coords = [np.array([np.random.choice(range(self.img_shape[0]), size=None)]), np.array([np.random.choice(range(self.img_shape[1]), size=None)])]
        traverse_diff = np.array([0., -0., 0.]).reshape(1,3)
        mask = 0.*(np.load(f"./tests/vtest_data/smap/mask.npy")[0,0,:,:])
        for i, (r, c) in enumerate(zip(*activated_coords)):
            mask[r, c] = 1.
        input_mask = torch.from_numpy(mask).float().to(self.device).reshape(1,1, self.img_shape[0], self.img_shape[1])
        z = 1.
        input_repr = (z*input_mask)
        input_repr = utils.to_3d(input_repr.reshape(1,1, self.img_shape[0], self.img_shape[1]), self.img_shape[0], self.img_shape[1], self.panel, self.img_shape, self.img_shape, self.smap3x3.camera_matrix_inv, self.device).reshape(1,3, self.img_shape[0], self.img_shape[1])
        
        actual = self.smap(torch.cat([input_repr, input_mask], dim=1), None, zoom)
        actual = (actual[:,-1:,(actual.shape[-2]-self.img_shape[0])//2:-(actual.shape[-2]-self.img_shape[0])//2, (actual.shape[-1]-self.img_shape[1])//2:-(actual.shape[-1]-self.img_shape[1])//2])
        actual = actual.detach().cpu().numpy().reshape(self.img_shape[0], self.img_shape[1])
        
        expected_activated_coords = np.where(mask>0.)
        for i, (r, c) in enumerate(zip(*activated_coords)):
            coord = (input_repr[0,:,r, c]).numpy().reshape(1,3)
            new_coord = coord+traverse_diff
            new_coord = torch.einsum('xz,yz->xy', torch.from_numpy(new_coord).float(), torch.from_numpy(self.camera).float()).cpu().numpy()
            
            new_r = new_coord[0,1]/new_coord[0,-1]-.5
            new_c = new_coord[0,0]/new_coord[0,-1]-.5
            expected_activated_coords[0][i] = round(new_r)
            expected_activated_coords[1][i] = round(new_c)
        expected = np.zeros_like(actual)
        expected[expected_activated_coords] = 1.
        
        try:
            np.testing.assert_almost_equal(actual, expected, decimal=3, err_msg='SMap forward test failed.')
        except Exception as e:
            raise e

if __name__ == "__main__":
    unittest.main()
