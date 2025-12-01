import unittest
import numpy as np
import torch
from smap import SMap, utils

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
        self.smap = SMap(self.img_shape[0], self.img_shape[1], self.camera, n=self.n, device=self.device).to(self.device)
        self.smap3x3 = self.smap.smap3x3
            
    def test_prepare_flows_for_coord(self):
        BATCH_SIZE = 1
        
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
            
            allow = self.smap.rectify_module.compute_allow_matrix(weights.reshape(BATCH_SIZE,1,3,3,1, height, width), target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))
            
            actual = self.smap.rectify_module.prepare_flows_for_coord(allow, target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))
            
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

            allow = self.smap.rectify_module.compute_allow_matrix(weights.reshape(BATCH_SIZE,1,3,3,1, height, width), target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))

            actual = self.smap.rectify_module.prepare_flows_for_coord(allow, target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))

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
        BATCH_SIZE = 1

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

            allow = self.smap.rectify_module.compute_allow_matrix(weights.reshape(BATCH_SIZE,1,3,3,1, height, width), target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))

            actual = self.smap.rectify_module.prepare_flows_for_mask(allow, target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))

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

            allow = self.smap.rectify_module.compute_allow_matrix(weights.reshape(BATCH_SIZE,1,3,3,1, height, width), target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))

            actual = self.smap.rectify_module.prepare_flows_for_mask(allow, target.reshape(BATCH_SIZE,1,1,1, self.img_shape[0], self.img_shape[1]))

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

    def test_SMap_forward(self):
        BATCH_SIZE = 1

        zoom = 0
        activated_coords = [np.array([np.random.choice(range(self.img_shape[0]), size=None)]), np.array([np.random.choice(range(self.img_shape[1]), size=None)])]
        traverse_diff = np.array([0., -0., 0.]).reshape(1,3)
        mask = 0.*(np.load(f"./tests/vtest_data/smap/mask.npy")[0,0,:,:])
        for i, (r, c) in enumerate(zip(*activated_coords)):
            mask[r, c] = 1.
        input_mask = torch.from_numpy(mask).float().to(self.device).reshape(1,1, self.img_shape[0], self.img_shape[1])
        z = 1.
        input_repr = (z*input_mask)
        input_repr = utils.to_3d(input_repr.reshape(BATCH_SIZE,1, self.img_shape[0], self.img_shape[1]), self.img_shape[0], self.img_shape[1], self.panel, self.img_shape, self.img_shape, self.smap3x3.camera_matrix_inv, self.device).reshape(BATCH_SIZE,3, self.img_shape[0], self.img_shape[1])
        
        _, actual = self.smap(torch.cat([input_repr, input_mask], dim=1), None, zoom)
        actual = (actual[:,-1,:,:]).detach().cpu().numpy().reshape(self.img_shape[0], self.img_shape[1])
        
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
