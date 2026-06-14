import unittest

import numpy as np
import torch

from smap import SMap


class RectifyUTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.img_shape = [8, 12]
        self.camera = np.array(
            [[2304.5479, 0, 1686.2379], [0, 2305.8757, -0.0151], [0, 0, 1.]],
            dtype=np.float32,
        )
        self.smap = SMap(
            self.img_shape[0],
            self.img_shape[1],
            self.camera,
            n=0,
            device=self.device,
        ).to(self.device)
        self.rectify_module = self.smap.rectify_module

    def _pre_allow_for_target(self, target_rc):
        height = self.img_shape[0] + 2
        width = self.img_shape[1] + 2
        weights = torch.zeros(1, 1, 3, 3, 1, height, width, device=self.device)
        target = torch.zeros(1, 1, 1, 1, self.img_shape[0], self.img_shape[1], device=self.device)

        # Active center-offset route at padded image coordinate (4, 5), which
        # maps back to target coordinate (3, 4) after removing the one-cell pad.
        weights[0, 0, 1, 1, 0, 4, 5] = 1.0
        target[0, 0, 0, 0, target_rc[0], target_rc[1]] = 1.0

        pre_allow = self.rectify_module.compute_pre_allow_matrices(weights, target)
        return weights, target, pre_allow

    def test_prepare_flows_blocks_matched_route(self):
        weights, target, pre_allow = self._pre_allow_for_target((3, 4))

        mask_flow = self.rectify_module.prepare_flows_for_mask(pre_allow, weights, target)
        coord_flow = self.rectify_module.prepare_flows_for_coord(pre_allow, weights, target)

        self.assertEqual(tuple(mask_flow.shape), (1, 9, 8, 12))
        self.assertEqual(tuple(coord_flow.shape), (1, 9, 8, 12))
        self.assertEqual(float(mask_flow.sum()), 783.0)
        self.assertEqual(float(coord_flow.sum()), 0.0)

    def test_prepare_flows_opens_misaligned_route(self):
        weights, target, pre_allow = self._pre_allow_for_target((3, 5))

        mask_flow = self.rectify_module.prepare_flows_for_mask(pre_allow, weights, target)
        coord_flow = self.rectify_module.prepare_flows_for_coord(pre_allow, weights, target)

        self.assertEqual(float(mask_flow.sum()), 864.0)
        self.assertEqual(float(coord_flow.sum()), 9.0)
        self.assertTrue(torch.all((coord_flow == 0.0) | (coord_flow == 1.0)))


if __name__ == "__main__":
    unittest.main()
