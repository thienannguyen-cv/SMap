import unittest

import numpy as np
import torch

from smap import SMap, rectify, utils


class SMapUTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.img_shape = [4, 4]
        self.active_rc = (self.img_shape[0] // 2, self.img_shape[1] // 2)
        self.camera = np.array(
            [[2304.5479, 0, 1686.2379], [0, 2305.8757, -0.0151], [0, 0, 1.]],
            dtype=np.float32,
        )
        y_im, x_im = np.where(np.ones(self.img_shape))
        self.y_im = torch.from_numpy(y_im).float() + 0.5
        self.x_im = torch.from_numpy(x_im).float() + 0.5

    def test_to_3d_current_signature(self):
        depth = torch.zeros(1, 1, self.img_shape[0], self.img_shape[1])
        row, col = self.active_rc
        depth[0, 0, row, col] = 10.0
        smap = SMap(self.img_shape[0], self.img_shape[1], self.camera, n=0)

        actual = utils.to_3d(
            depth,
            self.img_shape[0],
            self.img_shape[1],
            self.y_im,
            self.x_im,
            self.img_shape,
            self.img_shape,
            smap.smap3x3.camera_matrix_inv,
            self.device,
        )

        self.assertEqual(tuple(actual.shape), (1, 1, 3, self.img_shape[0], self.img_shape[1]))
        self.assertTrue(torch.isfinite(actual).all())

    def test_smap_forward_without_target_returns_mask_weights(self):
        x = torch.zeros(1, 1, 4, self.img_shape[0], self.img_shape[1])
        row, col = self.active_rc
        x[0, 0, 2, row, col] = 1.0
        x[0, 0, 3, row, col] = 1.0
        smap = SMap(self.img_shape[0], self.img_shape[1], self.camera, n=0)

        actual = smap(x)

        self.assertEqual(tuple(actual.shape), (1, 1, self.img_shape[0], self.img_shape[1]))
        self.assertTrue(torch.isfinite(actual).all())

    def test_rectify_modes_support_backward(self):
        for mode in (None, rectify.types.CAM, rectify.types.DEP, rectify.types.FUT):
            with self.subTest(mode=mode):
                x = torch.randn(1, 1, 4, self.img_shape[0], self.img_shape[1], requires_grad=True)
                target = torch.ones(1, 1, self.img_shape[0], self.img_shape[1])
                smap = SMap(self.img_shape[0], self.img_shape[1], self.camera, rectify_type=mode, n=0)

                weights = smap(x, target)
                self.assertEqual(tuple(weights.shape), (1, 9, self.img_shape[0], self.img_shape[1]))
                weights.sum().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
