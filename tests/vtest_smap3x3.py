import os
import tempfile
import unittest

import numpy as np
import torch

from smap import SMap, rectify, utils
from tools.testing.vtest.vtest_types import _DisabledTestbot


class SMap3x3VTestCase(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.img_shape = [4, 4]
        self.active_rc = (self.img_shape[0] // 2, self.img_shape[1] // 2)
        self.camera = np.array(
            [[2304.5479, 0, 1686.2379], [0, 2305.8757, -0.0151], [0, 0, 1.]],
            dtype=np.float32,
        )

    def test_disabled_testbot_is_identity_and_noop(self):
        bot = _DisabledTestbot()
        x = torch.randn(1, 3, 4)

        self.assertIs(bot(x), x)
        self.assertIs(bot.testbot_in(x), x)
        self.assertIs(bot.testbot_out(x), x)
        self.assertIsNone(bot.testbot_input(x, filename="ignored.npy"))
        self.assertIsNone(bot.testbot_target(x))
        bot.name = "absorbed"

    def test_smap3x3_go_returns_current_state_tuple(self):
        smap = SMap(self.img_shape[0], self.img_shape[1], self.camera, n=0)
        x = torch.zeros(1, 1, 4, self.img_shape[0], self.img_shape[1])
        row, col = self.active_rc
        x[0, 0, 2, row, col] = 1.0
        x[0, 0, 3, row, col] = 1.0

        state = smap.smap3x3.go(x, self.img_shape, is_last=True)

        self.assertEqual(len(state), 6)
        pre_x, pre_y, pre_z, pre_mask, panels, new_value = state
        self.assertEqual(tuple(pre_x.shape[-2:]), (6, 6))
        self.assertEqual(tuple(pre_y.shape[-2:]), (6, 6))
        self.assertEqual(tuple(pre_z.shape[-2:]), (6, 6))
        self.assertEqual(tuple(pre_mask.shape[-2:]), (6, 6))
        self.assertEqual(len(panels), 2)
        self.assertEqual(tuple(new_value.shape[-3:]), (4, 6, 6))

    def test_forward_with_debug_disabled_does_not_write_vtest_files(self):
        smap = SMap(
            self.img_shape[0],
            self.img_shape[1],
            self.camera,
            rectify_type=rectify.types.CAM,
            n=0,
        )
        x = torch.randn(1, 1, 4, self.img_shape[0], self.img_shape[1], requires_grad=True)
        target = torch.ones(1, 1, self.img_shape[0], self.img_shape[1])

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                weights = smap(x, target)
                weights.sum().backward()
                self.assertEqual(os.listdir(tmpdir), [])
            finally:
                os.chdir(cwd)

        self.assertIsInstance(smap.smap3x3.vtestcase, _DisabledTestbot)


if __name__ == "__main__":
    unittest.main()
