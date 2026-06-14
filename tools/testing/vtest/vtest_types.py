"""vtest testbots — instrumentation taps for the SMap forward/backward pass.

Each testbot wraps a point in the SMap graph and, in audit mode, captures the
tensor / its gradient and dumps it to ``.npy`` (the HDVO ground truth). All bots
share the same scaffold (``_TestBotBase``): an inner identity module, a
back-reference to the owning ``TestCase``, one backward hook, and name-prefixed
output paths. Subclasses only specify what to capture (``_on_backward``) and/or
what to record on the way in (``forward``).

When the audit flag is off, SMap installs ``_DisabledTestbot`` instead, so no
torch testbot is built, no hooks are registered, and no I/O happens.
"""
import pickle

import numpy as np
import torch

from smap import utils


class NoneBot(torch.nn.Module):
    """Identity wrapper: ``module(x)`` if a module is given, else ``x`` unchanged."""

    def __init__(self, module=None):
        super(NoneBot, self).__init__()
        self.module = module

    def forward(self, x):
        if self.module is not None:
            return self.module(x)
        return x + 0


class _TestBotBase(torch.nn.Module):
    """Common scaffold shared by every testbot.

    Wraps an optional inner module (default: identity :class:`NoneBot`), keeps a
    back-reference to the owning :class:`TestCase`, and registers a single
    backward hook that dispatches to :meth:`_on_backward`. By default both
    :meth:`_on_backward` and :meth:`forward` are pass-throughs; subclasses
    override the one(s) they need. Output paths are always
    ``out_path + testcase.name + suffix`` (the path the real code takes, since a
    bot's ``testcase`` is always wired by :class:`TestCase` before use).
    """

    def __init__(self, module=None, name="in", connet2name="out", offset=0):
        super(_TestBotBase, self).__init__()
        self.testcase = None
        self.module = module if module is not None else NoneBot()
        self.name = name
        self.connet2name = connet2name
        self.offset = offset
        self.input_representation = None
        self.target_representation = None
        self.module.register_backward_hook(self._dispatch_backward())

    # -- hook plumbing --
    def _dispatch_backward(self):
        def hook(module, grad_inputs, grad_outputs):
            self._on_backward(grad_inputs, grad_outputs)
        return hook

    def _on_backward(self, grad_inputs, grad_outputs):
        """Override to capture gradients on backward. Default: no-op."""
        pass

    def forward(self, x):
        return self.module(x)

    # -- shared output helpers --
    @staticmethod
    def _pick(grads):
        """Last non-None gradient in the tuple (the value the old loops kept)."""
        picked = None
        for grad in grads:
            if grad is not None:
                picked = grad
        return picked

    def _out(self, suffix):
        return self.testcase.out_path + self.testcase.name + suffix

    def _record_flow(self, key, value):
        self.testcase.activation_gradients[key] = value
        self.testcase.gradient_flows[(self.name, self.connet2name)] = None

    def _save_flow_info(self):
        flow_info = {"activation_gradients": self.testcase.activation_gradients,
                     "gradient_flows": self.testcase.gradient_flows}
        with open(self._out("_flow_info.pkl"), "wb") as f:
            pickle.dump(flow_info, f)


class TestBot_In(_TestBotBase):
    def __init__(self, module=None, offset_in=0, name="in", connet2name="out"):
        super(TestBot_In, self).__init__(module=module, name=name, connet2name=connet2name, offset=offset_in)

    def _on_backward(self, grad_inputs, grad_outputs):
        if self.name is None:
            return
        grad_in = self._pick(grad_inputs)
        H_in, W_in = (grad_in.shape[-2]), (grad_in.shape[-1])
        grad_in = (grad_in[:, :, self.offset, :, :]).reshape(H_in, W_in)
        self._record_flow(self.name, grad_in.cpu().numpy())
        np.save(self._out("_input_representation.npy"), self.input_representation.reshape(H_in, W_in))
        self._save_flow_info()

    def forward(self, x, mask):
        h, w = mask.shape[-2], mask.shape[-1]
        self.input_representation = mask.detach().cpu().numpy().reshape(h, w)
        return self.module(x)


class TestBot_Out(_TestBotBase):
    def __init__(self, module=None, offset_out=4, name="in", connet2name="out"):
        super(TestBot_Out, self).__init__(module=module, name=name, connet2name=connet2name, offset=offset_out)

    def _on_backward(self, grad_inputs, grad_outputs):
        if self.name is None:
            return
        grad_out = self._pick(grad_outputs)
        H_out, W_out = (grad_out.shape[-2]), (grad_out.shape[-1])
        grad_out = (grad_out[:, self.offset, :, :]).reshape(H_out, W_out)
        self._record_flow(self.connet2name, grad_out.cpu().numpy())


class TestBot_Input_3_3(_TestBotBase):
    def __init__(self, module=None, name="out"):
        super(TestBot_Input_3_3, self).__init__(module=module, name=name)

    def forward(self, x, filename="_input_representation.npy", dim=3):
        target_representation_shape = self.testcase.testbot_target.target_representation.shape
        H_target, W_target = (target_representation_shape[-2]), (target_representation_shape[-1])
        C_zoom, h, w = (x.shape[1]), (x.shape[-2]), (x.shape[-1])
        if dim == 4:
            x = x.reshape((x.shape[0]), -1, 3 * 3, (x.shape[-2]), (x.shape[-1]))[:, :1, :, :, :]
            x = x.reshape((x.shape[0]), -1, 3 * 3, (x.shape[-2]), (x.shape[-1]))
            C_zoom, h, w = (x.shape[1]), (x.shape[-2]), (x.shape[-1])
            x = (x[0, :, :, :]).reshape(1, -1, h, w)
        else:
            x = x[:, :1, :, :]
            C_zoom, h, w = (x.shape[1]), (x.shape[-2]), (x.shape[-1])
            x = (x[0, :, :, :]).reshape(-1, h, w)
        self.input_representation = x.detach().cpu().numpy()
        H_diff, W_diff = ((h - H_target) // 2), ((w - W_target) // 2)
        np.save(self._out(filename), self.input_representation[..., H_diff:(h - H_diff), W_diff:(w - W_diff)])
        return self.module(x)


class TestBot_Target(_TestBotBase):
    def __init__(self, module=None, name="out"):
        super(TestBot_Target, self).__init__(module=module, name=name)

    def forward(self, x):
        C_zoom, h, w = x.shape[1], x.shape[-2], x.shape[-1]
        if utils.DEBUG_FLAG:
            print(x.shape)
            print(f"target[0,0,0,0] = {(x[0,:,:,:]).reshape(C_zoom,1,h,w).permute(1,0,2,3)[0,0,0,0]}")
        C_zoom_2 = int(np.sqrt(C_zoom))
        current_zoom = int(np.log2(C_zoom_2))
        x = (x[0, :, :, :])
        for i in range(current_zoom):
            C_zoom_2 = (C_zoom_2 // 2)
            x = x.reshape(-1, C_zoom_2, 2, C_zoom_2, 2, h, w).permute(0, 1, 3, 5, 2, 6, 4)
            h, w = h * 2, w * 2
            x = x.reshape(-1, C_zoom_2, C_zoom_2, h, w)
        x = x.reshape(-1, h, w)
        self.target_representation = x.detach().cpu().numpy()
        np.save(self._out("_target_representation.npy"), self.target_representation)
        return self.module(x)


class TestBot_In_3_3(_TestBotBase):
    def __init__(self, module=None, offset_in=0, name="in", connet2name="out"):
        super(TestBot_In_3_3, self).__init__(module=module, name=name, connet2name=connet2name, offset=offset_in)

    def _on_backward(self, grad_inputs, grad_outputs):
        if self.name is None:
            return
        grad_in = self._pick(grad_inputs)
        grad_in = grad_in.reshape((grad_in.shape[0]), -1, 3 * 3, (grad_in.shape[-2]), (grad_in.shape[-1]))[:, :1, :, :, :]
        grad_in = grad_in.reshape((grad_in.shape[0]), -1, (grad_in.shape[-2]), (grad_in.shape[-1]))
        target_representation_shape = self.testcase.testbot_target.target_representation.shape
        H_target, W_target = (target_representation_shape[-2]), (target_representation_shape[-1])
        C_zoom, H_out, W_out = ((grad_in.shape[1]) // (3 * 3)), (grad_in.shape[-2]), (grad_in.shape[-1])

        C_zoom_2 = int(np.sqrt(C_zoom))
        current_zoom = int(np.log2(C_zoom_2))
        grad_in = (grad_in[0, :, :, :]).reshape(C_zoom, 3 * 3, H_out, W_out).permute(1, 0, 2, 3)
        if utils.DEBUG_FLAG:
            print(grad_in.shape)
            H_orig, W_orig = self.testcase.orig_shape
            H_diff, W_diff = ((H_out - H_orig) // 2), ((W_out - W_orig) // 2)
            print(f"in[4,0,0,0] = {(grad_in[:,:,H_diff:(H_out-H_diff),W_diff:(W_out-W_diff)])[4,0,0,0]}")
        for i in range(current_zoom):
            C_zoom_2 = (C_zoom_2 // 2)
            grad_in = grad_in.reshape(-1, C_zoom_2, 2, C_zoom_2, 2, H_out, W_out).permute(0, 1, 3, 5, 2, 6, 4)
            H_out, W_out = H_out * 2, W_out * 2
            grad_in = grad_in.reshape(-1, C_zoom_2, C_zoom_2, H_out, W_out)
        H_diff, W_diff = ((H_out - H_target) // 2), ((W_out - W_target) // 2)
        grad_in = (grad_in.reshape(-1, H_out, W_out)[:, H_diff:(H_out - H_diff), W_diff:(W_out - W_diff)]).reshape(-1, H_target, W_target)
        self._record_flow(self.name, grad_in.cpu().numpy())
        self._save_flow_info()

    def forward(self, x, mask=None):
        self.input_representation = None
        if mask is not None:
            h, w = mask.shape[-2], mask.shape[-1]
            self.input_representation = mask.detach().cpu().numpy().reshape(h, w)
        return self.module(x)


class TestBot_Out_3_3(_TestBotBase):
    def __init__(self, module=None, offset_out=4, name="in", connet2name="out"):
        super(TestBot_Out_3_3, self).__init__(module=module, name=name, connet2name=connet2name, offset=offset_out)

    def _on_backward(self, grad_inputs, grad_outputs):
        if self.name is None:
            return
        grad_out = self._pick(grad_outputs)
        grad_out = grad_out.reshape((grad_out.shape[0]), -1, 3 * 3, (grad_out.shape[-2]), (grad_out.shape[-1]))[:, :1, :, :, :]
        grad_out = grad_out.reshape((grad_out.shape[0]), -1, (grad_out.shape[-2]), (grad_out.shape[-1]))
        target_representation_shape = self.testcase.testbot_target.target_representation.shape
        H_target, W_target = (target_representation_shape[-2]), (target_representation_shape[-1])
        C_zoom, H_out, W_out = ((grad_out.shape[1]) // (3 * 3)), (grad_out.shape[-2]), (grad_out.shape[-1])

        C_zoom_2 = int(np.sqrt(C_zoom))
        current_zoom = int(np.log2(C_zoom_2))
        grad_out = (grad_out[0, :, :, :]).reshape(C_zoom, 3 * 3, H_out, W_out).permute(1, 0, 2, 3)
        if utils.DEBUG_FLAG:
            H_orig, W_orig = self.testcase.orig_shape
            H_diff, W_diff = ((H_out - H_orig) // 2), ((W_out - W_orig) // 2)
            print(grad_out.shape)
            print(f"out[4,0,0,0] = {(grad_out[:,:,H_diff:(H_out-H_diff),W_diff:(W_out-W_diff)])[4,0,0,0]}")
        for i in range(current_zoom):
            C_zoom_2 = (C_zoom_2 // 2)
            grad_out = grad_out.reshape(-1, C_zoom_2, 2, C_zoom_2, 2, H_out, W_out).permute(0, 1, 3, 5, 2, 6, 4)
            H_out, W_out = H_out * 2, W_out * 2
            grad_out = grad_out.reshape(-1, C_zoom_2, C_zoom_2, H_out, W_out)
        H_diff, W_diff = ((H_out - H_target) // 2), ((W_out - W_target) // 2)
        grad_out = (grad_out.reshape(-1, H_out, W_out)[:, H_diff:(H_out - H_diff), W_diff:(W_out - W_diff)]).reshape(-1, H_target, W_target)
        if self.connet2name == "out":
            self._record_flow(self.connet2name, grad_out.cpu().numpy())
        np.save(self._out(f"_gradient_flow_{self.connet2name}.npy"), grad_out.cpu().numpy()[np.newaxis, :, :, :])


class TestCase():
    """Owns a set of testbots, shared dump path/name, and the captured buffers."""

    _BOT_SLOTS = ("testbot_in", "testbot_out", "testbot_input", "testbot_target")

    def __init__(self, orig_shape=None, name="", testbot_in=None, testbot_out=None,
                 testbot_input=None, testbot_target=None, out_path=None):
        self.orig_shape = orig_shape
        self.out_path = out_path if out_path is not None else "./tests/vtest_data/output/"
        self.name = name
        self.activation_gradients = {}
        self.gradient_flows = {}
        for slot, bot in zip(self._BOT_SLOTS, (testbot_in, testbot_out, testbot_input, testbot_target)):
            if bot is not None:
                setattr(self, slot, bot)
                bot.testcase = self

    def get_testbot_in(self):
        return self.testbot_in

    def get_testbot_out(self):
        return self.testbot_out

    def get_testbot_input(self):
        return self.testbot_input

    def get_testbot_target(self):
        return self.testbot_target


class _DisabledTestbot:
    """No-op stand-in for the vtest testbot, installed when ``utils.DEBUG_FLAG`` is False.

    In audit mode the real :class:`TestCase`/``TestBot_*`` objects are built and
    register backward hooks that dump tensors to ``.npy`` (HDVO instrumentation).
    When the flag is off SMap installs this instead: nothing from this package is
    constructed, no backward hooks are registered, and no file I/O happens. The
    data path is preserved EXACTLY:

      - ``__call__`` / ``testbot_in`` / ``testbot_out`` -> return the tensor
        unchanged (the real ones return ``module(x) == x + 0``, an identity
        passthrough),
      - ``testbot_input`` / ``testbot_target`` -> no-ops (return value discarded),
      - attribute writes (``.name``, ``.orig_shape``, probe stubs) -> absorbed.
    """

    def __call__(self, x, *args, **kwargs):
        return x

    def testbot_in(self, x, *args, **kwargs):
        return x

    def testbot_out(self, x, *args, **kwargs):
        return x

    def testbot_input(self, *args, **kwargs):
        return None

    def testbot_target(self, *args, **kwargs):
        return None

    def __setattr__(self, name, value):
        pass
