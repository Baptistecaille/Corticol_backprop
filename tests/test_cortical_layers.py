"""Tests for the five cortical layer classes."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.cortical_layers import (
    L4Layer,
    L23Layer,
    L5Layer,
    L6Layer,
    L1Layer,
)
from cortical_column.config import (
    PATCH_SIZE,
    LATENT_DIM,
    N_CLASSES,
    L4_DIM,
    L23_DIM,
    L5_DIM,
    L6_DIM,
    L1_DIM,
)

BATCH = 8


# ---------------------------------------------------------------------------
# Device fixture: runs each test on CPU; also on MPS if available
# ---------------------------------------------------------------------------
def available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=available_devices())
def device(request):
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# Helper: check all parameter gradients are non-zero
# ---------------------------------------------------------------------------
def _assert_grad_flow(module: torch.nn.Module, loss: torch.Tensor):
    loss.backward()
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum().item() > 0, (
                f"All-zero gradient for parameter {name}"
            )


# ===========================================================================
# L4Layer
# ===========================================================================
class TestL4Layer:
    def test_output_shape(self, device):
        model = L4Layer().to(device)
        patch = torch.randn(BATCH, PATCH_SIZE ** 2, device=device)
        out = model(patch)
        assert out.shape == (BATCH, L4_DIM), (
            f"Expected ({BATCH}, {L4_DIM}), got {out.shape}"
        )

    def test_gradient_flow(self, device):
        model = L4Layer().to(device)
        patch = torch.randn(BATCH, PATCH_SIZE ** 2, device=device, requires_grad=True)
        out = model(patch)
        _assert_grad_flow(model, out.sum())
        assert patch.grad is not None
        assert patch.grad.abs().sum().item() > 0

    def test_no_inplace(self, device):
        model = L4Layer().to(device)
        patch = torch.randn(BATCH, PATCH_SIZE ** 2, device=device)
        out = model(patch)
        assert out.data_ptr() != patch.data_ptr()


# ===========================================================================
# L23Layer
# ===========================================================================
class TestL23Layer:
    def test_output_shape(self, device):
        model = L23Layer().to(device)
        bu = torch.randn(BATCH, L4_DIM, device=device)
        td = torch.randn(BATCH, L1_DIM, device=device)
        out = model(bu, td)
        assert out.shape == (BATCH, L23_DIM), (
            f"Expected ({BATCH}, {L23_DIM}), got {out.shape}"
        )

    def test_output_shape_no_lateral(self, device):
        model = L23Layer(use_lateral=False).to(device)
        bu = torch.randn(BATCH, L4_DIM, device=device)
        td = torch.randn(BATCH, L1_DIM, device=device)
        out = model(bu, td)
        assert out.shape == (BATCH, L23_DIM)

    def test_gradient_flow(self, device):
        model = L23Layer().to(device)
        bu = torch.randn(BATCH, L4_DIM, device=device, requires_grad=True)
        td = torch.randn(BATCH, L1_DIM, device=device, requires_grad=True)
        out = model(bu, td)
        _assert_grad_flow(model, out.sum())
        for inp in (bu, td):
            assert inp.grad is not None
            assert inp.grad.abs().sum().item() > 0

    def test_no_inplace(self, device):
        model = L23Layer().to(device)
        bu = torch.randn(BATCH, L4_DIM, device=device)
        td = torch.randn(BATCH, L1_DIM, device=device)
        out = model(bu, td)
        assert out.data_ptr() != bu.data_ptr()
        assert out.data_ptr() != td.data_ptr()


# ===========================================================================
# L5Layer
# ===========================================================================
class TestL5Layer:
    def test_output_shape(self, device):
        model = L5Layer().to(device)
        x_l23 = torch.randn(BATCH, L23_DIM, device=device)
        x_fb = torch.randn(BATCH, L6_DIM, device=device)
        out = model(x_l23, x_fb)
        assert out.shape == (BATCH, L5_DIM), (
            f"Expected ({BATCH}, {L5_DIM}), got {out.shape}"
        )

    def test_l5_dim_equals_latent_dim(self):
        assert L5_DIM == LATENT_DIM, (
            f"L5_DIM ({L5_DIM}) must equal LATENT_DIM ({LATENT_DIM})"
        )

    def test_gradient_flow(self, device):
        model = L5Layer().to(device)
        x_l23 = torch.randn(BATCH, L23_DIM, device=device, requires_grad=True)
        x_fb = torch.randn(BATCH, L6_DIM, device=device, requires_grad=True)
        out = model(x_l23, x_fb)
        _assert_grad_flow(model, out.sum())
        for inp in (x_l23, x_fb):
            assert inp.grad is not None
            assert inp.grad.abs().sum().item() > 0

    def test_no_inplace(self, device):
        model = L5Layer().to(device)
        x_l23 = torch.randn(BATCH, L23_DIM, device=device)
        x_fb = torch.randn(BATCH, L6_DIM, device=device)
        out = model(x_l23, x_fb)
        assert out.data_ptr() != x_l23.data_ptr()
        assert out.data_ptr() != x_fb.data_ptr()


# ===========================================================================
# L6Layer
# ===========================================================================
class TestL6Layer:
    def test_output_shape(self, device):
        model = L6Layer().to(device)
        x = torch.randn(BATCH, L5_DIM, device=device)
        out = model(x)
        assert out.shape == (BATCH, L6_DIM), (
            f"Expected ({BATCH}, {L6_DIM}), got {out.shape}"
        )

    def test_output_range_sigmoid(self, device):
        """Sigmoid output must be in (0, 1)."""
        model = L6Layer().to(device)
        x = torch.randn(BATCH, L5_DIM, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.min().item() > 0.0, "Sigmoid output should be > 0"
        assert out.max().item() < 1.0, "Sigmoid output should be < 1"

    def test_gradient_flow(self, device):
        model = L6Layer().to(device)
        x = torch.randn(BATCH, L5_DIM, device=device, requires_grad=True)
        out = model(x)
        _assert_grad_flow(model, out.sum())
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_no_inplace(self, device):
        model = L6Layer().to(device)
        x = torch.randn(BATCH, L5_DIM, device=device)
        out = model(x)
        assert out.data_ptr() != x.data_ptr()


# ===========================================================================
# L1Layer
# ===========================================================================
class TestL1Layer:
    def test_output_shape_default(self, device):
        """Default input_dim = N_CLASSES."""
        model = L1Layer().to(device)
        err = torch.randn(BATCH, N_CLASSES, device=device)
        out = model(err)
        assert out.shape == (BATCH, L1_DIM), (
            f"Expected ({BATCH}, {L1_DIM}), got {out.shape}"
        )

    def test_output_shape_latent_dim(self, device):
        """Flexible input_dim: use LATENT_DIM."""
        model = L1Layer(input_dim=LATENT_DIM).to(device)
        err = torch.randn(BATCH, LATENT_DIM, device=device)
        out = model(err)
        assert out.shape == (BATCH, L1_DIM)

    def test_gradient_flow(self, device):
        model = L1Layer().to(device)
        err = torch.randn(BATCH, N_CLASSES, device=device, requires_grad=True)
        out = model(err)
        _assert_grad_flow(model, out.sum())
        assert err.grad is not None
        assert err.grad.abs().sum().item() > 0

    def test_no_inplace(self, device):
        model = L1Layer().to(device)
        err = torch.randn(BATCH, N_CLASSES, device=device)
        out = model(err)
        assert out.data_ptr() != err.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
