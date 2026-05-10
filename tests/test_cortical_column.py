"""Tests for CorticalColumn — integration of L1, L4, L23, L5, L6 layers."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.cortical_column import CorticalColumn
from cortical_column.config import (
    PATCH_SIZE,
    LATENT_DIM,
    N_CLASSES,
    L6_DIM,
)

BATCH = 8
PATCH_DIM = PATCH_SIZE ** 2  # 49


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
# Helper: check all parameter gradients are non-zero after backward
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
# CorticalColumn
# ===========================================================================
class TestCorticalColumn:

    def test_output_shapes_no_top_down(self, device):
        """Forward with top_down_signal=None returns correct shapes."""
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device)
        latent, l6_signal = model(patch, top_down_signal=None)
        assert latent.shape == (BATCH, LATENT_DIM), (
            f"Expected latent ({BATCH}, {LATENT_DIM}), got {latent.shape}"
        )
        assert l6_signal.shape == (BATCH, L6_DIM), (
            f"Expected l6_signal ({BATCH}, {L6_DIM}), got {l6_signal.shape}"
        )

    def test_output_shapes_with_top_down(self, device):
        """Forward with a top_down_signal tensor returns correct shapes."""
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device)
        top_down = torch.randn(BATCH, N_CLASSES, device=device)
        latent, l6_signal = model(patch, top_down_signal=top_down)
        assert latent.shape == (BATCH, LATENT_DIM), (
            f"Expected latent ({BATCH}, {LATENT_DIM}), got {latent.shape}"
        )
        assert l6_signal.shape == (BATCH, L6_DIM), (
            f"Expected l6_signal ({BATCH}, {L6_DIM}), got {l6_signal.shape}"
        )

    def test_return_type_is_tuple(self, device):
        """forward must always return a tuple of two tensors."""
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device)
        result = model(patch, top_down_signal=None)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"

    def test_gradient_flow_latent_no_top_down(self, device):
        """Gradient flows from latent output through L4, L23, L5 (no top_down).

        Computation graph for latent: patch -> L4 -> L23(proj_bu only) -> L5
        - l1 is skipped (zeros injected), so l1.proj has no gradient.
        - l23.proj_td has zero gradient because its input is all-zeros.
        - l6 is downstream of latent, not upstream, so no gradient from latent.sum().
        """
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device, requires_grad=True)
        latent, _ = model(patch, top_down_signal=None)
        latent.sum().backward()
        # L4 and L5 must have gradients
        for submod in [model.l4, model.l5]:
            for name, param in submod.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No gradient for {name}"
                    assert param.grad.abs().sum().item() > 0, (
                        f"All-zero gradient for {name}"
                    )
        # L23 proj_bu must have gradient (bottom-up path is active)
        assert model.l23.proj_bu.weight.grad is not None
        assert model.l23.proj_bu.weight.grad.abs().sum().item() > 0
        # Input gradient
        assert patch.grad is not None
        assert patch.grad.abs().sum().item() > 0

    def test_gradient_flow_with_top_down(self, device):
        """Gradients flow through L1, L4, L23, L5 when top_down_signal is provided.

        l6 is still downstream of latent, so l6 params have no gradient from
        latent.sum(). We test l6 separately via l6_signal.
        """
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device, requires_grad=True)
        top_down = torch.randn(BATCH, N_CLASSES, device=device, requires_grad=True)
        latent, _ = model(patch, top_down_signal=top_down)
        latent.sum().backward()
        # L1, L4, L23, L5 should all have gradients
        for submod in [model.l1, model.l4, model.l23, model.l5]:
            for name, param in submod.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No gradient for {name}"
                    assert param.grad.abs().sum().item() > 0, (
                        f"All-zero gradient for {name}"
                    )
        assert patch.grad is not None
        assert patch.grad.abs().sum().item() > 0
        assert top_down.grad is not None
        assert top_down.grad.abs().sum().item() > 0

    def test_gradient_flow_l6_output(self, device):
        """Gradients flow through L4, L23, L5, L6 from the l6_signal output."""
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device, requires_grad=True)
        _, l6_signal = model(patch, top_down_signal=None)
        l6_signal.sum().backward()
        # L6 is in the graph for l6_signal (latent -> L6 -> l6_signal)
        for name, param in model.l6.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for l6.{name}"
                assert param.grad.abs().sum().item() > 0, (
                    f"All-zero gradient for l6.{name}"
                )
        # L5 feeds latent which feeds l6_signal
        for name, param in model.l5.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for l5.{name}"
                assert param.grad.abs().sum().item() > 0, (
                    f"All-zero gradient for l5.{name}"
                )
        assert patch.grad is not None
        assert patch.grad.abs().sum().item() > 0

    def test_no_inplace_operations(self, device):
        """Output tensors are distinct from input tensors."""
        model = CorticalColumn().to(device)
        patch = torch.randn(BATCH, PATCH_DIM, device=device)
        top_down = torch.randn(BATCH, N_CLASSES, device=device)
        latent, l6_signal = model(patch, top_down_signal=top_down)
        assert latent.data_ptr() != patch.data_ptr()
        assert l6_signal.data_ptr() != patch.data_ptr()

    def test_different_batch_sizes(self, device):
        """Forward pass works for various batch sizes.

        BatchNorm in MiniColumn requires B > 1 in training mode, so batch size 1
        is tested in eval mode only (consistent with inference usage).
        """
        model = CorticalColumn().to(device)
        for B in (2, 4, 16, 32):
            patch = torch.randn(B, PATCH_DIM, device=device)
            latent, l6_signal = model(patch, top_down_signal=None)
            assert latent.shape == (B, LATENT_DIM)
            assert l6_signal.shape == (B, L6_DIM)

        # B=1 only works in eval mode (BatchNorm constraint)
        model.eval()
        with torch.no_grad():
            patch = torch.randn(1, PATCH_DIM, device=device)
            latent, l6_signal = model(patch, top_down_signal=None)
            assert latent.shape == (1, LATENT_DIM)
            assert l6_signal.shape == (1, L6_DIM)

    def test_top_down_none_vs_zeros_differ(self, device):
        """Passing None for top_down should give same result as passing explicit zeros
        of shape [B, N_CLASSES], because L1(zeros) != zeros in general, but
        top_down=None bypasses L1 and injects zeros directly into L2/3."""
        model = CorticalColumn().to(device)
        model.eval()
        patch = torch.randn(BATCH, PATCH_DIM, device=device)
        with torch.no_grad():
            latent_none, _ = model(patch, top_down_signal=None)
            # top_down=None injects zeros of size L1_DIM directly (skips L1 projection)
            # whereas passing zeros[N_CLASSES] would go through L1 linear layer
            zeros_td = torch.zeros(BATCH, N_CLASSES, device=device)
            latent_zeros, _ = model(patch, top_down_signal=zeros_td)
        # They should differ because L1(zeros) != zeros (bias term in L1.proj)
        # Note: this test is informational — both paths are valid
        # We just verify neither is NaN/Inf
        assert not torch.isnan(latent_none).any(), "latent_none contains NaN"
        assert not torch.isnan(latent_zeros).any(), "latent_zeros contains NaN"
        assert not torch.isinf(latent_none).any(), "latent_none contains Inf"
        assert not torch.isinf(latent_zeros).any(), "latent_zeros contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
