"""Tests for CorticalNetwork."""

import torch
import pytest

from cortical_column import CorticalNetwork


@pytest.fixture
def net():
    """Return a default CorticalNetwork (16 columns, patch_size=7, latent=128, n_classes=10)."""
    return CorticalNetwork()


# ---------------------------------------------------------------------------
# extract_patches
# ---------------------------------------------------------------------------

class TestExtractPatches:
    def test_output_shape(self, net):
        images = torch.randn(4, 1, 28, 28)
        patches = net.extract_patches(images)
        assert patches.shape == (4, 16, 49), f"Expected (4,16,49), got {patches.shape}"

    def test_batch_size_1(self, net):
        images = torch.randn(1, 1, 28, 28)
        patches = net.extract_patches(images)
        assert patches.shape == (1, 16, 49)

    def test_patch_0_is_top_left_7x7(self, net):
        """Patch index 0 should correspond to rows 0-6, cols 0-6."""
        images = torch.randn(2, 1, 28, 28)
        patches = net.extract_patches(images)
        expected = images[:, 0, 0:7, 0:7].contiguous().view(2, 49)
        assert torch.allclose(patches[:, 0, :], expected)

    def test_patch_1_is_top_second_column(self, net):
        """Patch index 1 should correspond to rows 0-6, cols 7-13."""
        images = torch.randn(2, 1, 28, 28)
        patches = net.extract_patches(images)
        expected = images[:, 0, 0:7, 7:14].contiguous().view(2, 49)
        assert torch.allclose(patches[:, 1, :], expected)

    def test_patch_4_is_second_row_first_column(self, net):
        """Patch index 4 should correspond to rows 7-13, cols 0-6 (second row of patches)."""
        images = torch.randn(2, 1, 28, 28)
        patches = net.extract_patches(images)
        expected = images[:, 0, 7:14, 0:7].contiguous().view(2, 49)
        assert torch.allclose(patches[:, 4, :], expected)

    def test_patch_15_is_bottom_right(self, net):
        """Patch index 15 should correspond to rows 21-27, cols 21-27 (bottom-right)."""
        images = torch.randn(2, 1, 28, 28)
        patches = net.extract_patches(images)
        expected = images[:, 0, 21:28, 21:28].contiguous().view(2, 49)
        assert torch.allclose(patches[:, 15, :], expected)

    def test_patches_cover_full_image(self, net):
        """All 16 patches together should reconstruct every pixel of the image."""
        images = torch.randn(1, 1, 28, 28)
        patches = net.extract_patches(images)
        # Reconstruct all pixels by checking no pixel is missed
        # Sum of absolute values of patches == sum of absolute values of image
        assert torch.allclose(patches.abs().sum(), images.abs().sum(), atol=1e-5)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

class TestForward:
    def test_logits_shape(self, net):
        images = torch.randn(8, 1, 28, 28)
        logits, latents = net(images)
        assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"

    def test_latents_shape(self, net):
        images = torch.randn(8, 1, 28, 28)
        logits, latents = net(images)
        assert latents.shape == (8, 16, 128), f"Expected (8, 16, 128), got {latents.shape}"

    def test_batch_size_1(self, net):
        # BatchNorm requires > 1 sample during training; use eval mode
        net.eval()
        with torch.no_grad():
            images = torch.randn(1, 1, 28, 28)
            logits, latents = net(images)
        assert logits.shape == (1, 10)
        assert latents.shape == (1, 16, 128)

    def test_returns_tuple_of_two(self, net):
        images = torch.randn(3, 1, 28, 28)
        out = net(images)
        assert isinstance(out, tuple)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_through_logits(self, net):
        """Loss from logits should produce gradients in all column parameters."""
        images = torch.randn(4, 1, 28, 28)
        logits, latents = net(images)
        loss = logits.sum()
        loss.backward()

        # Classifier must have gradients
        assert net.classifier.weight.grad is not None
        assert net.classifier.bias.grad is not None

        # Each column's L4 proj weight should have gradients
        for i, col in enumerate(net.columns):
            assert col.l4.proj.weight.grad is not None, \
                f"Column {i} L4 proj weight has no gradient"

    def test_gradients_flow_through_latents(self, net):
        """Loss from latents should also backprop through all columns."""
        images = torch.randn(4, 1, 28, 28)
        logits, latents = net(images)
        loss = latents.sum()
        loss.backward()

        for i, col in enumerate(net.columns):
            assert col.l4.proj.weight.grad is not None, \
                f"Column {i} L4 proj weight has no gradient when backpropping through latents"


# ---------------------------------------------------------------------------
# Independent parameters (no weight sharing)
# ---------------------------------------------------------------------------

class TestIndependentParameters:
    def test_columns_have_different_l4_weights(self, net):
        """Different columns must have different initial weights (no weight sharing)."""
        w0 = net.columns[0].l4.proj.weight.data
        w1 = net.columns[1].l4.proj.weight.data
        assert not torch.allclose(w0, w1), \
            "columns[0] and columns[1] share L4 proj weights — weight sharing detected"

    def test_all_column_l4_weights_distinct(self, net):
        """All 16 columns must have distinct L4 proj weight matrices."""
        weights = [col.l4.proj.weight.data for col in net.columns]
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                assert not torch.allclose(weights[i], weights[j]), \
                    f"columns[{i}] and columns[{j}] share L4 proj weights"

    def test_parameter_update_does_not_affect_other_columns(self, net):
        """Manually updating one column's parameters must not change another column's."""
        # Record initial weight for column 1
        w1_before = net.columns[1].l4.proj.weight.data.clone()

        # Manually perturb column 0's weight
        with torch.no_grad():
            net.columns[0].l4.proj.weight.add_(1.0)

        w1_after = net.columns[1].l4.proj.weight.data
        assert torch.allclose(w1_before, w1_after), \
            "Modifying column 0 affected column 1 — they share a weight tensor"

    def test_n_columns_in_module_list(self, net):
        """ModuleList must contain exactly 16 columns."""
        assert len(net.columns) == 16

    def test_each_column_is_cortical_column_instance(self, net):
        """Every entry in net.columns must be a CorticalColumn."""
        from cortical_column import CorticalColumn
        for i, col in enumerate(net.columns):
            assert isinstance(col, CorticalColumn), \
                f"columns[{i}] is {type(col)}, expected CorticalColumn"
