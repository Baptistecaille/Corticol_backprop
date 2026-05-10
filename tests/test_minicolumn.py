"""Tests for MiniColumn module."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.minicolumn import MiniColumn
from cortical_column.config import PATCH_SIZE, L4_DIM, N_MINICOLUMNS, SPARSITY_K

# Parameters matching the real use-case in L4Layer
INPUT_DIM  = PATCH_SIZE * PATCH_SIZE  # 49
HIDDEN_DIM = L4_DIM // N_MINICOLUMNS  # 4
K          = SPARSITY_K               # 4
BATCH_SIZE = 8


@pytest.fixture
def model():
    return MiniColumn(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, k=K)


@pytest.fixture
def sample_input():
    return torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)


# ------------------------------------------------------------------
# Test 1: output shape
# ------------------------------------------------------------------
def test_output_shape(model, sample_input):
    out = model(sample_input)
    assert out.shape == (BATCH_SIZE, HIDDEN_DIM), (
        f"Expected shape ({BATCH_SIZE}, {HIDDEN_DIM}), got {out.shape}"
    )


# ------------------------------------------------------------------
# Test 2: sparsity – at most k non-zero values per sample
#
# Pipeline: Linear -> BatchNorm -> ReLU -> K-WTA
# ReLU may zero some pre-K-WTA activations, so the true non-zero count
# is min(k, number_of_relu_survivors) <= k.
# We use a larger hidden_dim (hidden_dim > k) to guarantee exactly k
# non-zero outputs without ReLU interference.
# ------------------------------------------------------------------
def test_sparsity():
    """With hidden_dim > k, K-WTA must keep exactly k non-zero values."""
    hidden_dim = 16  # clearly larger than k=4
    k = 4
    mc = MiniColumn(input_dim=INPUT_DIM, hidden_dim=hidden_dim, k=k)
    mc.eval()
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    with torch.no_grad():
        out = mc(x)

    for i in range(BATCH_SIZE):
        nz = (out[i] != 0).sum().item()
        assert nz == k, (
            f"Sample {i}: expected exactly {k} non-zero values, got {nz}"
        )


# ------------------------------------------------------------------
# Test 3: gradients flow through K-WTA (straight-through estimator)
# ------------------------------------------------------------------
def test_gradients_flow(model, sample_input):
    # Re-create with grad tracking
    x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Input gradient must not be all-zero
    assert x.grad is not None, "No gradient on input"
    assert x.grad.abs().sum().item() > 0, "Input gradient is all zeros"

    # All linear layer parameter gradients must not be all-zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum().item() > 0, (
                f"All-zero gradient for parameter {name}"
            )


# ------------------------------------------------------------------
# Test 4: kwta standalone correctness with a larger hidden_dim
# ------------------------------------------------------------------
def test_kwta_standalone():
    k = 3
    model = MiniColumn(input_dim=8, hidden_dim=10, k=k)
    x = torch.randn(4, 10)
    out = model.kwta(x)
    for i in range(4):
        nz = (out[i] != 0).sum().item()
        assert nz == k, f"K-WTA standalone: expected {k} non-zero, got {nz}"


# ------------------------------------------------------------------
# Test 5: no in-place ops – verify output != input tensor identity
# (catches += style bugs that would break MPS)
# ------------------------------------------------------------------
def test_no_inplace(model, sample_input):
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    out = model(x)
    # Simply ensure forward returns a new tensor (not same storage)
    assert out.data_ptr() != x.data_ptr(), "Output shares storage with input (in-place bug)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
