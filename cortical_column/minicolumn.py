"""MiniColumn module: sparse representation via K-Winner-Take-All."""

import torch
import torch.nn as nn
from torch import Tensor


class MiniColumn(nn.Module):
    """
    Processes an input vector (flattened patch) and produces
    a sparse representation via K-Winner-Take-All (K-WTA).

    Args:
        input_dim  : size of input vector (e.g. 49 for 7x7 patch)
        hidden_dim : internal representation dimension
        k          : number of activations kept (sparsity)
    """

    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        assert k < hidden_dim, (
            f"k ({k}) must be < hidden_dim ({hidden_dim}) for sparsity to have effect"
        )
        self.k = k
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.linear.weight)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def kwta(self, x: Tensor) -> Tensor:
        """
        K-Winner-Take-All: keeps the k highest values,
        zeros the rest. Differentiable via straight-through estimator.

        Uses scatter-based masking to guarantee exactly k active units,
        avoiding the tie-breaking issue of threshold >= comparisons.
        """
        topk_vals, topk_idx = torch.topk(x, self.k, dim=-1)
        # Build a binary mask with exactly k ones per row
        mask = torch.zeros_like(x).scatter_(-1, topk_idx, 1.0)

        # Sparse forward value
        sparse = x * mask

        # Straight-through estimator: forward uses sparse output,
        # backward passes gradient as if all units were active
        output = x + (sparse - x).detach()
        return output

    def forward(self, x: Tensor) -> Tensor:
        # Linear -> BatchNorm -> ReLU -> K-WTA
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.kwta(x)
        return x
