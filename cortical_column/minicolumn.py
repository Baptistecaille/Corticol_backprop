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
        """
        # Find the k-th largest value per sample
        topk_vals, _ = torch.topk(x, self.k, dim=-1)
        # Threshold: smallest of the top-k values
        threshold = topk_vals[..., -1].unsqueeze(-1)

        # Hard mask: 1 where value >= threshold, 0 elsewhere
        mask = (x >= threshold).float()

        # Straight-through estimator: forward uses sparse output,
        # backward passes gradient as if all units were active
        sparse = x * mask
        output = x + (sparse - x).detach()
        return output

    def forward(self, x: Tensor) -> Tensor:
        # Linear -> BatchNorm -> ReLU -> K-WTA
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.kwta(x)
        return x
