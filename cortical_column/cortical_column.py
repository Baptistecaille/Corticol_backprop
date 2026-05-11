"""CorticalColumn: assembles the 6 layers in biological processing order.

Information flow:
    patch → L4 → L2/3(+L1 feedback) → L5(+L6 feedback) → L6 → latent
"""

import torch
import torch.nn as nn
from torch import Tensor

from cortical_column.config import L1_DIM, L6_DIM, N_CLASSES, LATENT_DIM
from cortical_column.cortical_layers import L1Layer, L4Layer, L23Layer, L5Layer, L6Layer


class CorticalColumn(nn.Module):
    """
    Assembles the 6 layers in biological processing order.
    Receives a flattened patch of any dimension, produces a latent vector ∈ ℝ¹²⁸.

    Information flow:
        patch → L4 → L2/3(+L1 feedback) → L5(+L6 feedback) → L6 → latent

    Args:
        patch_dim  : flattened patch dimension (49 for MNIST, 192 for CIFAR-10 RGB 8×8)
        latent_dim : latent vector output dimension (128)
    """

    def __init__(self, patch_dim: int = 49, latent_dim: int = 128):
        super().__init__()
        assert latent_dim == LATENT_DIM, f"latent_dim must be {LATENT_DIM}, got {latent_dim}"
        self.patch_dim = patch_dim
        self.latent_dim = latent_dim
        self.l1  = L1Layer(input_dim=N_CLASSES)
        self.l4  = L4Layer(patch_dim=patch_dim)
        self.l23 = L23Layer()
        self.l5  = L5Layer()
        self.l6  = L6Layer()

    def forward(
        self,
        patch: Tensor,                       # [B, 49]
        top_down_signal: Tensor | None       # [B, L1_DIM] or None
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            latent    : Tensor[B, 128]  — main representation (L5)
            l6_signal : Tensor[B, 64]   — L6 output (to be used as feedback in recurrent extensions)
        """
        B = patch.shape[0]

        # 1. L4: normalized input
        x_l4 = self.l4(patch)

        # 2. L1: top-down error signal (zeros if None)
        if top_down_signal is not None:
            x_l1 = self.l1(top_down_signal)
        else:
            x_l1 = torch.zeros(B, L1_DIM, device=patch.device, dtype=patch.dtype)

        # 3. L2/3: fusion
        x_l23 = self.l23(x_l4, x_l1)

        # 4. L5: integration + latent
        #    l6_prev = zeros (stateless implementation, no recurrence across batches)
        l6_prev = torch.zeros(B, L6_DIM, device=patch.device, dtype=patch.dtype)
        latent = self.l5(x_l23, l6_prev)

        # 5. L6: gating
        l6_out = self.l6(latent)

        return latent, l6_out
