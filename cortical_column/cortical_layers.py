"""Five cortical layer classes for the cortical column network.

All tensors are (batch, features) — no spatial dimensions.
Designed for MPS compatibility: no in-place operations.
"""

import torch
import torch.nn as nn
from torch import Tensor

from cortical_column.config import (
    PATCH_SIZE,
    N_MINICOLUMNS,
    MINICOLUMN_HIDDEN_DIM,
    SPARSITY_K,
    N_CLASSES,
    L4_DIM,
    L23_DIM,
    L5_DIM,
    L6_DIM,
    L1_DIM,
)
from cortical_column.minicolumn import MiniColumn


class L4Layer(nn.Module):
    """
    Primary input layer. Receives a flattened patch of arbitrary dimension.
    Projects via N_MINICOLUMNS MiniColumns in parallel, aggregates sparse outputs.

    Default patch_dim = PATCH_SIZE² = 49  (MNIST, grayscale 7×7).
    For CIFAR-10 (RGB 8×8): patch_dim = 8² × 3 = 192.

    forward(patch: Tensor[B, patch_dim]) -> Tensor[B, L4_DIM]
    """

    def __init__(self, patch_dim: int = PATCH_SIZE ** 2):
        super().__init__()
        self.minicolumns = nn.ModuleList([
            MiniColumn(
                input_dim=patch_dim,
                hidden_dim=MINICOLUMN_HIDDEN_DIM,
                k=SPARSITY_K,
            )
            for _ in range(N_MINICOLUMNS)
        ])

        # N_MINICOLUMNS * MINICOLUMN_HIDDEN_DIM -> L4_DIM
        concat_dim = N_MINICOLUMNS * MINICOLUMN_HIDDEN_DIM
        self.proj = nn.Linear(concat_dim, L4_DIM)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')

        self.norm = nn.LayerNorm(L4_DIM, eps=1e-4)
        self.act = nn.ReLU()

    def forward(self, patch: Tensor) -> Tensor:
        # Each MiniColumn: [B, 49] -> [B, MINICOLUMN_HIDDEN_DIM]
        outputs = [mc(patch) for mc in self.minicolumns]
        # Concatenate: [B, N_MINICOLUMNS * MINICOLUMN_HIDDEN_DIM]
        x = torch.cat(outputs, dim=-1)
        # Cache pre-projection sparse tensor for sparsity monitoring (detached, no grad)
        self._sparse_cache: Tensor = x.detach()
        # Project -> norm -> activate
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class L23Layer(nn.Module):
    """
    Fuses bottom-up (L4) and top-down (L1) signals via separate linear projections
    concatenated together. Includes an optional lateral recurrent connection
    (residual add) for within-layer propagation.

    forward(
        x_bottom_up: Tensor[B, L4_DIM],
        x_top_down:  Tensor[B, L1_DIM]
    ) -> Tensor[B, L23_DIM]
    """

    def __init__(self, use_lateral: bool = True):
        super().__init__()
        self.use_lateral = use_lateral

        self.proj_bu = nn.Linear(L4_DIM, L23_DIM // 2)
        nn.init.kaiming_normal_(self.proj_bu.weight, nonlinearity='relu')

        self.proj_td = nn.Linear(L1_DIM, L23_DIM // 2)
        nn.init.kaiming_normal_(self.proj_td.weight, nonlinearity='relu')

        self.norm = nn.LayerNorm(L23_DIM, eps=1e-4)
        self.act = nn.ReLU()

        if use_lateral:
            self.lateral = nn.Linear(L23_DIM, L23_DIM)
            nn.init.kaiming_normal_(self.lateral.weight, nonlinearity='relu')

    def forward(self, x_bottom_up: Tensor, x_top_down: Tensor) -> Tensor:
        bu = self.proj_bu(x_bottom_up)   # [B, L23_DIM//2]
        td = self.proj_td(x_top_down)    # [B, L23_DIM//2]
        x = torch.cat([bu, td], dim=-1)  # [B, L23_DIM]
        x = self.norm(x)
        x = self.act(x)
        if self.use_lateral:
            x = x + self.lateral(x)      # residual add (no in-place +=)
        return x


class L5Layer(nn.Module):
    """
    Integrates L2/3 (bottom-up) and L6 (feedback) signals.
    Produces the main latent vector of the column.

    forward(
        x_l23:      Tensor[B, L23_DIM],
        x_feedback: Tensor[B, L6_DIM]
    ) -> Tensor[B, L5_DIM]   # = LATENT_DIM = 128
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(L23_DIM + L6_DIM, L5_DIM)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')

        self.norm = nn.LayerNorm(L5_DIM, eps=1e-4)
        self.act = nn.GELU()

    def forward(self, x_l23: Tensor, x_feedback: Tensor) -> Tensor:
        x = torch.cat([x_l23, x_feedback], dim=-1)  # [B, L23_DIM + L6_DIM]
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class L6Layer(nn.Module):
    """
    Cortical gain regulation. Receives L5 output, produces gating signal
    sent back to L5 and (future) thalamus.

    forward(x_l5: Tensor[B, L5_DIM]) -> Tensor[B, L6_DIM]
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(L5_DIM, L6_DIM)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')

        self.act = nn.Sigmoid()

    def forward(self, x_l5: Tensor) -> Tensor:
        x = self.proj(x_l5)
        x = self.act(x)
        return x


class L1Layer(nn.Module):
    """
    Transports error signal from network output to superficial layers (L2/3).
    Models apical dendrites of pyramidal neurons.

    forward(error_signal: Tensor[B, N_CLASSES or LATENT_DIM]) -> Tensor[B, L1_DIM]
    """

    def __init__(self, input_dim: int = N_CLASSES):
        super().__init__()
        self.proj = nn.Linear(input_dim, L1_DIM)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')

        self.act = nn.ReLU()

    def forward(self, error_signal: Tensor) -> Tensor:
        x = self.proj(error_signal)
        x = self.act(x)
        return x
