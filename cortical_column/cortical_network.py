"""CorticalNetwork: orchestrates 16 cortical columns over 16 patches of an MNIST image.

Each column is specialized for its own spatial region (no weight sharing).
Aggregates latents via mean pooling and classifies via a linear head.

Future: remove the classification head and use latents directly as
encoder for I-JEPA.
"""

import torch
import torch.nn as nn
from torch import Tensor

from cortical_column.config import N_PATCHES, PATCH_SIZE, LATENT_DIM, N_CLASSES
from cortical_column.cortical_column import CorticalColumn


class CorticalNetwork(nn.Module):
    """
    Orchestrates 16 cortical columns over the 16 patches of an MNIST image.
    Aggregates latents via mean pooling and classifies via a linear head.

    Each column is specialized for its spatial region (NO weight sharing).

    Future: remove the classification head and use latents directly as
    encoder for I-JEPA.

    Args:
        n_columns  : 16
        patch_size : 7
        latent_dim : 128
        n_classes  : 10
    """

    def __init__(
        self,
        n_columns: int = N_PATCHES,
        patch_size: int = PATCH_SIZE,
        latent_dim: int = LATENT_DIM,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        n_patches_from_grid = (28 // patch_size) ** 2
        assert n_columns == n_patches_from_grid, (
            f"n_columns={n_columns} does not match grid patch count {n_patches_from_grid}"
        )
        self.n_columns = n_columns
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.columns = nn.ModuleList([
            CorticalColumn(patch_dim=patch_size ** 2, latent_dim=latent_dim)
            for _ in range(n_columns)
        ])
        self.classifier = nn.Linear(latent_dim, n_classes)

    def extract_patches(self, images: Tensor) -> Tensor:
        """
        Slice images into 16 patches of 7x7.
        Input  : [B, 1, 28, 28]
        Output : [B, 16, 49]
        """
        assert images.shape[1:] == torch.Size([1, 28, 28]), \
            f"Expected [B,1,28,28], got {tuple(images.shape)}"
        B = images.shape[0]
        # unfold height (dim=2): size=7, step=7 -> [B, C, 4, W, 7] where W=28
        # unfold width  (dim=3): size=7, step=7 -> [B, C, 4, 4, 7, 7]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches: [B, 1, 4, 4, 7, 7]
        patches = patches.contiguous().view(B, -1, self.patch_size ** 2)
        # patches: [B, 16, 49]
        return patches

    def voting(self, latents: Tensor) -> Tensor:
        """
        Mean pooling over the 16 column latents.
        Input  : [B, 16, 128]
        Output : [B, 128]
        """
        return latents.mean(dim=1)

    def l4_sparsity(self) -> float:
        """
        Return mean fraction of non-zero K-WTA activations across all L4 mini-columns.

        Must be called after a forward pass — reads cached pre-projection sparse tensors
        from each column's L4Layer. Expected value ~SPARSITY_K / MINICOLUMN_HIDDEN_DIM = 0.25.

        Returns:
            float in [0, 1] — mean active fraction per mini-column unit
        """
        sparse_list = [col.l4._sparse_cache for col in self.columns]
        sparse = torch.stack(sparse_list, dim=1)  # [B, 16, N_MINICOLUMNS * MINICOLUMN_HIDDEN_DIM]
        return (sparse != 0).float().mean().item()

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
            logits  : Tensor[B, 10]      -- for MNIST loss
            latents : Tensor[B, 16, 128] -- for JEPA later
        """
        patches = self.extract_patches(images)          # [B, 16, 49]
        latents = []
        for i, col in enumerate(self.columns):
            lat, _ = col(patches[:, i, :], top_down_signal=None)  # l6 signal unused in stateless forward pass
            latents.append(lat)
        latents = torch.stack(latents, dim=1)           # [B, 16, 128]
        pooled = self.voting(latents)                   # [B, 128]
        logits = self.classifier(pooled)                # [B, 10]
        return logits, latents
