"""CorticalNetwork: orchestrates N cortical columns over N patches of an image.

Each column is specialized for its own spatial region (no weight sharing).
Aggregates latents via mean pooling and classifies via a linear head.

Supports any image resolution and channel count:
  - MNIST    : image_size=28, n_channels=1, patch_size=7  → 16 patches of  49 dims
  - CIFAR-10 : image_size=32, n_channels=3, patch_size=8  → 16 patches of 192 dims
"""

import torch
import torch.nn as nn
from torch import Tensor

from cortical_column.config import N_PATCHES, PATCH_SIZE, LATENT_DIM, N_CLASSES
from cortical_column.cortical_column import CorticalColumn


class CorticalNetwork(nn.Module):
    """
    Orchestrates N cortical columns over N patches of an image.
    Aggregates latents via mean pooling and classifies via a linear head.

    Each column is specialized for its spatial region (NO weight sharing).

    Args:
        n_columns  : number of columns = number of patches (16)
        patch_size : patch side length in pixels (7 for MNIST, 8 for CIFAR-10)
        n_channels : input image channels (1 = grayscale, 3 = RGB)
        latent_dim : column latent dimension (128)
        n_classes  : number of output classes (10)
    """

    def __init__(
        self,
        n_columns: int = N_PATCHES,
        patch_size: int = PATCH_SIZE,
        n_channels: int = 1,
        latent_dim: int = LATENT_DIM,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.n_columns = n_columns
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        patch_dim = patch_size ** 2 * n_channels   # 49 (MNIST) or 192 (CIFAR-10)
        self.columns = nn.ModuleList([
            CorticalColumn(patch_dim=patch_dim, latent_dim=latent_dim)
            for _ in range(n_columns)
        ])
        self.classifier = nn.Linear(latent_dim, n_classes)

    def extract_patches(self, images: Tensor) -> Tensor:
        """
        Slice images into non-overlapping patches (row-major order).

        Input  : [B, C, H, W]
        Output : [B, n_patches, C × patch_size²]

        Works for any (H, W, C, patch_size) — MNIST and CIFAR-10.
        """
        B, C, H, W = images.shape
        ps = self.patch_size
        # unfold spatial dims → [B, C, n_h, n_w, ps, ps]
        patches = (
            images
            .unfold(2, ps, ps)   # height → [B, C, n_h, W, ps]
            .unfold(3, ps, ps)   # width  → [B, C, n_h, n_w, ps, ps]
        )
        n_h, n_w = patches.shape[2], patches.shape[3]
        # reorder: [B, n_h, n_w, C, ps, ps] so channels merge cleanly with spatial
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # flatten to [B, n_h*n_w, C*ps*ps]
        return patches.view(B, n_h * n_w, C * ps * ps)

    def voting(self, latents: Tensor) -> Tensor:
        """
        Mean pooling over the N column latents.
        Input  : [B, N, 128]
        Output : [B, 128]
        """
        return latents.mean(dim=1)

    def l4_sparsity(self) -> float:
        """
        Return mean fraction of non-zero K-WTA activations across all L4 mini-columns.

        Must be called after a forward pass — reads cached pre-projection sparse tensors
        from each column's L4Layer. Expected value ~SPARSITY_K / MINICOLUMN_HIDDEN_DIM = 0.25.
        """
        sparse_list = [col.l4._sparse_cache for col in self.columns]
        sparse = torch.stack(sparse_list, dim=1)  # [B, N, N_MINICOLUMNS * MINICOLUMN_HIDDEN_DIM]
        return (sparse != 0).float().mean().item()

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
            logits  : Tensor[B, n_classes]   -- classification head
            latents : Tensor[B, n_columns, latent_dim] -- per-column representations
        """
        patches = self.extract_patches(images)              # [B, N, patch_dim]
        latents = []
        for i, col in enumerate(self.columns):
            lat, _ = col(patches[:, i, :], top_down_signal=None)
            latents.append(lat)
        latents = torch.stack(latents, dim=1)               # [B, N, 128]
        pooled  = self.voting(latents)                      # [B, 128]
        logits  = self.classifier(pooled)                   # [B, n_classes]
        return logits, latents
