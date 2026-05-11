"""Minimal ViT encoder for the I-JEPA baseline comparison.

Architecture choices for a fair comparison with CorticalNetwork on CIFAR-10:
  - Same number of patches : 16 (4×4 grid of 8×8×3 patches)
  - Same latent dimension  : 128  (= LATENT_DIM = VIT_EMBED_DIM)
  - Same JEPA components   : CorticalPredictor, BlockMaskingStrategy, jepa_loss
  - Only the encoder differs: attention-based (ViT) vs. layer-structured (cortical)

Parameter count (~1.2M encoder-only) is intentionally close to the cortical
column encoder (~1.3M) for a controlled comparison.

Encoding API mirrors CorticalNetwork where needed:
  vit.encode_context(patches, ctx_idx) → [B, n_ctx, 128]   (context encoder)
  vit.encode_all(patches)              → [B, 16,   128]   (target encoder / EMA)
  vit.forward(images)                  → (logits, latents) (supervised head)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cortical_column.config import LATENT_DIM, N_PATCHES, N_CLASSES
from cortical_column.config_cifar import (
    CIFAR_PATCH_SIZE,
    CIFAR_N_CHANNELS,
    CIFAR_PATCH_DIM,
    VIT_EMBED_DIM,
    VIT_DEPTH,
    VIT_N_HEADS,
    VIT_MLP_RATIO,
    VIT_DROPOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Pre-LN multi-head self-attention (batch_first)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out   # residual


class FFN(nn.Module):
    """Pre-LN feed-forward network (GELU)."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))   # residual


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.ffn  = FFN(embed_dim, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(self.attn(x))


# ─────────────────────────────────────────────────────────────────────────────
# ViT Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    Minimal Vision Transformer encoder for the I-JEPA baseline.

    Compared to CorticalNetwork (cortical column):
    ┌──────────────────────┬──────────────────────────────────────────────┐
    │ Property             │ Value (both models)                          │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Input patches        │ 16 × (8×8×3 = 192 dims)                     │
    │ Latent dimension     │ 128                                          │
    │ Output classes       │ 10                                           │
    │ JEPA predictor       │ shared CorticalPredictor                     │
    │ Masking strategy     │ shared BlockMaskingStrategy                  │
    └──────────────────────┴──────────────────────────────────────────────┘

    Encoder-only parameters:  ~1.2M
    Cortical column encoder:  ~1.3M

    The ViT has cross-patch attention in the encoder (the key architectural
    difference from the cortical column, where each column is independent).

    JEPA encoding API:
        encode_context(patches, ctx_idx) → [B, n_ctx, D]  — for online encoder
        encode_all(patches)              → [B, 16,   D]  — for EMA target encoder
    """

    def __init__(
        self,
        patch_dim: int   = CIFAR_PATCH_DIM,   # 192
        embed_dim: int   = VIT_EMBED_DIM,     # 128
        n_patches: int   = N_PATCHES,          # 16
        depth: int       = VIT_DEPTH,          # 6
        n_heads: int     = VIT_N_HEADS,        # 4
        mlp_ratio: float = VIT_MLP_RATIO,      # 4.0
        dropout: float   = VIT_DROPOUT,        # 0.0
        n_classes: int   = N_CLASSES,          # 10
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}"

        self.embed_dim  = embed_dim
        self.n_patches  = n_patches
        self.patch_size = CIFAR_PATCH_SIZE    # stored for extract_patches

        # Patch projection: linear(192 → 128)
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)

        # Learnable positional embeddings: one per patch position
        self.pos_embed = nn.Parameter(torch.zeros(n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (supervised training only)
        self.classifier = nn.Linear(embed_dim, n_classes)
        nn.init.zeros_(self.classifier.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Patch extraction (mirrors CorticalNetwork.extract_patches) ────────────

    def extract_patches(self, images: Tensor) -> Tensor:
        """
        Input  : [B, 3, 32, 32]
        Output : [B, 16, 192]
        """
        B, C, H, W = images.shape
        ps = self.patch_size
        patches = images.unfold(2, ps, ps).unfold(3, ps, ps)
        n_h, n_w = patches.shape[2], patches.shape[3]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(B, n_h * n_w, C * ps * ps)    # [B, 16, 192]

    # ── Encoding API for JEPA ─────────────────────────────────────────────────

    def encode_context(self, patches: Tensor, ctx_idx: Tensor) -> Tensor:
        """
        Encode only the context (visible) patches.
        The transformer attends only across context tokens — no information
        leakage from target patches (strict I-JEPA context encoding).

        Args:
            patches : [B, 16, 192]  — all patches (only ctx_idx positions used)
            ctx_idx : LongTensor[n_ctx] of patch indices

        Returns:
            Tensor[B, n_ctx, 128]
        """
        ctx_patches = patches[:, ctx_idx, :]               # [B, n_ctx, 192]
        ctx_pos     = self.pos_embed[ctx_idx]              # [n_ctx, 128]
        x = self.patch_proj(ctx_patches) + ctx_pos.unsqueeze(0)   # [B, n_ctx, 128]
        x = self.blocks(x)
        x = self.norm(x)
        return x                                           # [B, n_ctx, 128]

    def encode_all(self, patches: Tensor) -> Tensor:
        """
        Encode all 16 patches (used by the EMA target encoder).

        Args:
            patches : [B, 16, 192]

        Returns:
            Tensor[B, 16, 128]
        """
        x = self.patch_proj(patches) + self.pos_embed.unsqueeze(0)  # [B, 16, 128]
        x = self.blocks(x)
        x = self.norm(x)
        return x                                           # [B, 16, 128]

    # ── Supervised forward (mirrors CorticalNetwork.forward) ─────────────────

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
            logits  : Tensor[B, 10]       — classification head
            latents : Tensor[B, 16, 128]  — per-patch representations
        """
        patches = self.extract_patches(images)             # [B, 16, 192]
        latents = self.encode_all(patches)                 # [B, 16, 128]
        pooled  = latents.mean(dim=1)                      # [B, 128]
        logits  = self.classifier(pooled)                  # [B, 10]
        return logits, latents


# ─────────────────────────────────────────────────────────────────────────────
# EMA wrapper for ViT target encoder
# ─────────────────────────────────────────────────────────────────────────────

class ViTEMATargetEncoder(nn.Module):
    """
    EMA copy of ViTEncoder for use as the I-JEPA target encoder.
    API mirrors cortical_column.jepa.EMATargetEncoder.
    """

    import copy as _copy

    def __init__(self, encoder: "ViTEncoder", momentum: float):
        super().__init__()
        import copy
        self.momentum = momentum
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: "ViTEncoder", new_momentum: float | None = None) -> None:
        if new_momentum is not None:
            self.momentum = new_momentum
        tau = self.momentum
        for p_t, p_o in zip(self.target.parameters(), online.parameters()):
            p_t.data = tau * p_t.data + (1.0 - tau) * p_o.data

    @torch.no_grad()
    def encode_patches(self, patches: Tensor, indices: Tensor) -> Tensor:
        """
        Run all patches through target encoder, extract target positions.

        Args:
            patches : [B, 16, 192]
            indices : LongTensor[n_target]

        Returns:
            Tensor[B, n_target, 128]
        """
        all_latents = self.target.encode_all(patches)      # [B, 16, 128]
        return all_latents[:, indices, :]                  # [B, n_target, 128]
