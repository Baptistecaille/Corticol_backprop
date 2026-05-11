"""JEPA components for self-supervised cortical column pre-training.

Architecture (Joint Embedding Predictive Architecture, inspired by I-JEPA):

    Image [B, 1, 28, 28]
        ↓  extract_patches
    patches [B, 16, 49]
        ↓  BlockMaskingStrategy → context_idx, target_idx
        ├── Context columns (gradient flows)
        │       CorticalNetwork.columns[ctx_idx] → ctx_latents [B, n_ctx, 128]
        │                                           ↓
        │                               CorticalPredictor
        │                                           ↓
        │                           predicted_latents [B, n_tgt, 128]
        │                                           ↓  cosine loss ↑
        └── Target columns  (EMA encoder, stop-grad)
                EMATargetEncoder.columns[tgt_idx]  → tgt_latents [B, n_tgt, 128]

Key design choices
──────────────────
* Block masking  : spatially contiguous target regions force the predictor to
                   use lateral context across columns, mimicking long-range
                   horizontal connections in the cortex.
* EMA encoder    : prevents representational collapse by decoupling the target
                   latents from the online gradient update.  Analogous to slow
                   Hebbian consolidation (Complementary Learning Systems).
* Cosine loss    : scale-invariant, matches how cortical columns likely encode
                   magnitude (rate) separately from direction (pattern).
* Mask token     : the predictor receives a learnable mask token + positional
                   embedding at each target position — it must infer content
                   from context, not from the target patch itself.
"""

import copy
import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cortical_column.config import (
    LATENT_DIM,
    N_PATCHES,
    JEPA_GRID_SIZE,
    JEPA_TARGET_SCALE_MIN,
    JEPA_TARGET_SCALE_MAX,
    JEPA_N_TARGET_BLOCKS,
    JEPA_MIN_CONTEXT,
    JEPA_PREDICTOR_DEPTH,
    JEPA_PREDICTOR_HEADS,
    JEPA_PREDICTOR_DROPOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Positional embeddings
# ─────────────────────────────────────────────────────────────────────────────

class PatchPositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for the N_PATCHES patch locations.

    Each of the 16 positions in the 4×4 cortical map gets its own embedding
    vector, letting the predictor reason about spatial relationships between
    columns.

    Args:
        n_patches : total patches (16)
        embed_dim : embedding dimension (must equal LATENT_DIM for residual add)
    """

    def __init__(self, n_patches: int = N_PATCHES, embed_dim: int = LATENT_DIM):
        super().__init__()
        self.embedding = nn.Embedding(n_patches, embed_dim)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices : LongTensor[n] of patch indices in [0, N_PATCHES)
        Returns:
            Tensor[n, embed_dim]
        """
        return self.embedding(indices)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CorticalPredictor
# ─────────────────────────────────────────────────────────────────────────────

class CorticalPredictor(nn.Module):
    """
    Predicts target column latents from context column latents.

    Biologically, this models long-range horizontal axonal connections between
    cortical columns: columns that receive visual input "vote" to reconstruct
    what a masked, nearby column would have computed.

    Architecture
    ────────────
    Tokens fed to the Transformer:
        [ctx_latent_0 + pos_0 | ... | ctx_latent_{n-1} + pos_{n-1}
         | mask_token + pos_tgt_0 | ... | mask_token + pos_tgt_{m-1}]

    The transformer uses full self-attention (all tokens see each other).
    Predicted latents are read from the last m output positions.
    A thin Linear projection head refines the output.

    Args:
        latent_dim : column latent dimension (128)
        n_patches  : total patches (16)
        depth      : number of Pre-LN Transformer blocks
        n_heads    : attention heads (must divide latent_dim)
        mlp_ratio  : hidden-dim multiplier inside the FFN
        dropout    : dropout rate (set 0 for deterministic inference)
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        n_patches: int = N_PATCHES,
        depth: int = JEPA_PREDICTOR_DEPTH,
        n_heads: int = JEPA_PREDICTOR_HEADS,
        mlp_ratio: float = 4.0,
        dropout: float = JEPA_PREDICTOR_DROPOUT,
    ):
        super().__init__()
        assert latent_dim % n_heads == 0, (
            f"latent_dim={latent_dim} must be divisible by n_heads={n_heads}"
        )
        self.latent_dim = latent_dim
        self.n_patches = n_patches

        # Positional embeddings shared across context and target
        self.pos_emb = PatchPositionalEmbedding(n_patches, latent_dim)

        # Learnable mask token — replaces unseen target latents
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Pre-LayerNorm Transformer encoder (batch_first for [B, seq, dim])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=int(latent_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable than Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(latent_dim),
        )

        # Thin projection head (predictor head, not shared with encoder)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(
        self,
        context_latents: Tensor,   # [B, n_context, latent_dim]
        context_indices: Tensor,   # [n_context]  LongTensor of patch positions
        target_indices: Tensor,    # [n_target]   LongTensor of patch positions
    ) -> Tensor:
        """
        Returns:
            predicted_latents : Tensor[B, n_target, latent_dim]
        """
        B = context_latents.shape[0]
        n_context = context_latents.shape[1]
        n_target = target_indices.shape[0]

        # Context tokens: latent + positional embedding
        ctx_pos = self.pos_emb(context_indices)           # [n_context, D]
        ctx_tokens = context_latents + ctx_pos.unsqueeze(0)  # [B, n_context, D]

        # Target tokens: mask token + positional embedding
        tgt_pos = self.pos_emb(target_indices)            # [n_target, D]
        mask = self.mask_token.expand(B, n_target, -1)    # [B, n_target, D]
        tgt_tokens = mask + tgt_pos.unsqueeze(0)          # [B, n_target, D]

        # Full token sequence: context first, then target queries
        tokens = torch.cat([ctx_tokens, tgt_tokens], dim=1)  # [B, n_context+n_target, D]

        # Transformer forward
        out = self.transformer(tokens)                    # [B, n_context+n_target, D]

        # Extract and project predictions at target positions
        pred = out[:, n_context:, :]                      # [B, n_target, D]
        pred = self.head(pred)
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss
# ─────────────────────────────────────────────────────────────────────────────

def jepa_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """
    Cosine similarity loss in the latent space.

    loss = mean(1 - cos_sim(predicted, target))
          ∈ [0, 2],  optimal = 0

    Scale-invariant: a column that correctly predicts the *direction* of
    the target representation (not just its magnitude) gets zero loss.
    This is consistent with how L5 pyramidal neurons likely encode
    feature identity separately from firing rate.

    Args:
        predicted : Tensor[B, n_target, latent_dim]
        target    : Tensor[B, n_target, latent_dim]  (stop-gradient)
    Returns:
        scalar loss
    """
    pred_n = F.normalize(predicted, dim=-1)   # [B, n_target, D]
    tgt_n  = F.normalize(target,    dim=-1)   # [B, n_target, D]
    cos    = (pred_n * tgt_n).sum(dim=-1)     # [B, n_target]
    return (1.0 - cos).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Masking strategy
# ─────────────────────────────────────────────────────────────────────────────

class BlockMaskingStrategy:
    """
    Generates context / target patch splits using spatial block masking.

    The 16 patches form a 4×4 cortical map. JEPA_N_TARGET_BLOCKS independent
    rectangular blocks are sampled and unioned as the target region.  This
    forces the predictor to reason across spatially contiguous gaps, similar
    to how cortical columns in V1 complete occluded contours via horizontal
    connections.

    Target fraction is sampled uniformly in [JEPA_TARGET_SCALE_MIN,
    JEPA_TARGET_SCALE_MAX] each call.  At least JEPA_MIN_CONTEXT context
    patches are always preserved.

    Args:
        n_patches        : total patches (16)
        grid_size        : side length of the patch grid (4)
        target_scale     : (min, max) fraction of patches to mask as target
        n_target_blocks  : number of independent block masks per sample
        min_context      : minimum context patches to preserve
    """

    def __init__(
        self,
        n_patches: int = N_PATCHES,
        grid_size: int = JEPA_GRID_SIZE,
        target_scale: Tuple[float, float] = (JEPA_TARGET_SCALE_MIN, JEPA_TARGET_SCALE_MAX),
        n_target_blocks: int = JEPA_N_TARGET_BLOCKS,
        min_context: int = JEPA_MIN_CONTEXT,
    ):
        self.n_patches = n_patches
        self.grid_size = grid_size
        self.target_scale = target_scale
        self.n_target_blocks = n_target_blocks
        self.min_context = min_context

    def _sample_block(self, target_frac: float) -> set:
        """Sample one rectangular block; its area ≤ target_frac × n_patches."""
        max_area = max(1, int(self.n_patches * target_frac))
        h = random.randint(1, self.grid_size)
        max_w = max(1, max_area // h)
        w = random.randint(1, min(max_w, self.grid_size))
        top  = random.randint(0, self.grid_size - h)
        left = random.randint(0, self.grid_size - w)
        return {
            (top + dr) * self.grid_size + (left + dc)
            for dr in range(h)
            for dc in range(w)
        }

    def sample(self, device=None) -> Tuple[Tensor, Tensor]:
        """
        Sample one context / target split for a batch.

        Returns:
            context_indices : LongTensor[n_context]
            target_indices  : LongTensor[n_target]
        Both sorted in ascending order.
        """
        target_frac = random.uniform(*self.target_scale)

        target_set: set = set()
        for _ in range(self.n_target_blocks):
            target_set |= self._sample_block(target_frac)

        all_patches = set(range(self.n_patches))
        context_set = all_patches - target_set

        # Enforce minimum context
        if len(context_set) < self.min_context:
            excess = list(target_set)
            random.shuffle(excess)
            for idx in excess[: self.min_context - len(context_set)]:
                context_set.add(idx)
                target_set.discard(idx)

        # Edge case: no targets
        if not target_set:
            victim = random.choice(list(context_set))
            context_set.discard(victim)
            target_set.add(victim)

        ctx = torch.tensor(sorted(context_set), dtype=torch.long)
        tgt = torch.tensor(sorted(target_set),  dtype=torch.long)
        if device is not None:
            ctx, tgt = ctx.to(device), tgt.to(device)
        return ctx, tgt


# ─────────────────────────────────────────────────────────────────────────────
# 5. EMA target encoder
# ─────────────────────────────────────────────────────────────────────────────

class EMATargetEncoder(nn.Module):
    """
    Exponential Moving Average copy of the context encoder.

    Prevents representational collapse by providing targets that change
    slower than the online encoder.  Parameters are updated after each
    optimiser step (not during backprop):

        θ_target ← τ · θ_target + (1 − τ) · θ_online

    τ is optionally scheduled from JEPA_EMA_START → JEPA_EMA_END via a
    cosine schedule (mimicking slow biological consolidation gradually
    approaching a fixed memory).

    No gradients ever flow through the target encoder.

    Args:
        encoder  : the online context encoder (CorticalNetwork)
        momentum : initial τ (JEPA_EMA_START, e.g. 0.996)
    """

    def __init__(self, encoder: nn.Module, momentum: float):
        super().__init__()
        self.momentum = momentum
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module, new_momentum: float | None = None) -> None:
        """Update target weights and optionally set a new momentum value."""
        if new_momentum is not None:
            self.momentum = new_momentum
        tau = self.momentum
        for p_t, p_o in zip(self.target.parameters(), online.parameters()):
            p_t.data = tau * p_t.data + (1.0 - tau) * p_o.data

    @torch.no_grad()
    def encode_patches(
        self,
        patches: Tensor,    # [B, N_PATCHES, 49]
        indices: Tensor,    # [n_selected]
    ) -> Tensor:
        """
        Run selected patches through the frozen target encoder columns.

        Returns:
            Tensor[B, n_selected, latent_dim]
        """
        latents = []
        for i in indices.tolist():
            lat, _ = self.target.columns[i](patches[:, i, :], top_down_signal=None)
            latents.append(lat)
        return torch.stack(latents, dim=1)   # [B, n_selected, latent_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMA momentum schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_ema_schedule(
    step: int,
    total_steps: int,
    tau_start: float,
    tau_end: float,
) -> float:
    """
    Cosine schedule from tau_start → tau_end over total_steps.

    Following I-JEPA: momentum gradually approaches 1 so target updates
    slow down as training converges, stabilising the representation.
    """
    progress = step / max(1, total_steps)
    return tau_end - (tau_end - tau_start) * (math.cos(math.pi * progress) + 1) / 2
