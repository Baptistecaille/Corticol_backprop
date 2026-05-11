"""CIFAR-10 configuration for CorticalNetwork and ViT baseline.

Patch grid: 4×4 = 16 patches of 8×8×3 = 192 dims — same N_PATCHES=16 as MNIST.
This lets both models use the identical CorticalPredictor, BlockMaskingStrategy,
and jepa_loss without modification.
"""

# ── Image / patch geometry ──────────────────────────────────────────────────
CIFAR_IMAGE_SIZE   = 32        # pixels, square
CIFAR_N_CHANNELS   = 3        # RGB
CIFAR_PATCH_SIZE   = 8        # pixels — 4×4 grid → 16 patches
CIFAR_N_PATCHES    = 16       # (32 // 8) ** 2 — same as MNIST config
CIFAR_PATCH_DIM    = CIFAR_PATCH_SIZE ** 2 * CIFAR_N_CHANNELS   # 192
CIFAR_N_CLASSES    = 10

# Normalisation constants (CIFAR-10 channel means / stds)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# ── Shared latent / model dims ───────────────────────────────────────────────
# Keep identical to MNIST config so the same predictor/evaluator code works.
from cortical_column.config import LATENT_DIM, N_PATCHES   # noqa: E402
assert CIFAR_N_PATCHES == N_PATCHES, "Patch count must match for shared predictor."

# ── ViT baseline architecture ────────────────────────────────────────────────
VIT_EMBED_DIM  = LATENT_DIM    # 128 — same as cortical column for direct comparison
VIT_DEPTH      = 6             # transformer blocks
VIT_N_HEADS    = 4             # attention heads (must divide VIT_EMBED_DIM)
VIT_MLP_RATIO  = 4.0          # FFN hidden dim = embed_dim × mlp_ratio
VIT_DROPOUT    = 0.0

# ── Supervised training (CIFAR-10) ───────────────────────────────────────────
CIFAR_BATCH_SIZE      = 256
CIFAR_VAL_BATCH_SIZE  = 512
CIFAR_LR              = 1e-3
CIFAR_EPOCHS          = 50
CIFAR_WEIGHT_DECAY    = 1e-4

# ── JEPA training (CIFAR-10) — same as MNIST JEPA but CIFAR scale ────────────
CIFAR_JEPA_LR            = 1.5e-4
CIFAR_JEPA_WEIGHT_DECAY  = 0.05
CIFAR_JEPA_BATCH_SIZE    = 256
CIFAR_JEPA_EPOCHS        = 100
CIFAR_JEPA_WARMUP_EPOCHS = 15
CIFAR_JEPA_TRAIN_SUBSET  = None   # None = full 50k training set
