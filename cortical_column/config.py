"""Central hyperparameter configuration for cortical column network."""

# Patch and grid configuration
PATCH_SIZE     = 7       # pixels, square patch
N_PATCHES      = 16      # 4×4 grid on 28×28 image
N_MINICOLUMNS  = 16      # mini-columns per cortical column
LATENT_DIM     = 128     # latent vector dimension (L5 output)
SPARSITY_K     = 4       # K-WTA: top-k activations kept per mini-column
MINICOLUMN_HIDDEN_DIM = 16  # internal dim per mini-column; must be > SPARSITY_K
N_CLASSES      = 10      # MNIST

# Internal layer dimensions
L4_DIM   = 64
L23_DIM  = 128
L5_DIM   = 128   # == LATENT_DIM
L6_DIM   = 64
L1_DIM   = 32

# Training hyperparameters (supervised MNIST)
BATCH_SIZE      = 512   # large batch fully utilises GPU VRAM; scale down if OOM
VAL_BATCH_SIZE  = 1024  # no gradients during val → fits 2× in same memory
TRAIN_SUBSET    = 10000  # cap training set size for fast iteration; None = full 60 000
LR              = 1e-3
EPOCHS          = 20
DEVICE          = "cuda"  # auto-detected at runtime: cuda > mps > cpu
NUM_WORKERS     = 8       # parallel DataLoader workers; set to 0 for MPS/CPU
PREFETCH_FACTOR = 4       # batches queued ahead per worker (CUDA only)

# ── JEPA hyperparameters ────────────────────────────────────────────────────
# Self-supervised pre-training using Joint Embedding Predictive Architecture.
# Context columns → CorticalPredictor → predicted target latents
# Target columns  → EMA encoder (stop-grad) → actual target latents
# Loss: cosine similarity in latent space

# Masking (BlockMaskingStrategy)
JEPA_GRID_SIZE        = 4       # sqrt(N_PATCHES): 4×4 spatial grid
JEPA_TARGET_SCALE_MIN = 0.15    # min fraction of patches masked as target
JEPA_TARGET_SCALE_MAX = 0.40    # max fraction of patches masked as target
JEPA_N_TARGET_BLOCKS  = 4       # number of independent block masks per sample
JEPA_MIN_CONTEXT      = 6       # always keep ≥ 6 context columns

# CorticalPredictor (transformer)
JEPA_PREDICTOR_DEPTH  = 4       # transformer blocks
JEPA_PREDICTOR_HEADS  = 4       # attention heads (must divide LATENT_DIM=128)
JEPA_PREDICTOR_DROPOUT= 0.0     # dropout inside predictor (0 = disabled)

# Target encoder (EMA)
JEPA_EMA_START        = 0.996   # initial EMA momentum τ
JEPA_EMA_END          = 1.000   # final EMA momentum (cosine schedule toward 1)

# JEPA optimisation
JEPA_LR               = 1.5e-4  # peak learning rate (AdamW)
JEPA_WEIGHT_DECAY     = 0.05
JEPA_BATCH_SIZE       = 256
JEPA_EPOCHS           = 50
JEPA_WARMUP_EPOCHS    = 10      # linear LR warmup duration
JEPA_TRAIN_SUBSET     = None    # cap training samples; None = full 60,000
