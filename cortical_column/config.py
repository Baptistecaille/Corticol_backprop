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

# Training hyperparameters
BATCH_SIZE      = 512   # large batch fully utilises GPU VRAM; scale down if OOM
VAL_BATCH_SIZE  = 1024  # no gradients during val → fits 2× in same memory
TRAIN_SUBSET    = 10000  # cap training set size for fast iteration; None = full 60 000
LR              = 1e-3
EPOCHS          = 20
DEVICE          = "cuda"  # auto-detected at runtime: cuda > mps > cpu
NUM_WORKERS     = 8       # parallel DataLoader workers; set to 0 for MPS/CPU
PREFETCH_FACTOR = 4       # batches queued ahead per worker (CUDA only)
