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
BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 20
DEVICE     = "mps"   # fallback "cpu" if MPS unavailable
