# Cortical Backprop

Bio-inspired cortical column network implemented in PyTorch. The model splits an image into non-overlapping patches, processes each patch with an independent cortical column, and aggregates the resulting latents with a mean vote before classification.

The codebase supports two training modes:

- Supervised learning on MNIST and CIFAR-10.
- Self-supervised JEPA-style pretraining with a masked-patch prediction objective.

## What This Repository Contains

- A modular cortical hierarchy with mini-columns and layers L1 through L6.
- K-WTA sparsity inside the mini-columns.
- A patch-wise network with 16 independent columns and no weight sharing.
- MNIST and CIFAR-10 training scripts.
- JEPA pretraining, linear probe evaluation, and occlusion robustness evaluation.
- Unit and smoke tests for the core model pieces.

## Model Summary

For MNIST, the image is split into a 4x4 grid of 7x7 patches. For CIFAR-10, the image is split into a 4x4 grid of 8x8 patches. Each patch is processed by one cortical column, producing a latent vector of size 128. The 16 column latents are then mean-pooled and fed to a linear classifier.

Inside each column:

- L4 receives the flattened patch and passes it through parallel mini-columns.
- L2/3 fuses bottom-up and top-down signals.
- L5 produces the main latent representation.
- L6 provides a gating signal.
- L1 carries the top-down error signal used by the recurrent/feedback path.

The JEPA pipeline uses a context encoder, a predictor, and an EMA target encoder to reconstruct masked column latents from visible context.

## Repository Layout

- `cortical_column/`: core package with models, configs, training, and evaluation scripts.
- `cortical_column/baselines/`: baseline encoders such as the ViT implementation used for CIFAR-10 comparisons.
- `tests/`: smoke and unit tests for the column, layer, network, and training code.
- `data/`: local MNIST and CIFAR-10 datasets.
- `cortical_network_mnist.pt`, `cortical_network_jepa.pt`: example checkpoints saved at the repository root.
- `test_latents.npy`, `test_labels.npy`: cached outputs produced by MNIST supervised training.

## Requirements

- Python 3.10 or newer.
- PyTorch and torchvision.
- NumPy.
- Matplotlib for the linear-probe plots.
- PyTest for the test suite.

A minimal install looks like this:

```bash
pip install torch torchvision numpy matplotlib pytest
```

On Apple Silicon, the scripts automatically prefer MPS when CUDA is not available. On a GPU machine, CUDA is used first.

## Quick Start

Run everything from the repository root.

```bash
python -m cortical_column.train
```

This trains the supervised MNIST model, saves `cortical_network_mnist.pt`, and exports `test_latents.npy` and `test_labels.npy`.

To run JEPA pretraining after supervised warm-start:

```bash
python -m cortical_column.train_jepa --ckpt cortical_network_mnist.pt
```

The default output is `cortical_network_jepa.pt`.

To evaluate the learned representations with a linear probe:

```bash
python -m cortical_column.eval_linear_probe
```

To test how the model behaves under occlusion:

```bash
python -m cortical_column.eval_occlusion
```

## CIFAR-10 Workflows

Supervised training on CIFAR-10:

```bash
python -m cortical_column.cifar_train_supervised --model both
```

JEPA pretraining on CIFAR-10:

```bash
python -m cortical_column.cifar_train_jepa --model both
```

Comparison / evaluation:

```bash
python -m cortical_column.cifar_eval_comparison
```

## Tests

Run the test suite with:

```bash
pytest
```

The smoke test in `tests/test_train_smoke.py` checks that the main MNIST training path runs, the outputs have the expected shapes, and the sparsity statistic stays in range.

## Notes

- The package exposes `CorticalColumn` and `CorticalNetwork` from `cortical_column.__init__`.
- MNIST and CIFAR-10 are downloaded automatically into `data/` if they are not already present.
- The core design intentionally keeps one column per patch, so there is no weight sharing across spatial regions.
- `train.py` caches latent vectors from the validation set so they can be reused for later visualization or analysis.
