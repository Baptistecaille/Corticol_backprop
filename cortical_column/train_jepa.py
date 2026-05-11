"""JEPA self-supervised pre-training for the CorticalNetwork.

Usage
─────
From the project root:

    python -m cortical_column.train_jepa                             # from scratch
    python -m cortical_column.train_jepa --ckpt cortical_network_mnist.pt   # warm-start

The script:
1. Loads (optionally) pre-trained MNIST weights into the context encoder.
2. Builds an EMA target encoder as a frozen slow copy.
3. For each batch, samples a block mask (context / target split).
4. Runs context columns → CorticalPredictor → predicted latents.
5. Runs target columns through the EMA encoder → actual latents (stop-grad).
6. Minimises cosine similarity loss in the latent space.
7. Updates the EMA encoder (not the optimiser).
8. Saves a checkpoint containing context encoder + predictor weights.

The saved checkpoint is plug-and-play: load the context encoder back into
CorticalNetwork and use it as a frozen backbone for any downstream task.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Allow running as `python train_jepa.py` from the cortical_column/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.config import (
    LATENT_DIM,
    N_PATCHES,
    NUM_WORKERS,
    JEPA_LR,
    JEPA_WEIGHT_DECAY,
    JEPA_BATCH_SIZE,
    JEPA_EPOCHS,
    JEPA_WARMUP_EPOCHS,
    JEPA_TRAIN_SUBSET,
    JEPA_PREDICTOR_DEPTH,
    JEPA_PREDICTOR_HEADS,
    JEPA_PREDICTOR_DROPOUT,
    JEPA_EMA_START,
    JEPA_EMA_END,
)
from cortical_column.cortical_network import CorticalNetwork
from cortical_column.jepa import (
    BlockMaskingStrategy,
    CorticalPredictor,
    EMATargetEncoder,
    cosine_ema_schedule,
    jepa_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mnist_loader(
    batch_size: int,
    train: bool,
    num_workers: int,
    subset: int | None = None,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        root="./data", train=train, download=True, transform=transform
    )
    if subset is not None and train:
        indices = torch.randperm(len(dataset))[:subset]
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def lr_lambda_fn(
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_jepa(
    pretrained_ckpt: str | None = None,
    output_ckpt: str = "cortical_network_jepa.pt",
) -> tuple[CorticalNetwork, CorticalPredictor]:
    """
    Run JEPA self-supervised pre-training.

    Args:
        pretrained_ckpt : path to a supervised MNIST checkpoint to warm-start
                          the context encoder (highly recommended).
        output_ckpt     : where to save the final JEPA checkpoint.

    Returns:
        (context_encoder, predictor) — both in eval mode, on CPU.
    """
    device = get_device()
    print(f"Device      : {device}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    # JEPA is fully self-supervised → use full MNIST (no label subset)
    n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
    train_loader = get_mnist_loader(JEPA_BATCH_SIZE, train=True,  num_workers=n_workers, subset=JEPA_TRAIN_SUBSET)
    val_loader   = get_mnist_loader(JEPA_BATCH_SIZE, train=False, num_workers=n_workers)
    steps_per_epoch = len(train_loader)
    total_steps     = JEPA_EPOCHS * steps_per_epoch
    warmup_steps    = JEPA_WARMUP_EPOCHS * steps_per_epoch

    train_size = JEPA_TRAIN_SUBSET if JEPA_TRAIN_SUBSET is not None else 60_000
    print(f"Train size  : {train_size:,}{' (subset)' if JEPA_TRAIN_SUBSET is not None else ''}")
    print(f"Batch size  : {JEPA_BATCH_SIZE} | Steps/epoch: {steps_per_epoch}")
    print(f"Epochs      : {JEPA_EPOCHS} | Total steps: {total_steps:,}")
    print(f"LR          : {JEPA_LR} | Warmup: {JEPA_WARMUP_EPOCHS} epochs")
    print(f"EMA τ       : {JEPA_EMA_START} → {JEPA_EMA_END}")

    # ── Models ────────────────────────────────────────────────────────────────
    context_encoder = CorticalNetwork().to(device)

    if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
        state = torch.load(pretrained_ckpt, map_location=device, weights_only=True)
        context_encoder.load_state_dict(state)
        print(f"Loaded      : {pretrained_ckpt}")
    else:
        print("Warm-start  : none (training from scratch)")

    target_encoder = EMATargetEncoder(context_encoder, momentum=JEPA_EMA_START).to(device)

    predictor = CorticalPredictor(
        latent_dim=LATENT_DIM,
        n_patches=N_PATCHES,
        depth=JEPA_PREDICTOR_DEPTH,
        n_heads=JEPA_PREDICTOR_HEADS,
        dropout=JEPA_PREDICTOR_DROPOUT,
    ).to(device)

    masker = BlockMaskingStrategy()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Only context encoder + predictor are updated.
    # The classifier head in the context encoder is left in; it won't receive
    # gradients from the JEPA loss (not part of the computation graph).
    params = list(context_encoder.parameters()) + list(predictor.parameters())
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"Parameters  : {n_params:,} (encoder + predictor, trainable)")

    optimizer = torch.optim.AdamW(params, lr=JEPA_LR, weight_decay=JEPA_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda_fn(step, total_steps, warmup_steps),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, JEPA_EPOCHS + 1):
        context_encoder.train()
        predictor.train()

        train_loss_sum = 0.0
        n_context_mean = 0.0
        n_target_mean  = 0.0

        for images, _ in train_loader:
            images = images.to(device)           # [B, 1, 28, 28]
            patches = context_encoder.extract_patches(images)  # [B, 16, 49]

            # ── Masking ───────────────────────────────────────────────────────
            ctx_idx, tgt_idx = masker.sample(device=device)
            n_ctx = ctx_idx.shape[0]
            n_tgt = tgt_idx.shape[0]

            # ── Context encoder (gradient flows) ──────────────────────────────
            ctx_latents = []
            for i in ctx_idx.tolist():
                lat, _ = context_encoder.columns[i](
                    patches[:, i, :], top_down_signal=None
                )
                ctx_latents.append(lat)
            ctx_latents = torch.stack(ctx_latents, dim=1)    # [B, n_ctx, 128]

            # ── Target encoder (EMA, stop-grad) ───────────────────────────────
            tgt_latents = target_encoder.encode_patches(patches, tgt_idx)  # [B, n_tgt, 128]

            # ── Predictor ─────────────────────────────────────────────────────
            predicted = predictor(ctx_latents, ctx_idx, tgt_idx)  # [B, n_tgt, 128]

            # ── Loss ──────────────────────────────────────────────────────────
            loss = jepa_loss(predicted, tgt_latents)

            # ── Backprop ──────────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # ── EMA update (cosine schedule τ) ────────────────────────────────
            new_tau = cosine_ema_schedule(global_step, total_steps, JEPA_EMA_START, JEPA_EMA_END)
            target_encoder.update(context_encoder, new_momentum=new_tau)

            train_loss_sum += loss.item()
            n_context_mean += n_ctx
            n_target_mean  += n_tgt
            global_step    += 1

        # ── Validation (no labels needed) ─────────────────────────────────────
        context_encoder.eval()
        predictor.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                patches = context_encoder.extract_patches(images)
                ctx_idx, tgt_idx = masker.sample(device=device)

                ctx_latents = []
                for i in ctx_idx.tolist():
                    lat, _ = context_encoder.columns[i](
                        patches[:, i, :], top_down_signal=None
                    )
                    ctx_latents.append(lat)
                ctx_latents = torch.stack(ctx_latents, dim=1)

                tgt_latents = target_encoder.encode_patches(patches, tgt_idx)
                predicted   = predictor(ctx_latents, ctx_idx, tgt_idx)
                val_loss_sum += jepa_loss(predicted, tgt_latents).item()

        avg_train = train_loss_sum / steps_per_epoch
        avg_val   = val_loss_sum   / len(val_loader)
        lr_now    = optimizer.param_groups[0]["lr"]
        tau_now   = target_encoder.momentum
        n_ctx_avg = n_context_mean / steps_per_epoch
        n_tgt_avg = n_target_mean  / steps_per_epoch

        print(
            f"Epoch {epoch:3d}/{JEPA_EPOCHS}"
            f" | train_loss={avg_train:.4f}"
            f" | val_loss={avg_val:.4f}"
            f" | ctx={n_ctx_avg:.1f} tgt={n_tgt_avg:.1f}"
            f" | τ={tau_now:.5f}"
            f" | lr={lr_now:.2e}"
        )

        # ── Checkpoint (best val loss) ─────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "context_encoder": context_encoder.state_dict(),
                    "target_encoder":  target_encoder.target.state_dict(),
                    "predictor":       predictor.state_dict(),
                    "val_loss":        avg_val,
                },
                output_ckpt,
            )
            print(f"  ✓ checkpoint saved (val_loss={avg_val:.4f})")

    print(f"\nBest val_loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {output_ckpt}")

    context_encoder.eval().cpu()
    predictor.eval().cpu()
    return context_encoder, predictor


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA pre-training for CorticalNetwork")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a pretrained supervised checkpoint (warm-start). "
             "E.g. --ckpt cortical_network_mnist.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cortical_network_jepa.pt",
        help="Output checkpoint path.",
    )
    args = parser.parse_args()
    train_jepa(pretrained_ckpt=args.ckpt, output_ckpt=args.output)
