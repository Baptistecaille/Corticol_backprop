"""JEPA self-supervised training on CIFAR-10 for both CorticalNetwork and ViT.

Both models share:
  - Identical CorticalPredictor  (same architecture, same hyperparams)
  - Identical BlockMaskingStrategy
  - Identical jepa_loss (cosine similarity)
  - Identical EMA schedule
  - Identical dataset, batch size, epochs, LR

Only the encoder architecture differs:
  - CorticalNetwork : 16 independent cortical columns (no cross-patch attention)
  - ViTEncoder      : 16 patches with full self-attention across context tokens

Usage
─────
    python -m cortical_column.cifar_train_jepa
    python -m cortical_column.cifar_train_jepa --model cortical --ckpt cortical_cifar10_supervised.pt
    python -m cortical_column.cifar_train_jepa --model vit      --ckpt vit_cifar10_supervised.pt
    python -m cortical_column.cifar_train_jepa --model both

Output:
    cortical_cifar10_jepa.pt   (keys: context_encoder, target_encoder, predictor)
    vit_cifar10_jepa.pt        (keys: context_encoder, target_encoder, predictor)
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.config import LATENT_DIM, N_PATCHES, NUM_WORKERS
from cortical_column.config_cifar import (
    CIFAR_MEAN, CIFAR_STD,
    CIFAR_N_CHANNELS, CIFAR_PATCH_SIZE, CIFAR_N_CLASSES,
    CIFAR_JEPA_LR, CIFAR_JEPA_WEIGHT_DECAY,
    CIFAR_JEPA_BATCH_SIZE, CIFAR_JEPA_EPOCHS, CIFAR_JEPA_WARMUP_EPOCHS,
    CIFAR_JEPA_TRAIN_SUBSET,
)
from cortical_column.config import (
    JEPA_EMA_START, JEPA_EMA_END,
    JEPA_PREDICTOR_DEPTH, JEPA_PREDICTOR_HEADS, JEPA_PREDICTOR_DROPOUT,
)
from cortical_column.cortical_network import CorticalNetwork
from cortical_column.baselines.vit_encoder import ViTEncoder, ViTEMATargetEncoder
from cortical_column.jepa import (
    BlockMaskingStrategy, CorticalPredictor,
    EMATargetEncoder, cosine_ema_schedule, jepa_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def get_cifar10_loader(batch_size: int, train: bool, subset: int | None = None) -> DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf)
    if subset and train:
        idx = torch.randperm(len(dataset))[:subset]
        from torch.utils.data import Subset
        dataset = Subset(dataset, idx)
    n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=n_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


# ─────────────────────────────────────────────────────────────────────────────
# Cortical column JEPA step
# ─────────────────────────────────────────────────────────────────────────────

def cortical_jepa_step(
    encoder: CorticalNetwork,
    target_enc: EMATargetEncoder,
    predictor: CorticalPredictor,
    patches: torch.Tensor,
    ctx_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
) -> torch.Tensor:
    """One forward pass for cortical column JEPA."""
    ctx_latents = torch.stack(
        [encoder.columns[i](patches[:, i, :], top_down_signal=None)[0]
         for i in ctx_idx.tolist()],
        dim=1,
    )                                                       # [B, n_ctx, 128]
    tgt_latents = target_enc.encode_patches(patches, tgt_idx)   # [B, n_tgt, 128]
    predicted   = predictor(ctx_latents, ctx_idx, tgt_idx)      # [B, n_tgt, 128]
    return jepa_loss(predicted, tgt_latents)


# ─────────────────────────────────────────────────────────────────────────────
# ViT JEPA step
# ─────────────────────────────────────────────────────────────────────────────

def vit_jepa_step(
    encoder: ViTEncoder,
    target_enc: ViTEMATargetEncoder,
    predictor: CorticalPredictor,
    patches: torch.Tensor,
    ctx_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
) -> torch.Tensor:
    """One forward pass for ViT JEPA."""
    ctx_latents = encoder.encode_context(patches, ctx_idx)      # [B, n_ctx, 128]
    tgt_latents = target_enc.encode_patches(patches, tgt_idx)   # [B, n_tgt, 128]
    predicted   = predictor(ctx_latents, ctx_idx, tgt_idx)      # [B, n_tgt, 128]
    return jepa_loss(predicted, tgt_latents)


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_jepa_model(
    encoder: nn.Module,
    target_enc: nn.Module,
    predictor: CorticalPredictor,
    masker: BlockMaskingStrategy,
    step_fn,           # cortical_jepa_step or vit_jepa_step
    name: str,
    output_ckpt: str,
    device: torch.device,
    pretrained_ckpt: str | None = None,
):
    encoder = encoder.to(device)
    target_enc = target_enc.to(device)
    predictor = predictor.to(device)

    if pretrained_ckpt and os.path.isfile(pretrained_ckpt):
        state = torch.load(pretrained_ckpt, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
        print(f"  Warm-start  : {pretrained_ckpt}")
    else:
        print("  Warm-start  : none (from scratch)")

    train_loader = get_cifar10_loader(
        CIFAR_JEPA_BATCH_SIZE, train=True, subset=CIFAR_JEPA_TRAIN_SUBSET
    )
    val_loader = get_cifar10_loader(CIFAR_JEPA_BATCH_SIZE, train=False)

    steps_per_epoch = len(train_loader)
    total_steps  = CIFAR_JEPA_EPOCHS * steps_per_epoch
    warmup_steps = CIFAR_JEPA_WARMUP_EPOCHS * steps_per_epoch

    params = list(encoder.parameters()) + list(predictor.parameters())
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"  Params      : {n_params:,} (encoder + predictor)")

    optimizer = torch.optim.AdamW(
        params, lr=CIFAR_JEPA_LR, weight_decay=CIFAR_JEPA_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, total_steps, warmup_steps),
    )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, CIFAR_JEPA_EPOCHS + 1):
        encoder.train()
        predictor.train()
        train_loss_sum = 0.0

        for images, _ in train_loader:
            images  = images.to(device)
            patches = encoder.extract_patches(images)

            ctx_idx, tgt_idx = masker.sample(device=device)
            loss = step_fn(encoder, target_enc, predictor, patches, ctx_idx, tgt_idx)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            tau = cosine_ema_schedule(global_step, total_steps, JEPA_EMA_START, JEPA_EMA_END)
            target_enc.update(encoder, new_momentum=tau)

            train_loss_sum += loss.item()
            global_step    += 1

        # Validation
        encoder.eval()
        predictor.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images  = images.to(device)
                patches = encoder.extract_patches(images)
                ctx_idx, tgt_idx = masker.sample(device=device)
                val_loss_sum += step_fn(
                    encoder, target_enc, predictor, patches, ctx_idx, tgt_idx
                ).item()

        avg_train = train_loss_sum / steps_per_epoch
        avg_val   = val_loss_sum   / len(val_loader)
        lr_now    = optimizer.param_groups[0]["lr"]
        tau_now   = target_enc.momentum

        print(
            f"Epoch {epoch:3d}/{CIFAR_JEPA_EPOCHS}"
            f" | train={avg_train:.4f} | val={avg_val:.4f}"
            f" | τ={tau_now:.5f} | lr={lr_now:.2e}",
            end="",
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "context_encoder": encoder.state_dict(),
                    "target_encoder":  target_enc.target.state_dict(),
                    "predictor":       predictor.state_dict(),
                    "val_loss":        avg_val,
                },
                output_ckpt,
            )
            print("  ✓")
        else:
            print()

    print(f"  Best val_loss: {best_val_loss:.4f} → {output_ckpt}")
    return best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(which: str, cortical_ckpt: str | None, vit_ckpt: str | None):
    device = get_device()
    masker = BlockMaskingStrategy()
    results = {}

    if which in ("cortical", "both"):
        print(f"\n{'═'*60}")
        print("JEPA training — CorticalNetwork on CIFAR-10")
        print(f"{'═'*60}")
        encoder    = CorticalNetwork(
            patch_size=CIFAR_PATCH_SIZE, n_channels=CIFAR_N_CHANNELS,
            n_classes=CIFAR_N_CLASSES,
        )
        target_enc = EMATargetEncoder(encoder, momentum=JEPA_EMA_START)
        predictor  = CorticalPredictor(
            depth=JEPA_PREDICTOR_DEPTH, n_heads=JEPA_PREDICTOR_HEADS,
            dropout=JEPA_PREDICTOR_DROPOUT,
        )
        loss = train_jepa_model(
            encoder, target_enc, predictor, masker,
            cortical_jepa_step, "CorticalNetwork",
            "cortical_cifar10_jepa.pt", device,
            pretrained_ckpt=cortical_ckpt,
        )
        results["CorticalNetwork JEPA"] = loss

    if which in ("vit", "both"):
        print(f"\n{'═'*60}")
        print("JEPA training — ViT (I-JEPA baseline) on CIFAR-10")
        print(f"{'═'*60}")
        encoder    = ViTEncoder()
        target_enc = ViTEMATargetEncoder(encoder, momentum=JEPA_EMA_START)
        predictor  = CorticalPredictor(
            depth=JEPA_PREDICTOR_DEPTH, n_heads=JEPA_PREDICTOR_HEADS,
            dropout=JEPA_PREDICTOR_DROPOUT,
        )
        loss = train_jepa_model(
            encoder, target_enc, predictor, masker,
            vit_jepa_step, "ViT-Tiny",
            "vit_cifar10_jepa.pt", device,
            pretrained_ckpt=vit_ckpt,
        )
        results["ViT-Tiny JEPA"] = loss

    print(f"\n{'═'*40}")
    print("CIFAR-10 JEPA results (best val cosine loss):")
    for name, loss in results.items():
        print(f"  {name:<28} {loss:.4f}  (cos_sim ≈ {1-loss:.3f})")
    print(f"{'═'*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cortical", "vit", "both"], default="both")
    parser.add_argument("--ckpt",  type=str, default=None,
                        help="Supervised checkpoint for warm-starting (cortical model).")
    parser.add_argument("--ckpt_vit", type=str, default=None,
                        help="Supervised checkpoint for warm-starting (ViT model).")
    args = parser.parse_args()
    main(args.model, args.ckpt, args.ckpt_vit)
