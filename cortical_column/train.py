"""
Training CorticalNetwork on MNIST.
Device priority: CUDA > MPS > CPU.
CUDA optimizations: bf16 autocast when available, cudnn.benchmark,
pin_memory + non_blocking transfers, parallel DataLoader workers.
"""

import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from pathlib import Path
from torch.utils.data import DataLoader

from cortical_column.config import (
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    TRAIN_SUBSET,
    EPOCHS,
    LATENT_DIM,
    LR,
    N_CLASSES,
    N_PATCHES,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    PATCH_SIZE,
)
from cortical_column.cortical_network import CorticalNetwork


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(
    batch_size: int,
    device: torch.device,
    val_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Download MNIST, return (train_loader, val_loader).
    Normalization: mean=0.1307, std=0.3081.
    val_batch_size defaults to 2× batch_size (no gradients → fits more in VRAM).
    pin_memory, num_workers, and prefetch_factor are enabled for CUDA only.
    """
    if val_batch_size is None:
        val_batch_size = batch_size * 2

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to load MNIST. Install torchvision to run training."
        ) from exc

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    val_dataset   = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    if TRAIN_SUBSET is not None:
        indices = torch.randperm(len(train_dataset))[:TRAIN_SUBSET]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    use_cuda = device.type == "cuda"
    nw = NUM_WORKERS if use_cuda else 0
    cuda_kwargs = dict(
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and nw > 0,
        prefetch_factor=PREFETCH_FACTOR if (use_cuda and nw > 0) else None,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,     shuffle=True,  **cuda_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=val_batch_size, shuffle=False, **cuda_kwargs)
    return train_loader, val_loader


def train() -> None:
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        free, total = torch.cuda.mem_get_info()
        print(f"GPU RAM: {(total - free) / 1e9:.1f} / {total / 1e9:.1f} GB free")

    model = CorticalNetwork(
        n_columns=N_PATCHES,
        patch_size=PATCH_SIZE,
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    # torch.compile is disabled by default because it can drop the _sparse_cache
    # side effect used by l4_sparsity() and has been observed to destabilize this
    # model under mixed precision. Re-enable only after replacing that cache with
    # an explicit returned statistic and re-validating CUDA training.
    # model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    train_size = TRAIN_SUBSET if TRAIN_SUBSET is not None else 60_000
    print(f"Train batch: {BATCH_SIZE}  |  Val batch: {VAL_BATCH_SIZE}  |  Workers: {NUM_WORKERS}  |  Train samples: {train_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    # Prefer bf16 on CUDA when available. float16 is numerically fragile for this
    # model, so we fall back to plain fp32 unless bf16 is supported.
    use_bf16_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def maybe_autocast():
        return (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16_amp
            else nullcontext()
        )

    train_loader, val_loader = get_dataloaders(BATCH_SIZE, device, VAL_BATCH_SIZE)

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            # non_blocking=True overlaps host→device transfer with GPU compute
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            with maybe_autocast():
                logits, _ = model(images)
                loss = loss_fn(logits, labels)

            if not torch.isfinite(loss) and use_bf16_amp:
                print("  [warn] non-finite bf16 loss, retrying batch in fp32")
                with nullcontext():
                    logits, _ = model(images)
                    loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss={loss.item():.4f}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        if train_total == 0:
            raise RuntimeError("No finite training batches were processed; check numerical stability.")

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        mean_active_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with maybe_autocast():
                    logits, _ = model(images)
                    loss = loss_fn(logits, labels)

                if not torch.isfinite(loss) and use_bf16_amp:
                    with nullcontext():
                        logits, _ = model(images)
                        loss = loss_fn(logits, labels)

                if not torch.isfinite(loss):
                    print("  [warn] non-finite validation loss, skipping batch")
                    continue

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                # K-WTA sparsity from L4 pre-projection cache (~0.25 = k/hidden_dim = 4/16)
                mean_active_sum += model.l4_sparsity()
                n_val_batches += 1

        if val_total == 0 or n_val_batches == 0:
            raise RuntimeError("No finite validation batches were processed; check numerical stability.")

        val_loss /= val_total
        val_acc = val_correct / val_total
        mean_active = mean_active_sum / n_val_batches

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"mean_active={mean_active:.3f}"
        )

    # ---- Save model checkpoint ----
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint_path = repo_root / "cortical_network_mnist.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # ---- Save test latents for UMAP visualization ----
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            with maybe_autocast():
                _, latents = model(images)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latents = np.concatenate(all_latents, axis=0)  # [N_test, 16, 128]
    all_labels  = np.concatenate(all_labels,  axis=0)  # [N_test]

    np.save(repo_root / "test_latents.npy", all_latents)
    np.save(repo_root / "test_labels.npy",  all_labels)
    print(f"Test latents saved: {all_latents.shape}, labels: {all_labels.shape}")


if __name__ == "__main__":
    train()
