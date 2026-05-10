"""
Training CorticalNetwork on MNIST.
Device priority: CUDA > MPS > CPU.
CUDA optimizations: AMP (autocast + GradScaler), cudnn.benchmark,
pin_memory + non_blocking transfers, parallel DataLoader workers.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cortical_column.config import (
    BATCH_SIZE,
    EPOCHS,
    LATENT_DIM,
    LR,
    N_CLASSES,
    N_PATCHES,
    NUM_WORKERS,
    PATCH_SIZE,
)
from cortical_column.cortical_network import CorticalNetwork


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """
    Download MNIST, return (train_loader, val_loader).
    Normalization: mean=0.1307, std=0.3081 (standard MNIST values).
    pin_memory and num_workers are enabled only for CUDA for async host→device transfers.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    val_dataset   = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    use_cuda = device.type == "cuda"
    kwargs = dict(
        num_workers=NUM_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and NUM_WORKERS > 0,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader


def train() -> None:
    device = get_device()
    print(f"Using device: {device}")

    # cudnn.benchmark lets cuDNN auto-tune convolution algorithms for fixed input sizes.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = CorticalNetwork(
        n_columns=N_PATCHES,
        patch_size=PATCH_SIZE,
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    # Optional: uncomment for PyTorch 2.0+ graph compilation (significant speedup on CUDA)
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    # AMP: GradScaler prevents underflow with float16; no-op on CPU/MPS.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_loader, val_loader = get_dataloaders(BATCH_SIZE, device)

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

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss = loss_fn(logits, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

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

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(images)
                    loss = loss_fn(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                # K-WTA sparsity from L4 pre-projection cache (~0.25 = k/hidden_dim = 4/16)
                mean_active_sum += model.l4_sparsity()
                n_val_batches += 1

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
            with torch.amp.autocast("cuda", enabled=use_amp):
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
