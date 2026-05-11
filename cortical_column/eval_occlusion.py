"""Occlusion robustness evaluation: supervised vs JEPA vs JEPA+completion.

Tests three strategies under increasing patch occlusion (0 → 14 of 16
patches zeroed out):

    (A) Supervised encoder  : sees all 16 patches, some zeroed → classify
    (B) JEPA encoder        : same — sees zeroed patches as zeros
    (C) JEPA + completion   : context encoder only sees visible patches,
                              predictor fills in masked latents, then classify

(C) is the key demo: the predictor was trained to reconstruct masked column
latents from spatial context, so it should degrade far more gracefully than
(A) or (B) under heavy occlusion.

Biologically: (C) mirrors cortical completion — V1 columns receiving no
input are "filled in" by horizontal connections from neighbouring columns,
allowing the brain to perceive despite partial occlusion.

Usage
─────
    python -m cortical_column.eval_occlusion
    python -m cortical_column.eval_occlusion --occlusion 0 2 4 6 8 10 12 14
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortical_column.config import LATENT_DIM, N_PATCHES, NUM_WORKERS
from cortical_column.cortical_network import CorticalNetwork
from cortical_column.jepa import CorticalPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mnist_loader(batch_size: int, train: bool, num_workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def load_encoder(ckpt_path: str, key: str | None, device: torch.device) -> CorticalNetwork:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if key and isinstance(state, dict) and key in state:
        state = state[key]
    enc = CorticalNetwork().to(device)
    enc.load_state_dict(state)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def load_predictor(ckpt_path: str, device: torch.device) -> CorticalPredictor:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    pred = CorticalPredictor().to(device)
    pred.load_state_dict(state["predictor"])
    for p in pred.parameters():
        p.requires_grad_(False)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction under occlusion
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_standard(
    encoder: CorticalNetwork,
    loader: DataLoader,
    device: torch.device,
    n_occluded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard forward: zero out n_occluded random patches (same set per batch),
    mean-pool 16 latents → feature vector.

    This is what both the supervised and bare JEPA encoder see.
    """
    encoder.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        patches = encoder.extract_patches(images)   # [B, 16, 49]

        if n_occluded > 0:
            # Same random mask for every image in the batch (realistic: the
            # occluder covers the same spatial region in the scene)
            occ_idx = torch.randperm(N_PATCHES, device=device)[:n_occluded]
            patches[:, occ_idx, :] = 0.0

        latents = []
        for i, col in enumerate(encoder.columns):
            lat, _ = col(patches[:, i, :], top_down_signal=None)
            latents.append(lat)
        latents = torch.stack(latents, dim=1)       # [B, 16, 128]
        feats = latents.mean(dim=1).cpu()           # [B, 128]
        all_feats.append(feats)
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def extract_jepa_completion(
    encoder: CorticalNetwork,
    predictor: CorticalPredictor,
    loader: DataLoader,
    device: torch.device,
    n_occluded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JEPA completion forward: context columns see visible patches, predictor
    fills in the masked (occluded) column latents, all 16 latents pooled.

    This is the key advantage: the predictor was trained for exactly this —
    reconstructing target latents from surrounding context latents.
    """
    encoder.eval()
    predictor.eval()
    all_feats, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        patches = encoder.extract_patches(images)   # [B, 16, 49]

        if n_occluded == 0:
            # No occlusion: run all columns, no predictor needed
            latents = []
            for i, col in enumerate(encoder.columns):
                lat, _ = col(patches[:, i, :], top_down_signal=None)
                latents.append(lat)
            latents = torch.stack(latents, dim=1)   # [B, 16, 128]
        else:
            # Split patches into context (visible) and target (occluded)
            occ_idx = torch.randperm(N_PATCHES, device=device)[:n_occluded]
            occ_set = set(occ_idx.tolist())
            ctx_idx = torch.tensor(
                [i for i in range(N_PATCHES) if i not in occ_set],
                dtype=torch.long, device=device
            )
            tgt_idx = occ_idx.sort().values

            # Encode visible context columns (gradient-free)
            ctx_latents = []
            for i in ctx_idx.tolist():
                lat, _ = encoder.columns[i](patches[:, i, :], top_down_signal=None)
                ctx_latents.append(lat)
            ctx_latents = torch.stack(ctx_latents, dim=1)  # [B, n_ctx, 128]

            # Predict masked column latents from context
            pred_latents = predictor(ctx_latents, ctx_idx, tgt_idx)  # [B, n_occ, 128]

            # Reassemble all 16 latents in correct order
            latents = torch.zeros(images.shape[0], N_PATCHES, LATENT_DIM, device=device)
            for j, i in enumerate(ctx_idx.tolist()):
                latents[:, i, :] = ctx_latents[:, j, :]
            for j, i in enumerate(tgt_idx.tolist()):
                latents[:, i, :] = pred_latents[:, j, :]

        feats = latents.mean(dim=1).cpu()           # [B, 128]
        all_feats.append(feats)
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Linear head
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_head(
    feats: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 150,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
) -> nn.Linear:
    head = nn.Linear(LATENT_DIM, 10).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()
    f, l = feats.to(device), labels.to(device)
    for _ in range(n_epochs):
        head.train()
        opt.zero_grad()
        loss_fn(head(f), l).backward()
        opt.step()
        sched.step()
    return head


@torch.no_grad()
def accuracy(head: nn.Linear, feats: torch.Tensor, labels: torch.Tensor, device) -> float:
    head.eval()
    preds = head(feats.to(device)).argmax(dim=1).cpu()
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_occlusion_eval(
    supervised_ckpt: str = "cortical_network_mnist.pt",
    jepa_ckpt: str = "cortical_network_jepa.pt",
    occlusion_levels: list[int] | None = None,
    output_plot: str = "eval_occlusion.png",
    n_trials: int = 3,           # average over N random occlusion masks
):
    """
    Args:
        occlusion_levels : list of patch counts to occlude (0 = no occlusion)
        n_trials         : average accuracy over this many random occlusion masks
                           per level, to reduce mask-sampling variance
    """
    if occlusion_levels is None:
        occlusion_levels = [0, 2, 4, 6, 8, 10, 12, 14]

    device    = get_device()
    n_workers = NUM_WORKERS if torch.cuda.is_available() else 0
    print(f"Device : {device}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    sup_enc  = load_encoder(supervised_ckpt, key=None,              device=device)
    jepa_enc = load_encoder(jepa_ckpt,       key="context_encoder", device=device)
    predictor = load_predictor(jepa_ckpt, device=device)
    print(f"Loaded supervised : {supervised_ckpt}")
    print(f"Loaded JEPA       : {jepa_ckpt}\n")

    # ── Train linear heads on clean (0% occluded) full training set ───────────
    print("Training linear heads on clean full training set (60k)...")
    train_loader = get_mnist_loader(1024, train=True,  num_workers=n_workers)
    val_loader   = get_mnist_loader(1024, train=False, num_workers=n_workers)

    sup_train_feats,  sup_train_labels  = extract_standard(sup_enc,  train_loader, device, n_occluded=0)
    jepa_train_feats, jepa_train_labels = extract_standard(jepa_enc, train_loader, device, n_occluded=0)

    sup_head  = train_linear_head(sup_train_feats,  sup_train_labels,  device=device)
    jepa_head = train_linear_head(jepa_train_feats, jepa_train_labels, device=device)
    print("Linear heads trained.\n")

    # ── Sweep occlusion levels ────────────────────────────────────────────────
    results = {
        "supervised":      [],
        "JEPA":            [],
        "JEPA+completion": [],
    }

    header = f"{'occ_patches':>12}  {'supervised':>12}  {'JEPA':>12}  {'JEPA+completion':>16}"
    print(header)
    print("─" * len(header))

    for n_occ in occlusion_levels:
        acc_sup, acc_jepa, acc_comp = 0.0, 0.0, 0.0

        for _ in range(n_trials):
            # (A) Supervised
            vf, vl = extract_standard(sup_enc,  val_loader, device, n_occ)
            acc_sup  += accuracy(sup_head,  vf, vl, device)

            # (B) JEPA standard (sees zeroed patches)
            vf, vl = extract_standard(jepa_enc, val_loader, device, n_occ)
            acc_jepa += accuracy(jepa_head, vf, vl, device)

            # (C) JEPA + completion (predictor fills masked latents)
            vf, vl = extract_jepa_completion(jepa_enc, predictor, val_loader, device, n_occ)
            acc_comp += accuracy(jepa_head, vf, vl, device)

        acc_sup  /= n_trials
        acc_jepa /= n_trials
        acc_comp /= n_trials

        results["supervised"].append(acc_sup)
        results["JEPA"].append(acc_jepa)
        results["JEPA+completion"].append(acc_comp)

        print(
            f"{n_occ:>12}  "
            f"{acc_sup * 100:>11.2f}%  "
            f"{acc_jepa * 100:>11.2f}%  "
            f"{acc_comp * 100:>15.2f}%"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))

        styles = {
            "supervised":      dict(color="#2196F3", marker="o", linestyle="-",  linewidth=2),
            "JEPA":            dict(color="#FF5722", marker="s", linestyle="--", linewidth=2),
            "JEPA+completion": dict(color="#4CAF50", marker="^", linestyle="-",  linewidth=2.5),
        }

        for name, accs in results.items():
            ax.plot(
                occlusion_levels,
                [a * 100 for a in accs],
                label=name,
                markersize=8,
                **styles[name],
            )

        # Shade region where JEPA+completion > supervised
        sup_arr  = [a * 100 for a in results["supervised"]]
        comp_arr = [a * 100 for a in results["JEPA+completion"]]
        ax.fill_between(
            occlusion_levels, sup_arr, comp_arr,
            where=[c > s for c, s in zip(comp_arr, sup_arr)],
            alpha=0.15, color="#4CAF50", label="JEPA+completion advantage",
        )

        ax.set_xlabel("Patches occluded (out of 16)", fontsize=12)
        ax.set_ylabel("Val accuracy (%)", fontsize=12)
        ax.set_title(
            "Occlusion robustness: supervised vs JEPA vs JEPA+completion",
            fontsize=13,
        )
        ax.set_xticks(occlusion_levels)
        ax.set_xticklabels([f"{n}\n({n/16*100:.0f}%)" for n in occlusion_levels])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"\nPlot saved → {output_plot}")

    except ImportError:
        print("\n[INFO] matplotlib not installed — skipping plot")

    return results, occlusion_levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occlusion robustness: supervised vs JEPA")
    parser.add_argument("--supervised", default="cortical_network_mnist.pt")
    parser.add_argument("--jepa",       default="cortical_network_jepa.pt")
    parser.add_argument(
        "--occlusion", nargs="+", type=int,
        default=[0, 2, 4, 6, 8, 10, 12, 14],
        help="Number of patches to occlude (0 = clean).",
    )
    parser.add_argument("--output",  default="eval_occlusion.png")
    parser.add_argument("--trials",  type=int, default=3,
                        help="Random mask samples to average per occlusion level.")
    args = parser.parse_args()

    run_occlusion_eval(
        supervised_ckpt=args.supervised,
        jepa_ckpt=args.jepa,
        occlusion_levels=args.occlusion,
        output_plot=args.output,
        n_trials=args.trials,
    )
