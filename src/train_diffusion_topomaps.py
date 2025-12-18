# src/train_diffusion_topomaps.py

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from eeg_topomap_dataset import TopomapImageDataset
from diffusion_unet import UNetModel


# -------------------------
# 1. Diffusion scheduler
# -------------------------

def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """
    Linear beta schedule between beta_start and beta_end for T timesteps.
    """
    return torch.linspace(beta_start, beta_end, T)


# -------------------------
# 2. Training loop
# -------------------------

def train(
    data_root: str,
    dataset_name: str,
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 50,
    num_steps: int = 1000,
    lr: float = 2e-4,
    save_dir: str = "checkpoints",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & loader
    ds = TopomapImageDataset(root_dir=data_root, dataset=dataset_name, image_size=image_size)
    pin_mem = (device.type == "cuda")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_mem,
    )

    num_classes = ds.num_classes
    print(f"Detected num_classes={num_classes}")

    # Model
    model = UNetModel(
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 4),  # for 224x224
        num_res_blocks=2,
        num_classes=num_classes,
        time_emb_dim=256,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Diffusion parameters
    T_steps = num_steps
    betas = make_beta_schedule(T_steps).to(device)              # (T,)
    alphas = 1.0 - betas                                       # (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=0)              # (T,)

    # helpers for q(x_t | x_0)
    def q_sample(x0, t, noise=None):
        """
        x0: (B,3,H,W) in [-1,1] or [0,1] (we'll scale to [-1,1])
        t: (B,) int64
        noise: (B,3,H,W)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cum = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_alpha_cum * x0 + sqrt_one_minus_alpha_cum * noise, noise

    # Training
    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device)  # (B,3,H,W), in [0,1]
            y = y.to(device)  # (B,)

            # scale to [-1,1]
            x = x * 2.0 - 1.0

            bsz = x.size(0)
            t = torch.randint(0, T_steps, (bsz,), device=device, dtype=torch.long)

            x_t, noise = q_sample(x, t)

            pred_noise = model(x_t, t, y)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"[Epoch {epoch}] avg_loss = {avg_loss:.6f}")

        # save checkpoint
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir_path / f"diffusion_{dataset_name}_epoch{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "num_steps": T_steps,
                "betas": betas.cpu(),
                "alphas_cumprod": alphas_cumprod.cpu(),
                "num_classes": num_classes,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training completed.")


# -------------------------
# 3. CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on EEG topomaps")
    parser.add_argument("--data-root", type=str,
                        default="D:/EEG-Neurodiffusion/Topomaps",
                        help="Root folder containing seed_iv / stew / clas / all")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "seed_iv", "stew", "clas"],
                        help="Which dataset folder to use")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Number of diffusion steps T")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints_diffusion")

    args = parser.parse_args()

    train(
        data_root=args.data_root,
        dataset_name=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_steps=args.num_steps,
        lr=args.lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
