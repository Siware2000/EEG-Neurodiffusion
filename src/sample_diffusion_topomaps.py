# src/sample_diffusion_topomaps.py

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from diffusion_unet import UNetModel   # <--- use UNetModel directly


# -----------------------------
# Diffusion schedule (must match training)
# -----------------------------
def make_beta_schedule(T: int = 1000,
                       beta_start: float = 1e-4,
                       beta_end: float = 0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


# -----------------------------
# Sampling function
# -----------------------------
@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    num_samples: int,
    num_classes: int,
    img_size: int = 224,
    T: int = 1000,
    device: torch.device = torch.device("cpu"),
    class_id: int = None,
    batch_size: int = 16,
):
    """
    Generate num_samples images using the trained diffusion model.

    If class_id is not None, generate conditionally for that class.
    Otherwise randomly sample labels in [0, num_classes-1].
    """
    betas, alphas, alphas_cumprod = make_beta_schedule(T)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    imgs = []
    n_generated = 0

    while n_generated < num_samples:
        cur_bs = min(batch_size, num_samples - n_generated)

        # x_T ~ N(0, I)
        x_t = torch.randn(cur_bs, 3, img_size, img_size, device=device)

        if class_id is not None:
            y = torch.full((cur_bs,), int(class_id),
                           dtype=torch.long, device=device)
        else:
            y = torch.randint(0, num_classes, (cur_bs,), device=device)

        # reverse diffusion: T-1 ... 0
        for t_idx in reversed(range(T)):
            t = torch.full((cur_bs,),
                           t_idx,
                           device=device,
                           dtype=torch.long)

            eps_theta = model(x_t, t, y)  # predicted noise

            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            alpha_bar_t = alphas_cumprod[t_idx]

            if t_idx > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)

            # DDPM update
            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            ) + torch.sqrt(beta_t) * z

        imgs.append(x_t.cpu())
        n_generated += cur_bs

    imgs = torch.cat(imgs, dim=0)[:num_samples]  # (N, 3, H, W)
    # We trained with images in [0,1], so clamp to [0,1]
    imgs = torch.clamp(imgs, 0.0, 1.0)
    return imgs


def save_images_to_folder(images: torch.Tensor,
                          out_dir: Path,
                          prefix: str,
                          start_idx: int = 0):
    """
    images: (N, 3, H, W) in [0,1]
    """
    from torchvision.utils import save_image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        fn = out_dir / f"{prefix}_{start_idx + i:06d}.png"
        save_image(images[i], fn)


# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample synthetic topomaps from trained diffusion model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "seed_iv", "stew", "clas"],
        required=True,
        help="Which dataset's diffusion model to load.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes for this dataset (e.g., seed_iv=4, clas=3, stew=2, all=4).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="How many synthetic images per class.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (should match training, e.g., 224).",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1000,
        help="Number of diffusion steps (must match training).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to diffusion checkpoint. If None, will infer from dataset name.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="D:/EEG-Neurodiffusion/Topomaps_synth",
        help="Root folder for saving synthetic images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sampling batch size.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    num_classes = args.num_classes

    # ⚠️ IMPORTANT: these hyperparameters MUST match train_diffusion_topomaps.py
    # We infer from checkpoint errors that training used:
    #   base_channels = 64
    #   channel_mults = (1, 2, 4, 4)
    model = UNetModel(
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        num_classes=num_classes,
        time_emb_dim=256,
    ).to(device)

    # Load checkpoint
    if args.checkpoint is None:
        ckpt_name = f"diffusion_{args.dataset}_epoch50.pt"
        ckpt_path = Path("D:/EEG-Neurodiffusion/checkpoints_diffusion") / ckpt_name
    else:
        ckpt_path = Path(args.checkpoint)

    print(f"[INFO] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # Your training script saved:
    #   {
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": ...,
    #     ...
    #   }
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # fallback if you saved raw state_dict
        model.load_state_dict(state)
    model.eval()

    out_root = Path(args.out_root)
    (out_root / args.dataset).mkdir(parents=True, exist_ok=True)

    # For each class, sample images and save
    for c in range(num_classes):
        print(f"[CLASS {c}] Generating {args.samples_per_class} synthetic samples...")
        imgs = p_sample_loop(
            model=model,
            num_samples=args.samples_per_class,
            num_classes=num_classes,
            img_size=args.img_size,
            T=args.T,
            device=device,
            class_id=c,
            batch_size=args.batch_size,
        )

        class_dir = out_root / args.dataset / str(c)
        save_images_to_folder(imgs, class_dir,
                              prefix=f"{args.dataset}_synth_c{c}")
        print(f"  -> Saved to {class_dir}")

    print("[DONE] Synthetic sampling completed.")


if __name__ == "__main__":
    main()
