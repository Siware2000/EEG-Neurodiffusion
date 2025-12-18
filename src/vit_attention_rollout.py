import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -------------------------
# Build EXACT ViT used in your training (torchvision)
# -------------------------
def build_vit(num_classes: int):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


# -------------------------
# Patch MHA so it always returns attention weights
# -------------------------
def enable_attention_weights(model: nn.Module):
    for layer in model.encoder.layers:
        mha = layer.self_attention
        orig_forward = mha.forward

        def forward_with_weights(query, key, value, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False  # keep per-head weights
            return orig_forward(query, key, value, **kwargs)

        mha.forward = forward_with_weights


# -------------------------
# Attention rollout (Abnar & Zuidema-style)
# attentions: list of (B, heads, tokens, tokens)
# returns: (B, tokens, tokens)
# -------------------------
def attention_rollout(attentions, discard_ratio=0.0):
    # attentions: list of (B, heads, T, T)
    attn = [a.mean(dim=1) for a in attentions]  # -> list of (B, T, T)

    B, T, _ = attn[0].shape

    # Optionally discard lowest attentions (simple global prune)
    if discard_ratio > 0:
        new_attn = []
        for a in attn:
            flat = a.reshape(B, -1)
            n_discard = int(flat.shape[1] * discard_ratio)
            if n_discard > 0:
                idx = torch.argsort(flat, dim=1)[:, :n_discard]
                flat.scatter_(1, idx, 0)
                a = flat.reshape(B, T, T)
            new_attn.append(a)
        attn = new_attn

    # Identity (residual) with correct batch size
    I = torch.eye(T, device=attn[0].device).unsqueeze(0).repeat(B, 1, 1)

    result = I.clone()
    for a in attn:
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)
        result = torch.bmm(a, result)

    return result  # (B, T, T)



def save_overlay(img_tensor, mask_2d, out_path: Path):
    """
    img_tensor: (3,H,W) normalized [-1,1] (because you used mean=0.5,std=0.5)
    mask_2d: (H,W) in [0,1]
    """
    # unnormalize to [0,1]
    img = img_tensor.clone()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()

    mask_np = mask_2d.cpu().numpy()
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    plt.imshow(mask_np, alpha=0.45)  # overlay
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8)  # total images to save
    parser.add_argument("--discard-ratio", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = build_vit(num_classes=len(ds.classes)).to(device)

    # load checkpoint (handles both raw state_dict or dict wrapper)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # IMPORTANT: force attention weights to be returned
    enable_attention_weights(model)

    # collect attentions from each encoder layer
    attentions = []

    def hook_fn(module, inp, output):
        # output is (attn_output, attn_weights)
        attn_w = output[1]
        if attn_w is not None:
            attentions.append(attn_w.detach())

    hooks = []
    for layer in model.encoder.layers:
        hooks.append(layer.self_attention.register_forward_hook(hook_fn))

    saved = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)

        attentions.clear()
        with torch.no_grad():
            _ = model(imgs)

        if len(attentions) == 0:
            print("[WARN] No attention weights captured. Check torchvision version.")
            break

        rollout = attention_rollout(attentions, discard_ratio=args.discard_ratio)  # (B,T,T)
        # CLS token attention to patches = rollout[:,0,1:]
        cls_attn = rollout[:, 0, 1:]  # (B, num_patches)
        num_patches = cls_attn.shape[1]
        grid = int(np.sqrt(num_patches))
        cls_attn = cls_attn.reshape(-1, grid, grid)  # (B, Gh, Gw)

        # upsample mask to image size
        cls_attn_up = torch.nn.functional.interpolate(
            cls_attn.unsqueeze(1), size=(args.img_size, args.img_size),
            mode="bilinear", align_corners=False
        ).squeeze(1)  # (B,H,W)

        for i in range(imgs.shape[0]):
            if saved >= args.num_samples:
                break
            out_path = out_dir / f"attn_{saved:03d}_label{int(labels[i])}.png"
            save_overlay(imgs[i].cpu(), cls_attn_up[i].cpu(), out_path)
            saved += 1

        if saved >= args.num_samples:
            break

    for h in hooks:
        h.remove()

    print(f"[DONE] Saved {saved} attention-rollout overlays to: {out_dir}")


if __name__ == "__main__":
    main()
