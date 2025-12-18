# src/vit_attention_rollout_fixed.py
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from PIL import Image

import matplotlib.pyplot as plt


def build_vit(num_classes: int):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def attention_rollout(attn_mats, discard_ratio=0.0):
    """
    attn_mats: list of tensors, each [B, heads, T, T]
    returns: rollout [B, T, T]
    """
    # Average heads
    attn_mats = [a.mean(dim=1) for a in attn_mats]  # [B,T,T]

    # Add identity + normalize (residual)
    result = torch.eye(attn_mats[0].size(-1), device=attn_mats[0].device).unsqueeze(0)
    result = result.repeat(attn_mats[0].size(0), 1, 1)  # [B,T,T]

    for a in attn_mats:
        # optionally drop low attentions (except class token row)
        if discard_ratio > 0:
            B, T, _ = a.shape
            flat = a.view(B, -1)
            n_discard = int(flat.size(1) * discard_ratio)
            if n_discard > 0:
                _, idx = torch.topk(flat, k=flat.size(1) - n_discard, dim=1, largest=True)
                mask = torch.zeros_like(flat)
                mask.scatter_(1, idx, 1.0)
                a = (flat * mask).view(B, T, T)

        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)  # residual
        a = a / a.sum(dim=-1, keepdim=True)  # row-normalize

        result = torch.bmm(a, result)

    return result


def overlay_heatmap_on_image(img_tensor, heatmap, out_path):
    """
    img_tensor: [3,H,W] in normalized space [-1,1] or similar
    heatmap: [H,W] in [0,1]
    """
    # unnormalize from mean=0.5 std=0.5 -> [0,1]
    img = img_tensor.clone()
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)

    img_np = img.permute(1, 2, 0).cpu().numpy()
    hm_np = heatmap.cpu().numpy()

    plt.figure()
    plt.imshow(img_np)
    plt.imshow(hm_np, alpha=0.45)
    plt.axis("off")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="ImageFolder root (real data).")
    p.add_argument("--ckpt", required=True, help="ViT checkpoint.")
    p.add_argument("--out-dir", required=True, help="Output folder.")
    p.add_argument("--num-samples", type=int, default=8, help="How many images to visualize.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--discard-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(args.data_root, transform=tfm)

    # pick random indices
    idxs = random.sample(range(len(ds)), k=min(args.num_samples, len(ds)))
    subset = torch.utils.data.Subset(ds, idxs)
    loader = DataLoader(subset, batch_size=min(args.batch_size, len(subset)), shuffle=False, num_workers=0)

    num_classes = len(ds.classes)
    model = build_vit(num_classes=num_classes).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # Hook attentions from every encoder layer self-attention
    attentions = []

    def hook_fn(module, inputs, output):
        # torchvision MultiheadAttention sometimes returns (attn_output, attn_weights) OR only attn_output
        # Your earlier error: output was tuple
        if isinstance(output, tuple):
            # (attn_output, attn_weights?) depends on need_weights
            # In torchvision vit, need_weights=False so weights not returned.
            # But some internal may still give tuple; take first tensor.
            out = output[0]
        else:
            out = output
        # out could be attn_output [B,T,C] not weights -> so we must hook the module that has weights.
        # Therefore, instead hook into the layer's "self_attention" forward to capture qkv attention weights is hard.
        # Workaround: use forward_pre_hook on encoder layer to capture attention weights via internal attention module weights is unavailable.
        # So we will hook the module "encoder.layers[i].self_attention" and force need_weights=True by monkeypatch forward.
        pass

    # ---- Monkeypatch each layer self_attention to return weights ----
    # This is the cleanest way for torchvision ViT.
    for li, layer in enumerate(model.encoder.layers):
        mha = layer.self_attention
        orig_forward = mha.forward

        def make_forward(orig_fwd):
            def forward_with_weights(query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                attn_out, attn_w = orig_fwd(query, key, value, **kwargs)
                return attn_out, attn_w
            return forward_with_weights

        mha.forward = make_forward(orig_forward)

        def make_layer_hook(layer_index):
            def layer_hook(module, inputs, output):
                # output = (attn_out, attn_weights)
                attn_w = output[1]  # [B, heads, T, T]
                attentions.append(attn_w.detach())
            return layer_hook

        mha.register_forward_hook(make_layer_hook(li))

    # ---- Forward once to collect attentions ----
    batch = next(iter(loader))
    imgs, labels = batch
    imgs = imgs.to(device)

    with torch.no_grad():
        _ = model(imgs)

    # attentions list length = num_layers, each [B, heads, T, T]
    if len(attentions) == 0:
        raise RuntimeError("No attention weights captured. Check torchvision version.")

    B = imgs.size(0)
    T = attentions[0].size(-1)  # token count, typically 197 for 224 with 16x16 patches
    print(f"[INFO] Captured {len(attentions)} layers. tokens={T} batch={B}")

    rollout = attention_rollout(attentions, discard_ratio=args.discard_ratio)  # [B,T,T]

    # Class token attends to patches -> take [CLS] row (0) excluding CLS column
    cls_attn = rollout[:, 0, 1:]  # [B, T-1]
    num_patches = cls_attn.size(-1)
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"Patch tokens {num_patches} not a perfect square.")

    cls_map = cls_attn.reshape(B, grid_size, grid_size)

    # Upsample heatmap to image size
    cls_map = cls_map.unsqueeze(1)  # [B,1,gs,gs]
    cls_map = torch.nn.functional.interpolate(cls_map, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
    cls_map = cls_map.squeeze(1)  # [B,H,W]
    cls_map = (cls_map - cls_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
              (cls_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - cls_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-8)

    # Save overlays
    for i in range(B):
        out_img = out_dir / f"sample_{i+1}_overlay.png"
        overlay_heatmap_on_image(imgs[i].cpu(), cls_map[i].cpu(), out_img)
        print(f"[SAVED] {out_img}")

    print("[DONE] Attention rollout saved.")


if __name__ == "__main__":
    main()
