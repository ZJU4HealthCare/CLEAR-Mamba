#!/usr/bin/env python3
"""
Confusion Matrix evaluator for MedMamba / HyEviMamba classifiers.

Features
- Loads a trained checkpoint and runs inference on a validation folder (ImageFolder).
- Draws a confusion matrix (counts or normalized), and prints per-class Precision / Recall / Specificity.
- Robust to different model outputs: logits tensor, (logits, extra), or dict with 'logits'.
- Optional EDL flag (we ignore evidences for CM, but won't crash if model returns them).
- Works with ImageNet-like normalization by default; override via CLI if needed.

Usage (examples)
-----------------
# MedMamba
python confusion_mtx_eval_medmamba.py \
  --val-dir /path/to/val \
  --num-classes 5 \
  --arch medmamba \
  --weights /path/to/best.pth \
  --image-size 224

# HyEviMamba (EDL-enabled checkpoint)
python confusion_mtx_eval_medmamba.py \
  --val-dir /path/to/val \
  --num-classes 5 \
  --arch hyevimamba \
  --weights /path/to/best.pth \
  --image-size 224 \
  --edl 1

Notes
- If your model constructor signature differs, tweak the `build_model()` function below to match your VSSM init.
- If your checkpoint is a full pickled nn.Module, we will load-and-use it directly.
- Class names are auto-derived from ImageFolder, or you can pass --class-indices JSON like in your MobileNet script.
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Optional: import EDL utils if available (safe even if unused)
try:
    from loss_evidential import compute_uncertainty  # noqa: F401
except Exception:
    compute_uncertainty = None  # Not required for confusion matrix

# Try to import MedMamba backbone
_MAMBA_IMPORTED = False
try:
    from MedMamba import VSSM as MedMambaBackbone  # user repo: class name used in train.py
    _MAMBA_IMPORTED = True
except Exception as e:
    MedMambaBackbone = None


def parse_args():
    p = argparse.ArgumentParser(description="Confusion Matrix evaluator for (HyEvi)MedMamba")

    # Data
    p.add_argument('--val-dir', type=str, required=True,
                   help='Path to validation folder (torchvision ImageFolder).')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)

    # Model / weights
    p.add_argument('--arch', type=str, default='medmamba', choices=['medmamba', 'hyevimamba', 'auto'],
                   help="Model to build when checkpoint isn't a full pickled module.")
    p.add_argument('--num-classes', type=int, required=True,
                   help='Number of classes in the classifier head (needed unless your checkpoint is a full model).')
    p.add_argument('--weights', type=str, required=True, help='Path to checkpoint .pth/.pt')
    p.add_argument('--edl', type=int, default=0, help='If 1, tolerate EDL tuple outputs; we still use logits for CM.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Labels / plotting
    p.add_argument('--class-indices', type=str, default='',
                   help='Optional path to class_indices.json (index->label). Overrides ImageFolder classes.')
    p.add_argument('--normalize', action='store_true', help='Normalize confusion matrix by true label counts.')
    p.add_argument('--save', type=str, default='', help='If set, save confusion matrix image to this path.')

    return p.parse_args()


class ConfusionMatrix:
    def __init__(self, num_classes: int, labels: List[str]):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds: np.ndarray, targets: np.ndarray):
        for p, t in zip(preds, targets):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / max(1, np.sum(self.matrix))
        print(f"Overall Accuracy: {acc:.4f}")

        try:
            from prettytable import PrettyTable
            table = PrettyTable()
            table.field_names = ["Class", "Precision", "Recall", "Specificity"]
        except Exception:
            table = None

        rows = []
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            rows.append((self.labels[i], precision, recall, specificity))

        if table is not None:
            for name, pre, rec, spe in rows:
                table.add_row([name, f"{pre:.3f}", f"{rec:.3f}", f"{spe:.3f}"])
            print(table)
        else:
            print("Per-class metrics:")
            for name, pre, rec, spe in rows:
                print(f"{name:>15s} | P {pre:.3f} | R {rec:.3f} | S {spe:.3f}")

    def plot(self, normalize: bool = False, save_path: Optional[str] = None):
        mat = self.matrix.astype(np.float32)
        if normalize:
            col_sums = mat.sum(axis=0, keepdims=True)  # by true labels (columns)
            mat = np.divide(mat, np.maximum(col_sums, 1.0), out=np.zeros_like(mat), where=col_sums != 0)

        plt.figure(figsize=(8, 7))
        im = plt.imshow(mat, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45, ha='right')
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

        thresh = mat.max() / 2.0 if mat.size else 0
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                val = mat[y, x]
                txt = f"{val:.2f}" if normalize else f"{int(val)}"
                plt.text(x, y, txt, va='center', ha='center',
                         color='white' if val > thresh else 'black')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"Saved confusion matrix to: {save_path}")
        else:
            plt.show()


def default_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def try_extract_logits(output: torch.Tensor):
    """Best-effort extraction of logits from varied model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        # common patterns: (logits, evidences) or (logits, aux1, aux2, ...)
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item
    if isinstance(output, dict):
        for key in ['logits', 'cls_logits', 'pred', 'y_logits']:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    raise RuntimeError("Cannot extract logits from model output; please adapt try_extract_logits().")


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Instantiate a classifier head on top of MedMamba backbone.
    Adjust this to exactly match your training script (constructor signature, etc.).
    """
    if not _MAMBA_IMPORTED or MedMambaBackbone is None:
        raise ImportError("MedMamba backbone not found. Ensure MedMamba.py is in PYTHONPATH and exposes VSSM class.")

    # Many repos expose VSSM(args) or VSSM(num_classes=...). We try a couple of common signatures.
    last_err = None
    for kwargs in (
        dict(num_classes=num_classes),
        dict(in_chans=3, num_classes=num_classes),
    ):
        try:
            model = MedMambaBackbone(**kwargs)
            return model
        except Exception as e:  # keep trying
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to construct VSSM with tried kwargs. Please edit build_model() to match your VSSM signature. Last error: {last_err}")


def maybe_load_full_module(weights_path: str, device: str) -> Optional[nn.Module]:
    """If the checkpoint is a fully pickled model, load it directly; otherwise return None."""
    try:
        obj = torch.load(weights_path, map_location=device)
        if isinstance(obj, nn.Module):
            obj.eval()
            return obj
        return None
    except Exception:
        return None


def load_state_dict_safely(model: nn.Module, weights_path: str, device: str):
    ckpt = torch.load(weights_path, map_location=device)
    sd = None
    if isinstance(ckpt, dict):
        # common keys
        for k in ['model', 'state_dict', 'model_state', 'net', 'ema_state_dict']:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        if sd is None and all(isinstance(k, str) for k in ckpt.keys()):
            # maybe it is already a state_dict
            sd = ckpt
    elif isinstance(ckpt, nn.Module):
        model.load_state_dict(ckpt.state_dict(), strict=False)
        return
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_state_dict] Missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
    if unexpected:
        print(f"[load_state_dict] Unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Data
    tfm = default_transforms(args.image_size)
    val_set = datasets.ImageFolder(root=args.val-dir if hasattr(args, 'val-dir') else args.val_dir, transform=tfm)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Labels (from class_indices.json or dataset classes)
    if args.class_indices and os.path.isfile(args.class_indices):
        with open(args.class_indices, 'r', encoding='utf-8') as f:
            idx2name = json.load(f)
        # Expecting a dict like {"0": "cat", "1": "dog"}
        labels = [idx2name[str(i)] if str(i) in idx2name else str(i) for i in range(args.num_classes)]
    else:
        # torchvision ImageFolder has classes attribute sorted by folder name
        if len(val_set.classes) == args.num_classes:
            labels = val_set.classes
        else:
            labels = [str(i) for i in range(args.num_classes)]

    cm = ConfusionMatrix(num_classes=args.num_classes, labels=labels)

    # Model
    model = maybe_load_full_module(args.weights, args.device)
    if model is None:
        model = build_model(args.arch, args.num_classes)
        load_state_dict_safely(model, args.weights, args.device)
    model = model.to(device).eval()

    # Inference loop
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)
            try:
                logits = try_extract_logits(out)
            except RuntimeError:
                # Some repos wrap output inside tuple when EDL enabled; try first item as fallback
                if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    logits = out[0]
                else:
                    raise
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(tgts.numpy())

    preds_np = np.concatenate(all_preds)
    tgts_np = np.concatenate(all_tgts)

    cm.update(preds_np, tgts_np)
    cm.plot(normalize=args.normalize, save_path=args.save)
    cm.summary()


if __name__ == '__main__':
    main()
