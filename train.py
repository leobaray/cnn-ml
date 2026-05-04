import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Plot theme (GitHub Dark inspired)
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size": 11,
    "savefig.facecolor": "#0d1117",
    "savefig.edgecolor": "#0d1117",
}

C_TRAIN = "#58a6ff"
C_VAL = "#f78166"
C_LR = "#d2a8ff"
C_GRAD = "#3fb950"
C_BEST = "#ffd700"
C_PHASE = "#484f58"
CLASS_COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffd700",
                "#ff7b72", "#79c0ff", "#7ee787", "#e3b341", "#ffa657"]

BW_MODE = False

PLOT_STYLE_BW = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "black",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.alpha": 0.5,
    "text.color": "black",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.facecolor": "white",
    "legend.edgecolor": "#333333",
    "font.size": 11,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
}

BW_GRAYS = ["#000000", "#555555", "#999999", "#333333", "#777777",
            "#222222", "#666666", "#aaaaaa", "#444444", "#888888"]
BW_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)),
                 (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 1)), (0, (3, 1))]
BW_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]
BW_HATCHES = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "...", "ooo",
              "**", "OO"]

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    default_data_dir = str(base_dir / "datasets")
    default_output_dir = str(base_dir / "output")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing best_model.pt instead of training from scratch")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, just evaluate and generate plots from existing model")
    parser.add_argument("--bw", action="store_true",
                        help="Generate print-friendly black & white plots")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed / device setup
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def list_images_and_labels(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    class_dirs = sorted([d for d in os.listdir(data_dir) if (Path(data_dir) / d).is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_dirs)}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files, labels = [], []
    for c in class_dirs:
        p = Path(data_dir) / c
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts and f.stem != "00":
                files.append(str(f))
                labels.append(class_to_idx[c])
    return files, labels, class_dirs


def stratified_split(files, labels, seed, val_split, test_split):
    counts = np.bincount(labels)
    min_count = counts.min()
    if min_count < 2:
        sparse = [i for i, c in enumerate(counts) if c < 2]
        print(f"WARNING: classes {sparse} have < 2 samples — falling back to non-stratified split")
        stratify_arg = None
    else:
        stratify_arg = labels

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        files, labels, test_size=test_split, random_state=seed, stratify=stratify_arg
    )
    rel_val = val_split / (1.0 - test_split)
    stratify_arg2 = y_trainval if stratify_arg is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, random_state=seed, stratify=stratify_arg2
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------------------------------------------------------------------------
# Dataset & Transforms
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(self, paths, labels, num_classes, transform=None):
        self.paths = paths
        self.labels = labels
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = torch.zeros(self.num_classes)
        label[self.labels[idx]] = 1.0
        return img, label


def get_train_transform(img_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.15, hue=0.04),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])


# ---------------------------------------------------------------------------
# MixUp + CutMix
# ---------------------------------------------------------------------------

def mixup_batch(x, y, alpha=0.2):
    bs = x.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((bs, 1, 1, 1)).to(x.device)
    lam_y = lam.view(bs, 1)
    idx = torch.randperm(bs, device=x.device)
    return lam * x + (1.0 - lam) * x[idx], lam_y * y + (1.0 - lam_y) * y[idx]


def cutmix_batch(x, y, alpha=1.0):
    bs = x.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(bs, device=x.device)
    _, _, h, w = x.shape
    cut_r = math.sqrt(1.0 - lam)
    ch, cw = int(h * cut_r), int(w * cut_r)
    cy, cx = random.randint(0, h), random.randint(0, w)
    y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
    x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return x_cut, lam_actual * y + (1.0 - lam_actual) * y[idx]


def mix_data(x, y):
    if random.random() < 0.5:
        return mixup_batch(x, y, alpha=0.2)
    return cutmix_batch(x, y, alpha=1.0)


# ---------------------------------------------------------------------------
# Class weights / Loss
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    return torch.tensor(
        [float(total) / (num_classes * max(1, c)) for c in counts], dtype=torch.float32
    )


def soft_cross_entropy(logits, targets, class_weights=None, label_smoothing=0.0):
    nc = targets.size(-1)
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + label_smoothing / nc
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    if class_weights is not None:
        loss = loss * class_weights[targets.argmax(dim=-1)]
    return loss.mean()


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GeM Pooling
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=[-2, -1]).pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, device: torch.device) -> nn.Module:
    base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    for param in base.features.parameters():
        param.requires_grad = False

    # Replace AvgPool with GeM
    base.avgpool = GeM(p=3.0)

    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    base = base.to(device)
    if device.type == "cuda":
        base = base.to(memory_format=torch.channels_last)
    return base


def get_base_model(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    if hasattr(model, "module"):
        return model.module
    return model


def unfreeze_backbone(model):
    raw = get_base_model(model)
    num_stages = len(raw.features)
    for i, stage in enumerate(raw.features):
        if i >= num_stages - 3:
            for module in stage.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    for p in module.parameters():
                        p.requires_grad = False
                else:
                    for p in module.parameters(recurse=False):
                        p.requires_grad = True


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._fh = target_layer.register_forward_hook(self._save_act)
        self._bh = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, target_class=None):
        self.model.eval()
        x = x.requires_grad_(True)
        with torch.enable_grad():
            out = self.model(x)
            if target_class is None:
                target_class = out.argmax(dim=1).item()
            self.model.zero_grad()
            out[0, target_class].backward()
        w = self.gradients.mean(dim=[-2, -1], keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), target_class, F.softmax(out, dim=-1).detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def denormalize_image(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * STD + MEAN, 0, 1)


def make_gradcam_overlay(img_arr, heatmap, alpha=0.45):
    cmap = plt.cm.jet
    colored = cmap(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * img_arr + alpha * colored, 0, 1)


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, class_weights,
                    label_smoothing, device, grad_accum_steps=1, use_mix=True):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    num_optim_steps = 0
    optimizer.zero_grad()

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)

        original_labels = yb.argmax(-1)

        if use_mix:
            xb, yb = mix_data(xb, yb)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(xb)
            loss = soft_cross_entropy(logits, yb, class_weights, label_smoothing)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            correct += (logits.argmax(-1) == original_labels).sum().item()
            total += xb.size(0)

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            total_grad_norm += gn.item() if isinstance(gn, torch.Tensor) else gn
            num_optim_steps += 1

        total_loss += loss.item() * grad_accum_steps * xb.size(0)

    return total_loss / total, correct / total, total_grad_norm / max(num_optim_steps, 1)


@torch.no_grad()
def evaluate_loss(model, loader, class_weights, label_smoothing, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)
        logits = model(xb)
        loss = soft_cross_entropy(logits, yb, class_weights, label_smoothing)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(-1) == yb.argmax(-1)).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, optimizer, scheduler, scaler,
                class_weights, label_smoothing, device, epochs, grad_accum_steps=1,
                patience=7, best_model_path=None, use_mix=True, ema=None):
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "lr": [], "grad_norm": []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            class_weights, label_smoothing, device, grad_accum_steps, use_mix
        )
        val_loss, val_acc = evaluate_loss(
            model, val_loader, class_weights, label_smoothing, device
        )
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["grad_norm"].append(grad_norm)

        vram = ""
        if device.type == "cuda":
            mb = torch.cuda.max_memory_allocated() / 1e6
            vram = f" VRAM: {mb:.0f}MB"

        print(f"  Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) — "
              f"loss: {train_loss:.4f} acc: {train_acc:.4f} "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
              f"lr: {lr_now:.2e} grad: {grad_norm:.3f}{vram}")

        if ema is not None:
            ema.update(model)

        if best_model_path is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, best_model_path)
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    return history


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float()

    def apply(self, model):
        ref = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(ref[k].dtype) for k in self.shadow})


# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------

TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: torch.flip(x, [-1]),
    lambda x: torch.flip(x, [-2]),
    lambda x: torch.flip(x, [-1, -2]),
    lambda x: torch.rot90(x, 1, [-2, -1]),
    lambda x: torch.rot90(x, 2, [-2, -1]),
    lambda x: torch.rot90(x, 3, [-2, -1]),
]


@torch.no_grad()
def tta_predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)
        preds_sum = torch.zeros(xb.size(0), yb.size(1), device=device)
        for fn in TTA_TRANSFORMS:
            preds_sum += F.softmax(model(fn(xb)), dim=-1)
        all_preds.append((preds_sum / len(TTA_TRANSFORMS)).cpu().numpy())
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Collect predictions & embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, device, use_tta=True):
    if use_tta:
        probs, labels_oh = tta_predict(model, loader, device)
    else:
        model.eval()
        all_p, all_l = [], []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            all_p.append(F.softmax(model(xb), dim=-1).cpu().numpy())
            all_l.append(yb.numpy())
        probs, labels_oh = np.concatenate(all_p), np.concatenate(all_l)
    y_true = np.argmax(labels_oh, axis=1)
    y_pred = np.argmax(probs, axis=1)
    return probs, y_true, y_pred


@torch.no_grad()
def extract_embeddings(model, loader, device):
    raw = get_base_model(model)
    raw.eval()
    embeddings, labels = [], []
    hook_out = []

    def hook_fn(module, inp, out):
        hook_out.append(out.detach().cpu())

    # Hook after GELU (index 7) in classifier -> 256-dim
    handle = raw.classifier[7].register_forward_hook(hook_fn)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)
        raw(xb)
        embeddings.append(hook_out.pop())
        labels.append(yb.argmax(-1).numpy())
    handle.remove()
    return np.concatenate(embeddings), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _style():
    return PLOT_STYLE_BW if BW_MODE else PLOT_STYLE


def _txt():
    return "black" if BW_MODE else "#c9d1d9"


def _styled_figure(*args, **kwargs):
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(*args, **kwargs)
    return fig, ax


def _styled_subplots(nrows, ncols, **kwargs):
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(nrows, ncols, **kwargs)
    return fig, axes


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    svg_path = str(path).rsplit(".", 1)[0] + ".svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def merge_histories(*hists) -> dict:
    keys = set()
    for h in hists:
        keys.update(h.keys())
    out = {}
    for k in keys:
        out[k] = []
        for h in hists:
            out[k].extend(list(h.get(k, [])))
    return out


def save_training_history(history: dict, output_dir: str) -> None:
    p = Path(output_dir) / "history"
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def plot_history(history: dict, output_dir: str, phase_bounds=None, best_epoch=None):
    results = Path(output_dir) / "results"
    results.mkdir(parents=True, exist_ok=True)

    c_train = "black" if BW_MODE else C_TRAIN
    c_val = "#666666" if BW_MODE else C_VAL
    c_lr = "black" if BW_MODE else C_LR
    c_grad = "black" if BW_MODE else C_GRAD
    c_best = "black" if BW_MODE else C_BEST
    c_phase = "#999999" if BW_MODE else C_PHASE
    ls_val = "--" if BW_MODE else "-"

    def _add_markers(ax, epochs_range):
        if phase_bounds:
            for b in phase_bounds:
                if b < len(epochs_range):
                    ax.axvline(x=b, color=c_phase, linestyle="--", linewidth=1, alpha=0.7)
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color=c_best, linestyle=":", linewidth=1.2, alpha=0.8)

    with plt.rc_context(_style()):
        # Loss
        if "loss" in history and "val_loss" in history:
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = range(len(history["loss"]))
            ax.plot(epochs, history["loss"], label="train_loss", color=c_train, linewidth=2)
            ax.plot(epochs, history["val_loss"], label="val_loss", color=c_val,
                    linewidth=2, linestyle=ls_val)
            if best_epoch is not None:
                ax.scatter([best_epoch], [history["val_loss"][best_epoch]],
                           color=c_best, s=80, zorder=5, marker="*", label=f"best (ep {best_epoch+1})")
            _add_markers(ax, epochs)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            fig.tight_layout()
            _save(fig, results / "loss_curve.png")

        # Accuracy
        if "acc" in history and "val_acc" in history:
            fig, ax = plt.subplots(figsize=(10, 5))
            epochs = range(len(history["acc"]))
            ax.plot(epochs, history["acc"], label="train_acc", color=c_train, linewidth=2)
            ax.plot(epochs, history["val_acc"], label="val_acc", color=c_val,
                    linewidth=2, linestyle=ls_val)
            _add_markers(ax, epochs)
            ax.set_xlabel("epoch")
            ax.set_ylabel("accuracy")
            ax.legend()
            fig.tight_layout()
            _save(fig, results / "acc_curve.png")

        # LR
        if "lr" in history and history["lr"]:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history["lr"], color=c_lr, linewidth=2)
            _add_markers(ax, range(len(history["lr"])))
            ax.set_xlabel("epoch")
            ax.set_ylabel("learning rate")
            ax.set_yscale("log")
            fig.tight_layout()
            _save(fig, results / "lr_curve.png")

        # Gradient norm
        if "grad_norm" in history and history["grad_norm"]:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history["grad_norm"], color=c_grad, linewidth=2)
            _add_markers(ax, range(len(history["grad_norm"])))
            ax.set_xlabel("epoch")
            ax.set_ylabel("gradient norm")
            fig.tight_layout()
            _save(fig, results / "grad_norm_curve.png")


def plot_confusion_matrix(cm, class_names, output_path):
    cmap = "Greys" if BW_MODE else "Blues"
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title="Matriz de Confusao", ylabel="Classe Verdadeira", xlabel="Classe Predita")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        _save(fig, output_path)


def plot_per_class_f1(y_true, y_pred, class_names, output_path):
    f1s = f1_score(y_true, y_pred, labels=range(len(class_names)),
                   average=None, zero_division=0)
    nc = len(class_names)
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(10, 5))
        if BW_MODE:
            colors = ["#cccccc"] * nc
            hatches = [BW_HATCHES[i % len(BW_HATCHES)] for i in range(nc)]
            bars = ax.barh(class_names, f1s, color=colors, edgecolor="black")
            for bar, h in zip(bars, hatches):
                bar.set_hatch(h)
        else:
            colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(nc)]
            bars = ax.barh(class_names, f1s, color=colors, edgecolor="#30363d")
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", color=_txt(), fontsize=10)
        ax.set_xlabel("F1 Score")
        ax.set_xlim(0, 1.15)
        ax.set_title("F1 Score por Classe")
        fig.tight_layout()
        _save(fig, output_path)


def plot_roc_curves(probs, y_true, class_names, output_path):
    nc = len(class_names)
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(nc):
            binary = (y_true == i).astype(int)
            if binary.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary, probs[:, i])
            roc_auc = auc(fpr, tpr)
            if BW_MODE:
                ax.plot(fpr, tpr, color=BW_GRAYS[i % len(BW_GRAYS)], linewidth=2,
                        linestyle=BW_LINESTYLES[i % len(BW_LINESTYLES)],
                        marker=BW_MARKERS[i % len(BW_MARKERS)], markevery=max(1, len(fpr) // 8),
                        markersize=5, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
            else:
                color = CLASS_COLORS[i % len(CLASS_COLORS)]
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f"{class_names[i]} (AUC={roc_auc:.3f})")
        diag_style = "k--" if BW_MODE else "w--"
        ax.plot([0, 1], [0, 1], diag_style, alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save(fig, output_path)


def plot_calibration(probs, y_true, output_path, n_bins=10):
    max_probs = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(int)
    try:
        frac_pos, mean_pred = calibration_curve(correct, max_probs, n_bins=n_bins, strategy="uniform")
    except ValueError:
        return
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(7, 7))
        c_model = "black" if BW_MODE else C_TRAIN
        diag_style = "k--" if BW_MODE else "w--"
        ax.plot(mean_pred, frac_pos, "o-", color=c_model, linewidth=2, label="Modelo")
        ax.plot([0, 1], [0, 1], diag_style, alpha=0.3, linewidth=1, label="Calibrado perfeito")
        ax.set_xlabel("Confianca media predita")
        ax.set_ylabel("Fracao de acertos")
        ax.set_title("Curva de Calibracao")
        ax.legend()
        fig.tight_layout()
        _save(fig, output_path)


def plot_confidence_histogram(probs, y_true, output_path):
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct_conf = max_probs[preds == y_true]
    wrong_conf = max_probs[preds != y_true]

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, 1, 25)
        if BW_MODE:
            if len(correct_conf) > 0:
                ax.hist(correct_conf, bins=bins, alpha=0.7, color="#dddddd",
                        label="Corretos", edgecolor="black", hatch="///")
            if len(wrong_conf) > 0:
                ax.hist(wrong_conf, bins=bins, alpha=0.7, color="#888888",
                        label="Errados", edgecolor="black", hatch="\\\\\\")
        else:
            if len(correct_conf) > 0:
                ax.hist(correct_conf, bins=bins, alpha=0.7, color=C_TRAIN,
                        label="Corretos", edgecolor="#30363d")
            if len(wrong_conf) > 0:
                ax.hist(wrong_conf, bins=bins, alpha=0.7, color=C_VAL,
                        label="Errados", edgecolor="#30363d")
        ax.set_xlabel("Confianca")
        ax.set_ylabel("Quantidade")
        ax.set_title("Distribuicao de Confianca")
        ax.legend()
        fig.tight_layout()
        _save(fig, output_path)


def plot_error_grid(probs, y_true, y_pred, test_ds, class_names, output_path, max_images=16):
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx) == 0:
        print("  Nenhum erro encontrado no test set!")
        return

    wrong_idx = wrong_idx[:max_images]
    confs = probs[wrong_idx].max(axis=1)

    n = len(wrong_idx)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        title_color = "red" if BW_MODE else C_VAL
        for i, idx in enumerate(wrong_idx):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            img = Image.open(test_ds.paths[idx]).convert("RGB").resize((256, 256))
            ax.imshow(np.array(img))
            true_name = class_names[y_true[idx]]
            pred_name = class_names[y_pred[idx]]
            ax.set_title(f"Real: {true_name}\nPred: {pred_name} ({confs[i]*100:.0f}%)",
                         fontsize=9, color=title_color)
            ax.axis("off")

        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")

        fig.suptitle("Predicoes Erradas", fontsize=14, color=_txt())
        fig.tight_layout()
        _save(fig, output_path)


def plot_tsne(embeddings, labels, class_names, output_path):
    n = len(embeddings)
    if n < 5:
        print("  Poucos dados pra t-SNE, pulando.")
        return
    perp = min(30, n - 1)
    print(f"  t-SNE com perplexity={perp}...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, name in enumerate(class_names):
            mask = labels == i
            if BW_MODE:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=BW_GRAYS[i % len(BW_GRAYS)], label=name,
                           marker=BW_MARKERS[i % len(BW_MARKERS)],
                           s=40, alpha=0.9, edgecolors="black", linewidths=0.5)
            else:
                color = CLASS_COLORS[i % len(CLASS_COLORS)]
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name,
                           s=40, alpha=0.8, edgecolors="#30363d", linewidths=0.3)
        ax.set_title("t-SNE dos Embeddings (256-dim)")
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        _save(fig, output_path)


def plot_gradcam_samples(model, test_ds, class_names, output_dir, device, samples_per_class=3):
    raw = get_base_model(model)
    target_layer = raw.features[-1]
    cam = GradCAM(raw, target_layer)

    img_size = None
    samples = {i: [] for i in range(len(class_names))}

    for idx in range(len(test_ds)):
        cls = test_ds.labels[idx]
        if len(samples[cls]) < samples_per_class:
            samples[cls].append(idx)
        if all(len(v) >= samples_per_class for v in samples.values()):
            break

    all_items = []
    for cls in sorted(samples.keys()):
        for idx in samples[cls]:
            all_items.append((cls, idx))

    n = len(all_items)
    if n == 0:
        cam.remove()
        return

    cols = samples_per_class * 2  # original + gradcam
    rows = len(class_names)

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
        if rows == 1:
            axes = axes[np.newaxis, :]

        for r, cls_name in enumerate(class_names):
            for s in range(samples_per_class):
                item_idx = r * samples_per_class + s
                col_orig = s * 2
                col_cam = s * 2 + 1

                if item_idx >= n or all_items[item_idx][0] != r:
                    axes[r, col_orig].axis("off")
                    axes[r, col_cam].axis("off")
                    continue

                _, ds_idx = all_items[item_idx]
                tensor, _ = test_ds[ds_idx]
                inp = tensor.unsqueeze(0).to(device)
                if device.type == "cuda":
                    inp = inp.to(memory_format=torch.channels_last)

                heatmap, pred_cls, probs_out = cam(inp)
                img_denorm = denormalize_image(tensor)

                axes[r, col_orig].imshow(img_denorm)
                axes[r, col_orig].set_title(f"{cls_name}", fontsize=9, color=_txt())
                axes[r, col_orig].axis("off")

                overlay = make_gradcam_overlay(img_denorm, heatmap)
                axes[r, col_cam].imshow(overlay)
                conf = probs_out[0, pred_cls].item() * 100
                pred_name = class_names[pred_cls]
                axes[r, col_cam].set_title(f"GradCAM → {pred_name} ({conf:.0f}%)",
                                           fontsize=9, color=_txt())
                axes[r, col_cam].axis("off")

        fig.suptitle("GradCAM — Onde o Modelo Olha", fontsize=14, color=_txt())
        fig.tight_layout()
        _save(fig, Path(output_dir) / "results" / "gradcam_samples.png")

    cam.remove()


def plot_dashboard(history, cm, class_names, y_true, y_pred, output_path,
                   phase_bounds=None, best_epoch=None):
    c_train = "black" if BW_MODE else C_TRAIN
    c_val = "#666666" if BW_MODE else C_VAL
    c_lr = "black" if BW_MODE else C_LR
    c_grad = "black" if BW_MODE else C_GRAD
    c_best = "black" if BW_MODE else C_BEST
    c_phase = "#999999" if BW_MODE else C_PHASE
    ls_val = "--" if BW_MODE else "-"
    cmap = "Greys" if BW_MODE else "Blues"
    nc = len(class_names)

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

        def _markers(ax, n):
            if phase_bounds:
                for b in phase_bounds:
                    if b < n:
                        ax.axvline(x=b, color=c_phase, ls="--", lw=1, alpha=0.7)
            if best_epoch is not None and best_epoch < n:
                ax.axvline(x=best_epoch, color=c_best, ls=":", lw=1.2, alpha=0.8)

        # (0,0) Loss
        ax = axes[0, 0]
        ep = range(len(history["loss"]))
        ax.plot(ep, history["loss"], color=c_train, lw=2, label="train")
        ax.plot(ep, history["val_loss"], color=c_val, lw=2, ls=ls_val, label="val")
        if best_epoch is not None:
            ax.scatter([best_epoch], [history["val_loss"][best_epoch]],
                       color=c_best, s=60, zorder=5, marker="*")
        _markers(ax, len(ep))
        ax.set_title("Loss")
        ax.legend(fontsize=9)

        # (0,1) Accuracy
        ax = axes[0, 1]
        ax.plot(ep, history["acc"], color=c_train, lw=2, label="train")
        ax.plot(ep, history["val_acc"], color=c_val, lw=2, ls=ls_val, label="val")
        _markers(ax, len(ep))
        ax.set_title("Accuracy")
        ax.legend(fontsize=9)

        # (0,2) LR
        ax = axes[0, 2]
        if history.get("lr"):
            ax.plot(history["lr"], color=c_lr, lw=2)
            _markers(ax, len(history["lr"]))
            ax.set_yscale("log")
        ax.set_title("Learning Rate")

        # (1,0) Gradient norm
        ax = axes[1, 0]
        if history.get("grad_norm"):
            ax.plot(history["grad_norm"], color=c_grad, lw=2)
            _markers(ax, len(history["grad_norm"]))
        ax.set_title("Gradient Norm")

        # (1,1) Confusion matrix
        ax = axes[1, 1]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=10)
        ax.set_title("Confusion Matrix")

        # (1,2) Per-class F1
        ax = axes[1, 2]
        f1s = f1_score(y_true, y_pred, labels=range(nc),
                       average=None, zero_division=0)
        if BW_MODE:
            colors = ["#cccccc"] * nc
            hatches = [BW_HATCHES[i % len(BW_HATCHES)] for i in range(nc)]
            bars = ax.barh(class_names, f1s, color=colors, edgecolor="black")
            for bar, h in zip(bars, hatches):
                bar.set_hatch(h)
        else:
            colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(nc)]
            bars = ax.barh(class_names, f1s, color=colors, edgecolor="#30363d")
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", color=_txt(), fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_title("F1 per Class")

        # Legend for phase lines
        legend_elements = []
        if phase_bounds:
            legend_elements.append(Line2D([0], [0], color=c_phase, ls="--", lw=1, label="Phase boundary"))
        if best_epoch is not None:
            legend_elements.append(Line2D([0], [0], color=c_best, ls=":", lw=1.2, label=f"Best epoch ({best_epoch+1})"))
        if legend_elements:
            fig.legend(handles=legend_elements, loc="upper center", ncol=len(legend_elements),
                       fontsize=10, framealpha=0.8)

        fig.suptitle("Training Dashboard", fontsize=16, color=_txt(), y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_and_report(model, test_ds, test_loader, class_names, output_dir, device,
                        use_tta=True):
    results_path = Path(output_dir) / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"[Eval] Coletando predicoes (TTA={use_tta})...")
    probs, y_true, y_pred = collect_predictions(model, test_loader, device, use_tta)

    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    rep = classification_report(y_true, y_pred, labels=labels_range,
                                target_names=class_names, digits=4, zero_division=0)
    print(rep)
    with open(results_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)

    print("[Eval] Gerando plots...")
    plot_confusion_matrix(cm, class_names, str(results_path / "confusion_matrix.png"))
    plot_per_class_f1(y_true, y_pred, class_names, str(results_path / "f1_per_class.png"))
    plot_roc_curves(probs, y_true, class_names, str(results_path / "roc_curves.png"))
    plot_calibration(probs, y_true, str(results_path / "calibration_curve.png"))
    plot_confidence_histogram(probs, y_true, str(results_path / "confidence_histogram.png"))
    plot_error_grid(probs, y_true, y_pred, test_ds, class_names,
                    str(results_path / "error_grid.png"))

    print("[Eval] Extraindo embeddings pra t-SNE...")
    embeddings, emb_labels = extract_embeddings(model, test_loader, device)
    plot_tsne(embeddings, emb_labels, class_names, str(results_path / "tsne.png"))

    print("[Eval] Gerando GradCAM samples...")
    plot_gradcam_samples(model, test_ds, class_names, output_dir, device)

    return cm, probs, y_true, y_pred


# ---------------------------------------------------------------------------
# Config dump
# ---------------------------------------------------------------------------

def save_config(args, device, model, output_dir):
    config = {
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        config["gpu_name"] = torch.cuda.get_device_name(0)
        config["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem / 1e6
        config["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        config["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        config["tf32_enabled"] = torch.backends.cuda.matmul.allow_tf32

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_params"] = total_p
    config["trainable_params"] = train_p

    with open(Path(output_dir) / "models" / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(model, img_size, output_dir, device):
    try:
        raw = get_base_model(model)
        raw.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        if device.type == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last)
        onnx_path = Path(output_dir) / "models" / "best_model.onnx"
        torch.onnx.export(raw, dummy, str(onnx_path), opset_version=17,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    global BW_MODE
    BW_MODE = args.bw
    set_seed(args.seed)
    setup_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark} | TF32: {torch.backends.cuda.matmul.allow_tf32}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    Path(args.output_dir, "models").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "results").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "history").mkdir(parents=True, exist_ok=True)

    files, labels, class_names = list_images_and_labels(args.data_dir)
    print(f"Classes: {class_names} ({len(files)} images)")
    print(f"Effective batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = stratified_split(
        files, labels, args.seed, args.val_split, args.test_split
    )
    num_classes = len(class_names)

    train_ds = ImageDataset(tr_x, tr_y, num_classes, get_train_transform(args.img_size))
    val_ds = ImageDataset(va_x, va_y, num_classes, get_eval_transform(args.img_size))
    test_ds = ImageDataset(te_x, te_y, num_classes, get_eval_transform(args.img_size))

    pin = device.type == "cuda"
    nw = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=pin, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=pin, persistent_workers=True)

    class_weights = compute_class_weights(tr_y, num_classes).to(device)
    steps_per_epoch = math.ceil(len(tr_x) / args.batch_size)
    optim_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)
    model_path_best = Path(args.output_dir) / "models" / "best_model.pt"
    model_path_latest = Path(args.output_dir) / "models" / "latest_model.pt"

    if args.resume and model_path_best.exists():
        print(f"Resuming from {model_path_best}")
        model = build_model(num_classes, device)
        model.load_state_dict(torch.load(model_path_best, map_location=device, weights_only=True))
    else:
        model = build_model(num_classes, device)

    if hasattr(torch, "compile") and device.type == "cuda" and os.name != "nt":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile unavailable: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params:,} total | {trainable_params:,} trainable")

    if args.eval_only:
        # Skip training, just load model and evaluate
        print("\n[eval_only] Skipping training, loading model for evaluation...")
        eval_model = build_model(num_classes, device)
        eval_model.load_state_dict(torch.load(model_path_best, map_location=device, weights_only=True))
        history = None
        phase_bounds = None
        best_epoch = None

        # Try to load saved history for plots
        hist_path = Path(args.output_dir) / "history" / "training_history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            best_epoch = int(np.argmin(history["val_loss"])) if history.get("val_loss") else None
            print(f"[eval_only] Loaded training history ({len(history.get('loss', []))} epochs)")
    else:
        epochs_p1 = max(1, int(0.4 * args.epochs))
        epochs_p2 = max(1, int(0.3 * args.epochs))
        epochs_p3 = max(1, args.epochs - epochs_p1 - epochs_p2)

        # ==================== Phase 1: Head ====================
        print(f"\n{'='*60}")
        print(f"PHASE 1: Head Training ({epochs_p1} epochs)")
        print(f"{'='*60}")

        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=3e-4, weight_decay=1e-3
        )
        sch1 = warmup_cosine_schedule(opt1, 2 * optim_steps_per_epoch, epochs_p1 * optim_steps_per_epoch)

        hist1 = train_phase(
            model, train_loader, val_loader, opt1, sch1, scaler,
            class_weights, label_smoothing=0.1, device=device, epochs=epochs_p1,
            grad_accum_steps=args.grad_accum,
            patience=15, best_model_path=model_path_best, use_mix=True
        )

        # ==================== Phase 2: Fine-tune ====================
        print(f"\n{'='*60}")
        print(f"PHASE 2: Fine-tune ({epochs_p2} epochs)")
        print(f"{'='*60}")

        unfreeze_backbone(model)
        print(f"Trainable after unfreeze: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and "classifier" not in n]
        classifier_params = [p for n, p in model.named_parameters()
                             if p.requires_grad and "classifier" in n]

        opt2 = torch.optim.AdamW([
            {"params": backbone_params, "lr": 5e-6, "weight_decay": 5e-2},
            {"params": classifier_params, "lr": 5e-5, "weight_decay": 1e-3},
        ])
        sch2 = warmup_cosine_schedule(opt2, 2 * optim_steps_per_epoch, epochs_p2 * optim_steps_per_epoch)

        hist2 = train_phase(
            model, train_loader, val_loader, opt2, sch2, scaler,
            class_weights, label_smoothing=0.05, device=device, epochs=epochs_p2,
            grad_accum_steps=args.grad_accum,
            patience=15, best_model_path=model_path_best, use_mix=True
        )

        # ==================== Phase 3: EMA ====================
        print(f"\n{'='*60}")
        print(f"PHASE 3: EMA ({epochs_p3} epochs)")
        print(f"{'='*60}")

        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and "classifier" not in n]
        classifier_params = [p for n, p in model.named_parameters()
                             if p.requires_grad and "classifier" in n]

        opt3 = torch.optim.AdamW([
            {"params": backbone_params, "lr": 2e-6, "weight_decay": 5e-2},
            {"params": classifier_params, "lr": 2e-5, "weight_decay": 1e-3},
        ])
        sch3 = warmup_cosine_schedule(opt3, optim_steps_per_epoch, epochs_p3 * optim_steps_per_epoch)
        ema = EMAModel(model, decay=0.999)

        hist3 = train_phase(
            model, train_loader, val_loader, opt3, sch3, scaler,
            class_weights, label_smoothing=0.05, device=device, epochs=epochs_p3,
            grad_accum_steps=args.grad_accum,
            patience=None, best_model_path=None, use_mix=True, ema=ema
        )

        ema.apply(model)
        torch.save(model.state_dict(), model_path_latest)

        ema_val_loss, _ = evaluate_loss(model, val_loader, class_weights, 0.05, device)
        print(f"EMA val_loss: {ema_val_loss:.4f}")

        if model_path_best.exists():
            best_model = build_model(num_classes, device)
            best_model.load_state_dict(torch.load(model_path_best, map_location=device, weights_only=True))
            best_val_loss, _ = evaluate_loss(best_model, val_loader, class_weights, 0.05, device)
            print(f"Best checkpoint val_loss: {best_val_loss:.4f}")
            if ema_val_loss < best_val_loss:
                print("EMA is better — saving")
                torch.save(model.state_dict(), model_path_best)
                eval_model = model
            else:
                print("Checkpoint is better — keeping")
                eval_model = best_model
        else:
            torch.save(model.state_dict(), model_path_best)
            eval_model = model

        # Histories & plots
        history = merge_histories(hist1, hist2, hist3)
        phase_bounds = [len(hist1["loss"]), len(hist1["loss"]) + len(hist2["loss"])]
        best_epoch = int(np.argmin(history["val_loss"])) if history["val_loss"] else None

        save_training_history(history, args.output_dir)

    if history:
        plot_history(history, args.output_dir, phase_bounds, best_epoch)

    # Full evaluation
    cm, probs, y_true, y_pred = evaluate_and_report(
        eval_model, test_ds, test_loader, class_names, args.output_dir, device, use_tta=True
    )

    # Dashboard
    if history:
        print("[Eval] Gerando dashboard...")
        plot_dashboard(history, cm, class_names, y_true, y_pred,
                       str(Path(args.output_dir) / "results" / "dashboard.png"),
                       phase_bounds, best_epoch)

    export_onnx(eval_model, args.img_size, args.output_dir, device)

    # Save metadata & config
    meta = {"class_names": class_names, "img_size": args.img_size, "num_classes": num_classes}
    with open(Path(args.output_dir) / "models" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    save_config(args, device, eval_model, args.output_dir)

    if device.type == "cuda":
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
