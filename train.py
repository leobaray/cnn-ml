import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed / data helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images_and_labels(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    class_dirs = sorted([d for d in os.listdir(data_dir) if (Path(data_dir) / d).is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_dirs)}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
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
    def __init__(self, paths, labels, img_size, num_classes, transform=None):
        self.paths = paths
        self.labels = labels
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = torch.zeros(self.num_classes)
        label[self.labels[idx]] = 1.0
        return img, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(54),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

def mixup_batch(x, y, alpha=0.2):
    batch_size = x.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)
    lam_y = lam.view(batch_size, 1)
    indices = torch.randperm(batch_size, device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[indices]
    y_mixed = lam_y * y + (1.0 - lam_y) * y[indices]
    return x_mixed, y_mixed


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = torch.tensor(
        [float(total) / (num_classes * max(1, c)) for c in counts], dtype=torch.float32
    )
    return weights


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def soft_cross_entropy(logits, targets, class_weights=None, label_smoothing=0.0):
    num_classes = targets.size(-1)
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + label_smoothing / num_classes
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    if class_weights is not None:
        primary_class = targets.argmax(dim=-1)
        loss = loss * class_weights[primary_class]
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
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, device: torch.device) -> nn.Module:
    base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    for param in base.features.parameters():
        param.requires_grad = False

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

    return base.to(device)


def unfreeze_backbone(model):
    all_modules = list(model.features.modules())
    freeze_upto = max(0, len(all_modules) - 150)
    for i, module in enumerate(all_modules):
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in module.parameters():
                p.requires_grad = False
        elif i >= freeze_upto:
            for p in module.parameters(recurse=False):
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, class_weights,
                    label_smoothing, device, use_mixup=True):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if use_mixup:
            xb, yb = mixup_batch(xb, yb, alpha=0.2)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(xb)
            loss = soft_cross_entropy(logits, yb, class_weights, label_smoothing)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(-1) == yb.argmax(-1)).sum().item()
        total += xb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loss(model, loader, class_weights, label_smoothing, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = soft_cross_entropy(logits, yb, class_weights, label_smoothing)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(-1) == yb.argmax(-1)).sum().item()
        total += xb.size(0)

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, optimizer, scheduler, scaler,
                class_weights, label_smoothing, device, epochs, patience=7,
                best_model_path=None, use_mixup=True, swa_callback=None):
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            class_weights, label_smoothing, device, use_mixup
        )
        val_loss, val_acc = evaluate_loss(
            model, val_loader, class_weights, label_smoothing, device
        )

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs} — "
              f"loss: {train_loss:.4f} acc: {train_acc:.4f} "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if swa_callback is not None:
            swa_callback.update(model)

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
# SWA
# ---------------------------------------------------------------------------

class SWACallback:
    def __init__(self):
        self.swa_state = None
        self.n = 0

    def update(self, model):
        current = {k: v.clone().float() for k, v in model.state_dict().items()}
        if self.swa_state is None:
            self.swa_state = current
        else:
            for k in self.swa_state:
                self.swa_state[k] = (self.swa_state[k] * self.n + current[k]) / (self.n + 1)
        self.n += 1

    def apply(self, model):
        if self.swa_state is not None:
            restored = {}
            ref = model.state_dict()
            for k in self.swa_state:
                restored[k] = self.swa_state[k].to(ref[k].dtype)
            model.load_state_dict(restored)


# ---------------------------------------------------------------------------
# TTA (Test-Time Augmentation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def tta_predict(model, dataloader, device, n_augments=8):
    model.eval()
    all_preds = []
    all_labels = []

    for xb, yb in dataloader:
        xb = xb.to(device)
        preds = F.softmax(model(xb), dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())

    base_preds = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    aug_preds_sum = base_preds.copy()

    for _ in range(n_augments - 1):
        aug_batch = []
        for xb, yb in dataloader:
            xb = xb.to(device)
            if random.random() > 0.5:
                xb = torch.flip(xb, [-1])
            if random.random() > 0.5:
                xb = torch.flip(xb, [-2])
            delta = torch.empty(1, device=device).uniform_(-0.05, 0.05)
            xb = xb + delta
            preds = F.softmax(model(xb), dim=-1).cpu().numpy()
            aug_batch.append(preds)
        aug_preds_sum += np.concatenate(aug_batch, axis=0)

    return aug_preds_sum / n_augments, all_labels_np


# ---------------------------------------------------------------------------
# Callbacks helpers
# ---------------------------------------------------------------------------

def callbacks_pack(output_dir: str, best_model_path: Path):
    return {"patience": 7, "best_model_path": best_model_path}


# ---------------------------------------------------------------------------
# History / plotting / evaluation
# ---------------------------------------------------------------------------

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


def plot_history(history: dict, output_dir: str) -> None:
    p = Path(output_dir)
    if "loss" in history and "val_loss" in history:
        plt.figure()
        plt.plot(history["loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p / "results" / "loss_curve.png")
        plt.close()
    if "acc" in history and "val_acc" in history:
        plt.figure()
        plt.plot(history["acc"], label="train_acc")
        plt.plot(history["val_acc"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p / "results" / "acc_curve.png")
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
    plt.savefig(output_path)
    plt.close(fig)


def evaluate_and_report(model, test_loader, class_names, output_dir, device, use_tta=True):
    results_path = Path(output_dir) / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    if use_tta:
        print("[Eval] Running TTA (8 augments)...")
        preds, labels_oh = tta_predict(model, test_loader, device, n_augments=8)
        y_true = np.argmax(labels_oh, axis=1)
        y_pred = np.argmax(preds, axis=1)
    else:
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                p = F.softmax(model(xb), dim=-1).cpu().numpy()
                y_true.extend(np.argmax(yb.numpy(), axis=1))
                y_pred.extend(np.argmax(p, axis=1))

    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    rep = classification_report(y_true, y_pred, labels=labels_range,
                                target_names=class_names, digits=4, zero_division=0)
    print(rep)
    with open(results_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)
    plot_confusion_matrix(cm, class_names, str(results_path / "confusion_matrix.png"))


def export_onnx(model, img_size, output_dir, device):
    try:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        onnx_path = Path(output_dir) / "models" / "best_model.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"ONNX model exported to {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    Path(args.output_dir, "models").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "results").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "history").mkdir(parents=True, exist_ok=True)

    files, labels, class_names = list_images_and_labels(args.data_dir)
    print(f"Classes: {class_names} ({len(files)} images)")

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = stratified_split(
        files, labels, args.seed, args.val_split, args.test_split
    )
    num_classes = len(class_names)

    train_ds = ImageDataset(tr_x, tr_y, args.img_size, num_classes, get_train_transform())
    val_ds = ImageDataset(va_x, va_y, args.img_size, num_classes, get_eval_transform())
    test_ds = ImageDataset(te_x, te_y, args.img_size, num_classes, get_eval_transform())

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin, persistent_workers=True)

    class_weights = compute_class_weights(tr_y, num_classes).to(device)
    steps_per_epoch = math.ceil(len(tr_x) / args.batch_size)
    model_path_best = Path(args.output_dir) / "models" / "best_model.pt"
    model_path_latest = Path(args.output_dir) / "models" / "latest_model.pt"

    if model_path_best.exists():
        print(f"Loading existing model from {model_path_best}")
        model = build_model(num_classes, device)
        model.load_state_dict(torch.load(model_path_best, map_location=device, weights_only=True))
    else:
        model = build_model(num_classes, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params * 4 / 1e6:.1f} MB)")

    # Epoch split: 40% head, 30% fine-tune, 30% SWA
    epochs_p1 = max(1, int(0.4 * args.epochs))
    epochs_p2 = max(1, int(0.3 * args.epochs))
    epochs_p3 = max(1, args.epochs - epochs_p1 - epochs_p2)

    # ==================== Phase 1: Head ====================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Head Training ({epochs_p1} epochs)")
    print(f"{'='*60}")

    total_steps_p1 = epochs_p1 * steps_per_epoch
    warmup_steps_p1 = 5 * steps_per_epoch
    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-5
    )
    scheduler_p1 = warmup_cosine_schedule(optimizer_p1, warmup_steps_p1, total_steps_p1)

    hist1 = train_phase(
        model, train_loader, val_loader, optimizer_p1, scheduler_p1, scaler,
        class_weights, label_smoothing=0.1, device=device, epochs=epochs_p1,
        patience=7, best_model_path=model_path_best, use_mixup=True
    )

    # ==================== Phase 2: Fine-tune ====================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tune ({epochs_p2} epochs)")
    print(f"{'='*60}")

    unfreeze_backbone(model)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreeze: {trainable_p2:,}")

    total_steps_p2 = epochs_p2 * steps_per_epoch
    warmup_steps_p2 = 3 * steps_per_epoch
    optimizer_p2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=1e-5
    )
    scheduler_p2 = warmup_cosine_schedule(optimizer_p2, warmup_steps_p2, total_steps_p2)

    hist2 = train_phase(
        model, train_loader, val_loader, optimizer_p2, scheduler_p2, scaler,
        class_weights, label_smoothing=0.05, device=device, epochs=epochs_p2,
        patience=7, best_model_path=model_path_best, use_mixup=True
    )

    # ==================== Phase 3: SWA ====================
    print(f"\n{'='*60}")
    print(f"PHASE 3: SWA ({epochs_p3} epochs)")
    print(f"{'='*60}")

    optimizer_p3 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-6, weight_decay=1e-5
    )
    swa = SWACallback()

    hist3 = train_phase(
        model, train_loader, val_loader, optimizer_p3, None, scaler,
        class_weights, label_smoothing=0.05, device=device, epochs=epochs_p3,
        patience=None, best_model_path=None, use_mixup=True, swa_callback=swa
    )

    # Apply SWA weights
    swa.apply(model)
    torch.save(model.state_dict(), model_path_latest)

    # Compare SWA vs best checkpoint
    swa_val_loss, _ = evaluate_loss(model, val_loader, class_weights, 0.05, device)
    print(f"SWA val_loss: {swa_val_loss:.4f}")

    if model_path_best.exists():
        best_model = build_model(num_classes, device)
        best_model.load_state_dict(
            torch.load(model_path_best, map_location=device, weights_only=True)
        )
        best_val_loss, _ = evaluate_loss(best_model, val_loader, class_weights, 0.05, device)
        print(f"Best checkpoint val_loss: {best_val_loss:.4f}")
        if swa_val_loss < best_val_loss:
            print("SWA model is better — saving as best")
            torch.save(model.state_dict(), model_path_best)
            eval_model = model
        else:
            print("Checkpoint model is better — keeping it")
            eval_model = best_model
    else:
        torch.save(model.state_dict(), model_path_best)
        eval_model = model

    # Combine histories
    history_combined = merge_histories(hist1, hist2, hist3)
    save_training_history(history_combined, args.output_dir)
    plot_history(history_combined, args.output_dir)

    # Evaluate with TTA
    evaluate_and_report(eval_model, test_loader, class_names, args.output_dir, device, use_tta=True)
    export_onnx(eval_model, args.img_size, args.output_dir, device)

    print("\nDone!")


if __name__ == "__main__":
    main()
