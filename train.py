import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2S, efficientnet_v2
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
    tf.random.set_seed(seed)


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
# Image loading, augmentation, MixUp
# ---------------------------------------------------------------------------

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def augmentation_model():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomBrightness(0.15),
    ])


def make_ds(paths, labels, img_size, batch_size, shuffle, seed, num_classes, augment=None):
    paths = tf.convert_to_tensor(paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(p, y):
        x = load_image(p, img_size)
        if augment is not None:
            x = augment(x, training=True)
        return x, tf.one_hot(y, num_classes, dtype=tf.float32)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def mixup_batch(ds, alpha=0.2):
    def _mixup(x, y):
        batch_size = tf.shape(x)[0]
        lam = tf.random.gamma([batch_size, 1, 1, 1], alpha=alpha, beta=alpha)
        lam_y = tf.reshape(lam, [batch_size, 1])
        indices = tf.random.shuffle(tf.range(batch_size))
        x_mixed = lam * x + (1.0 - lam) * tf.gather(x, indices)
        y_mixed = lam_y * y + (1.0 - lam_y) * tf.gather(y, indices)
        return x_mixed, y_mixed
    return ds.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(labels: List[int], num_classes: int) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = {i: float(total) / (num_classes * max(1, counts[i])) for i in range(num_classes)}
    return weights


# ---------------------------------------------------------------------------
# WarmUp + Cosine Decay schedule
# ---------------------------------------------------------------------------

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.base_lr * (step / tf.maximum(warmup, 1.0))
        cos_step = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * tf.minimum(cos_step, 1.0)))
        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    if tf.config.list_physical_devices("GPU"):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    base = EfficientNetV2S(include_top=False, weights="imagenet",
                           input_shape=input_shape, pooling=None)
    base.trainable = False

    inp = layers.Input(shape=input_shape)
    x = layers.Rescaling(255.0, offset=0.0)(inp)
    x = efficientnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(inp, out)
    return model


def get_backbone(model):
    for l in model.layers:
        if isinstance(l, tf.keras.Model) and l.name.startswith("efficientnetv2"):
            return l
    return None


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------

def compile_phase1(model, epochs, steps_per_epoch, warmup_epochs=5):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = warmup_epochs * steps_per_epoch
    schedule = WarmUpCosineDecay(base_lr=3e-4, total_steps=total_steps,
                                warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-5,
                                         clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])


def compile_phase2(model, epochs, steps_per_epoch, warmup_epochs=3):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = warmup_epochs * steps_per_epoch
    schedule = WarmUpCosineDecay(base_lr=1e-5, total_steps=total_steps,
                                warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-5,
                                         clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])


def compile_phase3(model):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5,
                                         clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])


# ---------------------------------------------------------------------------
# SWA Callback
# ---------------------------------------------------------------------------

class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.swa_weights = None
        self.n = 0

    def on_epoch_end(self, epoch, logs=None):
        current_weights = self.model.get_weights()
        if self.swa_weights is None:
            self.swa_weights = [w.copy() for w in current_weights]
        else:
            for i, w in enumerate(current_weights):
                self.swa_weights[i] = (self.swa_weights[i] * self.n + w) / (self.n + 1)
        self.n += 1

    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            self.model.set_weights(self.swa_weights)


# ---------------------------------------------------------------------------
# TTA (Test-Time Augmentation)
# ---------------------------------------------------------------------------

def tta_predict(model, images_ds, n_augments=8):
    all_preds = []
    all_labels = []
    for xb, yb in images_ds:
        preds = model.predict(xb, verbose=0)
        all_preds.append(preds)
        all_labels.append(yb.numpy())

    base_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aug_preds_sum = base_preds.copy()
    for _ in range(n_augments - 1):
        aug_batch_preds = []
        for xb, yb in images_ds:
            x_aug = tf.image.random_flip_left_right(xb)
            x_aug = tf.image.random_flip_up_down(x_aug)
            brightness_delta = tf.random.uniform([], -0.05, 0.05)
            x_aug = tf.image.adjust_brightness(x_aug, brightness_delta)
            x_aug = tf.clip_by_value(x_aug, 0.0, 1.0)
            preds = model.predict(x_aug, verbose=0)
            aug_batch_preds.append(preds)
        aug_preds_sum += np.concatenate(aug_batch_preds, axis=0)

    return aug_preds_sum / n_augments, all_labels


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def callbacks_pack(output_dir: str, best_model_path: Path):
    early = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    ckpt = ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss",
                           save_best_only=True, verbose=0)
    return [early, ckpt]


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


def evaluate_and_report(model, test_ds, class_names, output_dir, use_tta=True):
    results_path = Path(output_dir) / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    if use_tta:
        print("[Eval] Running TTA (8 augments)...")
        preds, labels_oh = tta_predict(model, test_ds, n_augments=8)
        y_true = np.argmax(labels_oh, axis=1)
        y_pred = np.argmax(preds, axis=1)
    else:
        y_true, y_pred = [], []
        for xb, yb in test_ds:
            p = model.predict(xb, verbose=0)
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


def export_tflite(saved_model_path: Path, output_dir: str) -> None:
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(Path(output_dir) / "models" / "best_model.tflite", "wb") as f:
            f.write(tflite_model)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    Path(args.output_dir, "models").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "results").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "history").mkdir(parents=True, exist_ok=True)

    files, labels, class_names = list_images_and_labels(args.data_dir)
    print(f"Classes: {class_names} ({len(files)} images)")

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = stratified_split(
        files, labels, args.seed, args.val_split, args.test_split
    )
    num_classes = len(class_names)
    aug = augmentation_model()

    train_ds = make_ds(tr_x, tr_y, args.img_size, args.batch_size, shuffle=True,
                       seed=args.seed, num_classes=num_classes, augment=aug)
    train_ds_mixup = mixup_batch(train_ds, alpha=0.2)
    val_ds = make_ds(va_x, va_y, args.img_size, args.batch_size, shuffle=False,
                     seed=args.seed, num_classes=num_classes, augment=None).cache()
    test_ds = make_ds(te_x, te_y, args.img_size, args.batch_size, shuffle=False,
                      seed=args.seed, num_classes=num_classes, augment=None).cache()

    class_weights = compute_class_weights(tr_y, num_classes)
    steps_per_epoch = math.ceil(len(tr_x) / args.batch_size)
    model_path_best = Path(args.output_dir) / "models" / "best_model.keras"
    model_path_latest = Path(args.output_dir) / "models" / "latest_model.keras"

    # Build or load model
    if model_path_best.exists():
        print(f"Loading existing model from {model_path_best}")
        model = tf.keras.models.load_model(model_path_best)
    else:
        model = build_model((args.img_size, args.img_size, 3), num_classes)

    model.summary(print_fn=lambda x: print(x))

    # Epoch split: 40% head, 30% fine-tune, 30% SWA
    epochs_p1 = max(1, int(0.4 * args.epochs))
    epochs_p2 = max(1, int(0.3 * args.epochs))
    epochs_p3 = max(1, args.epochs - epochs_p1 - epochs_p2)

    base_callbacks = callbacks_pack(args.output_dir, model_path_best)

    # ==================== Phase 1: Head ====================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Head Training ({epochs_p1} epochs)")
    print(f"{'='*60}")

    compile_phase1(model, epochs_p1, steps_per_epoch)
    hist1 = model.fit(
        train_ds_mixup,
        epochs=epochs_p1,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=base_callbacks,
        verbose=1
    ).history

    # ==================== Phase 2: Fine-tune ====================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tune ({epochs_p2} epochs)")
    print(f"{'='*60}")

    backbone = get_backbone(model)
    if backbone is not None:
        backbone.trainable = True
        freeze_upto = max(0, len(backbone.layers) - 150)
        for i, layer in enumerate(backbone.layers):
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = (i >= freeze_upto)

    compile_phase2(model, epochs_p2, steps_per_epoch)
    hist2 = model.fit(
        train_ds_mixup,
        epochs=epochs_p2,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_pack(args.output_dir, model_path_best),
        verbose=1
    ).history

    # ==================== Phase 3: SWA ====================
    print(f"\n{'='*60}")
    print(f"PHASE 3: SWA ({epochs_p3} epochs)")
    print(f"{'='*60}")

    compile_phase3(model)
    swa = SWACallback()
    hist3 = model.fit(
        train_ds_mixup,
        epochs=epochs_p3,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[swa],
        verbose=1
    ).history

    # Save SWA model
    model.save(model_path_latest)

    # Compare SWA vs best checkpoint
    swa_val = model.evaluate(val_ds, verbose=0)
    swa_val_loss = swa_val[0]
    print(f"SWA val_loss: {swa_val_loss:.4f}")

    if model_path_best.exists():
        best_model = tf.keras.models.load_model(model_path_best)
        best_val = best_model.evaluate(val_ds, verbose=0)
        best_val_loss = best_val[0]
        print(f"Best checkpoint val_loss: {best_val_loss:.4f}")
        if swa_val_loss < best_val_loss:
            print("SWA model is better — saving as best")
            model.save(model_path_best)
            eval_model = model
        else:
            print("Checkpoint model is better — keeping it")
            eval_model = best_model
    else:
        model.save(model_path_best)
        eval_model = model

    # Combine histories
    history_combined = merge_histories(hist1, hist2, hist3)
    save_training_history(history_combined, args.output_dir)
    plot_history(history_combined, args.output_dir)

    # Evaluate with TTA
    evaluate_and_report(eval_model, test_ds, class_names, args.output_dir, use_tta=True)
    export_tflite(model_path_best, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
