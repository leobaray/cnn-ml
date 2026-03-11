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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import EfficientNetV2B0, efficientnet_v2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    default_data_dir = str(base_dir.parent / "datasets")
    default_output_dir = str(base_dir.parent / "output")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser.parse_args()

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
            if f.is_file() and f.suffix.lower() in exts:
                files.append(str(f))
                labels.append(class_to_idx[c])
    return files, labels, class_dirs

def stratified_split(files, labels, seed, val_split, test_split):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        files, labels, test_size=test_split, random_state=seed, stratify=labels
    )
    rel_val = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, random_state=seed, stratify=y_trainval
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

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
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def augmentation_model():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])

def compute_class_weights(labels: List[int], num_classes: int) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = {i: float(total) / (num_classes * max(1, counts[i])) for i in range(num_classes)}
    return weights

def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    if tf.config.list_physical_devices("GPU"):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
    base = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=input_shape, pooling=None)
    base.trainable = False
    inp = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: t * 255.0)(inp)
    x = efficientnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(inp, out)
    return model

def compile_with_scheduler(model, epochs, steps_per_epoch, wd=1e-5, base_lr=1e-4):
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=max(1, epochs * steps_per_epoch),
        alpha=1e-2
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=wd)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
    try:
        from tensorflow_addons.metrics import F1Score
        metrics.append(F1Score(num_classes=model.output_shape[-1], average="macro", name="f1_macro"))
    except Exception:
        pass
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

def compile_finetune(model, steps_per_epoch, wd=5e-6, base_lr=1e-5):
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=max(1, steps_per_epoch * 10),
        alpha=1e-2
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=wd)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
    try:
        from tensorflow_addons.metrics import F1Score
        metrics.append(F1Score(num_classes=model.output_shape[-1], average="macro", name="f1_macro"))
    except Exception:
        pass
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

def callbacks_pack(output_dir: str, best_model_path: Path):
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=0)
    tb = TensorBoard(log_dir=str(Path(output_dir) / "logs"), histogram_freq=1)
    ckpt = ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", save_best_only=True, verbose=0)
    return [early, reduce, tb, ckpt]

def merge_histories(h1: dict, h2: dict) -> dict:
    keys = set(h1.keys()) | set(h2.keys())
    out = {}
    for k in keys:
        out[k] = list(h1.get(k, [])) + list(h2.get(k, []))
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
           title="Matriz de Confusão", ylabel="Classe Verdadeira", xlabel="Classe Predita")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def evaluate_and_report(model: models.Model, test_ds: tf.data.Dataset, class_names: list, output_dir: str) -> None:
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        preds = model.predict(xb, verbose=0)
        y_true.extend(np.argmax(yb.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0)
    results_path = Path(output_dir) / "results"
    results_path.mkdir(parents=True, exist_ok=True)
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

def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)
    Path(args.output_dir, "models").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "logs").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "results").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "history").mkdir(parents=True, exist_ok=True)
    files, labels, class_names = list_images_and_labels(args.data_dir)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = stratified_split(files, labels, args.seed, args.val_split, args.test_split)
    num_classes = len(class_names)
    aug = augmentation_model()
    train_ds = make_ds(tr_x, tr_y, args.img_size, args.batch_size, shuffle=True, seed=args.seed, num_classes=num_classes, augment=aug)
    val_ds = make_ds(va_x, va_y, args.img_size, args.batch_size, shuffle=False, seed=args.seed, num_classes=num_classes, augment=None).cache()
    test_ds = make_ds(te_x, te_y, args.img_size, args.batch_size, shuffle=False, seed=args.seed, num_classes=num_classes, augment=None).cache()
    class_weights = compute_class_weights(tr_y, num_classes)
    steps_per_epoch = math.ceil(len(tr_x) / args.batch_size)
    model_path_best = Path(args.output_dir) / "models" / "best_model.keras"
    model_path_latest = Path(args.output_dir) / "models" / "latest_model.keras"
    if model_path_best.exists():
        model = tf.keras.models.load_model(model_path_best)
    else:
        model = build_model((args.img_size, args.img_size, 3), num_classes)
    epochs_stage1 = max(1, int(0.6 * args.epochs))
    epochs_stage2 = max(0, args.epochs - epochs_stage1)
    compile_with_scheduler(model, epochs_stage1, steps_per_epoch)
    hist1 = model.fit(
        train_ds,
        epochs=epochs_stage1,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_pack(args.output_dir, model_path_best),
        verbose=1
    ).history
    base_model = None
    for l in model.layers:
        if isinstance(l, tf.keras.Model) and l.name.startswith("efficientnetv2"):
            base_model = l
            break
    if base_model is not None:
        base_model.trainable = True
        freeze_upto = max(0, len(base_model.layers) - 100)
        for i, layer in enumerate(base_model.layers):
            trainable = i >= freeze_upto
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = trainable
    if epochs_stage2 > 0:
        compile_finetune(model, steps_per_epoch)
        hist2 = model.fit(
            train_ds,
            epochs=epochs_stage2,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=callbacks_pack(args.output_dir, model_path_best),
            verbose=1
        ).history
        history_combined = merge_histories(hist1, hist2)
    else:
        history_combined = hist1
    save_training_history(history_combined, args.output_dir)
    model.save(model_path_latest)
    if model_path_best.exists():
        model = tf.keras.models.load_model(model_path_best)
    evaluate_and_report(model, test_ds, class_names, args.output_dir)
    export_tflite(model_path_best, args.output_dir)
    plot_history(history_combined, args.output_dir)

if __name__ == "__main__":
    main()
