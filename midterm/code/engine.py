import os
import math
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from midterm.code.backward import model_backward
from midterm.code.checkpoint import save_checkpoint
from midterm.code.data import augment_batch, create_batches, load_csv_dataset, make_tta_batch, one_hot
from midterm.code.metrics import (
    classification_metrics,
    compute_loss,
    print_classification_report,
    print_confusion_matrix,
    print_metrics_summary,
)
from midterm.code.optimizers import Adam
from midterm.models.CNN import OptimizedCNN, model_forward


def compute_top_k_accuracy(AL, y_true, k=3):
    top_k = np.argsort(AL, axis=1)[:, -k:]
    return float(np.mean([int(label) in top_k[i] for i, label in enumerate(y_true)]))


def predict_arrays(X, parameters, image_size=(64, 64), num_classes=5, tta=False):
    if tta:
        probs = []
        for X_variant in make_tta_batch(X):
            AL_variant, _ = model_forward(
                X_variant,
                parameters,
                input_shape=(*image_size, X.shape[-1]),
                num_classes=num_classes,
            )
            probs.append(AL_variant)
        AL = np.mean(probs, axis=0).astype(np.float32)
    else:
        AL, _ = model_forward(
            X,
            parameters,
            input_shape=(*image_size, X.shape[-1]),
            num_classes=num_classes,
        )
    return np.argmax(AL, axis=1), AL


def evaluate_metrics(X, y, parameters, image_size=(64, 64), num_classes=5, batch_size=16, tta=False):
    losses = []
    preds = []
    probs = []

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]
        pred_batch, AL_batch = predict_arrays(
            X_batch,
            parameters,
            image_size=image_size,
            num_classes=num_classes,
            tta=tta,
        )
        losses.append(compute_loss(AL_batch, one_hot(y_batch, num_classes)) * len(X_batch))
        preds.append(pred_batch)
        probs.append(AL_batch)

    y_pred = np.concatenate(preds, axis=0)
    AL = np.concatenate(probs, axis=0)
    loss = float(np.sum(losses) / len(X))
    acc = compute_accuracy(pred_all, y)
    top3 = compute_top_k_accuracy(AL_all, y, k=min(3, num_classes))
    return loss, acc, top3


def evaluate_metrics(X, y, parameters, image_size=(64, 64), num_classes=10, batch_size=32):
    pred, AL = predict_arrays(X, parameters, image_size=image_size, num_classes=num_classes)
    loss = compute_loss(AL, one_hot(y, num_classes))
    metrics = classification_metrics(y, pred, num_classes)
    return loss, metrics, pred


def predict_arrays(X, parameters, image_size=(64, 64), num_classes=10):
    AL, _ = model_forward(
        X,
        parameters,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
    )
    return np.argmax(AL, axis=1), AL

def train_model(
    train_csv,
    val_csv,
    num_classes=5,
    epochs=50,
    batch_size=16,
    learning_rate=0.005,
    image_size=(64, 64),
    seed=42,
    augment=True,
    normalize=False,
    dropout_keep_prob=0.85,
    weight_decay=1e-4,
    clip_norm=5.0,
    patience=16,
    min_delta=1e-4,
    train_acc_sample=256,
    latest_checkpoint_path=None,
    report_interval=4,
    keep_aspect=True,
    add_structure=False,
    limit=None,
    checkpoint_path="midterm/outputs/best.pkl",
    param_log_interval=0,
    monitor="val_macro_f1",
    dropout_rate=None,
    early_stopping_patience=None,
    label_smoothing=0.0,
):
    if dropout_rate is not None:
        dropout_keep_prob = 1.0 - float(dropout_rate)
    if early_stopping_patience is not None:
        patience = early_stopping_patience

    print("Loading dataset...")
    X_train, y_train, class_names = load_csv_dataset(
        train_csv,
        image_size=image_size,
        limit=limit,
        normalize=normalize,
        keep_aspect=keep_aspect,
        add_structure=add_structure,
    )
    X_val, y_val, _ = load_csv_dataset(
        val_csv,
        image_size=image_size,
        limit=limit,
        normalize=normalize,
        keep_aspect=keep_aspect,
        add_structure=add_structure,
    )

    inferred_classes = infer_num_classes(y_train, y_val)
    if num_classes != inferred_classes:
        print(f"[WARN] num_classes={num_classes} does not match dataset labels. Using {inferred_classes} classes.")
        num_classes = inferred_classes
    class_names = class_names[:num_classes]

    print(
        f"Train: {len(X_train)} images | Val: {len(X_val)} images | "
        f"Image size: {image_size} | Classes: {num_classes}"
    )

    model = OptimizedCNN(input_shape=(*image_size, X_train.shape[-1]), num_classes=num_classes, seed=seed)
    parameters = model.get_parameters()
    optimizer = GradientDescent(
        parameters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
    )

    if latest_checkpoint_path is None:
        latest_checkpoint_path = checkpoint_path.replace("best", "latest")

    y_train_one_hot = one_hot(y_train, num_classes)
    if label_smoothing > 0.0:
        y_train_one_hot = y_train_one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes
    best_val_acc = -1.0
    best_val_loss = np.inf
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    update_count = 0
    history = []
    update_count = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        steps = int(math.ceil(len(X_train) / batch_size))
        lr_now = learning_rate * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(epochs, 1)))
        optimizer.set_learning_rate(lr_now)

        batch_iter = create_batches(X_train, Y_train, batch_size, seed=seed + epoch)
        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=steps, desc=f"Epoch {epoch:03d}/{epochs}")

        for step, (X_batch, Y_batch) in enumerate(batch_iter, start=1):
            if augment:
                X_batch = augment_batch(X_batch, seed=seed + epoch * 1000 + step)

            AL, caches = model_forward(
                X_batch,
                parameters,
                input_shape=(*image_size, X_batch.shape[-1]),
                num_classes=num_classes,
                training=True,
                dropout_keep_prob=dropout_keep_prob,
                seed=seed + epoch * 1000 + step,
            )
            loss = compute_loss(AL, Y_batch)
            grads = model_backward(AL, Y_batch, caches)
            parameters = optimizer.step(parameters, grads)
            update_count += 1
            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)
        sample_size = min(train_acc_sample, len(X_train))
        train_loss_eval, train_metrics, _ = evaluate_metrics(
            X_train[:sample_size],
            y_train[:sample_size],
            parameters,
            image_size=image_size,
            num_classes=num_classes,
            batch_size=batch_size,
            tta=False,
        )
        val_loss, val_metrics, val_pred, _ = evaluate_metrics(
            X_val,
            y_val,
            parameters,
            image_size=image_size,
            num_classes=num_classes,
            batch_size=batch_size,
            tta=False,
        )

        train_acc = train_metrics["accuracy"]
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc(sample): {train_acc * 100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:5.2f}% | "
            f"Val Macro F1: {val_macro_f1 * 100:5.2f}% | {elapsed:.1f}s"
        )

        if report_interval and (epoch == 1 or epoch % report_interval == 0):
            print_metrics_summary("Train(sample)", train_eval_loss, train_metrics)
            print_metrics_summary("Val", val_loss, val_metrics)
            print_confusion_matrix(val_metrics, class_names=class_names)

        improved = (
            val_macro_f1 > best_val_f1 + min_delta
            or (
                abs(val_macro_f1 - best_val_f1) <= min_delta
                and val_acc > best_val_acc + min_delta
            )
            or (
                abs(val_macro_f1 - best_val_f1) <= min_delta
                and abs(val_acc - best_val_acc) <= min_delta
                and val_loss < best_val_loss - min_delta
            )
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": float(train_eval_loss),
            "train_acc_sample": float(train_acc),
            "train_macro_f1_sample": float(train_metrics["macro_f1"]),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
        }
        history.append(epoch_metrics)

        metadata = {
            "image_size": tuple(image_size),
            "num_classes": num_classes,
            "normalize": normalize,
            "keep_aspect": keep_aspect,
            "add_structure": add_structure,
            "architecture": "rgb-conv5x5-16-pool-conv3x3-32-pool-conv3x3-64-pool-flatten-dense128",
            "input_channels": int(X_train.shape[-1]),
            "dropout_keep_prob": dropout_keep_prob,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "label_smoothing": label_smoothing,
            "monitor": monitor,
            "metrics": epoch_metrics,
            "history": history.copy(),
            "best_epoch": best_epoch,
            "best_val_acc": float(max(best_val_acc, val_acc)),
            "best_val_macro_f1": float(max(best_val_f1, val_macro_f1)),
        }
        save_checkpoint(
            parameters,
            epoch,
            val_loss,
            val_acc,
            class_names,
            latest_checkpoint_path,
            image_size=image_size,
            num_classes=num_classes,
            metadata={**metadata, "is_best": False},
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                parameters,
                epoch,
                val_loss,
                val_acc,
                class_names,
                checkpoint_path,
                image_size=image_size,
                num_classes=num_classes,
                metadata={
                    **metadata,
                    "is_best": True,
                    "best_epoch": best_epoch,
                    "best_val_acc": float(best_val_acc),
                    "best_val_loss": float(best_val_loss),
                    "best_val_macro_f1": float(best_val_f1),
                },
            )
            print(
                f"Saved best checkpoint: {checkpoint_path} | "
                f"Val Acc: {best_val_acc * 100:.2f}% | Val Macro F1: {best_val_f1 * 100:.2f}%"
            )
            print_classification_report(val_metrics, class_names=class_names)
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"No validation improvement for {patience} epochs. "
                    f"Best epoch: {best_epoch}"
                )
                break

    print(
        f"Done. Best Val Acc = {best_val_acc * 100:.2f}% | "
        f"Best Val Macro F1 = {best_val_f1 * 100:.2f}% at epoch {best_epoch}"
    )
    return parameters


def evaluate_csv(
    test_csv,
    parameters,
    num_classes=5,
    image_size=(64, 64),
    batch_size=16,
    normalize=False,
    keep_aspect=True,
    add_structure=False,
    tta=True,
):
    X_test, y_test, class_names = load_csv_dataset(
        test_csv,
        image_size=image_size,
        normalize=normalize,
        keep_aspect=keep_aspect,
        add_structure=add_structure,
    )
    loss, metrics, _, _ = evaluate_metrics(
        X_test,
        y_test,
        parameters,
        image_size=image_size,
        num_classes=num_classes,
        batch_size=batch_size,
        tta=tta,
    )
    print_metrics_summary("Test", loss, metrics)
    print_classification_report(metrics, class_names=class_names)
    print_confusion_matrix(metrics, class_names=class_names)
    return loss, metrics["accuracy"]


def predict_image(image, parameters, input_shape=(64, 64, 3), num_classes=5, tta=True):
    X = image[np.newaxis, ...].astype(np.float32, copy=False)
    if tta:
        probs = []
        for X_variant in make_tta_batch(X):
            AL_variant, _ = model_forward(X_variant, parameters, input_shape=input_shape, num_classes=num_classes)
            probs.append(AL_variant)
        AL = np.mean(probs, axis=0).astype(np.float32)
        pred_idx = int(np.argmax(AL, axis=1)[0])
    else:
        AL, _ = model_forward(X, parameters, input_shape=input_shape, num_classes=num_classes)
        pred_idx = int(np.argmax(AL, axis=1)[0])
    confidence = float(np.max(AL))
    return pred_idx, confidence
