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
from midterm.code.metrics import compute_accuracy, compute_loss, compute_top_k_accuracy
from midterm.code.optimizers import Adam
from midterm.models.CNN import OptimizedCNN, model_forward, predict


def summarize_weights_biases(parameters):
    parts = []
    for key in sorted(parameters):
        value = parameters[key]
        kind = "W" if key.startswith("W") else "b"
        parts.append(
            f"{key}({kind}): mean={value.mean():+.4e}, std={value.std():.4e}, "
            f"min={value.min():+.4e}, max={value.max():+.4e}"
        )
    return "\n".join(parts)


def evaluate_arrays(X, y, parameters, image_size=(96, 96), num_classes=10, batch_size=32, tta=False):
    losses = []
    preds = []
    probs = []

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]
        Y_batch = one_hot(y_batch, num_classes)
        if tta:
            tta_probs = []
            for X_variant in make_tta_batch(X_batch):
                AL_variant, _ = model_forward(
                    X_variant,
                    parameters,
                    input_shape=(*image_size, 3),
                    num_classes=num_classes,
                )
                tta_probs.append(AL_variant)
            AL = np.mean(tta_probs, axis=0).astype(np.float32)
        else:
            AL, _ = model_forward(X_batch, parameters, input_shape=(*image_size, 3), num_classes=num_classes)
        losses.append(compute_loss(AL, Y_batch) * len(X_batch))
        probs.append(AL)
        preds.append(np.argmax(AL, axis=1))

    AL_all = np.concatenate(probs, axis=0)
    pred_all = np.concatenate(preds, axis=0)
    loss = float(np.sum(losses) / len(X))
    acc = compute_accuracy(pred_all, y)
    top3 = compute_top_k_accuracy(AL_all, y, k=min(3, num_classes))
    return loss, acc, top3


def train_model(
    train_csv,
    val_csv,
    num_classes=10,
    epochs=60,
    batch_size=16,
    learning_rate=0.001,
    image_size=(96, 96),
    seed=42,
    augment=True,
    weight_decay=1e-4,
    clip_norm=5.0,
    train_acc_sample=None,
    limit=None,
    checkpoint_path="midterm/outputs/best.pkl",
    param_log_interval="epoch",
    monitor="val_acc",
    dropout_rate=0.25,
    early_stopping_patience=20,
):
    print("Loading dataset...")
    X_train, y_train, class_names = load_csv_dataset(train_csv, image_size=image_size, limit=limit)
    X_val, y_val, _ = load_csv_dataset(val_csv, image_size=image_size, limit=limit)

    print(f"Train: {len(X_train)} images | Val: {len(X_val)} images | Image size: {image_size}")

    model = OptimizedCNN(input_shape=(*image_size, 3), num_classes=num_classes, seed=seed)
    parameters = model.get_parameters()
    optimizer = Adam(
        parameters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
    )

    best_val_acc = -1.0
    best_epoch = 0
    update_count = 0
    y_train_one_hot = one_hot(y_train, num_classes)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        steps = int(math.ceil(len(X_train) / batch_size))

        # Cosine decay keeps early learning fast and late updates more stable.
        lr_now = learning_rate * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(epochs, 1)))
        optimizer.set_learning_rate(lr_now)

        batch_iter = create_batches(X_train, y_train_one_hot, batch_size, seed=seed + epoch)
        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=steps, desc=f"Epoch {epoch:03d}/{epochs}")

        for step, (X_batch, Y_batch) in enumerate(batch_iter, start=1):
            if augment:
                X_batch = augment_batch(X_batch, seed=seed + epoch * 1000 + step)

            AL, caches = model_forward(
                X_batch,
                parameters,
                input_shape=(*image_size, 3),
                num_classes=num_classes,
                training=True,
                dropout_rate=dropout_rate,
                seed=seed + epoch * 1000 + step,
            )
            loss = compute_loss(AL, Y_batch)
            grads = model_backward(AL, Y_batch, caches)
            parameters = optimizer.step(parameters, grads)
            update_count += 1
            total_loss += loss * len(X_batch)

            if isinstance(param_log_interval, int) and param_log_interval > 0:
                if update_count % param_log_interval == 0:
                    print(f"\nUpdate {update_count:05d} W/b summary")
                    print(summarize_weights_biases(parameters))

        train_loss = total_loss / len(X_train)
        sample_size = len(X_train) if train_acc_sample is None else min(train_acc_sample, len(X_train))
        train_eval_loss, train_acc, train_top3 = evaluate_arrays(
            X_train[:sample_size],
            y_train[:sample_size],
            parameters,
            image_size=image_size,
            num_classes=num_classes,
            batch_size=batch_size,
        )
        val_loss, val_acc, val_top3 = evaluate_arrays(
            X_val,
            y_val,
            parameters,
            image_size=image_size,
            num_classes=num_classes,
            batch_size=batch_size,
        )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:5.2f}% | "
            f"Val Top-3: {val_top3 * 100:5.2f}% | {elapsed:.1f}s"
        )

        if param_log_interval == "epoch":
            print("W/b after update")
            print(summarize_weights_biases(parameters))

        score = train_acc if monitor == "train_acc" else val_acc
        best_score = best_val_acc
        if score > best_score:
            best_val_acc = score
            best_epoch = epoch
            save_checkpoint(
                parameters,
                epoch,
                val_loss,
                val_acc,
                class_names,
                checkpoint_path,
                image_size=image_size,
                num_classes=num_classes,
            )
            print(
                f"Saved best model: epoch={epoch}, "
                f"train_acc={train_acc * 100:.2f}%, val_acc={val_acc * 100:.2f}%"
            )
        elif early_stopping_patience and epoch - best_epoch >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"No {monitor} improvement for {early_stopping_patience} epochs."
            )
            break

    metric_name = "Train Acc" if monitor == "train_acc" else "Val Acc"
    print(f"Done. Best {metric_name} = {best_val_acc * 100:.2f}% at epoch {best_epoch}")
    return parameters


def evaluate_csv(test_csv, parameters, num_classes=10, image_size=(96, 96), batch_size=32, tta=True):
    X_test, y_test, _ = load_csv_dataset(test_csv, image_size=image_size)
    loss, acc, top3 = evaluate_arrays(
        X_test,
        y_test,
        parameters,
        image_size=image_size,
        num_classes=num_classes,
        batch_size=batch_size,
        tta=tta,
    )
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}% | Test Top-3: {top3 * 100:.2f}%")
    return loss, acc


def predict_image(image, parameters, input_shape=(96, 96, 3), num_classes=10, tta=True):
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
        pred_idx = int(predict(X, parameters, input_shape=input_shape, num_classes=num_classes)[0])
    confidence = float(np.max(AL))
    return pred_idx, confidence
