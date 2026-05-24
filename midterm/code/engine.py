import os

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from midterm.code.backward import model_backward
from midterm.code.checkpoint import save_checkpoint
from midterm.code.data import augment_batch, create_batches, load_csv_dataset, one_hot
from midterm.code.metrics import (
    classification_metrics,
    compute_accuracy,
    compute_loss,
    print_classification_report,
    print_confusion_matrix,
    print_metrics_summary,
)
from midterm.code.optimizers import Adam
from midterm.models.CNN import CNN, model_forward, predict


def evaluate_arrays(X, y, parameters, image_size=(64, 64), num_classes=10):
    Y = one_hot(y, num_classes)
    AL, _ = model_forward(
        X,
        parameters,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
    )
    loss = compute_loss(AL, Y)
    acc = compute_accuracy(np.argmax(AL, axis=1), y)
    return loss, acc


def evaluate_metrics(X, y, parameters, image_size=(64, 64), num_classes=10):
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


def print_per_class_report(y_true, y_pred, class_names=None, num_classes=10):
    metrics = classification_metrics(y_true, y_pred, num_classes)
    print_classification_report(metrics, class_names=class_names)


def train_model(
    train_csv,
    val_csv,
    num_classes=10,
    epochs=20,
    batch_size=8,
    learning_rate=0.0003,
    checkpoint_path="midterm/outputs/best.pkl",
    image_size=(64, 64),
    seed=42,
    augment=True,
    normalize=True,
    dropout_keep_prob=0.65,
    weight_decay=1e-4,
    patience=8,
    min_delta=1e-4,
    train_acc_sample=128,
    latest_checkpoint_path=None,
    report_interval=5,
    keep_aspect=True,
    limit=None,
):
    print("Dang load du lieu...")
    X_train, y_train, class_names = load_csv_dataset(
        train_csv,
        image_size=image_size,
        limit=limit,
        normalize=normalize,
        keep_aspect=keep_aspect,
    )
    X_val, y_val, _ = load_csv_dataset(
        val_csv,
        image_size=image_size,
        limit=limit,
        normalize=normalize,
        keep_aspect=keep_aspect,
    )
    Y_train = one_hot(y_train, num_classes)

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = CNN(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes, seed=seed)
    parameters = model.get_parameters()
    optimizer = Adam(parameters, learning_rate=learning_rate, weight_decay=weight_decay)
    best_val_acc = -1.0
    best_val_loss = np.inf
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if latest_checkpoint_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pkl")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_iter = create_batches(X_train, Y_train, batch_size, seed=seed + epoch)
        steps = int(np.ceil(len(X_train) / batch_size))

        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=steps, desc=f"Epoch {epoch}/{epochs}")

        for step, (X_batch, Y_batch) in enumerate(batch_iter):
            if augment:
                X_batch = augment_batch(X_batch, seed=seed + epoch * 1000 + step)

            AL, caches = model_forward(
                X_batch,
                parameters,
                input_shape=(image_size[0], image_size[1], 3),
                num_classes=num_classes,
                training=True,
                dropout_keep_prob=dropout_keep_prob,
                seed=seed + epoch * 1000 + step,
            )
            loss = compute_loss(AL, Y_batch)
            grads = model_backward(AL, Y_batch, caches)
            parameters = optimizer.step(parameters, grads)
            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)

        sample_size = min(train_acc_sample, len(X_train))
        train_loss_eval, train_metrics, _ = evaluate_metrics(
            X_train[:sample_size],
            y_train[:sample_size],
            parameters,
            image_size=image_size,
            num_classes=num_classes,
        )
        val_loss, val_metrics, val_pred = evaluate_metrics(
            X_val,
            y_val,
            parameters,
            image_size=image_size,
            num_classes=num_classes,
        )
        train_acc = train_metrics["accuracy"]
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Eval Loss(sample): {train_loss_eval:.4f} | "
            f"Train Acc(sample): {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Macro F1: {val_macro_f1 * 100:.2f}%"
        )

        if report_interval and (epoch == 1 or epoch % report_interval == 0):
            print_metrics_summary("Train(sample)", train_loss_eval, train_metrics)
            print_metrics_summary("Val", val_loss, val_metrics)

        improved = (
            val_acc > best_val_acc + min_delta
            or (
                abs(val_acc - best_val_acc) <= min_delta
                and val_macro_f1 > best_val_f1 + min_delta
            )
            or (
                abs(val_acc - best_val_acc) <= min_delta
                and abs(val_macro_f1 - best_val_f1) <= min_delta
                and val_loss < best_val_loss - min_delta
            )
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_eval_loss": float(train_loss_eval),
            "train_acc_sample": float(train_acc),
            "train_macro_f1_sample": float(train_metrics["macro_f1"]),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
        }
        history.append(epoch_metrics)

        common_metadata = {
            "image_size": image_size,
            "normalize": normalize,
                    "architecture": "conv16-pool-conv32-pool-conv64-pool-flatten-dense128",
            "dropout_keep_prob": dropout_keep_prob,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "keep_aspect": keep_aspect,
            "metrics": epoch_metrics,
            "val_metrics": val_metrics,
            "history": history.copy(),
            "best_epoch": best_epoch,
            "best_val_acc": float(max(best_val_acc, val_acc)),
        }

        latest_path = save_checkpoint(
            parameters,
            epoch,
            val_loss,
            val_acc,
            class_names,
            latest_checkpoint_path,
            metadata={**common_metadata, "is_best": False},
        )
        print(f"Saved latest checkpoint: {latest_path}")

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_path = save_checkpoint(
                parameters,
                epoch,
                val_loss,
                val_acc,
                class_names,
                checkpoint_path,
                metadata={
                    **common_metadata,
                    "is_best": True,
                    "best_epoch": best_epoch,
                    "best_val_acc": float(best_val_acc),
                    "best_val_loss": float(best_val_loss),
                    "best_val_macro_f1": float(best_val_f1),
                },
            )
            print(
                f"Saved best checkpoint: {best_path} | "
                f"Val Acc: {best_val_acc * 100:.2f}% | "
                f"Val Macro F1: {best_val_f1 * 100:.2f}%"
            )
            print_classification_report(val_metrics, class_names=class_names)
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"val did not improve for {patience} epochs. "
                    f"Best Val Acc: {best_val_acc * 100:.2f}% "
                    f"at epoch {best_epoch}"
                )
                break

    return parameters


def evaluate_csv(test_csv, parameters, num_classes=10, image_size=(64, 64), normalize=False, keep_aspect=True):
    X_test, y_test, class_names = load_csv_dataset(
        test_csv,
        image_size=image_size,
        normalize=normalize,
        keep_aspect=keep_aspect,
    )
    loss, metrics, _ = evaluate_metrics(X_test, y_test, parameters, image_size=image_size, num_classes=num_classes)
    print_metrics_summary("Test", loss, metrics)
    print_classification_report(metrics, class_names=class_names)
    print_confusion_matrix(metrics, class_names=class_names)
    acc = metrics["accuracy"]
    return loss, acc


def predict_image(image, parameters, input_shape=(64, 64, 3), num_classes=10):
    X = image[np.newaxis, ...]
    pred_idx = int(predict(X, parameters, input_shape=input_shape, num_classes=num_classes)[0])
    AL, _ = model_forward(X, parameters, input_shape=input_shape, num_classes=num_classes)
    confidence = float(np.max(AL))
    return pred_idx, confidence

