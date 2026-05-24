import numpy as np


def compute_loss(AL, Y):
    return -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[0]


def compute_accuracy(y_pred, y_true):
    return float(np.mean(y_pred.astype(int) == y_true.astype(int)))


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1

    return matrix


def classification_metrics(y_true, y_pred, num_classes):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = np.sum(matrix)
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class = []
    for label in range(num_classes):
        tp = float(matrix[label, label])
        fp = float(np.sum(matrix[:, label]) - tp)
        fn = float(np.sum(matrix[label, :]) - tp)
        support = int(np.sum(matrix[label, :]))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        per_class.append({
            "label": label,
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(recall),
        })

    supports = np.array([item["support"] for item in per_class], dtype=np.float64)
    f1_values = np.array([item["f1"] for item in per_class], dtype=np.float64)
    precision_values = np.array([item["precision"] for item in per_class], dtype=np.float64)
    recall_values = np.array([item["recall"] for item in per_class], dtype=np.float64)

    valid = supports > 0
    weighted_den = np.sum(supports) + 1e-8

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values[valid])) if np.any(valid) else 0.0,
        "macro_recall": float(np.mean(recall_values[valid])) if np.any(valid) else 0.0,
        "macro_f1": float(np.mean(f1_values[valid])) if np.any(valid) else 0.0,
        "weighted_f1": float(np.sum(f1_values * supports) / weighted_den),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def print_metrics_summary(prefix, loss, metrics):
    print(
        f"{prefix} Loss: {loss:.4f} | "
        f"Acc: {metrics['accuracy'] * 100:.2f}% | "
        f"Macro F1: {metrics['macro_f1'] * 100:.2f}% | "
        f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%"
    )


def print_classification_report(metrics, class_names=None):
    print("Per-class metrics:")
    for item in metrics["per_class"]:
        label = item["label"]
        if item["support"] == 0:
            continue

        name = class_names[label] if class_names and label < len(class_names) else str(label)
        print(
            f"  {label:02d} {name}: "
            f"P {item['precision'] * 100:6.2f}% | "
            f"R {item['recall'] * 100:6.2f}% | "
            f"F1 {item['f1'] * 100:6.2f}% | "
            f"N {item['support']}"
        )


def print_confusion_matrix(metrics, class_names=None):
    matrix = metrics["confusion_matrix"]
    labels = [
        str(i) if not class_names or i >= len(class_names) else class_names[i].replace("Fruits_", "")[:10]
        for i in range(matrix.shape[0])
    ]
    print("Confusion matrix (rows=true, cols=pred):")
    print("true\\pred " + " ".join(f"{label:>10}" for label in labels))
    for i, row in enumerate(matrix):
        print(f"{labels[i]:>9} " + " ".join(f"{int(value):10d}" for value in row))

