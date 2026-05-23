import numpy as np


def compute_loss(AL, Y):
    AL = np.clip(AL, 1e-8, 1.0)
    return float(-np.sum(Y * np.log(AL)) / Y.shape[0])


def compute_accuracy(y_pred, y_true):
    return float(np.mean(y_pred.astype(int) == y_true.astype(int)))


def compute_top_k_accuracy(AL, y_true, k=3):
    y_true = y_true.astype(int)
    top_k = np.argsort(AL, axis=1)[:, -k:]
    return float(np.mean([label in top_k[idx] for idx, label in enumerate(y_true)]))
