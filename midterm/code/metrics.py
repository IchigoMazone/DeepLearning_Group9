import numpy as np


def compute_loss(AL, Y):
    return -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[0]


def compute_accuracy(y_pred, y_true):
    return float(np.mean(y_pred.astype(int) == y_true.astype(int)))

