import numpy as np


class GradientDescent:
    """Vanilla gradient descent optimizer.

    Formula:
        W = W - learning_rate * dW
        b = b - learning_rate * db

    This class does not use momentum or adaptive gradients. Weight decay and
    gradient clipping are optional stability helpers and can be set to 0/None.
    """

    def __init__(self, parameters=None, learning_rate=0.001, weight_decay=0.0, clip_norm=None):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def _clip(self, grad):
        if not self.clip_norm:
            return grad
        norm = np.linalg.norm(grad)
        if norm > self.clip_norm:
            return grad * (self.clip_norm / (norm + 1e-12))
        return grad

    def step(self, parameters, grads):
        for key in parameters:
            grad = grads[f"d{key}"].astype(np.float32, copy=False)
            if self.weight_decay > 0.0 and key.startswith("W"):
                grad = grad + self.weight_decay * parameters[key]
            grad = self._clip(grad)
            parameters[key] = (parameters[key] - self.learning_rate * grad).astype(np.float32, copy=False)
        return parameters