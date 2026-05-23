import numpy as np


class Adam:
    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
        clip_norm=5.0,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.t = 0
        self.m = {key: np.zeros_like(value, dtype=np.float32) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value, dtype=np.float32) for key, value in parameters.items()}

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def _clip(self, grad):
        if not self.clip_norm:
            return grad
        norm = np.linalg.norm(grad)
        if norm > self.clip_norm:
            grad = grad * (self.clip_norm / (norm + 1e-12))
        return grad

    def step(self, parameters, grads):
        self.t += 1

        for key, value in parameters.items():
            grad_key = f"d{key}"
            if grad_key not in grads:
                continue

            grad = grads[grad_key].astype(np.float32, copy=False)
            if self.weight_decay > 0 and key.startswith("W"):
                grad = grad + self.weight_decay * value
            grad = self._clip(grad)

            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)
            parameters[key] = value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            parameters[key] = parameters[key].astype(np.float32, copy=False)

        return parameters
