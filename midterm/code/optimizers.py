import numpy as np


class Adam:
    def __init__(self, parameters, learning_rate=0.0003, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}

    def step(self, parameters, grads):
        self.t += 1

        for key in parameters:
            grad = grads[f"d{key}"]
            if self.weight_decay > 0.0 and key.startswith("W"):
                grad = grad + self.weight_decay * parameters[key]

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            parameters[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return parameters

