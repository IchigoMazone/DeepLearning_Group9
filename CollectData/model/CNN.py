import numpy as np


class SimpleCNN:

    def __init__(self):
        self.weights = np.random.randn(64 * 64 * 3, 2) * 0.01
        self.bias = np.zeros((1, 2))

    def softmax(self, x):

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):

        logits = np.dot(x, self.weights) + self.bias

        probs = self.softmax(logits)

        return probs

    def predict(self, x):
        probs = self.forward(x)

        return np.argmax(probs, axis=1)
