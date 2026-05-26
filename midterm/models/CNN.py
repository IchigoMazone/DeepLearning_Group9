import numpy as np

from midterm.code.layers import (
    conv_forward,
    dense_forward,
    dropout_forward,
    global_avg_pool_forward,
    global_max_pool_forward,
    he_init,
    max_pool_forward,
    relu,
    softmax,
    to_numpy_nhwc,
)


class OptimizedCNN:
    """Small NumPy CNN tuned for the current 5-class fruit dataset."""

    def __init__(self, input_shape=(128, 128, 3), num_classes=5, seed=42):
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (height, width, channels)")
        if input_shape[0] % 4 != 0 or input_shape[1] % 4 != 0:
            raise ValueError("height and width must be divisible by 4")

        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.seed = seed
        self.params = self._init_params()

    def _init_params(self):
        _, _, channels = self.input_shape
        rng = np.random.default_rng(self.seed)
        return {
            "W1": he_init(rng, (3, 3, channels, 32), fan_in=3 * 3 * channels),
            "b1": np.zeros((1, 1, 1, 32), dtype=np.float32),
            "W2": he_init(rng, (3, 3, 32, 64), fan_in=3 * 3 * 32),
            "b2": np.zeros((1, 1, 1, 64), dtype=np.float32),
            "W3": he_init(rng, (3, 3, 64, 128), fan_in=3 * 3 * 64),
            "b3": np.zeros((1, 1, 1, 128), dtype=np.float32),
            "W4": he_init(rng, (256, 128), fan_in=256),
            "b4": np.zeros((1, 128), dtype=np.float32),
            "W5": he_init(rng, (128, self.num_classes), fan_in=128),
            "b5": np.zeros((1, self.num_classes), dtype=np.float32),
        }

    def get_parameters(self):
        return self.params

    def set_parameters(self, parameters):
        self.params = parameters

    def forward(self, X, training=False, dropout_keep_prob=1.0, seed=None):
        X = to_numpy_nhwc(X)
        if tuple(X.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {X.shape[1:]}")

        caches = {}
        Z1, caches["conv1"] = conv_forward(X, self.params["W1"], self.params["b1"], stride=1, pad=1)
        A1 = relu(Z1)
        P1, caches["pool1"] = max_pool_forward(A1, f=2, stride=2)

        Z2, caches["conv2"] = conv_forward(P1, self.params["W2"], self.params["b2"], stride=1, pad=1)
        A2 = relu(Z2)
        P2, caches["pool2"] = max_pool_forward(A2, f=2, stride=2)

        Z3, caches["conv3"] = conv_forward(P2, self.params["W3"], self.params["b3"], stride=1, pad=1)
        A3 = relu(Z3)

        G_avg, caches["gap"] = global_avg_pool_forward(A3)
        G_max, caches["gmp"] = global_max_pool_forward(A3)
        G = np.concatenate([G_avg, G_max], axis=1).astype(np.float32, copy=False)

        Z4, caches["dense1"] = dense_forward(G, self.params["W4"], self.params["b4"])
        A4 = relu(Z4)
        A4, caches["dropout1"] = dropout_forward(
            A4,
            keep_prob=dropout_keep_prob,
            seed=seed,
            training=training,
        )

        Z5, caches["dense2"] = dense_forward(A4, self.params["W5"], self.params["b5"])
        AL = softmax(Z5)
        caches.update({"Z1": Z1, "Z2": Z2, "Z3": Z3, "Z4": Z4, "Z5": Z5, "AL": AL})
        return AL, caches

    def predict(self, X):
        AL, _ = self.forward(X, training=False)
        return np.argmax(AL, axis=1)


def init_parameters(num_classes=5, input_shape=(128, 128, 3), seed=42):
    return OptimizedCNN(input_shape=input_shape, num_classes=num_classes, seed=seed).get_parameters()


def model_forward(
    X,
    parameters,
    input_shape=(128, 128, 3),
    num_classes=5,
    training=False,
    dropout_keep_prob=1.0,
    seed=None,
):
    model = OptimizedCNN(input_shape=input_shape, num_classes=num_classes)
    model.set_parameters(parameters)
    return model.forward(X, training=training, dropout_keep_prob=dropout_keep_prob, seed=seed)


def predict(X, parameters, input_shape=(128, 128, 3), num_classes=5):
    model = OptimizedCNN(input_shape=input_shape, num_classes=num_classes)
    model.set_parameters(parameters)
    return model.predict(X)