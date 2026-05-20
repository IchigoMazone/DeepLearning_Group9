import cv2
import numpy as np


def relu(Z):
    """
    Tac dung:
    - Ham kich hoat ReLU.

    Dau vao:
    - Z[np.ndarray]: Du lieu dau vao.

    Dau ra:
    - np.ndarray: Du lieu sau ReLU.
    """

    return np.maximum(0, Z)


def softmax(Z):
    """
    Tac dung:
    - Chuyen logits thanh xac suat du doan.

    Dau vao:
    - Z[np.ndarray]: Logits shape (batch_size, num_classes).

    Dau ra:
    - np.ndarray: Xac suat shape (batch_size, num_classes).
    """

    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def to_numpy_nhwc(X):
    """
    Tac dung:
    - Chuan hoa input ve shape NHWC: (batch_size, height, width, channels).

    Ho tro:
    - NHWC: (batch_size, height, width, channels).
    - NCHW: (batch_size, channels, height, width).

    Ghi chu:
    - Khong import torch trong file model.
    - Neu X la torch.Tensor, ham chi goi detach/cpu/numpy neu object co san.
    """

    if hasattr(X, "detach"):
        X = X.detach()
    if hasattr(X, "cpu"):
        X = X.cpu()
    if hasattr(X, "numpy"):
        X = X.numpy()

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 4:
        raise ValueError(f"X must have 4 dimensions, got shape {X.shape}")

    if X.shape[-1] == 3:
        return X

    if X.shape[1] == 3:
        return np.transpose(X, (0, 2, 3, 1))

    raise ValueError(
        "X must be NHWC (m, H, W, 3) or NCHW (m, 3, H, W), "
        f"got shape {X.shape}"
    )


def he_init(rng, shape, fan_in):
    """
    Tac dung:
    - Khoi tao trong so theo He initialization cho ReLU.
    """

    return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


def conv_forward(A_prev, W, b, stride=1, pad=1):
    """
    Tac dung:
    - Forward qua lop convolution.
    - Su dung cv2.filter2D de tang toc so voi viec truot cua so bang vong lap pixel.

    Dau vao:
    - A_prev[np.ndarray]: Shape (m, H, W, C_prev).
    - W[np.ndarray]: Shape (f, f, C_prev, C_out).
    - b[np.ndarray]: Shape (1, 1, 1, C_out).
    - stride[int]: Buoc truot.
    - pad[int]: Padding.

    Dau ra:
    - Z[np.ndarray]: Output shape (m, H_out, W_out, C_out).
    - cache[tuple]: Cache phuc vu backward sau nay.
    """

    m, H_prev, W_prev, C_prev = A_prev.shape
    f, _, C_filter, C_out = W.shape

    if C_prev != C_filter:
        raise ValueError(f"Input channels and filter channels do not match: {C_prev} != {C_filter}")

    H_out = int((H_prev + 2 * pad - f) / stride) + 1
    W_out = int((W_prev + 2 * pad - f) / stride) + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError("Invalid conv output shape. Check input size, filter size, stride and pad.")

    Z = np.zeros((m, H_out, W_out, C_out), dtype=np.float32)

    for i in range(m):
        image = A_prev[i]

        if pad > 0:
            image = cv2.copyMakeBorder(
                image,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )

        for c_out in range(C_out):
            feature_map = np.zeros(image.shape[:2], dtype=np.float32)

            for c_in in range(C_prev):
                kernel = W[:, :, c_in, c_out].astype(np.float32)
                channel = image[:, :, c_in].astype(np.float32)
                feature_map += cv2.filter2D(
                    channel,
                    ddepth=-1,
                    kernel=kernel,
                    borderType=cv2.BORDER_CONSTANT,
                )

            start = f // 2
            end_h = start + H_out * stride
            end_w = start + W_out * stride
            Z[i, :, :, c_out] = feature_map[start:end_h:stride, start:end_w:stride] + b[0, 0, 0, c_out]

    cache = (A_prev, W, b, stride, pad)
    return Z, cache


def max_pool_forward(A_prev, f=2, stride=2):
    """
    Tac dung:
    - Forward qua lop max pooling.

    Dau vao:
    - A_prev[np.ndarray]: Shape (m, H, W, C).
    - f[int]: Kich thuoc cua so pooling.
    - stride[int]: Buoc truot.

    Dau ra:
    - A[np.ndarray]: Output sau pooling.
    - cache[tuple]: Cache phuc vu backward sau nay.
    """

    m, H_prev, W_prev, C_prev = A_prev.shape
    H_out = int((H_prev - f) / stride) + 1
    W_out = int((W_prev - f) / stride) + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError("Invalid pooling output shape. Check input size, f and stride.")

    A = np.zeros((m, H_out, W_out, C_prev), dtype=np.float32)

    for i in range(m):
        for h in range(H_out):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(W_out):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                window = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, :]
                A[i, h, w, :] = np.max(window, axis=(0, 1))

    cache = (A_prev, f, stride)
    return A, cache


def global_avg_pool_forward(A_prev):
    """
    Tac dung:
    - Giam moi feature map ve 1 gia tri trung binh.
    - Cach nay giam so tham so dense, do overfit hon flatten truc tiep.
    """

    A = np.mean(A_prev, axis=(1, 2))
    cache = A_prev.shape
    return A, cache


def flatten_forward(A_prev):
    """
    Tac dung:
    - Chuyen feature map 4D thanh ma tran 2D.
    """

    return A_prev.reshape(A_prev.shape[0], -1), A_prev.shape


def dense_forward(A_prev, W, b):
    """
    Tac dung:
    - Forward qua fully-connected layer.
    """

    Z = np.dot(A_prev, W) + b
    cache = (A_prev, W, b)
    return Z, cache


class CNN:
    """
    Tac dung:
    - Xay dung mo hinh CNN de phan loai 10 loai hoa qua.

    Kien truc:
    - Input
    - Conv 5x5, 8 filters -> ReLU -> MaxPool 2x2
    - Conv 3x3, 16 filters -> ReLU -> MaxPool 2x2
    - Conv 3x3, 32 filters -> ReLU
    - Global Average Pooling
    - Dense 64 -> ReLU
    - Dense num_classes -> Softmax

    Ghi chu:
    - File nay chi code mo hinh va forward/predict.
    - Khong code train trong file nay.
    - Khong su dung torch.nn, tensorflow, keras hay model co san.
    """

    def __init__(self, input_shape=(64, 64, 3), num_classes=10, seed=42):
        """
        Dau vao:
        - input_shape[tuple]: Shape cua anh sau resize, mac dinh (64, 64, 3).
        - num_classes[int]: So lop phan loai.
        - seed[int]: Seed khoi tao trong so.
        """

        if len(input_shape) != 3:
            raise ValueError("input_shape must be (height, width, channels)")

        if input_shape[0] % 4 != 0 or input_shape[1] % 4 != 0:
            raise ValueError("height and width must be divisible by 4 because model has 2 max-pooling layers")

        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.seed = seed
        self.params = self._init_params()

    def _init_params(self):
        """
        Tac dung:
        - Khoi tao tham so cua model.

        Dau ra:
        - params[dict]: Dictionary luu W, b cua tung layer.
        """

        H, W, C = self.input_shape
        rng = np.random.default_rng(self.seed)


        params = {
            "W1": he_init(rng, (5, 5, C, 8), fan_in=5 * 5 * C),
            "b1": np.zeros((1, 1, 1, 8), dtype=np.float32),
            "W2": he_init(rng, (3, 3, 8, 16), fan_in=3 * 3 * 8),
            "b2": np.zeros((1, 1, 1, 16), dtype=np.float32),
            "W3": he_init(rng, (3, 3, 16, 32), fan_in=3 * 3 * 16),
            "b3": np.zeros((1, 1, 1, 32), dtype=np.float32),
            "W4": he_init(rng, (32, 64), fan_in=32),
            "b4": np.zeros((1, 64), dtype=np.float32),
            "W5": he_init(rng, (64, self.num_classes), fan_in=64),
            "b5": np.zeros((1, self.num_classes), dtype=np.float32),
        }

        return params

    def forward(self, X):
        """
        Tac dung:
        - Forward qua toan bo CNN.

        Dau vao:
        - X[np.ndarray]: Batch anh shape NHWC hoac NCHW.

        Dau ra:
        - AL[np.ndarray]: Xac suat du doan shape (batch_size, num_classes).
        - caches[dict]: Cache cua cac layer de train/backward sau nay.
        """

        X = to_numpy_nhwc(X)

        if tuple(X.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {X.shape[1:]}")

        caches = {}

        Z1, caches["conv1"] = conv_forward(X, self.params["W1"], self.params["b1"], stride=1, pad=2)
        A1 = relu(Z1)
        P1, caches["pool1"] = max_pool_forward(A1, f=2, stride=2)

        Z2, caches["conv2"] = conv_forward(P1, self.params["W2"], self.params["b2"], stride=1, pad=1)
        A2 = relu(Z2)
        P2, caches["pool2"] = max_pool_forward(A2, f=2, stride=2)

        Z3, caches["conv3"] = conv_forward(P2, self.params["W3"], self.params["b3"], stride=1, pad=1)
        A3 = relu(Z3)

        G, caches["gap"] = global_avg_pool_forward(A3)

        Z4, caches["dense1"] = dense_forward(G, self.params["W4"], self.params["b4"])
        A4 = relu(Z4)

        Z5, caches["dense2"] = dense_forward(A4, self.params["W5"], self.params["b5"])
        AL = softmax(Z5)

        caches["Z1"] = Z1
        caches["A1"] = A1
        caches["Z2"] = Z2
        caches["A2"] = A2
        caches["P2"] = P2
        caches["Z3"] = Z3
        caches["A3"] = A3
        caches["Z4"] = Z4
        caches["A4"] = A4
        caches["Z5"] = Z5
        caches["AL"] = AL

        return AL, caches

    def predict(self, X):
        """
        Tac dung:
        - Du doan class index cua batch anh.
        """

        AL, _ = self.forward(X)
        return np.argmax(AL, axis=1)

    def get_parameters(self):
        """
        Tac dung:
        - Tra ve tham so model de train.py cap nhat va luu checkpoint.
        """

        return self.params

    def set_parameters(self, params):
        """
        Tac dung:
        - Nap tham so da train vao model.
        """

        self.params = params


def init_parameters(num_classes=10, input_shape=(64, 64, 3), seed=42):
    """
    Tac dung:
    - Wrapper tao tham so, giup train.py co the goi theo style ham.
    """

    model = CNN(input_shape=input_shape, num_classes=num_classes, seed=seed)
    return model.get_parameters()


def model_forward(X, parameters, input_shape=(64, 64, 3), num_classes=10):
    """
    Tac dung:
    - Wrapper forward, giup train.py co the truyen parameters rieng.
    """

    model = CNN(input_shape=input_shape, num_classes=num_classes)
    model.set_parameters(parameters)
    return model.forward(X)


def predict(X, parameters, input_shape=(64, 64, 3), num_classes=10):
    """
    Tac dung:
    - Wrapper predict, dung cho main.py khi load checkpoint.
    """

    model = CNN(input_shape=input_shape, num_classes=num_classes)
    model.set_parameters(parameters)
    return model.predict(X)


if __name__ == "__main__":
    model = CNN(input_shape=(64, 64, 3), num_classes=10)
    X_demo = np.random.rand(2, 64, 64, 3).astype(np.float32)
    AL, _ = model.forward(X_demo)

    print("Output shape:", AL.shape)
    print("Predict:", model.predict(X_demo))
