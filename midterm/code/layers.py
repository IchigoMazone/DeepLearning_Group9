import numpy as np

from midterm.code.im2col import im2col_nhwc


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def to_numpy_nhwc(X):
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
    return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


def conv_forward(A_prev, W, b, stride=1, pad=1):
    m, H_prev, W_prev, C_prev = A_prev.shape
    f, _, C_filter, C_out = W.shape

    if C_prev != C_filter:
        raise ValueError(f"Input channels and filter channels do not match: {C_prev} != {C_filter}")

    H_out = int((H_prev + 2 * pad - f) / stride) + 1
    W_out = int((W_prev + 2 * pad - f) / stride) + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError("Invalid conv output shape. Check input size, filter size, stride and pad.")

    X_col, _ = im2col_nhwc(A_prev, filter_size=f, stride=stride, pad=pad)
    W_col = W.reshape(f * f * C_prev, C_out)
    Z_col = np.dot(X_col, W_col) + b.reshape(1, C_out)
    Z = Z_col.reshape(m, H_out, W_out, C_out).astype(np.float32)

    cache = (A_prev, W, b, stride, pad, X_col)
    return Z, cache


def max_pool_forward(A_prev, f=2, stride=2):
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
                A[i, h, w, :] = np.max(
                    A_prev[i, vert_start:vert_end, horiz_start:horiz_end, :],
                    axis=(0, 1),
                )

    cache = (A_prev, f, stride)
    return A, cache


def global_avg_pool_forward(A_prev):
    A = np.mean(A_prev, axis=(1, 2))
    cache = A_prev.shape
    return A, cache


def flatten_forward(A_prev):
    A = A_prev.reshape(A_prev.shape[0], -1)
    cache = A_prev.shape
    return A, cache


def dense_forward(A_prev, W, b):
    Z = np.dot(A_prev, W) + b
    cache = (A_prev, W, b)
    return Z, cache


def dropout_forward(A_prev, keep_prob=1.0, seed=None, training=True):
    if not training or keep_prob >= 1.0:
        return A_prev, None

    if keep_prob <= 0.0:
        raise ValueError("keep_prob must be in (0, 1]")

    rng = np.random.default_rng(seed)
    mask = (rng.random(A_prev.shape) < keep_prob).astype(np.float32)
    A = A_prev * mask / keep_prob
    cache = (mask, keep_prob)
    return A.astype(np.float32), cache
