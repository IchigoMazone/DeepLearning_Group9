import numpy as np

from midterm.code.im2col import col2im_nhwc, im2col_nhwc


def relu(Z):
    return np.maximum(Z, 0.0).astype(np.float32, copy=False)


def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0.0
    return dZ


def dropout_forward(A, keep_prob=1.0, training=False, seed=None):
    if not training or keep_prob >= 1.0:
        return A, None
    if keep_prob <= 0.0:
        raise ValueError("keep_prob must be in (0, 1]")

    rng = np.random.default_rng(seed)
    mask = (rng.random(A.shape) < keep_prob).astype(np.float32)
    return (A * mask / keep_prob).astype(np.float32, copy=False), (mask, keep_prob)

def dropout_backward(dA, cache):
    if cache is None:
        return dA
    mask, keep_prob = cache
    return (dA * mask / keep_prob).astype(np.float32, copy=False)


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z).astype(np.float32, copy=False)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def to_numpy_nhwc(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 4:
        raise ValueError(f"X must have 4 dimensions, got {X.shape}")
    if X.shape[-1] in {1, 3, 4, 5}:
        return X
    if X.shape[1] in {1, 3, 4, 5}:
        return np.transpose(X, (0, 2, 3, 1))
    raise ValueError(f"Unsupported input shape: {X.shape}")


def he_init(rng, shape, fan_in):
    return (rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def conv_forward(A_prev, W, b, stride=1, pad=0):
    m, H_prev, W_prev, C_prev = A_prev.shape
    f, _, C_filter, C_out = W.shape

    if C_prev != C_filter:
        raise ValueError(f"Channel mismatch: input={C_prev}, filter={C_filter}")

    H_out = (H_prev + 2 * pad - f) // stride + 1
    W_out = (W_prev + 2 * pad - f) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError("Invalid convolution output shape")

    X_col, _ = im2col_nhwc(A_prev, filter_size=f, stride=stride, pad=pad)
    W_col = W.reshape(f * f * C_prev, C_out)
    Z_col = X_col @ W_col + b.reshape(1, C_out)
    Z = Z_col.reshape(m, H_out, W_out, C_out).astype(np.float32, copy=False)
    cache = (A_prev, W, stride, pad, X_col)
    return Z, cache


def max_pool_forward(A_prev, f=2, stride=2):
    m, H_prev, W_prev, C = A_prev.shape
    H_out = (H_prev - f) // stride + 1
    W_out = (W_prev - f) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError("Invalid max-pool output shape")

    if f == stride and H_prev % f == 0 and W_prev % f == 0:
        windows = A_prev.reshape(m, H_out, f, W_out, f, C)
        A = windows.max(axis=(2, 4)).astype(np.float32, copy=False)
        cache = (A_prev, f, stride, A, True)
        return A, cache

    A = np.empty((m, H_out, W_out, C), dtype=np.float32)
    for h in range(H_out):
        v_start = h * stride
        v_end = v_start + f
        for w in range(W_out):
            h_start = w * stride
            h_end = h_start + f
            A[:, h, w, :] = np.max(A_prev[:, v_start:v_end, h_start:h_end, :], axis=(1, 2))

    cache = (A_prev, f, stride, A, False)
    return A, cache


def global_avg_pool_forward(A_prev):
    A = np.mean(A_prev, axis=(1, 2), dtype=np.float32)
    return A.astype(np.float32, copy=False), A_prev.shape




def global_max_pool_forward(A_prev):
    A = np.max(A_prev, axis=(1, 2))
    return A.astype(np.float32, copy=False), (A_prev, A)

def flatten_forward(A_prev):
    A = A_prev.reshape(A_prev.shape[0], -1)
    cache = A_prev.shape
    return A, cache


def dense_forward(A_prev, W, b):
    Z = np.dot(A_prev, W) + b
    cache = (A_prev, W, b)
    return Z, cache


