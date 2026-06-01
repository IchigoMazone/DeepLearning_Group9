import numpy as np

from midterm.code.im2col import col2im_nhwc


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def dense_backward(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[0]
    dW = np.dot(A_prev.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev = np.dot(dZ, W.T)
    return dA_prev, dW.astype(np.float32), db.astype(np.float32)


def dropout_backward(dA, cache):
    if cache is None:
        return dA
    mask, keep_prob = cache
    return (dA * mask / keep_prob).astype(np.float32)


def flatten_backward(dA, cache):
    return dA.reshape(cache).astype(np.float32)


def global_avg_pool_backward(dA, cache):
    m, h, w, c = cache
    return np.ones((m, h, w, c), dtype=np.float32) * dA[:, None, None, :] / (h * w)


def global_max_pool_backward(dA, cache):
    A_prev, A = cache
    mask = A_prev == A[:, None, None, :]
    denom = np.sum(mask, axis=(1, 2), keepdims=True)
    return (mask * dA[:, None, None, :] / np.maximum(denom, 1)).astype(np.float32)


def max_pool_backward(dA, cache):
    A_prev, f, stride = cache[:3]
    m, _, _, C_prev = A_prev.shape
    _, H, W, _ = dA.shape
    dA_prev = np.zeros_like(A_prev, dtype=np.float32)

    for i in range(m):
        for h in range(h_out):
            v0 = h * stride
            v1 = v0 + f
            for w in range(w_out):
                h0 = w * stride
                h1 = h0 + f
                window = A_prev[i, v0:v1, h0:h1, :]
                max_values = A_out[i, h, w, :] if fast_path and A_out is not None else np.max(window, axis=(0, 1))
                mask = window == max_values
                denom = np.sum(mask, axis=(0, 1), keepdims=True)
                dA_prev[i, v0:v1, h0:h1, :] += mask * dA[i, h, w, :] / np.maximum(denom, 1)

    return dA_prev


def conv_backward(dZ, cache):
    A_prev, W, stride, pad, X_col = cache
    m, _, _, C_prev = A_prev.shape
    f, _, _, C_out = W.shape
    _, H_out, W_out, _ = dZ.shape

    m, _, _, c_prev = A_prev.shape
    f, _, _, c_out = W.shape
    _, h_out, w_out, _ = dZ.shape

    dZ_col = dZ.reshape(m * h_out * w_out, c_out)
    W_col = W.reshape(f * f * c_prev, c_out)

    dW_col = np.dot(X_col.T, dZ_col) / m
    dW = dW_col.reshape(W.shape).astype(np.float32)
    db = (np.sum(dZ_col, axis=0).reshape(1, 1, 1, c_out) / m).astype(np.float32)

    dX_col = np.dot(dZ_col, W_col.T)
    dA_prev = col2im_nhwc(dX_col, X_shape=A_prev.shape, filter_size=f, stride=stride, pad=pad)
    return dA_prev.astype(np.float32), dW, db


def model_backward(AL, Y, caches):
    grads = {}
    dZ5 = (AL - Y).astype(np.float32)
    dA4, grads["dW5"], grads["db5"] = dense_backward(dZ5, caches["dense2"])

    dA4 = dropout_backward(dA4, caches.get("dropout1"))
    dZ4 = relu_backward(dA4, caches["Z4"])
    dF, grads["dW4"], grads["db4"] = dense_backward(dZ4, caches["dense1"])

    dP3 = flatten_backward(dF, caches["flatten"])
    dA3 = max_pool_backward(dP3, caches["pool3"])

    dA3 = flatten_backward(dG, caches["flatten"])
    dA3 = max_pool_backward(dA3, caches["pool3"])
    dZ3 = relu_backward(dA3, caches["Z3"])
    dP2, grads["dW3"], grads["db3"] = conv_backward(dZ3, caches["conv3"])

    dA2 = max_pool_backward(dP2, caches["pool2"])
    dZ2 = relu_backward(dA2, caches["Z2"])
    dP1, grads["dW2"], grads["db2"] = conv_backward(dZ2, caches["conv2"])

    dA1 = max_pool_backward(dP1, caches["pool1"])
    dZ1 = relu_backward(dA1, caches["Z1"])
    _, grads["dW1"], grads["db1"] = conv_backward(dZ1, caches["conv1"])
    return grads
