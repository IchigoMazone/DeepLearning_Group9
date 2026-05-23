from midterm.code.layers import (
    conv_backward,
    dense_backward,
    dropout_backward,
    global_avg_pool_backward,
    max_pool_backward,
    relu_backward,
)


def model_backward(AL, Y, caches):
    m = Y.shape[0]
    grads = {}

    dZ5 = (AL - Y) / m
    dA4, grads["dW5"], grads["db5"] = dense_backward(dZ5, caches["dense2"])

    dA4 = dropout_backward(dA4, caches.get("dropout1"))
    dZ4 = relu_backward(dA4, caches["Z4"])
    dG, grads["dW4"], grads["db4"] = dense_backward(dZ4, caches["dense1"])

    dA3 = global_avg_pool_backward(dG, caches["gap"])
    dZ3 = relu_backward(dA3, caches["Z3"])
    dP2, grads["dW3"], grads["db3"] = conv_backward(dZ3, caches["conv3"])

    dA2 = max_pool_backward(dP2, caches["pool2"])
    dZ2 = relu_backward(dA2, caches["Z2"])
    dP1, grads["dW2"], grads["db2"] = conv_backward(dZ2, caches["conv2"])

    dA1 = max_pool_backward(dP1, caches["pool1"])
    dZ1 = relu_backward(dA1, caches["Z1"])
    _, grads["dW1"], grads["db1"] = conv_backward(dZ1, caches["conv1"])

    return grads
