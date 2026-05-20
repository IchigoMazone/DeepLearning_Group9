import numpy as np


def im2col_nhwc(X, filter_size, stride=1, pad=0):
    """
    Chuyen batch anh NHWC thanh ma tran cac cua so truot.

    Dau vao:
    - X: (m, H, W, C)
    - filter_size: kich thuoc filter f
    - stride: buoc truot
    - pad: padding

    Dau ra:
    - cols: (m * H_out * W_out, f * f * C)
    - out_shape: (m, H_out, W_out, C)
    """

    if pad > 0:
        X_pad = np.pad(
            X,
            ((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode="constant",
        )
    else:
        X_pad = X

    m, H_pad, W_pad, C = X_pad.shape
    f = filter_size
    H_out = (H_pad - f) // stride + 1
    W_out = (W_pad - f) // stride + 1

    shape = (m, H_out, W_out, f, f, C)
    strides = (
        X_pad.strides[0],
        stride * X_pad.strides[1],
        stride * X_pad.strides[2],
        X_pad.strides[1],
        X_pad.strides[2],
        X_pad.strides[3],
    )

    windows = np.lib.stride_tricks.as_strided(
        X_pad,
        shape=shape,
        strides=strides,
        writeable=False,
    )

    cols = windows.reshape(m * H_out * W_out, f * f * C)
    return cols, (m, H_out, W_out, C)


def col2im_nhwc(cols, X_shape, filter_size, stride=1, pad=0):
    """
    Dua gradient dang column ve lai shape anh NHWC.

    Dau vao:
    - cols: (m * H_out * W_out, f * f * C)
    - X_shape: shape goc cua X, (m, H, W, C)

    Dau ra:
    - dX: (m, H, W, C)
    """

    m, H, W, C = X_shape
    f = filter_size
    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    H_out = (H_pad - f) // stride + 1
    W_out = (W_pad - f) // stride + 1

    dX_pad = np.zeros((m, H_pad, W_pad, C), dtype=cols.dtype)
    cols_reshaped = cols.reshape(m, H_out, W_out, f, f, C)

    for i in range(f):
        i_end = i + stride * H_out
        for j in range(f):
            j_end = j + stride * W_out
            dX_pad[:, i:i_end:stride, j:j_end:stride, :] += cols_reshaped[:, :, :, i, j, :]

    if pad == 0:
        return dX_pad
    return dX_pad[:, pad:-pad, pad:-pad, :]

