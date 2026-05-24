import numpy as np


def im2col_nhwc(X, filter_size, stride=1, pad=0):
    """
    Chuyển batch ảnh NHWC thành ma trận các cửa sổ trượt (im2col).
    Dùng cho conv_forward.
    """
    if pad > 0:
        X_pad = np.pad(
            X,
            ((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=0
        )
    else:
        X_pad = X

    m, H_pad, W_pad, C = X_pad.shape
    f = filter_size
    H_out = (H_pad - f) // stride + 1
    W_out = (W_pad - f) // stride + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Invalid output size: H_out={H_out}, W_out={W_out}. "
                        f"Check input size, filter_size={f}, stride={stride}, pad={pad}")

    # Sử dụng as_strided để lấy các cửa sổ
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
        X_pad, shape=shape, strides=strides, writeable=False
    )

    cols = windows.reshape(m * H_out * W_out, f * f * C)
    return cols, (m, H_out, W_out, C)


def col2im_nhwc(cols, X_shape, filter_size, stride=1, pad=0):
    """
    Chuyển gradient từ dạng cột về lại shape ảnh (col2im).
    Dùng cho conv_backward.
    """
    m, H, W, C = X_shape
    f = filter_size
    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    H_out = (H_pad - f) // stride + 1
    W_out = (W_pad - f) // stride + 1

    dX_pad = np.zeros((m, H_pad, W_pad, C), dtype=cols.dtype)
    cols_reshaped = cols.reshape(m, H_out, W_out, f, f, C)

    # Accumulate gradients
    for i in range(f):
        for j in range(f):
            dX_pad[:, i:i + H_out*stride:stride,
                      j:j + W_out*stride:stride, :] += cols_reshaped[:, :, :, i, j, :]

    # Remove padding
    if pad == 0:
        return dX_pad
    return dX_pad[:, pad:-pad, pad:-pad, :]