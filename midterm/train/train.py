import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from midterm.models.CNN import CNN, model_forward, predict, relu


def load_image_numpy(path, size=(64, 64)):
    """
    Tac dung:
    - Doc anh bang cv2, resize va chuan hoa ve [0, 1].

    Dau vao:
    - path[str]: Duong dan anh.
    - size[tuple]: (height, width).

    Dau ra:
    - np.ndarray: Anh RGB shape (height, width, 3).
    """

    path = os.path.normpath(path)
    image_data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_csv_dataset(csv_path, image_size=(64, 64), limit=None):
    """
    Tac dung:
    - Doc file csv gom image_path, label, class_name thanh X, y.
    """

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    iterator = df["image_path"]
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Load {os.path.basename(csv_path)}")

    X = np.array([load_image_numpy(path, image_size) for path in iterator], dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    class_names = (
        df.sort_values("label")
        .drop_duplicates("label")["class_name"]
        .tolist()
    )

    return X, y, class_names


def split_dataframe(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Tac dung:
    - Chia dataset.csv thanh train/val/test DataFrame theo tung class.
    """

    df = pd.read_csv(csv_path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def make_split_csv(dataset_csv, output_dir="midterm/datasets/cf", train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Tac dung:
    - Tao train.csv, val.csv, test.csv tu dataset.csv.
    """

    os.makedirs(output_dir, exist_ok=True)
    train_df, val_df, test_df = split_dataframe(dataset_csv, train_ratio, val_ratio, seed)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def one_hot(y, num_classes):
    """
    Tac dung:
    - Chuyen label so thanh one-hot vector.
    """

    Y = np.zeros((len(y), num_classes), dtype=np.float32)
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def create_batches(X, Y, batch_size, seed=None):
    """
    Tac dung:
    - Tron va chia batch.
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], Y[batch_idx]


def compute_loss(AL, Y):
    """
    Tac dung:
    - Tinh categorical cross-entropy.
    """

    return -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[0]


def compute_accuracy(y_pred, y_true):
    """
    Tac dung:
    - Tinh accuracy.
    """

    return float(np.mean(y_pred.astype(int) == y_true.astype(int)))


def extract_cnn_features(X):
    """
    Tac dung:
    - Trich dac trung anh bang cac phep convolution/pooling OpenCV co dinh.
    - Day la feature extractor nhanh hon full CNN tu trainable conv.

    Dau vao:
    - X[np.ndarray]: Batch anh RGB shape (m, H, W, 3), gia tri [0, 1].

    Dau ra:
    - features[np.ndarray]: Ma tran dac trung shape (m, n_features).
    """

    kernels = [
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32),
        cv2.getGaborKernel((9, 9), 3.0, 0, 8.0, 0.5, 0, ktype=cv2.CV_32F),
        cv2.getGaborKernel((9, 9), 3.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F),
        cv2.getGaborKernel((9, 9), 3.0, np.pi / 2, 8.0, 0.5, 0, ktype=cv2.CV_32F),
        cv2.getGaborKernel((9, 9), 3.0, 3 * np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F),
    ]

    features = []
    for image in X:
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)

        parts = []

        # Mau sac la tin hieu rat manh trong bai toan hoa qua.
        hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).ravel()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).ravel()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).ravel()
        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        hist /= np.sum(hist) + 1e-8
        parts.append(hist)

        # Anh mau kich thuoc nho giu lai bo cuc tong quat.
        small_rgb = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
        parts.append(small_rgb.reshape(-1).astype(np.float32))

        # Conv bank + pooling: bat canh, texture, huong.
        conv_parts = []
        for kernel in kernels:
            response = cv2.filter2D(gray, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
            response = np.abs(response)
            pooled = cv2.resize(response, (8, 8), interpolation=cv2.INTER_AREA)
            pooled = pooled / (np.max(pooled) + 1e-8)
            conv_parts.append(pooled.reshape(-1))

        parts.append(np.concatenate(conv_parts).astype(np.float32))
        features.append(np.concatenate(parts).astype(np.float32))

    return np.asarray(features, dtype=np.float32)


def standardize_features(X_train, X_val):
    """
    Tac dung:
    - Chuan hoa feature theo mean/std cua train.
    """

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def init_mlp(input_dim, hidden_dim=256, num_classes=10, seed=42):
    """
    Tac dung:
    - Khoi tao classifier MLP train tren feature da trich.
    """

    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / input_dim),
        "b1": np.zeros((1, hidden_dim), dtype=np.float32),
        "W2": rng.standard_normal((hidden_dim, num_classes)).astype(np.float32) * np.sqrt(2.0 / hidden_dim),
        "b2": np.zeros((1, num_classes), dtype=np.float32),
    }


def mlp_forward(X, params):
    """
    Tac dung:
    - Forward classifier MLP.
    """

    Z1 = np.dot(X, params["W1"]) + params["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(A1, params["W2"]) + params["b2"]
    AL = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
    AL = AL / np.sum(AL, axis=1, keepdims=True)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
    return AL, cache


def mlp_backward(AL, Y, cache, params):
    """
    Tac dung:
    - Backward classifier MLP.
    """

    m = Y.shape[0]
    dZ2 = AL - Y
    dW2 = np.dot(cache["A1"].T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, params["W2"].T)
    dZ1 = relu_backward(dA1, cache["Z1"])
    dW1 = np.dot(cache["X"].T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def adam_update(params, grads, state, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Tac dung:
    - Cap nhat tham so bang Adam.
    """

    if state is None:
        state = {
            "t": 0,
            "m": {key: np.zeros_like(value) for key, value in params.items()},
            "v": {key: np.zeros_like(value) for key, value in params.items()},
        }

    state["t"] += 1
    t = state["t"]

    for key in params:
        state["m"][key] = beta1 * state["m"][key] + (1 - beta1) * grads[key]
        state["v"][key] = beta2 * state["v"][key] + (1 - beta2) * (grads[key] ** 2)
        m_hat = state["m"][key] / (1 - beta1 ** t)
        v_hat = state["v"][key] / (1 - beta2 ** t)
        params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    return params, state


def relu_backward(dA, Z):
    """
    Tac dung:
    - Backward qua ReLU.
    """

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def dense_backward(dZ, cache):
    """
    Tac dung:
    - Backward qua fully-connected layer.
    """

    A_prev, W, _ = cache
    m = A_prev.shape[0]

    dW = np.dot(A_prev.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev = np.dot(dZ, W.T)

    return dA_prev, dW, db


def flatten_backward(dA, cache):
    """
    Tac dung:
    - Backward qua flatten layer.
    """

    return dA.reshape(cache)


def global_avg_pool_backward(dA, cache):
    """
    Tac dung:
    - Backward qua global average pooling.
    """

    m, H, W, C = cache
    return np.ones((m, H, W, C), dtype=np.float32) * dA[:, None, None, :] / (H * W)


def max_pool_backward(dA, cache):
    """
    Tac dung:
    - Backward qua max pooling.
    """

    A_prev, f, stride = cache
    m, H_prev, W_prev, C_prev = A_prev.shape
    _, H, W, _ = dA.shape
    dA_prev = np.zeros_like(A_prev, dtype=np.float32)

    for i in range(m):
        for h in range(H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(C_prev):
                    window = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    mask = window == np.max(window)
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]

    return dA_prev


def conv_backward(dZ, cache):
    """
    Tac dung:
    - Backward qua convolution layer.

    Ghi chu:
    - Forward dung cv2 de nhanh hon.
    - Backward van cai dat bang NumPy de tinh gradient cho W, b, A_prev.
    """

    A_prev, W, _, stride, pad = cache
    m, H_prev, W_prev, C_prev = A_prev.shape
    f, _, _, C_out = W.shape
    _, H, W_out, _ = dZ.shape

    dA_prev = np.zeros_like(A_prev, dtype=np.float32)
    dW = np.zeros_like(W, dtype=np.float32)
    db = np.zeros((1, 1, 1, C_out), dtype=np.float32)

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
    )
    dA_prev_pad = np.pad(
        dA_prev,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
    )

    for i in range(m):
        for h in range(H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(W_out):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                for c in range(C_out):
                    grad = dZ[i, h, w, c]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * grad
                    dW[:, :, :, c] += a_slice * grad
                    db[:, :, :, c] += grad

    if pad == 0:
        dA_prev = dA_prev_pad
    else:
        dA_prev = dA_prev_pad[:, pad:-pad, pad:-pad, :]

    dW /= m
    db /= m

    return dA_prev, dW, db


def model_backward(AL, Y, caches):
    """
    Tac dung:
    - Backward toan bo model CNN.
    """

    grads = {}

    dZ5 = AL - Y
    dA4, grads["dW5"], grads["db5"] = dense_backward(dZ5, caches["dense2"])

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


def update_parameters(parameters, grads, learning_rate=0.001):
    """
    Tac dung:
    - Cap nhat tham so bang gradient descent.
    """

    for key in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5"]:
        parameters[key] -= learning_rate * grads[f"d{key}"]

    return parameters


def adam_update_cnn(parameters, grads, state=None, learning_rate=0.0003, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Tac dung:
    - Cap nhat weight/bias cua CNN bang Adam sau moi batch.
    """

    keys = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5"]

    if state is None:
        state = {
            "t": 0,
            "m": {key: np.zeros_like(parameters[key]) for key in keys},
            "v": {key: np.zeros_like(parameters[key]) for key in keys},
        }

    state["t"] += 1
    t = state["t"]

    for key in keys:
        grad = grads[f"d{key}"]
        state["m"][key] = beta1 * state["m"][key] + (1 - beta1) * grad
        state["v"][key] = beta2 * state["v"][key] + (1 - beta2) * (grad ** 2)
        m_hat = state["m"][key] / (1 - beta1 ** t)
        v_hat = state["v"][key] / (1 - beta2 ** t)
        parameters[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    return parameters, state


def save_checkpoint(parameters, epoch, val_loss, val_acc, class_names, checkpoint_path):
    """
    Tac dung:
    - Luu checkpoint model.
    """

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "parameters": parameters,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "class_names": class_names,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


def save_feature_checkpoint(
    parameters,
    epoch,
    val_loss,
    val_acc,
    class_names,
    checkpoint_path,
    feature_mean,
    feature_std,
):
    """
    Tac dung:
    - Luu checkpoint cho pipeline feature extractor + MLP.
    """

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_type": "feature_mlp",
        "parameters": parameters,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "class_names": class_names,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path):
    """
    Tac dung:
    - Doc checkpoint da luu.
    """

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint


def train(
    train_csv,
    val_csv,
    num_classes=10,
    epochs=30,
    batch_size=16,
    learning_rate=0.001,
    checkpoint_path="midterm/outputs/best.pkl",
    image_size=(64, 64),
    seed=42,
    limit=None,
):
    """
    Tac dung:
    - Huan luyen CNN trainable tu train.csv va val.csv.
    - Sau moi batch, cap nhat weight/bias bang Adam.
    """

    print("Dang load du lieu...")
    X_train, y_train, class_names = load_csv_dataset(train_csv, image_size=image_size, limit=limit)
    X_val, y_val, _ = load_csv_dataset(val_csv, image_size=image_size, limit=limit)

    Y_train = one_hot(y_train, num_classes)
    Y_val = one_hot(y_val, num_classes)

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = CNN(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes, seed=seed)
    parameters = model.get_parameters()
    adam_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_iter = create_batches(X_train, Y_train, batch_size, seed=seed + epoch)
        steps = int(np.ceil(len(X_train) / batch_size))

        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=steps, desc=f"Epoch {epoch}/{epochs}")

        for X_batch, Y_batch in batch_iter:
            AL, caches = model_forward(
                X_batch,
                parameters,
                input_shape=(image_size[0], image_size[1], 3),
                num_classes=num_classes,
            )
            loss = compute_loss(AL, Y_batch)
            grads = model_backward(AL, Y_batch, caches)
            parameters, adam_state = adam_update_cnn(parameters, grads, adam_state, learning_rate)
            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)
        AL_train, _ = model_forward(
            X_train,
            parameters,
            input_shape=(image_size[0], image_size[1], 3),
            num_classes=num_classes,
        )
        train_acc = compute_accuracy(np.argmax(AL_train, axis=1), y_train)
        AL_val, _ = model_forward(
            X_val,
            parameters,
            input_shape=(image_size[0], image_size[1], 3),
            num_classes=num_classes,
        )
        val_loss = compute_loss(AL_val, Y_val)
        val_pred = np.argmax(AL_val, axis=1)
        val_acc = compute_accuracy(val_pred, y_val)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(parameters, epoch, val_loss, val_acc, class_names, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

    return parameters


def evaluate(test_csv, parameters, num_classes=10, image_size=(64, 64)):
    """
    Tac dung:
    - Danh gia model tren test.csv.
    """

    X_test, y_test, _ = load_csv_dataset(test_csv, image_size=image_size)
    Y_test = one_hot(y_test, num_classes)

    AL, _ = model_forward(
        X_test,
        parameters,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
    )
    loss = compute_loss(AL, Y_test)
    acc = compute_accuracy(np.argmax(AL, axis=1), y_test)

    print(f"Test Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}%")
    return loss, acc


def evaluate_feature_checkpoint(test_csv, checkpoint, num_classes=10, image_size=(64, 64)):
    """
    Tac dung:
    - Danh gia checkpoint feature_mlp tren test.csv.
    """

    X_test, y_test, _ = load_csv_dataset(test_csv, image_size=image_size)
    Y_test = one_hot(y_test, num_classes)
    F_test = extract_cnn_features(X_test)
    F_test = (F_test - checkpoint["feature_mean"]) / checkpoint["feature_std"]

    AL, _ = mlp_forward(F_test, checkpoint["parameters"])
    loss = compute_loss(AL, Y_test)
    acc = compute_accuracy(np.argmax(AL, axis=1), y_test)
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}%")
    return loss, acc


def predict_feature_image(image_path, checkpoint, image_size=(64, 64)):
    """
    Tac dung:
    - Predict 1 anh bang checkpoint feature_mlp.
    """

    image = load_image_numpy(image_path, image_size)
    F = extract_cnn_features(image[np.newaxis, ...])
    F = (F - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    AL, _ = mlp_forward(F, checkpoint["parameters"])
    pred_idx = int(np.argmax(AL, axis=1)[0])
    confidence = float(np.max(AL))
    return pred_idx, confidence


if __name__ == "__main__":
    train(
        train_csv="midterm/datasets/cf/train.csv",
        val_csv="midterm/datasets/cf/val.csv",
        num_classes=10,
        epochs=5,
        batch_size=8,
        learning_rate=0.001,
        checkpoint_path="midterm/outputs/best.pkl",
        image_size=(64, 64),
    )
