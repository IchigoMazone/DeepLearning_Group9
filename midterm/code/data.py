import os

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path):
    path = os.path.normpath(path)
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def resize_with_padding(image, size=(64, 64), fill_value=0):
    target_h, target_w = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), fill_value, dtype=resized.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def load_image_numpy(path, size=(64, 64), normalize=False, keep_aspect=True):
    path = resolve_path(path)
    image_data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if keep_aspect:
        image = resize_with_padding(image, size=size, fill_value=0)
    else:
        image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    if normalize:
        image = (image - 0.5) / 0.5
    return image.astype(np.float32)


def load_csv_dataset(csv_path, image_size=(64, 64), limit=None, normalize=False, keep_aspect=True):
    df = pd.read_csv(csv_path)
    df["resolved_path"] = df["image_path"].apply(resolve_path)
    exists_mask = df["resolved_path"].apply(os.path.exists)
    missing = int((~exists_mask).sum())
    if missing:
        print(f"[WARN] Skip {missing} missing images from {csv_path}")
        df = df[exists_mask].reset_index(drop=True)

    if limit is not None:
        df = df.head(limit)

    if df.empty:
        raise FileNotFoundError(
            f"No valid images found in {csv_path}. "
            "Regenerate/extract the dataset before training."
        )

    iterator = df["resolved_path"]
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Load {os.path.basename(csv_path)}")

    X = np.array([
        load_image_numpy(path, image_size, normalize=normalize, keep_aspect=keep_aspect)
        for path in iterator
    ], dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    class_names = (
        df.sort_values("label")
        .drop_duplicates("label")["class_name"]
        .tolist()
    )
    return X, y, class_names


def split_dataframe(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42):
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
    os.makedirs(output_dir, exist_ok=True)
    train_df, val_df, test_df = split_dataframe(dataset_csv, train_ratio, val_ratio, seed)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, val_path, test_path


def make_filtered_csv(dataset_csv, output_csv, class_names):
    df = pd.read_csv(dataset_csv)
    df = df[df["class_name"].isin(class_names)].copy()
    if df.empty:
        raise ValueError(f"No rows match selected classes: {class_names}")

    label_map = {class_name: index for index, class_name in enumerate(class_names)}
    df["old_label"] = df["label"]
    df["label"] = df["class_name"].map(label_map).astype(np.int64)
    df = df[["image_path", "label", "class_name", "old_label"]]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes), dtype=np.float32)
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def create_batches(X, Y, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], Y[batch_idx]


def augment_batch(X, seed=None):
    rng = np.random.default_rng(seed)
    X_aug = np.array(X, copy=True)
    clip_min = -1.0 if np.min(X_aug) < 0.0 else 0.0

    for i in range(len(X_aug)):
        if rng.random() < 0.5:
            X_aug[i] = np.flip(X_aug[i], axis=1)

        if rng.random() < 0.45:
            h, w = X_aug[i].shape[:2]
            angle = rng.uniform(-10.0, 10.0)
            scale = rng.uniform(0.92, 1.06)
            tx = rng.uniform(-0.06, 0.06) * w
            ty = rng.uniform(-0.06, 0.06) * h
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            X_aug[i] = cv2.warpAffine(
                X_aug[i],
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(clip_min, clip_min, clip_min),
            )

        if rng.random() < 0.5:
            alpha = rng.uniform(0.85, 1.15)
            beta = rng.uniform(-0.08, 0.08)
            X_aug[i] = np.clip(X_aug[i] * alpha + beta, clip_min, 1.0)

        if rng.random() < 0.25:
            h, w = X_aug[i].shape[:2]
            erase_h = int(rng.integers(max(1, h // 12), max(2, h // 5)))
            erase_w = int(rng.integers(max(1, w // 12), max(2, w // 5)))
            y0 = int(rng.integers(0, max(1, h - erase_h + 1)))
            x0 = int(rng.integers(0, max(1, w - erase_w + 1)))
            X_aug[i, y0:y0 + erase_h, x0:x0 + erase_w, :] = clip_min

    return X_aug.astype(np.float32)
