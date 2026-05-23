import os

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

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


def load_image_numpy(path, size=(96, 96)):
    if cv2 is None:
        raise ImportError("opencv-python is required to load images. Install it with: pip install opencv-python")

    path = resolve_path(path)
    image_data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_csv_dataset(csv_path, image_size=(96, 96), limit=None):
    df = pd.read_csv(csv_path)
    df["resolved_path"] = df["image_path"].apply(resolve_path)

    exists_mask = df["resolved_path"].apply(os.path.exists)
    missing = int((~exists_mask).sum())
    if missing:
        print(f"[WARN] {missing} images are missing in {csv_path}")
        df = df[exists_mask].reset_index(drop=True)

    if limit is not None:
        df = df.head(limit)

    if df.empty:
        raise FileNotFoundError(f"No valid images in {csv_path}")

    iterator = df["resolved_path"]
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Loading {os.path.basename(csv_path)}")

    X = np.array([load_image_numpy(path, image_size) for path in iterator], dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    class_names = df.sort_values("label").drop_duplicates("label")["class_name"].tolist()
    return X, y, class_names


def split_dataframe(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    df = pd.read_csv(csv_path).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if train_test_split is not None:
        temp_ratio = 1.0 - train_ratio
        val_ratio_in_temp = val_ratio / temp_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df["label"],
            random_state=seed,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1.0 - val_ratio_in_temp,
            stratify=temp_df["label"],
            random_state=seed,
        )
        return (
            train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
            val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
            test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        )

    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("label"):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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

    print(f"Split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_path, val_path, test_path


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


def _as_cv_float32(img):
    return np.ascontiguousarray(np.clip(img, 0.0, 1.0).astype(np.float32, copy=False))


def make_tta_batch(X):
    variants = [
        X.astype(np.float32, copy=False),
        np.flip(X, axis=2).astype(np.float32, copy=False),
        _as_cv_float32(X * 1.08),
        _as_cv_float32(X * 0.92),
    ]
    return variants


def augment_batch(X, seed=None):
    if cv2 is None:
        raise ImportError("opencv-python is required for data augmentation. Install it with: pip install opencv-python")

    rng = np.random.default_rng(seed)
    X_aug = X.copy().astype(np.float32)

    for i in range(len(X_aug)):
        img = _as_cv_float32(X_aug[i])
        orig_h, orig_w = img.shape[:2]

        if rng.random() < 0.7:
            scale = rng.uniform(0.82, 1.0)
            crop_h = max(1, int(orig_h * scale))
            crop_w = max(1, int(orig_w * scale))
            top = rng.integers(0, orig_h - crop_h + 1)
            left = rng.integers(0, orig_w - crop_w + 1)
            img = img[top:top + crop_h, left:left + crop_w]
            img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            img = _as_cv_float32(img)

        if rng.random() < 0.5:
            img = _as_cv_float32(np.fliplr(img))

        if rng.random() < 0.8:
            alpha = rng.uniform(0.75, 1.25)
            beta = rng.uniform(-0.12, 0.12)
            img = _as_cv_float32(img * alpha + beta)

        if rng.random() < 0.6:
            h, w = img.shape[:2]
            angle = rng.uniform(-25, 25)
            transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, transform, (w, h), borderMode=cv2.BORDER_REFLECT)
            img = _as_cv_float32(img)

        if rng.random() < 0.5:
            h, w = img.shape[:2]
            zoom = rng.uniform(0.85, 1.15)
            new_h, new_w = max(1, int(h * zoom)), max(1, int(w * zoom))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if zoom >= 1.0:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img[start_h:start_h + h, start_w:start_w + w]
            else:
                pad_h = h - new_h
                pad_w = w - new_w
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
            img = _as_cv_float32(img)

        if rng.random() < 0.4:
            noise = rng.normal(0, 0.02, img.shape).astype(np.float32)
            img = _as_cv_float32(img + noise)

        if rng.random() < 0.5:
            img = _as_cv_float32(img)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 360
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.2), 0.0, 1.0)
            img = _as_cv_float32(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

        if rng.random() < 0.25:
            blur_kernel = 3
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
            img = _as_cv_float32(img)

        if img.shape[:2] != (orig_h, orig_w):
            img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        X_aug[i] = _as_cv_float32(img)

    return X_aug.astype(np.float32, copy=False)
