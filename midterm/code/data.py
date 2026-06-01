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




def add_structure_channels(image):
    gray = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(sobel_x, sobel_y)
    edge = edge / (float(edge.max()) + 1e-6)
    return np.concatenate([image, gray_f[..., None], edge[..., None]], axis=2).astype(np.float32)

def load_image_numpy(path, size=(64, 64), normalize=False, keep_aspect=True, add_structure=False):
    if cv2 is None:
        raise ImportError("opencv-python is required to load images. Install it with: pip install opencv-python")

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
    if add_structure:
        image = add_structure_channels(image)
    if normalize:
        image = (image - 0.5) / 0.5
    return image.astype(np.float32)


def load_csv_dataset(csv_path, image_size=(64, 64), limit=None, normalize=False, keep_aspect=True, add_structure=False):
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

    X = np.array([
        load_image_numpy(path, image_size, normalize=normalize, keep_aspect=keep_aspect, add_structure=add_structure)
        for path in iterator
    ], dtype=np.float32)
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
    X_aug = np.array(X, copy=True).astype(np.float32)
    base_channels = min(3, X_aug.shape[-1])
    clip_min = -1.0 if np.min(X_aug[..., :base_channels]) < 0.0 else 0.0
    clip_max = 1.0

    for i in range(len(X_aug)):
        img_full = X_aug[i]
        img = img_full[..., :base_channels]
        orig_h, orig_w = img.shape[:2]

        if rng.random() < 0.65:
            scale = rng.uniform(0.88, 1.0)
            crop_h = max(1, int(orig_h * scale))
            crop_w = max(1, int(orig_w * scale))
            top = int(rng.integers(0, orig_h - crop_h + 1))
            left = int(rng.integers(0, orig_w - crop_w + 1))
            img = img[top:top + crop_h, left:left + crop_w]
            img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        if rng.random() < 0.5:
            img = np.fliplr(img)

        if rng.random() < 0.55:
            h, w = img.shape[:2]
            angle = rng.uniform(-16.0, 16.0)
            scale = rng.uniform(0.94, 1.06)
            tx = rng.uniform(-0.04, 0.04) * w
            ty = rng.uniform(-0.04, 0.04) * h
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        if rng.random() < 0.7:
            alpha = rng.uniform(0.82, 1.18)
            beta = rng.uniform(-0.08, 0.08)
            img = img * alpha + beta

        if rng.random() < 0.35:
            noise = rng.normal(0, 0.018, img.shape).astype(np.float32)
            img = img + noise

        if rng.random() < 0.30:
            hsv_ready = np.clip(img, 0.0, 1.0).astype(np.float32)
            hsv = cv2.cvtColor(hsv_ready, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6.0, 6.0)) % 360.0
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.90, 1.12), 0.0, 1.0)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if rng.random() < 0.15:
            h, w = img.shape[:2]
            erase_h = int(rng.integers(max(1, h // 14), max(2, h // 6)))
            erase_w = int(rng.integers(max(1, w // 14), max(2, w // 6)))
            y0 = int(rng.integers(0, max(1, h - erase_h + 1)))
            x0 = int(rng.integers(0, max(1, w - erase_w + 1)))
            img[y0:y0 + erase_h, x0:x0 + erase_w, :] = clip_min

        img = np.clip(img, clip_min, clip_max)
        if X_aug.shape[-1] > 3:
            X_aug[i] = add_structure_channels(img)
        else:
            X_aug[i] = img

    return X_aug.astype(np.float32, copy=False)
