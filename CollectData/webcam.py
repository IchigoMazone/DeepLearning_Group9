import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIDTERM_ROOT = PROJECT_ROOT / "midterm"
DATASET_CSV = MIDTERM_ROOT / "datasets" / "dataset_5.csv"
CHECKPOINT_PATH = MIDTERM_ROOT / "outputs" / "best_cf5_numpy.pkl"
COLLECT_DIR = Path(__file__).resolve().parent / "new_dataset"

sys.path.append(str(PROJECT_ROOT))

from midterm.code.checkpoint import load_checkpoint
from midterm.code.engine import predict_image


FONT = cv2.FONT_HERSHEY_SIMPLEX
PREDICT_EVERY_SECONDS = 0.6


def load_label_map(dataset_csv=DATASET_CSV):
    labels = {}
    with open(dataset_csv, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = int(row["label"])
            labels[label] = row.get("class_name", str(label))
    if not labels:
        raise ValueError(f"No labels found in {dataset_csv}")
    return dict(sorted(labels.items()))


def ensure_collect_dirs(label_map, collect_dir=COLLECT_DIR):
    collect_dir.mkdir(parents=True, exist_ok=True)
    for label in label_map:
        (collect_dir / str(label)).mkdir(parents=True, exist_ok=True)


def next_image_path(label, collect_dir=COLLECT_DIR):
    folder = collect_dir / str(label)
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    index = len(list(folder.glob("*.jpg")))
    return folder / f"label_{label}_{timestamp}_{index:05d}.jpg"


def resize_with_padding_bgr(frame, size):
    target_h, target_w = size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
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

def frame_to_model_input(frame_bgr, image_size, normalize=False, keep_aspect=True, add_structure=False):
    if keep_aspect:
        frame_bgr = resize_with_padding_bgr(frame_bgr, image_size)
    else:
        frame_bgr = cv2.resize(frame_bgr, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = frame_rgb.astype(np.float32) / 255.0
    if add_structure:
        image = add_structure_channels(image)
    if normalize:
        image = (image - 0.5) / 0.5
    return image.astype(np.float32)


def clean_name(name):
    return name.replace("Fruits_", "").replace("Fruits", "").strip("_")


def draw_panel(frame, label_map, prediction_text, confidence, saved_text):
    h, w = frame.shape[:2]
    panel_h = 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Keys: 0-4 save image | q quit", (12, 28), FONT, 0.72, (255, 255, 255), 2)
    labels_text = " | ".join(f"{label}:{clean_name(name)}" for label, name in label_map.items())
    cv2.putText(frame, labels_text, (12, 58), FONT, 0.55, (210, 240, 255), 1)

    if prediction_text:
        cv2.putText(
            frame,
            f"Predict: {clean_name(prediction_text)} ({confidence * 100:.1f}%)",
            (12, 90),
            FONT,
            0.72,
            (80, 255, 80),
            2,
        )
    else:
        cv2.putText(frame, "Predict: loading model...", (12, 90), FONT, 0.72, (120, 220, 255), 2)

    if saved_text:
        cv2.putText(frame, saved_text, (12, h - 18), FONT, 0.7, (0, 255, 255), 2)


def run_webcam(camera_index=0):
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {DATASET_CSV}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    label_map = load_label_map()
    ensure_collect_dirs(label_map)

    checkpoint = load_checkpoint(str(CHECKPOINT_PATH))
    image_size = tuple(checkpoint.get("image_size", (64, 64)))
    num_classes = int(checkpoint.get("num_classes", len(label_map)))
    class_names = checkpoint.get("class_names") or [label_map.get(i, str(i)) for i in range(num_classes)]
    parameters = checkpoint["parameters"]
    normalize = checkpoint.get("normalize", False)
    keep_aspect = checkpoint.get("keep_aspect", True)
    add_structure = checkpoint.get("add_structure", False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Webcam is running.")
    print("Press 0-4 to save current frame into CollectData/new_dataset/<label>.")
    print("Press q to quit.")

    prediction_text = ""
    confidence = 0.0
    saved_text = ""
    saved_until = 0.0
    last_predict_at = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break

            raw_frame = frame.copy()
            now = time.time()
            if now - last_predict_at >= PREDICT_EVERY_SECONDS:
                image = frame_to_model_input(
                    raw_frame,
                    image_size,
                    normalize=normalize,
                    keep_aspect=keep_aspect,
                    add_structure=add_structure,
                )
                pred_idx, confidence = predict_image(
                    image=image,
                    parameters=parameters,
                    input_shape=(*image_size, image.shape[-1]),
                    num_classes=num_classes,
                    tta=True,
                )
                prediction_text = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                last_predict_at = now

            if now > saved_until:
                saved_text = ""

            draw_panel(frame, label_map, prediction_text, confidence, saved_text)
            cv2.imshow("Fruit webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if ord("0") <= key <= ord("9"):
                label = key - ord("0")
                if label in label_map:
                    image_path = next_image_path(label)
                    cv2.imwrite(str(image_path), raw_frame)
                    saved_text = f"Saved label {label} -> {image_path.name}"
                    saved_until = time.time() + 1.5
                    print(saved_text)
                else:
                    print(f"Label {label} is not in dataset.csv")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
