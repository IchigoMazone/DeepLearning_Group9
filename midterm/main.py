import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from midterm.code.checkpoint import load_checkpoint
from midterm.code.data import load_image_numpy, make_split_csv
from midterm.code.engine import evaluate_csv, predict_image, train_model


CONFIG = {
    "dataset_csv": "midterm/datasets/dataset.csv",
    "split_dir": "midterm/datasets/cf",
    "train_csv": "midterm/datasets/cf/train.csv",
    "val_csv": "midterm/datasets/cf/val.csv",
    "test_csv": "midterm/datasets/cf/test.csv",
    "target_size": (64, 64),
    "num_classes": 10,
    "epochs": 40,
    "batch_size": 16,
    "lr": 0.001,
    "best_model": "midterm/outputs/best.pkl",
}


def ensure_split_csv():
    paths = [CONFIG["train_csv"], CONFIG["val_csv"], CONFIG["test_csv"]]
    if all(os.path.exists(path) for path in paths):
        return tuple(paths)

    if not os.path.exists(CONFIG["dataset_csv"]):
        raise FileNotFoundError(f"Cannot find dataset csv: {CONFIG['dataset_csv']}")

    return make_split_csv(
        dataset_csv=CONFIG["dataset_csv"],
        output_dir=CONFIG["split_dir"],
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )


def run_train():
    train_csv, val_csv, _ = ensure_split_csv()
    train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        num_classes=CONFIG["num_classes"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["lr"],
        checkpoint_path=CONFIG["best_model"],
        image_size=CONFIG["target_size"],
    )


def run_eval(resume_path):
    if not resume_path:
        resume_path = CONFIG["best_model"]

    _, _, test_csv = ensure_split_csv()
    checkpoint = load_checkpoint(resume_path)
    evaluate_csv(
        test_csv=test_csv,
        parameters=checkpoint["parameters"],
        num_classes=CONFIG["num_classes"],
        image_size=CONFIG["target_size"],
    )


def run_predict(resume_path, image_path):
    if not resume_path:
        resume_path = CONFIG["best_model"]
    if not image_path:
        raise ValueError("Can truyen --image de predict")

    checkpoint = load_checkpoint(resume_path)
    class_names = checkpoint.get("class_names")
    image = load_image_numpy(image_path, CONFIG["target_size"])

    pred_idx, confidence = predict_image(
        image=image,
        parameters=checkpoint["parameters"],
        input_shape=(CONFIG["target_size"][0], CONFIG["target_size"][1], 3),
        num_classes=CONFIG["num_classes"],
    )

    if class_names and pred_idx < len(class_names):
        print(f"Prediction: {pred_idx} - {class_names[pred_idx]} | Confidence: {confidence * 100:.2f}%")
    else:
        print(f"Prediction: {pred_idx} | Confidence: {confidence * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="CNN Fruit Classification")
    parser.add_argument("--mode", default="train", choices=["split", "train", "eval", "predict"])
    parser.add_argument("--resume", type=str, default=None, help="Duong dan checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Duong dan anh can du doan")
    args = parser.parse_args()

    if args.mode == "split":
        train_csv, val_csv, test_csv = ensure_split_csv()
        print("Train CSV:", train_csv)
        print("Val CSV:", val_csv)
        print("Test CSV:", test_csv)
    elif args.mode == "train":
        run_train()
    elif args.mode == "eval":
        run_eval(args.resume)
    elif args.mode == "predict":
        run_predict(args.resume, args.image)


if __name__ == "__main__":
    main()
