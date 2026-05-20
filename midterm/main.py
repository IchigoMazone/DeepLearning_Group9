import argparse
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from midterm.models.CNN import predict
from midterm.train.train import (
    evaluate,
    evaluate_feature_checkpoint,
    load_checkpoint,
    load_image_numpy,
    make_split_csv,
    predict_feature_image,
    train,
)


CONFIG = {
    "dataset_csv": "midterm/datasets/dataset.csv",
    "split_dir": "midterm/datasets/cf",
    "train_csv": "midterm/datasets/cf/train.csv",
    "val_csv": "midterm/datasets/cf/val.csv",
    "test_csv": "midterm/datasets/cf/test.csv",
    "target_size": (64, 64),
    "num_classes": 10,
    "epochs": 20,
    "batch_size": 8,
    "lr": 0.0003,
    "best_model": "midterm/outputs/best.pkl",
}


def ensure_split_csv():
    """
    Tac dung:
    - Neu chua co train/val/test csv thi tao tu dataset.csv.
    """

    has_split = all(
        os.path.exists(path)
        for path in [CONFIG["train_csv"], CONFIG["val_csv"], CONFIG["test_csv"]]
    )

    if has_split:
        return CONFIG["train_csv"], CONFIG["val_csv"], CONFIG["test_csv"]

    if not os.path.exists(CONFIG["dataset_csv"]):
        raise FileNotFoundError(
            "Cannot find split csv files or dataset.csv. "
            f"Expected dataset csv: {CONFIG['dataset_csv']}"
        )

    return make_split_csv(
        dataset_csv=CONFIG["dataset_csv"],
        output_dir=CONFIG["split_dir"],
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )


def run_train():
    """
    Tac dung:
    - Train model tu csv.
    """

    train_csv, val_csv, _ = ensure_split_csv()

    train(
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
    """
    Tac dung:
    - Danh gia model tren test.csv.
    """

    if not resume_path:
        resume_path = CONFIG["best_model"]

    _, _, test_csv = ensure_split_csv()
    checkpoint = load_checkpoint(resume_path)
    if checkpoint.get("model_type") == "feature_mlp":
        evaluate_feature_checkpoint(
            test_csv=test_csv,
            checkpoint=checkpoint,
            num_classes=CONFIG["num_classes"],
            image_size=CONFIG["target_size"],
        )
    else:
        evaluate(
            test_csv=test_csv,
            parameters=checkpoint["parameters"],
            num_classes=CONFIG["num_classes"],
            image_size=CONFIG["target_size"],
        )


def run_predict(resume_path, image_path):
    """
    Tac dung:
    - Du doan 1 anh dau vao.
    """

    if not resume_path:
        resume_path = CONFIG["best_model"]
    if not image_path:
        raise ValueError("Can truyen --image de predict")

    checkpoint = load_checkpoint(resume_path)
    class_names = checkpoint.get("class_names")

    if checkpoint.get("model_type") == "feature_mlp":
        pred_idx, confidence = predict_feature_image(
            image_path=image_path,
            checkpoint=checkpoint,
            image_size=CONFIG["target_size"],
        )
    else:
        img = load_image_numpy(image_path, CONFIG["target_size"])
        X = img[np.newaxis, ...]
        pred_idx = int(
            predict(
                X,
                checkpoint["parameters"],
                input_shape=(CONFIG["target_size"][0], CONFIG["target_size"][1], 3),
                num_classes=CONFIG["num_classes"],
            )[0]
        )
        confidence = None

    if class_names and pred_idx < len(class_names):
        if confidence is None:
            print(f"Prediction: {pred_idx} - {class_names[pred_idx]}")
        else:
            print(f"Prediction: {pred_idx} - {class_names[pred_idx]} | Confidence: {confidence * 100:.2f}%")
    else:
        print(f"Prediction: {pred_idx}")


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
