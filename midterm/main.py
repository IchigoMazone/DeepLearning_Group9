import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from midterm.code.checkpoint import load_checkpoint
from midterm.code.data import load_image_numpy, make_filtered_csv, make_split_csv
from midterm.code.engine import evaluate_csv, predict_image, train_model


CONFIG = {
    "dataset_csv": "midterm/datasets/dataset_5.csv",
    "source_dataset_csv": "midterm/datasets/dataset_fruits1.csv",
    "split_dir": "midterm/datasets/cf_5",
    "train_csv": "midterm/datasets/cf_5/train.csv",
    "val_csv": "midterm/datasets/cf_5/val.csv",
    "test_csv": "midterm/datasets/cf_5/test.csv",
    "target_size": (64, 64),
    "num_classes": 5,
    "selected_classes": [
        "Fruits_Cucumber",
        "Fruits_Grapes",
        "Fruits_Kiwi",
        "Fruits_Orange",
        "Fruits_Pomegranate",
    ],
    "epochs": 80,
    "batch_size": 8,
    "lr": 0.0007,
    "dropout_keep_prob": 0.7,
    "weight_decay": 1e-4,
    "patience": 10,
    "report_interval": 5,
    "keep_aspect": True,
    "best_model": "midterm/outputs/best_cf5_numpy.pkl",
    "latest_model": "midterm/outputs/latest_cf5_numpy.pkl",
}


def ensure_split_csv():
    paths = [CONFIG["train_csv"], CONFIG["val_csv"], CONFIG["test_csv"]]
    if all(os.path.exists(path) for path in paths):
        return tuple(paths)

    if not os.path.exists(CONFIG["dataset_csv"]) and os.path.exists(CONFIG["source_dataset_csv"]):
        make_filtered_csv(
            dataset_csv=CONFIG["source_dataset_csv"],
            output_csv=CONFIG["dataset_csv"],
            class_names=CONFIG["selected_classes"],
        )

    if not os.path.exists(CONFIG["dataset_csv"]):
        raise FileNotFoundError(f"Cannot find dataset CSV: {CONFIG['dataset_csv']}")

    return make_split_csv(
        dataset_csv=CONFIG["dataset_csv"],
        output_dir=CONFIG["split_dir"],
        train_ratio=0.7,
        val_ratio=0.15,
        seed=CONFIG["seed"],
    )


def run_train(args):
    train_csv, val_csv, _ = ensure_split_csv()
    train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        num_classes=CONFIG["num_classes"],
        epochs=args.epochs or CONFIG["epochs"],
        batch_size=args.batch_size or CONFIG["batch_size"],
        learning_rate=args.lr or CONFIG["lr"],
        image_size=CONFIG["target_size"],
        dropout_keep_prob=CONFIG["dropout_keep_prob"],
        weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"],
        latest_checkpoint_path=CONFIG["latest_model"],
        report_interval=CONFIG["report_interval"],
        keep_aspect=CONFIG["keep_aspect"],
    )


def run_eval(resume_path=None):
    if not resume_path:
        resume_path = CONFIG["best_model"]

    _, _, test_csv = ensure_split_csv()
    checkpoint = load_checkpoint(resume_path)
    image_size = tuple(checkpoint.get("image_size", CONFIG["target_size"]))
    num_classes = int(checkpoint.get("num_classes", CONFIG["num_classes"]))
    evaluate_csv(
        test_csv=test_csv,
        parameters=checkpoint["parameters"],
        num_classes=CONFIG["num_classes"],
        image_size=CONFIG["target_size"],
        normalize=checkpoint.get("normalize", False),
        keep_aspect=checkpoint.get("keep_aspect", CONFIG["keep_aspect"]),
    )


def run_predict(resume_path=None, image_path=None):
    if not resume_path:
        resume_path = CONFIG["best_model"]
    if not image_path:
        raise ValueError("Please pass --image <image_path>")

    checkpoint = load_checkpoint(resume_path)
    image_size = tuple(checkpoint.get("image_size", CONFIG["target_size"]))
    num_classes = int(checkpoint.get("num_classes", CONFIG["num_classes"]))
    class_names = checkpoint.get("class_names")
    image = load_image_numpy(
        image_path,
        CONFIG["target_size"],
        normalize=checkpoint.get("normalize", False),
        keep_aspect=checkpoint.get("keep_aspect", CONFIG["keep_aspect"]),
    )

    image = load_image_numpy(image_path, image_size)
    pred_idx, confidence = predict_image(
        image=image,
        parameters=checkpoint["parameters"],
        input_shape=(*image_size, 3),
        num_classes=num_classes,
        tta=CONFIG["tta"],
    )

    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)
    print(f"Prediction: {label} | Confidence: {confidence * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="NumPy CNN Fruit & Vegetable Classification")
    parser.add_argument("--mode", default="train", choices=["split", "train", "eval", "predict"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--image", type=str, default=None, help="Image path for prediction")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation during training")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--param-log-interval",
        default=0,
        help="'epoch' to print W/b after every epoch, 0 to disable, or an integer update interval",
    )
    args = parser.parse_args()

    if str(args.param_log_interval).isdigit():
        args.param_log_interval = int(args.param_log_interval)
    elif str(args.param_log_interval).lower() in {"0", "none", "false", "off"}:
        args.param_log_interval = 0

    if args.mode == "split":
        train_csv, val_csv, test_csv = ensure_split_csv()
        print(f"Train CSV: {train_csv}")
        print(f"Val CSV: {val_csv}")
        print(f"Test CSV: {test_csv}")
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args.resume)
    elif args.mode == "predict":
        run_predict(args.resume, args.image)


if __name__ == "__main__":
    main()
