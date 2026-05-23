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
    "target_size": (128,128),
    "num_classes": 10,
    "epochs": 60,
    "batch_size": 16,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "clip_norm": 5.0,
    "augment": True,
    "monitor": "val_acc",
    "dropout_rate": 0.35,
    "early_stopping_patience": 20,
    "tta": True,
    "best_model": "midterm/outputs/best.pkl",
    "seed": 42,
}


def ensure_split_csv():
    paths = [CONFIG["train_csv"], CONFIG["val_csv"], CONFIG["test_csv"]]
    if all(os.path.exists(path) for path in paths):
        return tuple(paths)

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
        weight_decay=CONFIG["weight_decay"],
        clip_norm=CONFIG["clip_norm"],
        augment=args.augment if args.augment else CONFIG["augment"] and not args.no_augment,
        seed=CONFIG["seed"],
        checkpoint_path=args.output or CONFIG["best_model"],
        param_log_interval=args.param_log_interval,
        monitor=CONFIG["monitor"],
        dropout_rate=CONFIG["dropout_rate"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
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
        num_classes=num_classes,
        image_size=image_size,
        batch_size=CONFIG["batch_size"],
        tta=CONFIG["tta"],
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
