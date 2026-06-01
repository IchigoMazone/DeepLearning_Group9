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
    "input_channels": 3,
    "num_classes": 5,
    "epochs": 50,
    "batch_size": 16,
    "lr": 0.005,
    "weight_decay": 1e-4,
    "clip_norm": 5.0,
    "augment": True,
    "normalize": False,
    "keep_aspect": True,
    "seed": 42,
    "tta": True,
    "best_model": "midterm/outputs/best_cf5_numpy.pkl",
    "latest_model": "midterm/outputs/latest_cf5_numpy.pkl",
}


def ensure_split_csv(force=False):
    paths = [CONFIG["train_csv"], CONFIG["val_csv"], CONFIG["test_csv"]]
    if not force and all(os.path.exists(path) for path in paths):
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
    train_csv, val_csv, _ = ensure_split_csv(force=args.resplit)
    train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        num_classes=CONFIG["num_classes"],
        epochs=args.epochs or CONFIG["epochs"],
        batch_size=args.batch_size or CONFIG["batch_size"],
        learning_rate=args.lr or CONFIG["lr"],
        image_size=CONFIG["target_size"],
        seed=CONFIG["seed"],
        augment=CONFIG["augment"] and not args.no_augment,
        normalize=CONFIG["normalize"],
        add_structure=CONFIG["add_structure"],
        dropout_keep_prob=CONFIG["dropout_keep_prob"],
        label_smoothing=CONFIG["label_smoothing"],
        weight_decay=CONFIG["weight_decay"],
        clip_norm=CONFIG["clip_norm"],
        patience=CONFIG["patience"],
        checkpoint_path=args.output or CONFIG["best_model"],
        latest_checkpoint_path=CONFIG["latest_model"],
        report_interval=CONFIG["report_interval"],
        keep_aspect=CONFIG["keep_aspect"],
        seed=CONFIG["seed"],
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
        normalize=checkpoint.get("normalize", False),
        keep_aspect=checkpoint.get("keep_aspect", CONFIG["keep_aspect"]),
        add_structure=checkpoint.get("add_structure", CONFIG["add_structure"]),
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

    image = load_image_numpy(
        image_path,
        image_size,
        normalize=checkpoint.get("normalize", False),
        keep_aspect=checkpoint.get("keep_aspect", CONFIG["keep_aspect"]),
        add_structure=checkpoint.get("add_structure", CONFIG["add_structure"]),
    )
    pred_idx, confidence = predict_image(
        image=image,
        parameters=checkpoint["parameters"],
        input_shape=(*image_size, checkpoint.get("input_channels", image.shape[-1])),
        num_classes=num_classes,
        tta=CONFIG["tta"],
    )

    label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)
    print(f"Prediction: {label} | Confidence: {confidence * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="NumPy CNN 5-class fruit classification")
    parser.add_argument("--mode", default="train", choices=["split", "train", "eval", "predict"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--image", type=str, default=None, help="Image path for prediction")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    args = parser.parse_args()

    if args.mode == "split":
        train_csv, val_csv, test_csv = ensure_split_csv(force=True)
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
