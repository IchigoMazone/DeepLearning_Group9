from midterm.code.engine import train_model


if __name__ == "__main__":
    CONFIG = {
        "train_csv": "midterm/datasets/cf/train.csv",
        "val_csv": "midterm/datasets/cf/val.csv",
        "num_classes": 5,
        "epochs": 80,
        "batch_size": 16,
        "learning_rate": 0.0007,
        "image_size": (128, 128),
        "weight_decay": 1e-4,
        "clip_norm": 5.0,
        "augment": True,
        "normalize": False,
        "keep_aspect": True,
        "dropout_keep_prob": 0.72,
        "patience": 16,
        "seed": 42,
        "checkpoint_path": "midterm/outputs/best.pkl",
        "latest_checkpoint_path": "midterm/outputs/latest.pkl",
        "report_interval": 4,
        "label_smoothing": 0.05,
    }

    train_model(**CONFIG)