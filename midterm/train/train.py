from midterm.code.engine import train_model


if __name__ == "__main__":
<<<<<<< HEAD
    train(
        train_csv="midterm/datasets/cf_5/train.csv",
        val_csv="midterm/datasets/cf_5/val.csv",
        num_classes=5,
        epochs=80,
        batch_size=8,
        learning_rate=0.0007,
        checkpoint_path="midterm/outputs/best_cf5_numpy.pkl",
        image_size=(64, 64),
    )
=======
    CONFIG = {
        "train_csv": "midterm/datasets/cf/train.csv",
        "val_csv": "midterm/datasets/cf/val.csv",
        "num_classes": 10,
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "image_size": (96, 96),
        "weight_decay": 5e-4,
        "clip_norm": 5.0,
        "augment": True,
        "seed": 42,
        "checkpoint_path": "midterm/outputs/best.pkl",
        "monitor": "val_acc",
        "dropout_rate": 0.25,
        "early_stopping_patience": 20,
    }
>>>>>>> a46c8f5637c5403a4c253e1f7e4ab525fcca6d53

    train_model(
        train_csv=CONFIG["train_csv"],
        val_csv=CONFIG["val_csv"],
        num_classes=CONFIG["num_classes"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        image_size=CONFIG["image_size"],
        weight_decay=CONFIG["weight_decay"],
        clip_norm=CONFIG["clip_norm"],
        augment=CONFIG["augment"],
        seed=CONFIG["seed"],
        checkpoint_path=CONFIG["checkpoint_path"],
        param_log_interval=0,
        monitor=CONFIG["monitor"],
        dropout_rate=CONFIG["dropout_rate"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
    )
