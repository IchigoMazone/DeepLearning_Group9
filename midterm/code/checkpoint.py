import os
import pickle
from datetime import datetime


def save_checkpoint(parameters, epoch, val_loss, val_acc, class_names, checkpoint_path, metadata=None):
    checkpoint_path = os.path.normpath(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    abs_path = os.path.abspath(checkpoint_path)
    checkpoint = {
        "parameters": parameters,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "class_names": class_names,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": abs_path,
    }
    if metadata:
        checkpoint.update(metadata)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    return abs_path


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)
