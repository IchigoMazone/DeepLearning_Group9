import os
import pickle
from datetime import datetime


def save_checkpoint(parameters, epoch, val_loss, val_acc, class_names, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "parameters": parameters,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "class_names": class_names,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)

