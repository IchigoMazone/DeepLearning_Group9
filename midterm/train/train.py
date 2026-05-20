from midterm.code.checkpoint import load_checkpoint
from midterm.code.data import load_image_numpy, make_split_csv
from midterm.code.engine import evaluate_csv as evaluate
from midterm.code.engine import predict_image
from midterm.code.engine import train_model as train


if __name__ == "__main__":
    train(
        train_csv="midterm/datasets/cf/train.csv",
        val_csv="midterm/datasets/cf/val.csv",
        num_classes=10,
        epochs=20,
        batch_size=8,
        learning_rate=0.0003,
        checkpoint_path="midterm/outputs/best.pkl",
        image_size=(64, 64),
    )

