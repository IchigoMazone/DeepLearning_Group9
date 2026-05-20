import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from midterm.code.backward import model_backward
from midterm.code.checkpoint import save_checkpoint
from midterm.code.data import augment_batch, create_batches, load_csv_dataset, one_hot
from midterm.code.metrics import compute_accuracy, compute_loss
from midterm.code.optimizers import Adam
from midterm.models.CNN import CNN, model_forward, predict


def evaluate_arrays(X, y, parameters, image_size=(64, 64), num_classes=10):
    Y = one_hot(y, num_classes)
    AL, _ = model_forward(
        X,
        parameters,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
    )
    loss = compute_loss(AL, Y)
    acc = compute_accuracy(np.argmax(AL, axis=1), y)
    return loss, acc


def train_model(
    train_csv,
    val_csv,
    num_classes=10,
    epochs=20,
    batch_size=8,
    learning_rate=0.0003,
    checkpoint_path="midterm/outputs/best.pkl",
    image_size=(64, 64),
    seed=42,
    augment=True,
    train_acc_sample=128,
    limit=None,
):
    print("Dang load du lieu...")
    X_train, y_train, class_names = load_csv_dataset(train_csv, image_size=image_size, limit=limit)
    X_val, y_val, _ = load_csv_dataset(val_csv, image_size=image_size, limit=limit)
    Y_train = one_hot(y_train, num_classes)

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = CNN(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes, seed=seed)
    parameters = model.get_parameters()
    optimizer = Adam(parameters, learning_rate=learning_rate)
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_iter = create_batches(X_train, Y_train, batch_size, seed=seed + epoch)
        steps = int(np.ceil(len(X_train) / batch_size))

        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=steps, desc=f"Epoch {epoch}/{epochs}")

        for step, (X_batch, Y_batch) in enumerate(batch_iter):
            if augment:
                X_batch = augment_batch(X_batch, seed=seed + epoch * 1000 + step)

            AL, caches = model_forward(
                X_batch,
                parameters,
                input_shape=(image_size[0], image_size[1], 3),
                num_classes=num_classes,
            )
            loss = compute_loss(AL, Y_batch)
            grads = model_backward(AL, Y_batch, caches)
            parameters = optimizer.step(parameters, grads)
            total_loss += loss * len(X_batch)

        train_loss = total_loss / len(X_train)

        sample_size = min(train_acc_sample, len(X_train))
        train_loss_eval, train_acc = evaluate_arrays(
            X_train[:sample_size],
            y_train[:sample_size],
            parameters,
            image_size=image_size,
            num_classes=num_classes,
        )
        val_loss, val_acc = evaluate_arrays(
            X_val,
            y_val,
            parameters,
            image_size=image_size,
            num_classes=num_classes,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc(sample): {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(parameters, epoch, val_loss, val_acc, class_names, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

    return parameters


def evaluate_csv(test_csv, parameters, num_classes=10, image_size=(64, 64)):
    X_test, y_test, _ = load_csv_dataset(test_csv, image_size=image_size)
    loss, acc = evaluate_arrays(X_test, y_test, parameters, image_size=image_size, num_classes=num_classes)
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}%")
    return loss, acc


def predict_image(image, parameters, input_shape=(64, 64, 3), num_classes=10):
    X = image[np.newaxis, ...]
    pred_idx = int(predict(X, parameters, input_shape=input_shape, num_classes=num_classes)[0])
    AL, _ = model_forward(X, parameters, input_shape=input_shape, num_classes=num_classes)
    confidence = float(np.max(AL))
    return pred_idx, confidence

