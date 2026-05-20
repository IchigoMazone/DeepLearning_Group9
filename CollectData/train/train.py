import cv2
import pandas as pd
import numpy as np

from models.cnn import SimpleCNN


class Trainer:

    def __init__(self):
        self.model = SimpleCNN()

    def load_data(self, csv_path):

        df = pd.read_csv(csv_path)

        images = []
        labels = []

        for _, row in df.iterrows():

            img = cv2.imread(row['image'])

            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            img = img.flatten()

            images.append(img)

            labels.append(row['label'])

        return np.array(images), np.array(labels)

    def train(self):
        x_train, y_train = self.load_data("datasets/train.csv")

        predictions = self.model.forward(x_train)

        predicted_labels = np.argmax(predictions, axis=1)

        accuracy = np.mean(predicted_labels == y_train)

        print(f"Training Accuracy: {accuracy * 100:.2f}%")

    if __name__ == "__main__":
        trainer = Trainer()

        trainer.train()