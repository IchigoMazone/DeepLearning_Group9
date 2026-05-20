import os
import csv


def create_csv(
        dataset_dir="datasets/captured_images",
        output_csv="datasets/dataset.csv"
):

    data = []

    for label in ["0", "1"]:

        class_dir = os.path.join(dataset_dir, label)

        if not os.path.exists(class_dir):
            continue

        for image_name in os.listdir(class_dir):

            image_path = os.path.join(class_dir, image_name)

            data.append([image_path, int(label)])
    with open(output_csv, mode="w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["image", "label"])

        writer.writerows(data)

    print(f"CSV saved to: {output_csv}")

if __name__ == "__main__":
        create_csv()