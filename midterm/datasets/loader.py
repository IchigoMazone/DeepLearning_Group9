import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from datasets.preprocessing.loader import Dataset

if __name__ == "__main__":

    # Khoi tao class
    dataset = Dataset(
        zip_path="Fruits_1.zip",
        extract_dir="datasets/processed_fruits1",
        dataset_dir="midterm/datasets",
        objects=["Fruits"],
        status=[
            "Banana",
            "Cucumber",
            "Grapes",
            "Kiwi",
            "Mango",
            "Orange",
            "Pear",
            "Pineapple",
            "Pomegranate",
            "Watermelon",
        ]
    )

    # Giai nen thu muc goc
    dataset.extract_zip(
        flatten=False,
        progress=True,
        default=True,
        message=False,
        chunk_size=1024 * 64
    )

    # Giai nen cac thu muc con
    dataset.extract_all(
        progress=True,
        info=False
    )

    # Khoi tao dieu kien cho bo Fruits_1
    dataset.fit(
        merge=False,
        param=None,
        status=None
    )

    # Khoi tao file dataset
    dataset.transform(
        name_file="dataset_fruits1.csv",
        index=False,
        end=False
    )

    # Xoa file khong su dung den
    # dataset.clear_folder(
    #     folder=os.path.dirname(dataset.zip_path),
    #     mode="file",
    #     exclude=["__init__.py"]
    # )
    #
    # # Xoa folder de upload code len github
    # dataset.clear_folder(
    #     folder=dataset.extract_dir,
    #     mode="folder",
    #     exclude=["__init__.py"]
    # )









