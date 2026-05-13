from datasets.preprocessing.loader import Dataset
import os

if __name__ == "__main__":
    
    # Khoi tao class
    dataset = Dataset(
        zip_path="datasets/raw/Augmented-Resized Image.zip",
        extract_dir="datasets/processed",
        dataset_dir="project-middle/datasets",
        objects=["Orange", "Mango", "Grape", "Banana"],
        status=["Rotten", "Fresh", "Formalin-mixed"]
    )

    # # Giai nen thu muc goc
    # dataset.extract_zip(
    #     flatten=False,
    #     progress=True,
    #     default=True,
    #     message=False,
    #     chunk_size=1024 * 64
    # )

    # # Giai nen cac thu muc con
    # dataset.extract_all(
    #     progress=True,
    #     info=False
    # )

    # # Khoi tao dieu kien voi gop 4 loai trai cay voi ca 3 truong hop
    # dataset.fit(
    #     merge=True, 
    #     param=None, 
    #     status=None
    # )

    # # Khoi tao file dataset
    # dataset.transform(
    #     name_file="dataset.csv",
    #     index=False,
    #     end=False
    # )

    # Xoa file khong su dung den
    dataset.clear_folder(
        folder=os.path.dirname(dataset.zip_path),
        mode="file",
        exclude=["__init__.py"]
    )

    #Xoa folder de upload code len github
    dataset.clear_folder(
        folder=dataset.extract_dir,
        mode="folder",
        exclude=["__init__.py"]
    )




