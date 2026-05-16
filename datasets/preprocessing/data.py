from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchvision import transforms
import pandas as pd
import torch
import os

def train_val_test_split(dataset, train=0.7, val=0.15, pin_memory=False, num_workers=4, batch_size=32, random_state=42):
    
    """
    Tac dung:
    - Chia tap du lieu theo ty le
    
    Dau vao:
    - dataset[class]: Truyen du lieu dau vao
    - train[float]: Ty le sample train / tong sample
    - val[float]: Ty le sample val / tong sample
    - pin-memory[bool]: Tang toc voi GPU
    - num_workers[int]: So luong hoat dong
    - batch_size[int]: So sample tren mot don vi
    - random_state[int]: Mot so bat ki de giu data khong thay doi

    Dau ra:
    - train_ts[class]: Tra ve du lieu chia theo batch_size cua train
    - val_ts[class]: Tra ve du lieu chia theo batch_size cua val
    - test_ts[class]: Tra ve du lieu chia theo batch_size cua test

    Nguon: TrinhNhuNhat_14052026.
    """

    from datasets.preprocessing.transform import TransformDataset
        
    n = len(dataset)
    n_test = int(n * (1 - train - val))
    n_val = int(n * val) 
    n_train = n - n_test - n_val
        
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_state)
    )

    train_ts = DataLoader(
        dataset=train_ds,
        shuffle=True,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_ts = DataLoader(
        dataset=val_ds,
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    test_ts = DataLoader(
        dataset=test_ds,
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_ts, val_ts, test_ts
    
    
def train_val_test_dataset(
        root=None, 
        train=0.7, 
        val=0.15, 
        stratify=True, 
        folder=None, 
        download=False, 
        batch_size=32,
        pin_memory=True,
        num_workers=4,
        active=True,
        random_state=42, 
        **kwargs
    ):

    """
    Tac dung:
    - Tach file dataset goc thanh 3 file train, val, test

    Dau vao:
    - root[str]: Duong dan file dataset
    - train[float]: Ty le sample train / tong sample
    - val[float]: Ty le sample val / tong sample
    - stratify[bool]: Giu ty le phan phoi giua cac class
    - download[bool]: Co download file du lieu ve khong
    - batch_size[int]: Kich thuoc anh moi block train
    - pin_memory[bool]: Co bat tang toc bang gpu khong?
    - num_workers[int]: So luong hoat dong
    - active[bool]: Vi tri lap cua transform 3 o truong hop 2 tf
    - random_state[int]: Yeu to giu ty le random

    Dau ra:
    - train_ts[class]: Tra ve du lieu chia theo batch_size cua train
    - val_ts[class]: Tra ve du lieu chia theo batch_size cua val
    - test_ts[class]: Tra ve du lieu chia theo batch_size cua test

    Nguon: TrinhNhuNhat_14052026.
    """

    from datasets.preprocessing.transform import TransformDataset

    if not os.path.exists(root):
        raise FileExistsError(f"file already exists {root}")
    
    if download and not folder:
        raise ValueError(f"download is True when folder must be str")
    
    if len(kwargs) < 1 or len(kwargs) > 3:
        raise ValueError(f"kwargs there may be 1, 2 or 3 items, got {len(kwargs)}")

    df = pd.read_csv(root)
    df = shuffle(df, random_state=random_state).reset_index(drop=True)

    image_paths = df["image_path"].tolist()
    labels = df["label"].tolist()
    class_names = df["class_name"].tolist()

    temp = round(1.0 - train, 10)
    val_after = round(val / temp, 10)
    test = round(1.0 - val_after, 10)

    X_train, X_temp, y_train, y_temp, class_train, class_temp = train_test_split(
        image_paths,
        labels,
        class_names,
        test_size=temp,
        stratify=labels if stratify else None,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test, class_val, class_test = train_test_split(
        X_temp,
        y_temp,
        class_temp,
        test_size=test,
        stratify=y_temp if stratify else None,
        random_state=random_state
    )

    df_train = pd.DataFrame({
        "image_path": X_train,
        "label": y_train,
        "class_name": class_train
    })

    df_val = pd.DataFrame({
        "image_path": X_val,
        "label": y_val,
        "class_name": class_val
    })

    df_test = pd.DataFrame({
        "image_path": X_test,
        "label": y_test,
        "class_name": class_test
    })

    func = list(kwargs.values())
    len_tf = len(func)

    train_tf = func[0]
    valnn_tf = func[0] if len_tf == 1 else func[1]
    testn_tf = func[0] if len_tf == 1 else ((func[0] if active else func[1]) if len_tf == 2 else func[2])

    train_ds = TransformDataset(
        datasets=df_train,
        transform=train_tf
    )

    val_ds = TransformDataset(
        datasets=df_val,
        transform=valnn_tf
    )

    test_ds = TransformDataset(
        datasets=df_test,
        transform=testn_tf
    )

    train_ts = DataLoader(
        dataset=train_ds,
        shuffle=True,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_ts = DataLoader(
        dataset=val_ds,
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    test_ts = DataLoader(
        dataset=test_ds,
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    if download and folder:
        os.makedirs(folder, exist_ok=True)
        df_train.to_csv(f"{folder}/train.csv", index=False)
        df_val.to_csv(f"{folder}/val.csv", index=False)
        df_test.to_csv(f"{folder}/test.csv", index=False)

        return None

    return train_ts, val_ts, test_ts

def load_datasets(root=None, pin_memory=True, active=True, batch_size=32, num_workers=4, **kwargs):

    """
    Tac dung:
    - Doc file da duoc tach roi tra ve train, val, test

    Dau vao:
    - root[str]: Duong dan file dataset
    - batch_size[int]: Kich thuoc anh moi block train
    - active[bool]: Vi tri lap cua transform 3 o truong hop 2 tf
    - pin_memory[bool]: Co bat tang toc bang gpu khong?
    - num_workers[int]: So luong hoat dong

    Dau ra:
    - train_ts[class]: Tra ve du lieu chia theo batch_size cua train
    - val_ts[class]: Tra ve du lieu chia theo batch_size cua val
    - test_ts[class]: Tra ve du lieu chia theo batch_size cua test

    Nguon: TrinhNhuNhat_16052026.
    """

    from datasets.preprocessing.transform import TransformDataset

    file_train = f"{root}/train.csv"
    file_val = f"{root}/val.csv"
    file_test = f"{root}/test.csv"

    func = list(kwargs.values())
    len_tf = len(func)

    train_tf = func[0]
    valnn_tf = func[0] if len_tf == 1 else func[1]
    testn_tf = func[0] if len_tf == 1 else ((func[0] if active else func[1]) if len_tf == 2 else func[2])
    
    train_ds = TransformDataset(
        root=file_train,
        transform=train_tf
    )

    val_ds = TransformDataset(
        root=file_val,
        transform=valnn_tf
    )

    test_ds = TransformDataset(
        root=file_test,
        transform=testn_tf
    )

    train_ts = DataLoader(
        dataset=train_ds,
        shuffle=True,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    val_ts = DataLoader(
        dataset=val_ds,
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    test_ts = DataLoader(
        dataset=test_ds,
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_ts, val_ts, test_ts

if __name__ == "__main__":

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # dataset = train_val_test_dataset(
    #     root="midterm/datasets/dataset.csv",
    #     folder="midterm/datasets/cf",
    #     train=0.8,
    #     val=0.1,
    #     stratify=False,
    #     download=False,
    #     random_state=42,
    #     active=True,
    #     transformX=transforms,g
    #     transformY=transforms,
    #     transformZ=transforms,
    # )

    # train_loader, val_loader, test_loader = dataset

    train_loader, val_loader, test_loader = load_datasets(
        root="midterm/datasets/cf", 
        pin_memory=True, 
        active=True, 
        batch_size=32, 
        num_workers=4, 
        transformX=train_transform,
        transformY=train_transform,
    )
    
    print(f"Type: {type(train_loader)}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Batch size: {train_loader.batch_size}")

    print("\n" * 2)
    print(f"Type: {type(val_loader)}")
    print(f"Number of batches: {len(val_loader)}")
    print(f"Batch size: {val_loader.batch_size}")

    print("\n" * 2)
    print(f"Type: {type(test_loader)}")
    print(f"Number of batches: {len(test_loader)}")
    print(f"Batch size: {test_loader.batch_size}")


    


