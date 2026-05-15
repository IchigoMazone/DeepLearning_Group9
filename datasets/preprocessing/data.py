from torch.utils.data import DataLoader, random_split
# from sklearn.model_selection import train_test_split
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
    
    
def train_val_test_dataset(root=None, train=0.7, val=0.15, statify=True, download=False, random_state=42):

    """
    Tac dung:
    - Tach file dataset goc thanh 3 file train, val, test

    Dau vao:
    - root[str]: Duong dan file dataset
    - train[float]: Ty le sample train / tong sample
    - val[float]: Ty le sample val / tong sample
    - statify[bool]: Giu ty le phan phoi giua cac class
    - download[bool]: Co download file du lieu ve khong
    - random_state[int]: Yeu to giu ty le random

    Dau ra:
    - void: Khong tra ve

    Nguon: TrinhNhuNhat_14052026.
    """

    if not os.path.exists(root):
        raise FileExistsError(f"file already exists {root}")

    df = pd.read_csv(root)

    image_paths = df["image_path"].tolist()
    labels = df["label"].tolist()
    class_names = df["class_name"].tolist()

    n = 1000
    test = round(1.0 - train - val, 10)
    n_train = int(n * train)
    n_after = n - n_train

    val_after = round(val / (val + test), 10)
    n_val = int(n_after * val_after)
    n_test = n_after - n_val

    print(n_train, n_val, n_test)
    print(n_train / n + n_val / n + n_test / n)



if __name__ == "__main__":

    train_val_test_dataset(
        root="midterm/datasets/dataset.csv",
        train=0.7,
        val=0.15,
        statify=False,
        download=False,
        random_state=42
    )
    


