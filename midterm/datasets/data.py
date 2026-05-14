from torch.utils.data import DataLoader, random_split
import torch

def train_val_test_split(dataset, train=0.7, val=0.15, pin_memory=False, num_workers=4, batch_size=32, random_state=42):
        
        """
        Tac dung:
        - Chia tap du lieu theo ty le

        Dau vao:
        dataset[class]: Truyen du lieu dau vao
        train[float]: Ty le sample train / tong sample
        val[float]: Ty le sample val / tong sample
        pin-memory[bool]: Tang toc voi GPU
        num_workers[int]: So luong hoat dong
        batch_size[int]: So sample tren mot don vi
        random_state[int]: Mot so bat ki de giu data khong thay doi

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

