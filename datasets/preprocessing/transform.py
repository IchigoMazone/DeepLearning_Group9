from torch.utils.data import Dataset
from datasets.preprocessing.data import train_val_test_split
from torchvision import transforms
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

class TransformDataset(Dataset):

    """
    Nguon: TrinhNhuNhat_13-14052026.
    """

    def __init__(self, root=None, transform=None):

        if not os.path.exists(root):
            raise FileExistsError(f"file already exists: {root}")

        self.image_paths, self.labels, self.class_name = self._load_dataset(csv_path=root)
        self.root = root
        self.transform = transform

    def _load_dataset(self, csv_path=None):

        """
        Tac dung: 
        - Doc file datasets dang csv file

        Dau vao:
        - self: class
        - csv[str|None]: Duong dan file 

        Dau ra: 
        - list[str]: Danh sach duong dan anh
        - list[str]: Danh sach nhan
        - list[str]: Danh sach ten nhan

        Nguon: TrinhNhuNhat_13052026.
        """

        df = pd.read_csv(csv_path)

        return (
            df["image_path"].tolist(),
            df["label"].tolist(),
            df["class_name"].tolist()
        )
    
    def __len__(self):

        """
        Tac dung:
        - Lay so sample data

        Dau vao:
        - self: class

        Dau ra:
        - int: So luong sample

        Nguon: TrinhNhuNhat_13052026.
        """

        return len(self.labels)
    
    def __getitem__(self, index):

        """
        Tac dung:
        - Tra ve image_path, label theo index

        Dau vao:
        - self: class 
        - index[int]: Chi so trong data

        Dau ra:
        - image[Tensor]: Tensor cua anh
        - label[int]: Nhan so

        Nguon: TrinhNhuNhat_13052026.
        """

        image_path, label = self.image_paths[index], self.labels[index]
        image = self._read_image(image=image_path)
        image = self.transform(image) if self.transform else image
        return image, int(label)
    
    def _read_image(self, image):

        """
        Tac dung:
        - Doc anh va tra ve Tensor cua anh

        Dau vao:
        - self: class
        - image[str]: Duong dan cua anh

        Dau ra:
        - Tensor: Du lieu cua anh
        """

        with Image.open(image) as img:
            return img.convert("RGB")
    
    def _denormalize(self, image):

        """
        Tac dung:
        - Chuan hoa lai khoang gia tri cua anh

        Dau vao:
        - self: class
        - image[Tensor]: Du lieu cua anh

        Dau ra:
        - Tensor: Du lieu da duoc chuan hoa lai

        Nguon: TrinhNhuNhat_14052026.
        """

        return (image * 0.5) + 0.5 
    
    def _convert_to_jpg(self, input_path, output_path=None, quality=95):

        """
        Tac dung:
        - Tac dung: Chuyen doi anh tu cac dinh dang duoi khac ve .jpg

        Dau vao:
        - self: class
        - input_path[str]: Duong dan anh ban dau
        - output_path[str]: Ten anh sau khi chuyen doi
        - quality[int]: Chat luong anh

        Dau ra:
        - output_path[str]: Duong dan anh sau luu

        Nguon: TrinhNhuNhat_14052026.
        """

        image = Image.open(input_path)
        image = image.convert("RGB") if image.mode != "RGB" else image
        output_path = os.path.splitext(input_path)[0] + ".jpg" if output_path is None else output_path
        image.save(output_path, "JPEG", quality=quality)
        return output_path
    
    def _show_image(self, tensor_img):

        """
        Tac dung:
        - Bieu dien anh cua tung sample

        Dau vao:
        self: class
        tensor_img[Tensor]: Du lieu cua anh

        Dau ra:
        - plt.show(): Bieu dien anh tren man hinh

        Nguon: TrinhNhuNhat_14052026.
        """

        img = tensor_img.permute(1, 2, 0).cpu().numpy() 
        img = self._denormalize(img)
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def stats(self):

        """
        Tac dung: 
        - Dem so luong cua moi class

        Dau vao:
        self: class

        Dau ra:
        - Counter: Du lieu dang dict cua cac class theo class va so luong

        Nguon: TrinhNhuNhat_14052026.
        """

        return Counter(self.labels)
    
    def get_class_map(self, option="class"):

        """
        Tac dung:
        - Tra ve mapping cua class: label va label: class

        Dau vao: 
        - self: class
        - option["class"|"label"|"all"]: Lua chon de in ra

        Dau ra:
        - class_to_label[dict]: Tra ve dict du lieu class theo label
        - label_to_class[dict]: Tra ve dict du lieu labeL theo class

        Nguon: TrinhNhuNhat_14052026.
        """

        if not isinstance(option, str):
            raise TypeError(f"option must be str, got {type(option).__name__}")
        
        if option not in ["class", "label", "all"]:
            raise ValueError(f"option must in class or label, got {option}")

        class_to_label = {}
        label_to_class = {}

        for index, class_name in enumerate(self.class_name):
            if class_name not in class_to_label:
                cls, lbl = self.class_name[index], int(self.labels[index])
                class_to_label[cls] = lbl
                label_to_class[lbl] = cls

        if option == "all": return class_to_label, label_to_class
        return class_to_label if option == "class" else label_to_class

    def show_sample(self, index):

        """
        Tac dung:
        - Bieu dien anh cua tung sample

        Dau vao:
        self: class
        index[int]: Dia chi cua sample

        Dau ra:
        - plt.show(): Bieu dien anh tren man hinh

        Nguon: TrinhNhuNhat_14052026.
        """

        image_path, _ = self.__getitem__(index=index)
        self._show_image(image_path)


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = TransformDataset(
        root="midterm/datasets/dataset.csv",
        transform=transform
    )

    # dataset.show_sample(index=10)
    # print(dataset.get_class_map(option="all"))
    # print(dataset.stats())

    # loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=4
    # )

    # for image, label in loader:
    #     print(image.shape)
    #     print(label.shape)

    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, 
        train=0.7,
        val=0.15,
        pin_memory=True,
        batch_size=32,
        num_workers=4,
        random_state=42
    )

    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
 
   


    





    
