from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

class TransformDataset(Dataset):

    def __init__(self, csv_path=None, transform=True):

        if not os.path.exists(csv_path):
            raise FileExistsError(f"file already exists: {csv_path}")

        self.image_paths = None
        self.labels = None
        self.class_name = None
        self.csv_path = csv_path
        self.transform = transform

    def _load_dataset(self, csv_path=None):
        df = pd.read_csv(csv_path)

        return (
            df["image_path"].tolist(),
            df["label"].tolist(),
            df["class_name"].tolist()
        )
    
    @property
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path, label = self.image_paths[index], self.labels[index]
        image = Image.open(image_path)
        image = self._transform(image=image) if self.transform else image
        return image, label
    
    def _transform(self, image, width=224, height=224):
        image = image.resize((width, height))
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )(image)
        return image
    
    def show_image(self, tensor_img):
        img = tensor_img.permute(1, 2, 0)  
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
    def fit(self):
        image_path, label, class_name = self._load_dataset(csv_path=self.csv_path)
        self.image_paths = image_path
        self.labels = label
        self.class_name = class_name


if __name__ == "__main__":

    transform = TransformDataset(
        csv_path="project-middle/datasets/dataset.csv"
    )

    transform.fit()
    # print(transform.__len__)
    image, label = transform.__getitem__(10)
    transform.show_image(image)



    
