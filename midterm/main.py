from datasets.preprocessing.transform import TransformDataset
from datasets.preprocessing.data import train_val_test_split
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])