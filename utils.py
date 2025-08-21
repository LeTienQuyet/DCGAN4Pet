import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def prepare_data(root_dir="datasets", batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = PetDataset(os.path.join(root_dir, "train"), transform)
    dev_dataset = PetDataset(os.path.join(root_dir, "dev"), transform)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False, pin_memory=True)

    return train_dataloader, dev_dataloader