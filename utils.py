import os
import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image

def plot_loss(num_epochs, lossesGen, lossesDis, save_pth):
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lossesGen, label='Generator Loss')
    plt.plot(epochs, lossesDis, label='Discriminator Loss')
    plt.title('Generator & Discriminator loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_pth, "loss.png"), dpi=300, bbox_inches='tight')

def get_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=30)
    parser.add_argument("--root_dir", type=str, help="Directory of dataset", default="afhq_v2")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0002)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument("--beta1", type=float, help="First betas of optimizer", default=0.5)
    parser.add_argument("--beta2", type=float, help="Second betas of optimizer", default=0.999)
    parser.add_argument("--num_dims", type=int, help="No. dimensions of latent space", default=100)
    parser.add_argument("--step", type=int, help="No. of epoch to save generate images", default=3)
    parser.add_argument("--save_pth", type=str, help="Directory save anything", default="output")
    parser.add_argument("--decay", type=float, help="Decay of EMA", default=0.999)

    args = parser.parse_args()

    return args

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
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_dev = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    train_dataset = PetDataset(os.path.join(root_dir, "train"), transform_train)
    dev_dataset = PetDataset(os.path.join(root_dir, "val"), transform_dev)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_dataloader, dev_dataloader