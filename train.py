import os
import argparse
import random

from utils import prepare_data
from tqdm import tqdm
from DCGAN import weights_init, Generator, Discriminator
from calculate_fid_score import calculate_fid_score
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torchvision.utils import make_grid
import torch.optim as optim

def train_model(train_dataloader, epoch, gen, dis, genOptimizer, disOptimizer, criterion, num_dims, device, save_pth):
    gen.train()
    dis.train()

    total_lossGen, total_lossDis = 0.0, 0.0

    real_label, fake_label = 1.0, 0.0
    for img in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch", colour="RED"):
        real_data = img.to(device)
        batch = real_data.size(0)

        # Training Discriminator
        dis.zero_grad()
            # For real data
        label = torch.full((batch, ), real_label, device=device)
        output = dis(real_data)
        lossDis_real = criterion(output, label)
        lossDis_real.backward()

            # For fake data
        noises = torch.randn(batch, num_dims, 1, 1, device=device)
        fake_data = gen(noises)
        label.fill_(fake_label)
        output = dis(fake_data.detach())
        lossDis_fake = criterion(output, label)
        lossDis_fake.backward()

            # Compute total loss and update Discriminator
        lossDis = lossDis_fake + lossDis_real
        total_lossDis += lossDis.item()
        disOptimizer.step()

        # Trainging Generator
        gen.zero_grad()
        label.fill_(real_label)
        output = dis(fake_data)
        lossGen = criterion(output, label)
        lossGen.backward()
        total_lossGen += lossGen.item()
            # update Generator
        genOptimizer.step()

    # Save last model
    torch.save(
        gen.state_dict(),
        os.path.join(save_pth, "generator_last.pt")
    )
    torch.save(
        dis.state_dict(),
        os.path.join(save_pth, "discriminator_last.pt")
    )

    print(f"    Training Loss:  Generator = {total_lossGen:.4f}, Discriminator = {total_lossDis:.4f}")
    return total_lossGen, total_lossDis

def main(num_epochs, root_dir, batch_size, lr, beta1, beta2, num_dims, step, save_pth):
    # Make a folder save checkpoint
    os.makedirs(save_pth, exist_ok=True)

    best_fid_score = float("inf")

    # Fixed random
    seed = 412
    random.seed(seed)
    torch.manual_seed(seed)

    # Prepare data for training and evaluation
    train_dataloader, dev_dataloader = prepare_data(root_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Generator with weights have mean = 0, std = 0.02
    gen = Generator(num_dims=num_dims).to(device)
    gen.apply(weights_init)

    # Create Discriminator with weights have mean = 0, std = 0.02
    dis = Discriminator().to(device)
    dis.apply(weights_init)

    # Create Adam optimizer with lr = 0.0002, beta1 = 0.5, beta2 = 0.999
    genOptimizer = optim.Adam(gen.parameters(), lr, (beta1, beta2))
    disOptimizer = optim.Adam(dis.parameters(), lr, (beta1, beta2))

    # Create loss function
    criterion = nn.BCELoss()

    print(f"Details of Training:")
    print(f"    Epochs = {num_epochs}, Batch size = {batch_size}, Latent dims = {num_dims}, Device = {device}")
    print(f"    Optimizer with Learning rate = {lr}, betas = {beta1, beta2}")
    print("Ready Training !!!")

    lossesGen, lossesDis = [], []

    fixed_noises = torch.randn(64, num_dims, 1, 1, device=device)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    fake_images_list = []

    for epoch in range(num_epochs):
        total_lossGen, total_lossDis = train_model(train_dataloader, epoch, gen, dis, genOptimizer, disOptimizer, criterion, num_dims, device, save_pth)
        lossesDis.append(total_lossDis)
        lossesGen.append(total_lossGen)

        fid_score =  calculate_fid_score(gen, dev_dataloader, batch_size, num_dims, device)
        print(f"    FID score = {fid_score}")
        if fid_score < best_fid_score:
            # Save best model
            torch.save(
                gen.state_dict(),
                os.path.join(save_pth, "generator_best.pt")
            )
            torch.save(
                dis.state_dict(),
                os.path.join(save_pth, "discriminator_best.pt")
            )
            best_fid_score = fid_score
            print(f"Save best model at epoch {epoch+1} !!!")

        # Generate fake images
        if epoch % step == 0 or epoch == num_epochs-1:
            with torch.no_grad():
                fake_images = gen(fixed_noises).detach().cpu()
                grid = make_grid(fake_images, nrow=8, padding=2, normalize=True)
                img = grid.permute(1, 2, 0).numpy()

                im = ax.imshow(img, animated=True)
                txt = ax.text(
                    0.5, 1.05, f"Epoch {epoch + 1}",
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=14, color="red"
                )
                fake_images_list.append([im, txt])

    ani = animation.ArtistAnimation(fig, fake_images_list, interval=1500, repeat_delay=1000, blit=True)
    ani.save(os.path.join(save_pth, "training_image.gif"), writer="pillow")

    print(f"Completed Training with {num_epochs} epochs !!!")

    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lossesGen, label='Generator Loss', marker='o')
    plt.plot(epochs, lossesDis, label='Discriminator Loss', marker='o')
    plt.title('Generator & Discriminator loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_pth, "loss.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=30)
    parser.add_argument("--root_dir", type=str, help="Directory of dataset", default="datasets")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0002)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument("--beta1", type=float, help="First betas of optimizer", default=0.5)
    parser.add_argument("--beta2", type=float, help="Second betas of optimizer", default=0.999)
    parser.add_argument("--num_dims", type=int, help="No. dimensions of latent space", default=100)
    parser.add_argument("--step", type=int, help="No. of epoch to save generate images", default=3)
    parser.add_argument("--save_pth", type=str, help="Directory save anything", default="output")

    args = parser.parse_args()

    main(
        num_epochs=args.epoch,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        num_dims=args.num_dims,
        step=args.step,
        save_pth=args.save_pth
    )