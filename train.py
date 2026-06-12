import os
import random
import copy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import torch
import torch.optim as optim

from utils import prepare_data, get_args, plot_loss
from tqdm import tqdm
from DCGAN import weights_init, Generator, Discriminator
from calculate_fid_score import calculate_fid_score
from torchvision.utils import make_grid

def save_checkpoint(model, filename, save_pth):
    torch.save(
        model.state_dict(),
        os.path.join(save_pth, f"{filename}.pt")
    )

def update_ema(gen_ema, gen, decay=0.999):
    with torch.no_grad():
        for p_ema, p in zip(gen_ema.parameters(), gen.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

# Train model in 1 epoch
def train_model(train_dataloader, epoch, gen, gen_ema, dis, genOptimizer, disOptimizer, criterion, num_dims, device, save_pth, decay):
    gen.train()
    dis.train()
    total_lossGen, total_lossDis = 0.0, 0.0

    real_label, fake_label = 1.0, 0.0
    for img in tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch", colour="RED"):
        real_data = img.to(device)
        batch = real_data.size(0)

        # Training Discriminator
        disOptimizer.zero_grad()
            # For real data
        label = torch.full((batch, ), real_label, device=device)
        output = dis(real_data)
        lossDis_real = criterion(output, label)
        lossDis_real.backward()

            # For fake data
        noises = torch.randn(batch, num_dims, 1, 1, device=device)
        fake_data = gen(noises)
        label.fill_(fake_label)
        output = dis(fake_data.detach()) # Prevent gradient to Generator when training Discriminator
        lossDis_fake = criterion(output, label)
        lossDis_fake.backward()

            # Compute total loss and update Discriminator
        lossDis = lossDis_fake + lossDis_real
        total_lossDis += lossDis.item()
        disOptimizer.step()

        # Trainging Generator
        genOptimizer.zero_grad()
        label.fill_(real_label)
        output = dis(fake_data)
        lossGen = criterion(output, label)
        lossGen.backward()
        total_lossGen += lossGen.item()
            # update Generator
        genOptimizer.step()

        # Update EMA
        update_ema(gen_ema, gen, decay=decay)

    # Save last model
    save_checkpoint(gen, "generator_last", save_pth)
    save_checkpoint(gen_ema, f"generator_ema_{decay}_last", save_pth)
    save_checkpoint(dis, "discriminator_last", save_pth)

    print(f"    Training Loss:  Generator = {total_lossGen:.4f}, Discriminator = {total_lossDis:.4f}")
    return total_lossGen, total_lossDis

def main(num_epochs, root_dir, batch_size, lr, beta1, beta2, num_dims, step, save_pth, decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make a folder save checkpoint
    os.makedirs(save_pth, exist_ok=True)

    best_fid_score = float("inf")

    # Fixed random
    seed = 412
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Prepare data for training and evaluation
    train_dataloader, dev_dataloader = prepare_data(root_dir, batch_size)

    # Create Generator with weights have mean = 0, std = 0.02
    gen = Generator(num_dims=num_dims).to(device)
    gen.apply(weights_init)

    gen_ema = copy.deepcopy(gen).eval() # EMA model (for evaluation only)
    for p in gen_ema.parameters():
        p.requires_grad_(False)

    # Create Discriminator with weights have mean = 0, std = 0.02
    dis = Discriminator().to(device)
    dis.apply(weights_init)

    gen_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    dis_params = sum(p.numel() for p in dis.parameters() if p.requires_grad)

    # Create Adam optimizer with lr = 0.0002, beta1 = 0.5, beta2 = 0.999
    genOptimizer = optim.Adam(gen.parameters(), lr, (beta1, beta2))
    disOptimizer = optim.Adam(dis.parameters(), lr, (beta1, beta2))

    # Create loss function
    criterion = nn.BCEWithLogitsLoss()

    print(f"Details of Training:")
    print(f"    Total params {gen_params + dis_params:,}")
    print(f"    Epochs = {num_epochs}, Batch size = {batch_size}, Latent dims = {num_dims}, Device = {device}")
    print(f"    Optimizer with Learning rate = {lr}, betas = {beta1, beta2}")
    print("Ready Training !!!")

    lossesGen, lossesDis = [], []

    fixed_noises = torch.randn(64, num_dims, 1, 1, device=device)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    fake_images_list = []

    for epoch in range(1, num_epochs+1):
        total_lossGen, total_lossDis = train_model(train_dataloader, epoch, gen, gen_ema, dis, genOptimizer, disOptimizer, criterion, num_dims, device, save_pth, decay)
        lossesDis.append(total_lossDis)
        lossesGen.append(total_lossGen)

        fid_score =  calculate_fid_score(gen_ema, dev_dataloader, num_dims, device)
        print(f"    FID score = {fid_score}")
        if fid_score < best_fid_score:
            # Save best model
            save_checkpoint(gen, "generator_best", save_pth)
            save_checkpoint(gen_ema, f"generator_ema_{decay}_best", save_pth)
            save_checkpoint(dis, "discriminator_best", save_pth)

            best_fid_score = fid_score
            print(f"Save best model at epoch {epoch} !!!\n")

        # Generate fake images
        if epoch % step == 0 or epoch == 1:
            with torch.no_grad():
                fake_images = gen_ema(fixed_noises).detach().cpu()
                grid = make_grid(fake_images, nrow=8, padding=2, normalize=True)
                img = grid.permute(1, 2, 0).numpy()

                im = ax.imshow(img, animated=True)
                txt = ax.text(
                    0.5, 1.05, f"Epoch {epoch}",
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=14, color="red"
                )
                fake_images_list.append([im, txt])

    ani = animation.ArtistAnimation(fig, fake_images_list, interval=1500, repeat_delay=1000, blit=True)
    ani.save(os.path.join(save_pth, "training_image.gif"), writer="pillow")

    plot_loss(num_epochs, lossesGen, lossesDis, save_pth)

    print(f"\nCompleted Training with best FID =  {best_fid_score:.4f}  !!!")

if __name__ == "__main__":
    args = get_args()

    main(
        num_epochs=args.epoch,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        num_dims=args.num_dims,
        step=args.step,
        save_pth=args.save_pth,
        decay=args.decay
    )