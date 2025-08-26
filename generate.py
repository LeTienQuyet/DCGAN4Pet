from DCGAN import Generator
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import argparse

def main(num_images=64, filename="default", gen_ckpt="models/generator_best.pt", num_dims=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = Generator(num_dims=num_dims).to(device)
    gen.load_state_dict(
        torch.load(gen_ckpt, map_location=device, weights_only=True)
    )
    gen.eval()

    noises = torch.randn(num_images, num_dims, 1, 1, device=device)
    with torch.no_grad():
        images = gen(noises).detach().cpu()

    grid = make_grid(images, nrow=8, padding=2, normalize=True)
    grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyper-parameters for generate images")

    parser.add_argument("--num_images", type=int, help="Number of images to generate", default=64)
    parser.add_argument("--filename", type=str, help="File name of the image", default="default")
    parser.add_argument("--gen_ckpt", type=str, help="Checkpoint of generator", default="models/generator_best.pt")
    parser.add_argument("--num_dims", type=int, help="No. dimensions of latent space", default=100)

    args = parser.parse_args()

    main(
        num_images=args.num_images,
        filename=args.filename,
        gen_ckpt=args.gen_ckpt,
        num_dims=args.num_dims
    )