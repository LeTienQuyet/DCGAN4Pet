from torcheval.metrics import FrechetInceptionDistance
import torch
import torch.nn.functional as F

def calculate_fid_score(generator, dev_dataloader, batch_size=128, num_dims=100, device="cuda"):
    fid = FrechetInceptionDistance().to(device)
    generator.eval()

    with torch.no_grad():
        for real_images in dev_dataloader:
            real_images = real_images.to(device)
            fid.update(real_images, is_real=True)

            # Fake image
            noises = torch.randn(real_images.size(0), num_dims, 1, 1, device=device)
            fake_images = generator(noises)

            # Range from [-1, 1] to [0, 1]
            fake_images = fake_images.mul(0.5).add(0.5).clamp(0, 1)

            # Resize 64 x 64 => 299 x 299
            fake_images = F.interpolate(fake_images, size=(299, 299), mode="bilinear", align_corners=False)

            fid.update(fake_images, is_real=False)
    return fid.compute().item()