import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(w.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(w.bias.data, 0.0)

class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

class UpProjection(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            PixelNorm()
        )

    def forward(self, x):
        return self.scale_factor * self.projection(x)

class Generator(nn.Module):
    def __init__(self, num_dims=100, num_channels=3):
        super().__init__()

        # ( 1 x 1 x num_dims => 4 x 4 x 1024)
        self.trconv1 = nn.ConvTranspose2d(
            in_channels=num_dims, out_channels=1024,
            kernel_size=4, stride=1, padding=0, bias=False
        )
        self.pn1 = PixelNorm()

        # ( 4 x 4 x 1024 => 8 x 8 x 512)
        self.trconv2 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.residual_2 = UpProjection(in_channels=1024, out_channels=512)
        self.pn2 = PixelNorm()

        # ( 8 x 8 x 512 => 16 x 16 x 256)
        self.trconv3 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.residual_3 = UpProjection(in_channels=512, out_channels=256)
        self.pn3 = PixelNorm()

        # ( 16 x 16 x 256 => 32 x 32 x 128)
        self.trconv4 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.residual_4 = UpProjection(in_channels=256, out_channels=128)
        self.pn4 = PixelNorm()

        # ( 32 x 32 x 128 => 64 x 64 x 3)
        self.trconv5 = nn.ConvTranspose2d(
            in_channels=128, out_channels=num_channels,
            kernel_size=4, stride=2, padding=1, bias=True
        )

    def forward(self, x):
        x = F.gelu(self.pn1(self.trconv1(x)))

        residual_2 = self.residual_2(x)
        x = F.gelu(self.pn2(self.trconv2(x) + residual_2))

        residual_3 = self.residual_3(x)
        x = F.gelu(self.pn3(self.trconv3(x) + residual_3))

        residual_4 = self.residual_4(x)
        x = F.gelu(self.pn4(self.trconv4(x) + residual_4))

        x = torch.tanh(self.trconv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        # ( 64 x 64 x 3 => 32 x 32 x 64)
        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=64,
            kernel_size=4, stride=2, padding=1, bias=True
        )

        # ( 32 x 32 x 64 => 16 x 16 x 128)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=128)

        # ( 16 x 16 x 128 => 8 x 8 x 256)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=256)

        # ( 8 x 8 x 256 => 4 x 4 x 512)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(num_features=512)

        # ( 4 x 4 x 512 => 1 x 1 x 1)
        self.conv5 = nn.Conv2d(
            in_channels=512, out_channels=1,
            kernel_size=4, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = self.conv5(x)
        x = x.view(-1)
        return x