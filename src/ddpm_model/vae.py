import torch
from src.ddpm_model.attention import SelfAttention
from torch import nn
from torch.nn import functional as F

def reparameterise(mean, log_var, noise):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mean + std*eps

class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class AttentionBlockVAE(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x

class EncoderVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # simply following from - https://github.com/kjsman/stable-diffusion-pytorch/blob/main/stable_diffusion_pytorch/encoder.py
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlockVAE(128, 128),
            ResidualBlockVAE(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlockVAE(128, 256),
            ResidualBlockVAE(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlockVAE(256, 512),
            ResidualBlockVAE(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            AttentionBlockVAE(512), 
            ResidualBlockVAE(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x, noise):
        
        for module in self.encoder:
            if getattr(module, "stride", None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1) # split the tensor in half along the channel dimension
        log_variance = torch.clamp(log_variance, -30, 20) # makes sure the log variance does not become too small or too large
        x = reparameterise(mean, log_variance, noise)
        x *= 0.18215 # doesn't say why this was used
        return x

class DecoderVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential()

    def forward(self, x):
        pass


class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlockVAE(512, 512),
            AttentionBlockVAE(512),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            ResidualBlockVAE(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlockVAE(512, 256),
            ResidualBlockVAE(256, 256),
            ResidualBlockVAE(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlockVAE(256, 128),
            ResidualBlockVAE(128, 128),
            ResidualBlockVAE(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x

if __name__ == "__main__":
    encoder = EncoderVAE()
    x = torch.randn((1, 3, 64, 64))
    noise = torch.randn((1, 8, 32, 32))
    z = encoder(x, noise)
    
    print (z.shape)