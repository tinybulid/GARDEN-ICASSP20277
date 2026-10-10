import random

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("cond", "uncond", "none")


class ResidualBlock(nn.Module):
    def __init__(self, ch, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.norm1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3)
        self.norm2 = nn.InstanceNorm2d(ch)
        self.film = nn.Linear(cond_dim, 2 * ch) if cond_dim else None

    def forward(self, x, e=None):
        h = self.norm1(self.conv1(F.pad(x, (1, 1, 1, 1), mode="reflect")))
        if self.film is not None:
            scale, shift = self.film(e)[..., None, None].chunk(2, 1)
            h = h * (1 + scale) + shift
        h = F.relu(h)
        h = self.norm2(self.conv2(F.pad(h, (1, 1, 1, 1), mode="reflect")))
        return x + h


class Generator(nn.Module):
    def __init__(self, n_sources=None, ngf=64, n_blocks=9, emb_dim=128):
        super().__init__()
        self.embed = nn.Embedding(n_sources, emb_dim) if n_sources else None
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(1, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), nn.InstanceNorm2d(ngf * 4), nn.ReLU(True),
        )
        cond = emb_dim if n_sources else None
        self.blocks = nn.ModuleList(ResidualBlock(ngf * 4, cond) for _ in range(n_blocks))
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, 7), nn.Tanh(),
        )

    def forward(self, x, s=None):
        e = self.embed(s) if self.embed is not None else None
        h = self.head(x)
        for block in self.blocks:
            h = block(h, e)
        return self.tail(h)


def patch_discriminator(ndf=64):
    return nn.Sequential(
        nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 8, 1, 4, 1, 1),
    )


class ImagePool:
    def __init__(self, size=50):
        self.size = size
        self.items = []

    def __call__(self, batch):
        out = []
        for x in batch.detach():
            x = x[None]
            if len(self.items) < self.size:
                self.items.append(x)
                out.append(x)
            elif random.random() < 0.5:
                i = random.randrange(self.size)
                out.append(self.items[i].clone())
                self.items[i] = x
            else:
                out.append(x)
        return torch.cat(out)


def lr_lambda(hold=200, total=400):
    return lambda epoch: 1.0 if epoch < hold else 1.0 - (epoch - hold) / (total - hold)


class CycleGAN(nn.Module):
    def __init__(self, n_sources, mode="cond", lambda_cyc=10.0, pool_size=50):
        super().__init__()
        self.mode = mode
        self.G = Generator()
        self.D_A = patch_discriminator()
        self.F = None if mode == "none" else Generator(n_sources if mode == "cond" else None)
        self.D_B = None if mode == "none" else patch_discriminator()
        self.lambda_cyc = lambda_cyc
        self.pool_a = ImagePool(pool_size)
        self.pool_b = ImagePool(pool_size)

    def generator_loss(self, xa, xb, sb):
        fake_a = self.G(xb)
        loss = 0.5 * (self.D_A(fake_a) - 1).pow(2).mean()
        if self.F is None:
            return loss, fake_a.detach(), None
        s = sb[torch.randint(len(sb), (len(xa),), device=sb.device)]
        fake_b = self.F(xa, s)
        loss = loss + 0.5 * (self.D_B(fake_b) - 1).pow(2).mean()
        cyc = (self.G(fake_b) - xa).abs().mean() + (self.F(fake_a, sb) - xb).abs().mean()
        return loss + self.lambda_cyc * cyc, fake_a.detach(), fake_b.detach()

    def discriminator_loss(self, xa, xb, fake_a, fake_b):
        fake_a = self.pool_a(fake_a)
        loss = 0.5 * (self.D_A(xa) - 1).pow(2).mean() + 0.5 * self.D_A(fake_a).pow(2).mean()
        if self.D_B is None:
            return loss
        fake_b = self.pool_b(fake_b)
        return loss + 0.5 * (self.D_B(xb) - 1).pow(2).mean() + 0.5 * self.D_B(fake_b).pow(2).mean()
