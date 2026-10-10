import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import LabelEmbedding


def sinusoid(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], 1)


def gn(ch):
    return nn.GroupNorm(32, ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, edim):
        super().__init__()
        self.norm1 = gn(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(edim, 2 * cout)
        self.norm2 = gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, e):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(e))[..., None, None].chunk(2, 1)
        h = self.norm2(h) * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = gn(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x, e=None):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).transpose(2, 3).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class Stage(nn.Module):
    def __init__(self, cin, cout, edim, attn):
        super().__init__()
        self.res = ResBlock(cin, cout, edim)
        self.attn = Attention(cout) if attn else None

    def forward(self, x, e):
        x = self.res(x, e)
        return x if self.attn is None else self.attn(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, e=None):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, e=None):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Denoiser(nn.Module):
    def __init__(self, n_devices=10, n_scenes=10, in_ch=8, ch=128, mults=(1, 2, 4), n_res=2, dim=512):
        super().__init__()
        self.ch = ch
        self.t_mlp = nn.Sequential(nn.Linear(ch, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.device_emb = LabelEmbedding(n_devices, dim)
        self.scene_emb = LabelEmbedding(n_scenes, dim)
        self.null_d = nn.Parameter(torch.zeros(dim))
        self.null_c = nn.Parameter(torch.zeros(dim))

        last = len(mults) - 1
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        skips, cur = [ch], ch
        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            for _ in range(n_res):
                self.down.append(Stage(cur, ch * m, dim, attn=i == last))
                cur = ch * m
                skips.append(cur)
            if i < last:
                self.down.append(Down(cur))
                skips.append(cur)

        self.mid = nn.ModuleList([ResBlock(cur, cur, dim), Attention(cur), ResBlock(cur, cur, dim)])

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(n_res + 1):
                self.up.append(Stage(cur + skips.pop(), ch * m, dim, attn=i == last))
                cur = ch * m
            if i > 0:
                self.up.append(Up(cur))

        self.out = nn.Sequential(gn(cur), nn.SiLU(), nn.Conv2d(cur, in_ch, 3, padding=1))

    def embed(self, t, d, c, drop_d, drop_c):
        ed = self.device_emb(d)
        ec = self.scene_emb(c)
        ed = torch.where(drop_d[:, None], self.null_d.expand_as(ed), ed)
        ec = torch.where(drop_c[:, None], self.null_c.expand_as(ec), ec)
        return self.t_mlp(sinusoid(t, self.ch)) + ed + ec

    def forward(self, z, t, d, c, drop_d, drop_c):
        e = self.embed(t, d, c, drop_d, drop_c)
        h = self.inp(z)
        hs = [h]
        for layer in self.down:
            h = layer(h, e)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, e)
        for layer in self.up:
            h = layer(h, e) if isinstance(layer, Up) else layer(torch.cat([h, hs.pop()], 1), e)
        return self.out(h)


class LatentDiffusion(nn.Module):
    def __init__(self, n_devices=10, n_scenes=10, T=1000, beta_start=1e-4, beta_end=0.02,
                 p_scene=0.5, p_device=0.2, latent_shape=(8, 64, 16)):
        super().__init__()
        self.net = Denoiser(n_devices, n_scenes, in_ch=latent_shape[0])
        self.T = T
        self.p_scene, self.p_device = p_scene, p_device
        self.latent_shape = latent_shape
        betas = torch.linspace(beta_start, beta_end, T)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))

    def loss(self, z0, d, c):
        n = z0.size(0)
        t = torch.randint(0, self.T, (n,), device=z0.device)
        eps = torch.randn_like(z0)
        ab = self.abar[t].view(-1, 1, 1, 1)
        zt = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
        drop_c = torch.rand(n, device=z0.device) < self.p_scene
        drop_d = drop_c & (torch.rand(n, device=z0.device) < self.p_device / self.p_scene)
        return F.mse_loss(self.net(zt, t, d, c, drop_d, drop_c), eps)

    def guided_eps(self, z, t, d, c, wd, wc):
        n = z.size(0)
        on = torch.ones(n, dtype=torch.bool, device=z.device)
        off = ~on
        drop_d = torch.cat([on, off, off])
        drop_c = torch.cat([on, on, off])
        tt = torch.full((3 * n,), int(t), device=z.device)
        e0, ed, edc = self.net(z.repeat(3, 1, 1, 1), tt, d.repeat(3), c.repeat(3), drop_d, drop_c).chunk(3)
        return e0 + wd * (ed - e0) + wc * (edc - ed)

    @torch.no_grad()
    def sample(self, d, c, steps=50, wd=6.0, wc=2.0):
        z = torch.randn(len(d), *self.latent_shape, device=d.device)
        ts = torch.arange(0, self.T, self.T // steps, device=d.device).flip(0)
        for i, t in enumerate(ts):
            ab = self.abar[t]
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(ab)
            eps = self.guided_eps(z, t, d, c, wd, wc)
            x0 = (z - (1 - ab).sqrt() * eps) / ab.sqrt()
            z = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return z
