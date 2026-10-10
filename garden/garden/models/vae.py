import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.resnet import ResnetBlock2D

from ..utils import LabelEmbedding

AUDIOLDM = "cvssp/audioldm-s-full-v2"


class ConditionalVAE(nn.Module):
    def __init__(self, n_devices=10, n_scenes=10, dim=512, kl_weight=1e-8, pretrained=AUDIOLDM):
        super().__init__()
        self.ae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
        self.device_emb = LabelEmbedding(n_devices, dim)
        self.scene_emb = LabelEmbedding(n_scenes, dim)
        self.kl_weight = kl_weight
        self.scale = self.ae.config.scaling_factor
        self._e = None
        self.film = nn.ModuleList()
        for block in [m for m in self.ae.modules() if isinstance(m, ResnetBlock2D)]:
            lin = nn.Linear(dim, 2 * block.norm2.num_channels)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            block.norm2.register_forward_hook(self._modulate(lin))
            self.film.append(lin)

    def _modulate(self, lin):
        def hook(module, args, out):
            scale, shift = lin(F.silu(self._e))[..., None, None].chunk(2, 1)
            return out * (1 + scale) + shift
        return hook

    def condition(self, d, c):
        self._e = self.device_emb(d) + self.scene_emb(c)

    def encode(self, x, d, c):
        self.condition(d, c)
        return self.ae.encode(x).latent_dist

    def decode(self, z, d, c):
        self.condition(d, c)
        return self.ae.decode(z).sample

    def loss(self, x, d, c):
        post = self.encode(x, d, c)
        rec = self.decode(post.sample(), d, c)
        return F.l1_loss(rec, x) + self.kl_weight * post.kl().mean()

    @torch.no_grad()
    def to_latent(self, x, d, c):
        return self.encode(x, d, c).sample() * self.scale

    @torch.no_grad()
    def from_latent(self, z, d, c):
        return self.decode(z / self.scale, d, c).clamp(-1, 1)