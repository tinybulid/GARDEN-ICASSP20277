import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules
from torchvision.ops import Conv2dNormActivation

from ..data.features import N_MELS, to_db, to_unit
from ..utils import CKPT_DIR
from .cyclegan import Generator

BACKBONES = ("resnet50", "efficientnet_b0", "cnn14", "ast", "vit_b16")
N_CLASSES = 10
PANNS_CKPT = CKPT_DIR / "Cnn14_mAP=0.431.pth"
AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x, pool=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool)


class Cnn14(nn.Module):
    def __init__(self, n_classes=N_CLASSES, pretrained=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", ConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.head = nn.Linear(2048, n_classes)
        if pretrained:
            state = torch.load(PANNS_CKPT, map_location="cpu")["model"]
            own = self.state_dict()
            self.load_state_dict({k: v for k, v in state.items() if k in own and v.shape == own[k].shape}, strict=False)

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(6):
            block = getattr(self, f"conv_block{i + 1}")
            x = F.dropout(block(x, (1, 1) if i == 5 else (2, 2)), 0.2, self.training)
        x = x.mean(3)
        x = x.max(2).values + x.mean(2)
        x = F.dropout(x, 0.5, self.training)
        x = F.dropout(F.relu(self.fc1(x)), 0.5, self.training)
        return self.head(x)


class AST(nn.Module):
    def __init__(self, n_classes=N_CLASSES, pretrained=True):
        super().__init__()
        from transformers import ASTConfig, ASTForAudioClassification
        kw = dict(num_mel_bins=N_MELS, max_length=64, num_labels=n_classes)
        if pretrained:
            self.model = ASTForAudioClassification.from_pretrained(AST_NAME, ignore_mismatched_sizes=True, **kw)
        else:
            self.model = ASTForAudioClassification(ASTConfig(**kw))

    def forward(self, x):
        return self.model(input_values=x.squeeze(1).transpose(1, 2)).logits


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.dequant(x)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.gamma * (x * nx) + self.beta + x
        return self.quant(x)


class CPMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate, stride):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)
        exp_conv = Conv2dNormActivation(in_channels, exp_channels, kernel_size=1, stride=1,
                                        norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, inplace=False)
        depth_conv = Conv2dNormActivation(exp_channels, exp_channels, kernel_size=3, stride=stride, padding=1,
                                          groups=exp_channels, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, inplace=False)
        proj_conv = Conv2dNormActivation(exp_channels, out_channels, kernel_size=1, stride=1,
                                         norm_layer=nn.BatchNorm2d, activation_layer=None, inplace=False)
        self.after_block_norm = GRN()
        self.after_block_activation = nn.ReLU()
        self.use_shortcut = in_channels == out_channels and (stride == 1 or stride == (1, 1))
        self.shortcut = nn.Sequential() if self.use_shortcut else None
        self.block = nn.Sequential(exp_conv, depth_conv, proj_conv)
        self.skip_add = FloatFunctional()

    def forward(self, x):
        result = self.block(x)
        if self.use_shortcut:
            result = self.skip_add.add(result, self.shortcut(x))
        result = self.after_block_norm(result)
        result = self.after_block_activation(result)
        return result


class CPMobile(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_classes = config["n_classes"]
        in_channels = config["in_channels"]
        base_channels = config["base_channels"]
        channels_multiplier = config["channels_multiplier"]
        expansion_rate = config["expansion_rate"]
        n_blocks = config["n_blocks"]
        strides = config["strides"]
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels, channels_per_stage[0] // 4, kernel_size=3, stride=2, inplace=False),
            Conv2dNormActivation(channels_per_stage[0] // 4, channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU, kernel_size=3, stride=2, inplace=False),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id], channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id], strides=strides, expansion_rate=expansion_rate)
            self.stages.add_module(f"s{stage_id + 1}", stage)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(channels_per_stage[-1], n_classes, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, strides, expansion_rate):
        stage = nn.Sequential()
        for _ in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f"b{block_id}"
            self.total_block_count += 1
            stride = strides.get(bname, (1, 1))
            stage.add_module(bname, CPMobileBlock(in_channels, out_channels, expansion_rate, stride))
            in_channels = out_channels
        return stage

    def forward(self, x):
        x = self.quant(x)
        x = self.in_c(x)
        x = self.stages(x)
        x = self.feed_forward(x)
        logits = x.squeeze(2).squeeze(2)
        return self.dequant(logits)

    def fuse_model(self):
        for module_name, module_instance in self.named_modules():
            if module_name == "in_c":
                fuse_modules(module_instance[0], ["0", "1", "2"], inplace=True)
                fuse_modules(module_instance[1], ["0", "1", "2"], inplace=True)
            elif isinstance(module_instance, CPMobileBlock):
                fuse_modules(module_instance.block[0], ["0", "1", "2"], inplace=True)
                fuse_modules(module_instance.block[1], ["0", "1", "2"], inplace=True)
                fuse_modules(module_instance.block[2], ["0", "1"], inplace=True)
            elif module_name == "feed_forward":
                fuse_modules(module_instance, ["0", "1"], inplace=True)


def cp_mobile(n_classes=N_CLASSES):
    return CPMobile({
        "n_classes": n_classes,
        "in_channels": 1,
        "base_channels": 32,
        "channels_multiplier": 2.0,
        "expansion_rate": 3.0,
        "n_blocks": [3, 3, 2],
        "strides": {"b2": (2, 2), "b4": (2, 1)},
    })


def build_backbone(name, n_classes=N_CLASSES, pretrained=True):
    if name == "cpmobile":
        return cp_mobile(n_classes)
    if name == "cnn14":
        return Cnn14(n_classes, pretrained)
    if name == "ast":
        return AST(n_classes, pretrained)
    if name == "vit_b16":
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=n_classes,
                                 in_chans=1, img_size=(N_MELS, 64))
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes, in_chans=1)


class SceneClassifier(nn.Module):
    def __init__(self, backbone, n_classes=N_CLASSES, pretrained=True):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.net = build_backbone(backbone, n_classes, pretrained)

    def forward(self, x):
        return self.net(self.norm(x))


@torch.no_grad()
def to_reference(G, x, is_ref):
    if (~is_ref).any():
        x = x.clone()
        x[~is_ref] = to_db(G(to_unit(x[~is_ref])))
    return x


class GardenClassifier(nn.Module):
    def __init__(self, backbone, G, n_classes=N_CLASSES, pretrained=True):
        super().__init__()
        self.G = G.eval().requires_grad_(False)
        self.clf = SceneClassifier(backbone, n_classes, pretrained)

    def forward(self, x, is_ref):
        return self.clf(to_reference(self.G, x, is_ref))


def load_classifier(ckpt):
    b = ckpt["backbone"]
    if ckpt["garden"]:
        model = GardenClassifier(b, Generator(), N_CLASSES, pretrained=False)
    else:
        model = SceneClassifier(b, N_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model
