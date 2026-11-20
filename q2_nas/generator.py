"""
Q2.2 - Automated architecture generator: turns a sampled configuration dict
(see search_space.py) into a trainable torch.nn.Module.
"""
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "hardswish": lambda: nn.Hardswish(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
}


def conv_bn_act(cin, cout, k, stride, act, groups=1):
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act is not None:
        layers.append(ACTIVATIONS[act]())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Standard KxK convolution + BN + activation (residual when shapes match)."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = conv_bn_act(cin, cout, k, stride, act)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


class DepthwiseSeparableBlock(nn.Module):
    """KxK depthwise conv followed by 1x1 pointwise conv (MobileNetV1 style)."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(cin, cin, k, stride, act, groups=cin),
                                  conv_bn_act(cin, cout, 1, 1, act))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


class MBConvBlock(nn.Module):
    """Inverted residual: 1x1 expand (x4) -> KxK depthwise -> 1x1 linear projection."""

    def __init__(self, cin, cout, k, stride, act, expand=4):
        super().__init__()
        hid = cin * expand
        self.body = nn.Sequential(conv_bn_act(cin, hid, 1, 1, act),
                                  conv_bn_act(hid, hid, k, stride, act, groups=hid),
                                  conv_bn_act(hid, cout, 1, 1, None))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


BLOCKS = {"conv": ConvBlock, "dws": DepthwiseSeparableBlock, "mbconv": MBConvBlock}


class GeneratedNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.arch = arch
        self.resolution = arch["resolution"]
        first_act = arch["stages"][0]["activation"]
        self.stem = conv_bn_act(3, arch["stem_channels"], 3, 1, first_act)
        layers, cin = [], arch["stem_channels"]
        for i, st in enumerate(arch["stages"]):
            for d in range(st["depth"]):
                stride = 2 if (i > 0 and d == 0) else 1
                layers.append(BLOCKS[st["block"]](cin, st["channels"], st["kernel"], stride, st["activation"]))
                cin = st["channels"]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(arch.get("dropout", 0.0)), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(self.stem(x))
        return self.classifier(torch.flatten(self.pool(x), 1))


def build_from_config(arch, num_classes=10):
    return GeneratedNet(arch, num_classes)
