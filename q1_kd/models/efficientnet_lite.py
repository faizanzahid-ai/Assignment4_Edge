"""
EfficientNet-Lite0 (TensorFlow "lite" variant of EfficientNet-B0) for 32x32 inputs.

Lite changes w.r.t. EfficientNet-B0 (to be friendly to mobile accelerators and
post-training quantisation):
  * no Squeeze-and-Excitation blocks,
  * ReLU6 instead of Swish,
  * stem (32 ch) and head (1280 ch) are not scaled.
CIFAR adaptation: stride-1 stem and stride-1 second stage (total stride 8).
"""
import torch
import torch.nn as nn

from .mobilenetv2 import conv_bn_relu6, _init_weights


class MBConv(nn.Module):
    """Mobile inverted bottleneck (no SE) with optional residual connection."""

    def __init__(self, cin, cout, kernel, stride, expand, drop_connect=0.0):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.drop_connect = drop_connect
        layers = []
        if expand != 1:
            layers.append(conv_bn_relu6(cin, hidden, k=1))
        layers += [conv_bn_relu6(hidden, hidden, k=kernel, stride=stride, groups=hidden),
                   nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_connect > 0:   # stochastic depth
            keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_connect
            out = out * keep / (1.0 - self.drop_connect)
        return x + out


class EfficientNetLite0(nn.Module):
    # (expand, kernel, stride, out_channels, repeats) -- B0 stages, strides adapted for CIFAR
    CONFIG = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
              (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]
    feature_channels = (112, 1280)

    def __init__(self, num_classes=100, dropout=0.2, drop_connect=0.1):
        super().__init__()
        self.stem = conv_bn_relu6(3, 32, 3, 1)
        total = sum(c[4] for c in self.CONFIG)
        blocks, cin, idx = [], 32, 0
        for e, k, s, c, n in self.CONFIG:
            for i in range(n):
                blocks.append(MBConv(cin, c, k, s if i == 0 else 1, e, drop_connect * idx / total))
                cin, idx = c, idx + 1
        split = sum(c[4] for c in self.CONFIG[:5])     # after the 112-channel stage (8x8)
        self.stage_mid = nn.Sequential(*blocks[:split])
        self.stage_final = nn.Sequential(*blocks[split:])
        self.head = conv_bn_relu6(cin, 1280, k=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        mid = self.stage_mid(self.stem(x))          # 112 x 8 x 8
        final = self.head(self.stage_final(mid))    # 1280 x 4 x 4
        logits = self.classifier(torch.flatten(self.pool(final), 1))
        return (logits, [mid, final]) if return_features else logits
