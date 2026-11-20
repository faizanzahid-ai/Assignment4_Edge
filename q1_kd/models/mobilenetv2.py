"""
MobileNetV2 (Sandler et al., 2018) for 32x32 inputs.

Inverted residual blocks: 1x1 expansion -> 3x3 depthwise -> 1x1 linear
projection, with a residual connection when stride == 1 and in == out.
CIFAR adaptation: stride-1 stem and stride-1 second stage, so the total
down-sampling factor is 8 (32 -> 4) instead of 32.
"""
import torch
import torch.nn as nn


def conv_bn_relu6(cin, cout, k=3, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand_ratio):
        super().__init__()
        hidden = cin * expand_ratio
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_relu6(cin, hidden, k=1))                     # expansion
        layers += [
            conv_bn_relu6(hidden, hidden, k=3, stride=stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, cout, 1, bias=False),                            # linear bottleneck
            nn.BatchNorm2d(cout),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class MobileNetV2(nn.Module):
    # (expansion t, output channels c, repeats n, stride s) -- strides adapted for CIFAR
    CONFIG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
              (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    feature_channels = (96, 1280)

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        cin = int(32 * width_mult)
        self.stem = conv_bn_relu6(3, cin, 3, 1)
        blocks = []
        for t, c, n, s in self.CONFIG:
            cout = int(c * width_mult)
            for i in range(n):
                blocks.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
        # split after the 96-channel stage (8x8) to expose a mid-level feature map
        split = sum(n for _, _, n, _ in self.CONFIG[:5])
        self.stage_mid = nn.Sequential(*blocks[:split])
        self.stage_final = nn.Sequential(*blocks[split:])
        self.head = conv_bn_relu6(cin, 1280, k=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        mid = self.stage_mid(self.stem(x))          # 96 x 8 x 8
        final = self.head(self.stage_final(mid))    # 1280 x 4 x 4
        logits = self.classifier(torch.flatten(self.pool(final), 1))
        return (logits, [mid, final]) if return_features else logits


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
