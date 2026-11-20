"""
SqueezeNet v1.1 (Iandola et al., 2016) for 32x32 inputs.

Fire module = 1x1 "squeeze" conv followed by parallel 1x1 and 3x3 "expand"
convs whose outputs are concatenated.  The classifier is a 1x1 conv producing
one map per class followed by global average pooling (no FC layer).
CIFAR adaptation: 3x3 stride-1 stem, BatchNorm after every conv for stable
training from scratch, and max-pools placed so the maps are 32->16->8->4.
"""
import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, cin, squeeze, expand):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1, bias=False),
                                     nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.expand1 = nn.Sequential(nn.Conv2d(squeeze, expand, 1, bias=False),
                                     nn.BatchNorm2d(expand), nn.ReLU(inplace=True))
        self.expand3 = nn.Sequential(nn.Conv2d(squeeze, expand, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(expand), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1(s), self.expand3(s)], dim=1)


class SqueezeNet(nn.Module):
    feature_channels = (384, 512)

    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage_mid = nn.Sequential(
            Fire(64, 16, 64), Fire(128, 16, 64), nn.MaxPool2d(2),     # 32 -> 16
            Fire(128, 32, 128), Fire(256, 32, 128), nn.MaxPool2d(2),  # 16 -> 8
            Fire(256, 48, 192), Fire(384, 48, 192),                   # 384 x 8 x 8
        )
        self.stage_final = nn.Sequential(
            nn.MaxPool2d(2),                                          # 8 -> 4
            Fire(384, 64, 256), Fire(512, 64, 256),                   # 512 x 4 x 4
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        mid = self.stage_mid(self.stem(x))
        final = self.stage_final(mid)
        logits = torch.flatten(self.pool(self.classifier(final)), 1)
        return (logits, [mid, final]) if return_features else logits
