"""
ResNet-50 teacher adapted to 32x32 CIFAR images.

The ImageNet stem (7x7 stride-2 conv + max-pool) shrinks a 32x32 image to 8x8
before the first residual stage, which destroys most spatial information.  We
therefore replace it by a 3x3 stride-1 conv and remove the max-pool, so the four
stages work at 32 -> 16 -> 8 -> 4 resolution.  The backbone is initialised from
ImageNet weights (the new 3x3 stem is initialised from the centre of the
pretrained 7x7 kernels) and then fine-tuned on CIFAR-100.
"""
import torch
import torch.nn as nn
import torchvision


class ResNet50Teacher(nn.Module):
    feature_channels = (1024, 2048)  # channels of [layer3, layer4] outputs

    def __init__(self, num_classes=100, pretrained=True, stem_stride=1):
        """stem_stride=1 -> stages at 32/16/8/4 (best accuracy, GPU).
        stem_stride=2 -> stages at 16/8/4/2 (4x fewer FLOPs, used for CPU-only runs)."""
        super().__init__()
        self.stem_stride = stem_stride
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        old_conv = net.conv1
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                net.conv1.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layer2(self.layer1(x))
        mid = self.layer3(x)            # 1024 x 8 x 8
        final = self.layer4(mid)        # 2048 x 4 x 4
        logits = self.fc(torch.flatten(self.pool(final), 1))
        return (logits, [mid, final]) if return_features else logits
