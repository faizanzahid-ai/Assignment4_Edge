"""
Model registry for Question 1.

Every model exposes the same interface:
    logits = model(x)
    logits, feats = model(x, return_features=True)
where `feats` = [mid_level_feature_map, final_feature_map].  For 32x32 inputs
the mid-level map is 8x8 and the final map is 4x4 for *all* networks, which
makes feature-based distillation between teacher and students straightforward.
"""
from .teacher import ResNet50Teacher
from .mobilenetv2 import MobileNetV2
from .squeezenet import SqueezeNet
from .efficientnet_lite import EfficientNetLite0

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def build_model(arch, num_classes=100, pretrained=False, stem_stride=1):
    arch = arch.lower()
    if arch == "resnet50":
        return ResNet50Teacher(num_classes, pretrained=pretrained, stem_stride=stem_stride)
    if arch == "mobilenetv2":
        return MobileNetV2(num_classes)
    if arch == "squeezenet":
        return SqueezeNet(num_classes)
    if arch == "efficientnet_lite0":
        return EfficientNetLite0(num_classes)
    raise ValueError(f"unknown architecture '{arch}'")
