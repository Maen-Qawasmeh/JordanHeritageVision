"""EfficientNet-B0 backbone and its classifier head."""
from torch import nn
from torchvision import models


def create(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def head(model):
    return model.classifier


def last_blocks(model, count):
    return list(model.features.children())[-count:]
