"""ResNet-101 backbone and its FC classification head."""
from torch import nn
from torchvision import models


def create(num_classes, pretrained=True):
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    model = models.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def head(model):
    return model.fc


def last_blocks(model, count):
    return list(model.layer4.children())[-count:]
