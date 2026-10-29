"""VGG-16 backbone and its fully connected classification head."""
from torch import nn
from torchvision import models


def create(num_classes, pretrained=True):
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def head(model):
    return model.classifier


def last_blocks(model, count):
    # Final convolution layers, not arbitrary ReLU/MaxPool children.
    convolutions = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    return convolutions[-count:]
