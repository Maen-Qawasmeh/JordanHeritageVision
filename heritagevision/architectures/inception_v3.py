"""InceptionV3 classifier with 299x299 inputs and no auxiliary output.

Torchvision requires aux_logits=True while loading pretrained ImageNet weights.
Once weights are loaded, the auxiliary branch is disabled so our training
pipeline receives a single logits tensor of shape [batch_size, num_classes].
"""
from torch import nn
from torchvision import models


def create(num_classes, pretrained=True):
    if pretrained:
        # The pretrained checkpoint contains an auxiliary classifier.
        # Torchvision requires it to exist WHILE loading that checkpoint.
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            aux_logits=True,
            transform_input=True,
        )
        # Our training loop expects one logits tensor, not InceptionOutputs.
        model.aux_logits = False
        model.AuxLogits = None
    else:
        # For resume/evaluation, create the exact same architecture without
        # downloading ImageNet weights, then load our saved model_state.
        model = models.inception_v3(
            weights=None,
            aux_logits=False,
            transform_input=True,
            init_weights=False,
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def head(model):
    return model.fc


def last_blocks(model, count):
    return [getattr(model, name) for name in ("Mixed_7a", "Mixed_7b", "Mixed_7c")[-count:]]
