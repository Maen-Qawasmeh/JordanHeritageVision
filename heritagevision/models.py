"""Small architecture registry; each model lives in its own file."""
from .architectures import efficientnet_b0, resnet101, vgg16, inception_v3

ARCHITECTURES = {
    'efficientnet_b0': efficientnet_b0,
    'resnet101': resnet101,
    'vgg16': vgg16,
    'inception_v3': inception_v3,
}
SUPPORTED_MODELS = tuple(ARCHITECTURES)
DEFAULT_IMAGE_SIZES = {
    'efficientnet_b0': 224,
    'resnet101': 224,
    'vgg16': 224,
    'inception_v3': 299,
}


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name not in ARCHITECTURES:
        raise ValueError(f'Unknown model {model_name!r}; supported: {SUPPORTED_MODELS}')
    return ARCHITECTURES[model_name].create(num_classes, pretrained)


def freeze_backbone(model, model_name='efficientnet_b0'):
    """Train only newly replaced classifier output (or the classifier module)."""
    architecture = ARCHITECTURES[model_name]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Only NEW final layer is trainable: VGG's earlier pretrained FC layers
    # stay frozen to avoid large parameter/VRAM consumption.
    module = architecture.head(model)
    final_layer = module[-1] if hasattr(module, '__getitem__') else module
    for parameter in final_layer.parameters():
        parameter.requires_grad_(True)


def unfreeze_last_blocks(model, model_name='efficientnet_b0', last_n=2):
    if last_n < 1:
        raise ValueError('last_n must be positive')
    freeze_backbone(model, model_name)
    architecture = ARCHITECTURES[model_name]
    for block in architecture.last_blocks(model, last_n):
        for parameter in block.parameters():
            parameter.requires_grad_(True)
