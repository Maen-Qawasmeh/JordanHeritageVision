"""Smoke-test the model structure and a forward pass before training."""

import torch

from heritagevision.models import build_model, freeze_backbone


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6

    # Do not download weights for this structural test.
    model = build_model("efficientnet_b0", num_classes, pretrained=False)
    freeze_backbone(model)
    model = model.to(device).eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Device:", device)
    print("Trainable parameters:", trainable)
    print("Frozen parameters:", frozen)

    # Simulate two 224x224 RGB images.
    images = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(images)

    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))
    assert logits.shape == (2, num_classes)
    print("Model forward pass PASSED.")
    print("NOTE: Random weights; output is not a trained prediction.")


if __name__ == "__main__":
    main()
