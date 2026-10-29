"""Load saved checkpoints with the right network and image transform."""
from pathlib import Path
import torch
from .models import build_model
from .data import build_transform


def load_checkpoint(path: str | Path, device: torch.device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No checkpoint at {path}')
    # Trust checkpoints you created yourself; never open arbitrary downloads.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt['model_name'], len(ckpt['classes']), pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt, build_transform(False, ckpt['image_size'])
