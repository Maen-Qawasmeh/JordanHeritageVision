"""Verify the local PyTorch CUDA installation."""
import torch

print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print('Running on CPU. For NVIDIA GPU support, install the CUDA-enabled PyTorch build.')
