"""Export the trained JordanHeritageVision EfficientNetB0 to ONNX and verify it.

Run from the JordanHeritageVision project root on Windows:
    python export_efficientnet_onnx.py

Requires: torch, torchvision, onnx, onnxscript, onnxruntime, numpy.
Does not modify the original checkpoint, model weights or dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from heritagevision.checkpoints import load_checkpoint
from heritagevision.data import IMAGENET_MEAN, IMAGENET_STD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        default='outputs/experiment_02/checkpoints/best_model.pt',
        help='Your trained, trusted EfficientNetB0 checkpoint',
    )
    parser.add_argument(
        '--output',
        default='web/model/efficientnet_b0.onnx',
        help='Output ONNX path (will create parent directories)',
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint not found: {checkpoint_path}')

    # Export the CPU Float32 weights. The browser will run this model via WASM.
    model, info, _ = load_checkpoint(checkpoint_path, torch.device('cpu'))
    if info['model_name'] != 'efficientnet_b0':
        parser.error(f"Expected efficientnet_b0, got {info['model_name']!r}")

    classes = list(info['classes'])
    image_size = int(info['image_size'])
    if image_size <= 0 or len(classes) != 6:
        parser.error('Checkpoint must contain six classes and a valid image_size')

    model.eval()
    example_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Checkpoint: {checkpoint_path}')
    print(f'Exporting: {output_path} | {image_size}x{image_size} | classes={classes}')
    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        input_names=['images'],
        output_names=['logits'],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )

    # Validate ONNX shape and numerical closeness to the source PyTorch model.
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(output_path))
    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        torch_logits = model(example_input).numpy()
    onnx_logits = session.run(['logits'], {'images': example_input.numpy()})[0]
    if onnx_logits.shape != (1, len(classes)):
        raise SystemExit(f'Unexpected ONNX output shape: {onnx_logits.shape}')

    max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    match = bool(np.allclose(torch_logits, onnx_logits, atol=1e-3, rtol=1e-3))
    print(f'PyTorch vs ONNX max absolute logit difference: {max_abs_diff:.6g}')
    if not match:
        raise SystemExit('FAIL: ONNX/PyTorch logits differ too much; do not publish yet.')

    metadata = {
        'model_name': 'efficientnet_b0',
        'classes': classes,  # Preserve the exact trained class order.
        'image_size': image_size,
        'resize_short_side': round(image_size * 256 / 224),
        'crop_type': 'center',
        'mean': list(IMAGENET_MEAN),
        'std': list(IMAGENET_STD),
        'input_layout': 'NCHW',
        'input_dtype': 'float32',
        'onnx_input': 'images',
        'onnx_output': 'logits',
    }
    metadata_path = output_path.with_name('model_metadata.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print('PASS: ONNX output agrees with PyTorch (sample test).')
    print(f'Model: {output_path} ({output_path.stat().st_size / 1024**2:.2f} MiB)')
    print(f'Metadata: {metadata_path}')
    print('NEXT: implement browser preprocessing to match resize/crop/normalize,')
    print('then compare browser predictions on a real sample image before deployment.')


if __name__ == '__main__':
    main()
