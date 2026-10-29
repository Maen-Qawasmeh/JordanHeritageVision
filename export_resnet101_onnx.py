"""Export JordanHeritageVision ResNet101 for browser inference (no retraining).

From the project root:
    python export_resnet101_onnx.py
    python export_resnet101_onnx.py --image "path/to/real_photo.jpg"

Requires: torch, torchvision, onnx, onnxscript, onnxruntime, numpy, Pillow.
Loads only a trusted checkpoint created by your own training pipeline.
Does NOT modify your checkpoint, data, existing website, or README.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from heritagevision.checkpoints import load_checkpoint
from heritagevision.data import IMAGENET_MEAN, IMAGENET_STD, build_transform


def pack_weights_fp16(source: Path, destination: Path) -> int:
    """Store large FP32 weights as FP16 and Cast them back to FP32 at runtime.

    Computation stays float32, making this more compatible with browser WASM
    than converting all convolution operators to float16. It is *lossy* due
    to rounding weights, so the converted model is verified below.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = onnx.load(str(source))
    casts = []
    packed = 0
    graph_inputs = {item.name for item in graph.graph.input}
    for idx, initializer in enumerate(graph.graph.initializer):
        if initializer.data_type != TensorProto.FLOAT or initializer.name in graph_inputs:
            continue
        arr = numpy_helper.to_array(initializer)
        if arr.size < 1024:  # Keep small tensors and most batch-norm statistics in FP32.
            continue
        original_name = initializer.name
        storage_name = f'__jhv_fp16_weight_{idx}'
        packed_value = numpy_helper.from_array(arr.astype(np.float16), name=storage_name)
        initializer.CopyFrom(packed_value)
        casts.append(helper.make_node(
            'Cast', inputs=[storage_name], outputs=[original_name],
            to=TensorProto.FLOAT, name=f'JHV_RestoreFloat32_{idx}',
        ))
        packed += 1

    if not casts:
        raise RuntimeError('No compatible FP32 weight tensors found to pack.')
    for pos, node in enumerate(casts):
        graph.graph.node.insert(pos, node)
    onnx.checker.check_model(graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(graph, str(destination), save_as_external_data=False)
    return packed


def logits_from_torch(model, images):
    with torch.inference_mode():
        return model(images).numpy()


def check_predictions(session, model, example, description):
    expected = logits_from_torch(model, example)
    obtained = session.run(['logits'], {'images': example.numpy()})[0]
    if obtained.shape != expected.shape:
        raise RuntimeError(f'{description}: unexpected shape {obtained.shape}')
    max_diff = float(np.abs(expected - obtained).max())
    matches = np.allclose(expected, obtained, atol=0.05, rtol=0.01)
    same_class = np.array_equal(expected.argmax(axis=1), obtained.argmax(axis=1))
    print(f'{description}: max logit diff={max_diff:.6f}; '
          f'PyTorch top-1={expected.argmax(axis=1).tolist()}, '
          f'ONNX top-1={obtained.argmax(axis=1).tolist()}')
    if not matches or not same_class:
        raise RuntimeError(f'{description}: prediction verification FAILED. Do not publish this ONNX.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('outputs/experiment_04_resnet101/checkpoints/best_model.pt'))
    parser.add_argument('--output', type=Path,
                        default=Path('web/resnet101_export/resnet101.onnx'))
    parser.add_argument('--image', type=Path, default=None,
                        help='Optional real image used only to compare PyTorch vs ONNX inference.')
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if args.image is not None and not args.image.is_file():
        parser.error(f'Image not found: {args.image}')

    import onnx
    import onnxruntime as ort

    model, ckpt, _ = load_checkpoint(args.checkpoint, torch.device('cpu'))
    if ckpt['model_name'] != 'resnet101':
        parser.error(f"Wrong model: {ckpt['model_name']!r}, expected 'resnet101'.")
    classes = list(ckpt['classes'])
    image_size = int(ckpt['image_size'])
    if image_size != 224 or len(classes) != 6:
        parser.error('Expected 224x224 input and exactly six landmark classes.')

    model.eval()
    torch.manual_seed(42)
    sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print('ResNet101 checkpoint:', args.checkpoint)
    print('Class order:', classes)
    print('Exporting full-precision ONNX in a temporary directory...')

    # ResNet101 FP32 (~170MB) must not be committed: GitHub blocks files >100MiB.
    with tempfile.TemporaryDirectory(prefix='jhv_resnet101_') as tempdir:
        fp32 = Path(tempdir) / 'resnet101_fp32.onnx'
        torch.onnx.export(
            model, (sample,), str(fp32), input_names=['images'], output_names=['logits'],
            opset_version=18, dynamo=True, external_data=False,
        )
        onnx.checker.check_model(str(fp32))
        baseline_session = ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider'])
        check_predictions(baseline_session, model, sample, 'FP32 ONNX test')
        del baseline_session

        print('Packing FP32 weights into FP16 storage with FP32 Cast operations...')
        packed_count = pack_weights_fp16(fp32, args.output)

    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    check_predictions(session, model, sample, 'Packed ONNX random-input test')

    if args.image:
        from PIL import Image, ImageOps
        with Image.open(args.image) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            real_input = build_transform(False, image_size)(img).unsqueeze(0)
        check_predictions(session, model, real_input, 'Packed ONNX real-photo test')

    metadata = {
        'model_name': 'resnet101', 'classes': classes, 'image_size': image_size,
        'resize_short_side': round(image_size * 256 / 224), 'crop_type': 'center',
        'mean': list(IMAGENET_MEAN), 'std': list(IMAGENET_STD),
        'input_layout': 'NCHW', 'input_dtype': 'float32',
        'onnx_input': 'images', 'onnx_output': 'logits',
        'weight_storage': 'float16_with_float32_cast',
    }
    metadata_path = args.output.parent / 'resnet101_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    mib = args.output.stat().st_size / (1024 * 1024)
    print(f'PASS: packed {packed_count} weight tensors; ONNX verified on CPU.')
    print(f'ONNX: {args.output} ({mib:.2f} MiB)')
    print(f'Metadata: {metadata_path}')
    if mib >= 100:
        print('WARNING: Model >=100MiB; do not add it to GitHub regular Git history.')
    elif mib >= 50:
        print('NOTE: Model is >50MiB; Git will warn but allows files under 100MiB.')
    print('NEXT: verify accuracy agreement on validation images and test ONNX Runtime Web.')
    print('Do NOT copy into docs/ or commit until browser inference is verified.')


if __name__ == '__main__':
    main()
