"""Create Grad-CAM visual explanations for selected misclassified TEST images.

Run `analyze_errors.py` first to produce misclassifications.csv. This script
READS that file and your saved checkpoint, generates Grad-CAM for the model's
predicted and true classes, and never updates model weights or split manifests.

Examples:
    python generate_gradcam.py \
      --checkpoint outputs/experiment_04_resnet101/checkpoints/best_model.pt \
      --split-dir data/splits_final \
      --errors-csv outputs/error_analysis/resnet101/misclassifications.csv \
      --output-dir outputs/gradcam/resnet101 --top-n 12
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms

from heritagevision.checkpoints import load_checkpoint
from heritagevision.data import build_transform


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def target_layer_for(model, model_name):
    """Last spatial feature map before classification, by model family."""
    if model_name == 'resnet101':
        return model.layer4[-1]
    if model_name == 'efficientnet_b0':
        return model.features[-1]
    if model_name == 'vgg16':
        # After the last in-place ReLU and before the final max pool.
        return model.features[-2]
    if model_name == 'inception_v3':
        return model.Mixed_7c
    raise ValueError(f'Unsupported architecture: {model_name}')


class GradCAM:
    """Grad-CAM using an activation forward hook and tensor gradient hook."""

    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError('Target layer must output a [batch, channels, H, W] tensor')
        self.activations = output
        output.register_hook(self._capture_grad)

    def _capture_grad(self, grad):
        self.gradients = grad

    def explain(self, logits, class_idx: int, retain_graph: bool):
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        logits[0, class_idx].backward(retain_graph=retain_graph)
        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM hooks did not capture activations/gradients')
        activation = self.activations.detach()
        gradient = self.gradients.detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False)
        heatmap = heatmap[0, 0]
        heatmap -= heatmap.min()
        peak = heatmap.max()
        if float(peak) > 1e-10:
            heatmap /= peak
        return heatmap.float().cpu().numpy()

    def close(self):
        self.handle.remove()
        self.activations = None
        self.gradients = None


def overlay(image: np.ndarray, cam: np.ndarray):
    colors = plt.get_cmap('turbo')(cam)[..., :3]
    alpha = (cam[..., None] * 0.55).astype(np.float32)
    return np.clip((1 - alpha) * image + alpha * colors, 0, 1)


def make_panel(image, predicted_cam, true_cam, path, true_label, predicted_label, confidence):
    display = [image, overlay(image, predicted_cam), overlay(image, true_cam)]
    headers = ['Input (evaluation crop)', f'Grad-CAM: Predicted {predicted_label}',
               f'Grad-CAM: True {true_label}']
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0))
    fig.patch.set_facecolor('#101827')
    for axis, picture, heading in zip(axes, display, headers):
        axis.imshow(picture)
        axis.set_title(heading, fontsize=11, color='white', pad=11)
        axis.axis('off')
    fig.suptitle(f'{path}  |  Model confidence: {confidence:.1%} (uncalibrated)',
                 fontsize=12, color='#e2e8f0')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, default=Path('data/splits_final'))
    parser.add_argument('--errors-csv', type=Path, required=True,
                        help='Output misclassifications.csv from analyze_errors.py')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--top-n', type=int, default=12)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args(argv)

    if args.top_n < 1:
        parser.error('--top-n must be >= 1')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device = torch.device(device_name)

    meta_path = args.split_dir / 'metadata.json'
    test_path = args.split_dir / 'test.csv'
    for p in (args.checkpoint, args.errors_csv, meta_path, test_path):
        if not p.is_file():
            parser.error(f'Missing required file: {p}')

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    model, ckpt, _ = load_checkpoint(args.checkpoint, device)
    classes = ckpt['classes']
    if classes != meta['classes']:
        raise SystemExit('STOP: class order mismatch between checkpoint and split metadata')
    for split in ('train', 'val', 'test'):
        expected = ckpt.get('manifest_fingerprints', {}).get(split)
        if expected and sha256_file(args.split_dir / f'{split}.csv') != expected:
            raise SystemExit(f'STOP: {split}.csv fingerprint differs from checkpoint')
    if not ckpt.get('manifest_fingerprints'):
        print('NOTE: Legacy checkpoint missing manifest fingerprints. Verify the dataset manually.')

    with test_path.open(newline='', encoding='utf-8-sig') as stream:
        test_rows = list(csv.DictReader(stream))
    expected_labels = {r['rel_path']: r['label'] for r in test_rows}

    with args.errors_csv.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'rel_path', 'true_label', 'predicted_label', 'confidence'}
        if not required.issubset(reader.fieldnames or []):
            parser.error(f'Error CSV is missing columns: {required}')
        selected = list(reader)[:args.top_n]
    if not selected:
        raise SystemExit('No errors found. There are no misclassifications to explain.')

    out = args.output_dir or args.checkpoint.parent.parent / 'gradcam'
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    size = int(ckpt['image_size'])
    source_dir = Path(meta['data_dir'])

    # Explain the exact deterministic evaluation crop, NOT the augmented training image.
    to_display = transforms.Compose([
        transforms.Resize(round(size * 256 / 224)),
        transforms.CenterCrop(size),
    ])
    preprocess = build_transform(training=False, image_size=size)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    gradcam = GradCAM(model, target_layer_for(model, ckpt['model_name']))
    gradcam.input_size = size

    written = 0
    output_rows = []
    try:
        for rank, record in enumerate(selected, start=1):
            rel_path = record['rel_path']
            true_label = record['true_label']
            if expected_labels.get(rel_path) != true_label:
                raise SystemExit(f'STOP: error CSV mismatch with current test manifest at {rel_path}')
            src = source_dir / Path(rel_path)
            if not src.is_file():
                raise FileNotFoundError(f'Missing image: {src}')
            with Image.open(src) as opened:
                pil = ImageOps.exif_transpose(opened).convert('RGB')
                original_crop = np.asarray(to_display(pil), dtype=np.float32) / 255.0
                tensor = preprocess(pil).unsqueeze(0).to(device)
            tensor.requires_grad_(True)
            with torch.enable_grad():
                logits = model(tensor)
                if not isinstance(logits, torch.Tensor) or logits.shape != (1, len(classes)):
                    raise RuntimeError('Unexpected model output; expected logits [1, num_classes]')
                probabilities = logits.detach().float().softmax(dim=1)[0]
                pred_idx = int(probabilities.argmax())
                predicted_label = classes[pred_idx]
                if predicted_label != record['predicted_label']:
                    raise SystemExit(f'STOP: stale error report or checkpoint mismatch at {rel_path}; '
                                     f'CSV says {record["predicted_label"]}, model says {predicted_label}')
                true_idx = classes.index(true_label)
                pred_heatmap = gradcam.explain(logits, pred_idx, retain_graph=True)
                true_heatmap = gradcam.explain(logits, true_idx, retain_graph=False)
            conf = float(probabilities[pred_idx])
            fig = make_panel(original_crop, pred_heatmap, true_heatmap,
                             rel_path, true_label, predicted_label, conf)
            file_path = out / f'{rank:03d}_{true_label}_as_{predicted_label}_gradcam.png'
            fig.savefig(file_path, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            written += 1
            output_rows.append({
                'rel_path': rel_path, 'true_label': true_label,
                'predicted_label': predicted_label,
                'softmax_confidence': f'{conf:.6f}', 'gradcam_file': file_path.name,
            })
            print(f'[{written}/{len(selected)}] Saved {file_path.name}')
            del tensor, logits
    finally:
        gradcam.close()

    with (out / 'gradcam_index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'rel_path', 'true_label', 'predicted_label',
            'softmax_confidence', 'gradcam_file'))
        writer.writeheader()
        writer.writerows(output_rows)
    (out / 'README.txt').write_text(
        'Grad-CAM panels are qualitative visual explanations, not causal proofs.\n'
        'Left: deterministic evaluation crop. Middle: predicted-class map. '
        'Right: true-class map.\n'
        'Higher heat intensity means relatively greater contribution of spatial features.\n'
        'Confidence is an uncalibrated softmax score.\n'
        'Do not use held-out TEST errors to tune and retest a new model on the same holdout.\n'
        'Check image licenses before publishing generated panels.\n', encoding='utf-8')
    print(f'Finished: {written} Grad-CAM explanations. Saved to {out.resolve()}')
    print('This script does NOT train a model or modify its checkpoint.')


if __name__ == '__main__':
    main()
