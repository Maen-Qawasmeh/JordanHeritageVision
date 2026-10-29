"""Post-hoc classification error analysis for JordanHeritageVision.

Reads a selected model checkpoint and the existing final TEST manifest. Does NOT
train, modify dataset files, select a new model, or change a checkpoint.

Run from the project root, e.g.:
    python analyze_errors.py \
      --checkpoint outputs/experiment_04_resnet101/checkpoints/best_model.pt \
      --split-dir data/splits_final \
      --output-dir outputs/error_analysis/resnet101 \
      --batch-size 8

Results:
  predictions.csv, misclassifications.csv, confusion_pairs.csv, summary.json,
  error_gallery.png, and errors/*.png (annotated source-image previews).
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from heritagevision.checkpoints import load_checkpoint
from heritagevision.data import LandmarkDataset


PREDICTION_FIELDS = (
    'rel_path', 'true_label', 'predicted_label', 'correct',
    'confidence', 'true_label_probability', 'second_predicted_label',
    'second_predicted_probability', 'top1_top2_margin',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def font(size: int):
    # Pillow's bundled fallback also works when these fonts are unavailable.
    for name in ('DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotated_preview(path: Path, record: dict, output: Path, tile_w=460, tile_h=390):
    """Show the original image, not the transformed tensor fed to the model."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    image.thumbnail((tile_w - 20, tile_h - 108))
    canvas = Image.new('RGB', (tile_w, tile_h), '#111827')
    ox = (tile_w - image.width) // 2
    oy = 8 + (tile_h - 108 - image.height) // 2
    canvas.paste(image, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    title = font(20)
    small = font(15)
    info_y = tile_h - 100
    draw.text((12, info_y), f"True: {record['true_label']}", font=title, fill='#e2e8f0')
    draw.text((12, info_y+28), f"Predicted: {record['predicted_label']} ({record['confidence']*100:.1f}%)",
              font=small, fill='#fb7185')
    display_name = record['rel_path']
    if len(display_name) > 51:
        display_name = '...' + display_name[-48:]
    draw.text((12, info_y+56), display_name, font=small, fill='#94a3b8')
    canvas.save(output)
    return canvas


def save_gallery(tiles, out_path: Path, cols=4):
    if not tiles:
        return False
    tile_w, tile_h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    pad = 12
    header = 76
    sheet = Image.new('RGB', (cols * tile_w + (cols + 1) * pad,
                              header + rows * tile_h + (rows + 1) * pad), '#0b1220')
    painter = ImageDraw.Draw(sheet)
    painter.text((pad+6, 12), 'JordanHeritageVision  |  High-confidence errors',
                 font=font(26), fill='#f1f5f9')
    painter.text((pad+6, 47), 'True / Predicted labels from the held-out TEST set',
                 font=font(16), fill='#94a3b8')
    for i, tile in enumerate(tiles):
        col, row = i % cols, i // cols
        sheet.paste(tile, (pad + col*(tile_w+pad), header + pad + row*(tile_h+pad)))
    sheet.save(out_path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Saved best_model.pt selected using VALIDATION data.')
    parser.add_argument('--split-dir', type=Path, default=Path('data/splits_final'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--top-n', type=int, default=24,
                        help='Maximum number of individual error previews (default: 24).')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.top_n < 0:
        parser.error('batch-size must be >= 1 and top-n must be >= 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')

    split_dir = args.split_dir
    metadata_file = split_dir / 'metadata.json'
    manifest_file = split_dir / 'test.csv'
    for path in (metadata_file, manifest_file, args.checkpoint):
        if not path.is_file():
            parser.error(f'Missing required file: {path}')

    meta = json.loads(metadata_file.read_text(encoding='utf-8'))
    print('Loading selected checkpoint ...')
    model, ckpt, _ = load_checkpoint(args.checkpoint, device)
    classes = ckpt['classes']
    if classes != meta['classes']:
        raise SystemExit('STOP: checkpoint class order differs from split metadata')

    # Same fingerprint safeguards used by evaluate.py in the model comparison.
    recorded_fingerprints = ckpt.get('manifest_fingerprints', {})
    for split in ('train', 'val', 'test'):
        expected = recorded_fingerprints.get(split)
        if expected:
            actual = sha256_file(split_dir / f'{split}.csv')
            if actual != expected:
                raise SystemExit(f'STOP: {split}.csv differs from the checkpoint manifest fingerprint')
    if not recorded_fingerprints:
        print('NOTE: Legacy checkpoint has no manifest fingerprints. Verify split-dir manually.')

    dataset = LandmarkDataset(manifest_file, meta['data_dir'], classes,
                              training=False, image_size=int(ckpt['image_size']))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    records = []
    position = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=(device.type == 'cuda')))
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.size(1) != len(classes):
                raise RuntimeError('Unexpected model output. Expected logits tensor [batch, num_classes].')
            probabilities = torch.softmax(logits.float(), dim=1).cpu()
            top_values, top_indices = probabilities.topk(min(2, len(classes)), dim=1)
            for i, true_idx in enumerate(labels.tolist()):
                pred_idx = int(top_indices[i, 0])
                second_idx = int(top_indices[i, 1]) if len(classes) > 1 else pred_idx
                first_prob = float(top_values[i, 0])
                second_prob = float(top_values[i, 1]) if len(classes) > 1 else 0.0
                records.append({
                    'rel_path': dataset.samples[position]['rel_path'],
                    'true_label': classes[true_idx],
                    'predicted_label': classes[pred_idx],
                    'correct': pred_idx == true_idx,
                    'confidence': first_prob,
                    'true_label_probability': float(probabilities[i, true_idx]),
                    'second_predicted_label': classes[second_idx],
                    'second_predicted_probability': second_prob,
                    'top1_top2_margin': first_prob - second_prob,
                })
                position += 1

    if len(records) != len(dataset):
        raise RuntimeError('Unexpected prediction count; samples may not match the manifest')

    errors = [r for r in records if not r['correct']]
    errors.sort(key=lambda r: r['confidence'], reverse=True)
    confusion = Counter((r['true_label'], r['predicted_label']) for r in errors)
    out = args.output_dir if args.output_dir else args.checkpoint.parent.parent / 'error_analysis'
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'predictions.csv', PREDICTION_FIELDS, records)
    write_csv(out / 'misclassifications.csv', PREDICTION_FIELDS, errors)
    pairs = [
        {'true_label': true, 'predicted_label': pred, 'count': amount}
        for (true, pred), amount in sorted(confusion.items(), key=lambda x: (-x[1], x[0]))
    ]
    write_csv(out / 'confusion_pairs.csv', ('true_label','predicted_label','count'), pairs)

    recalls = {}
    for label in classes:
        subset = [r for r in records if r['true_label'] == label]
        correct = sum(r['correct'] for r in subset)
        recalls[label] = {'correct': correct, 'support': len(subset),
                          'recall': correct / len(subset) if subset else None}
    summary = {
        'model_name': ckpt['model_name'],
        'checkpoint': str(args.checkpoint),
        'split_dir': str(split_dir),
        'split': 'test',
        'samples': len(records),
        'correct': len(records) - len(errors),
        'errors': len(errors),
        'accuracy': (len(records) - len(errors)) / len(records),
        'per_class_recall': recalls,
        'common_confusions': pairs[:10],
        'note': ('Softmax confidence is NOT calibrated probability of correctness. '
                 'Use this report for post-hoc diagnostics; do not tune on the held-out test set.'),
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Save top-N original-image previews; do NOT add test images to Git without permission.
    tiles = []
    if args.top_n > 0 and errors:
        error_images_dir = out / 'errors'
        error_images_dir.mkdir(exist_ok=True)
        for rank, record in enumerate(errors[:args.top_n], 1):
            original = Path(meta['data_dir']) / Path(record['rel_path'])
            filename = f'{rank:03d}_{record["true_label"]}_as_{record["predicted_label"]}.png'
            tiles.append(annotated_preview(original, record, error_images_dir / filename))
        save_gallery(tiles, out / 'error_gallery.png')

    print(f'Model: {ckpt["model_name"]} | Device: {device} | Test samples: {len(records)}')
    print(f'Correct: {len(records) - len(errors)} | Errors: {len(errors)} | Accuracy: {summary["accuracy"]:.3%}')
    print(f'Saved: {out.resolve()}')
    print('Files: predictions.csv, misclassifications.csv, confusion_pairs.csv, summary.json')
    print(f'Image previews: {len(tiles)}' + (' (see error_gallery.png and errors/)' if tiles else ''))
    print('NOTE: Confidence is a softmax score, not a calibrated correctness probability.')
    print('IMPORTANT: Do not use held-out test errors to tune and retest on the same test set.')


if __name__ == '__main__':
    main()
