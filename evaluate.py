"""Final held-out test evaluation. Do not run until models are selected on validation."""
import argparse
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from heritagevision.checkpoints import load_checkpoint
from heritagevision.data import LandmarkDataset


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split-dir', default='data/splits_final')
    p.add_argument('--output-dir', default=None,
                   help='Defaults to a test_reports directory alongside the checkpoint')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint, _ = load_checkpoint(args.checkpoint, device)
    split_dir = Path(args.split_dir)
    meta = json.loads((split_dir / 'metadata.json').read_text(encoding='utf-8'))
    if checkpoint['classes'] != meta['classes']:
        raise SystemExit('Checkpoint class order does not match split metadata')
    for name in ('train', 'val', 'test'):
        recorded = checkpoint.get('manifest_fingerprints', {}).get(name)
        if recorded and recorded != sha256_file(split_dir / f'{name}.csv'):
            raise SystemExit(f'Wrong dataset: {name}.csv fingerprint mismatch')
    classes = checkpoint['classes']
    dataset = LandmarkDataset(split_dir / 'test.csv', meta['data_dir'], classes,
                              training=False, image_size=checkpoint['image_size'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    truth, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    idx = list(range(len(classes)))
    acc = accuracy_score(truth, preds)
    report = classification_report(truth, preds, labels=idx,
                                   target_names=classes, zero_division=0, digits=4)
    out = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent.parent / 'test_reports'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'classification_report.txt').write_text(
        f'Model: {checkpoint["model_name"]}\nTest accuracy: {acc:.4%} on {len(dataset)} samples\n\n{report}',
        encoding='utf-8')
    cm = confusion_matrix(truth, preds, labels=idx)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(idx, classes, rotation=35, ha='right')
    ax.set_yticks(idx, classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    print(f'TEST accuracy: {acc:.4%}\n{report}\nReports: {out.resolve()}')


if __name__ == '__main__':
    main()
