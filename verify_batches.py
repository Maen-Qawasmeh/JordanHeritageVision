"""
What does the script check?
- Labels: Verifies that the class name in the CSV matches the corresponding image folder.
- File Paths: Confirms that all 3,688 image paths exist.
- Data Leakage: Ensures that the same image path does not appear in more than one split (e.g., training, validation, or test sets).
- Batch Size: Confirms that each batch contains 8 images.
- Image Tensors: Checks the dimensions and data type, and ensures there are no NaN (Not a Number) or Inf (Infinity) values.
- Visualization: Saves 8 images from the training set, with the class name displayed above each image.

"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save figures without requiring a GUI window.
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from heritagevision.data import LandmarkDataset, IMAGENET_MEAN, IMAGENET_STD


SPLIT_DIR = Path('data/splits')
meta = json.loads((SPLIT_DIR / 'metadata.json').read_text(encoding='utf-8'))
classes = meta['classes']
data_dir = Path(meta['data_dir'])
print('Class mapping:', {name: i for i, name in enumerate(classes)})

# Ensure each manifest label matches its image folder and every file exists.
paths_per_split = {}
for split in ('train', 'val', 'test'):
    with (SPLIT_DIR / f'{split}.csv').open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    rel_paths = [row['rel_path'] for row in rows]
    assert len(rel_paths) == len(set(rel_paths)), f'Duplicate path within {split}'
    for row in rows:
        assert row['label'] in classes, f"Unknown class: {row['label']}"
        assert Path(row['rel_path']).parts[0] == row['label'], f"Label/folder mismatch: {row}"
        assert (data_dir / row['rel_path']).is_file(), f"Missing image: {row['rel_path']}"
    paths_per_split[split] = set(rel_paths)
    print(f'{split}: {len(rows)} image paths and labels OK')

assert not (paths_per_split['train'] & paths_per_split['val'])
assert not (paths_per_split['train'] & paths_per_split['test'])
assert not (paths_per_split['val'] & paths_per_split['test'])
print('Split overlap: NONE (by relative file path)')

# Load a batch from all three splits, using the actual project Dataset class.
for split in ('train', 'val', 'test'):
    dataset = LandmarkDataset(
        SPLIT_DIR / f'{split}.csv', data_dir, classes,
        training=(split == 'train'), image_size=224,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=(split == 'train'), num_workers=0)
    images, labels = next(iter(loader))
    assert images.shape == (8, 3, 224, 224), f'Unexpected image shape in {split}'
    assert labels.shape == (8,), f'Unexpected labels shape in {split}'
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert torch.isfinite(images).all(), f'Non-finite pixel values in {split}'
    assert bool(((labels >= 0) & (labels < len(classes))).all()), f'Out-of-range labels in {split}'
    print(f'{split} batch: images={tuple(images.shape)}, labels={tuple(labels.shape)} — OK')

    if split == 'train':
        # Reverse ImageNet normalization for a human-readable image grid.
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        for i, ax in enumerate(axes.flat):
            image = (images[i].cpu() * std + mean).clamp(0, 1)
            ax.imshow(image.permute(1, 2, 0).numpy())
            ax.set_title(classes[labels[i].item()])
            ax.axis('off')
        fig.suptitle('Training batch (with augmentation)')
        fig.tight_layout()
        target = Path('outputs/reports/sample_batch.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=140)
        plt.close(fig)
        print(f'Saved image preview: {target.resolve()}')

print('\nBasic manifest and batch checks PASSED.')
print('NOTE: This does not detect visually similar or duplicated image content.')
