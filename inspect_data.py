"""Inspect the local dataset without modifying it."""
import argparse
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def scan_dataset(root: Path):
    if not root.is_dir():
        raise SystemExit(f'Dataset folder not found: {root}')
    counts = {}
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and not folder.name.startswith('.'):
            n = sum(1 for p in folder.rglob('*')
                    if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if n:
                counts[folder.name] = n
    if not counts:
        raise SystemExit('No class folders with images found. Expected: ROOT/ClassName/image.jpg')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    args = parser.parse_args()
    counts = scan_dataset(Path(args.data_dir))
    print('\nDataset overview')
    for name, count in counts.items():
        print(f'  {name:<25} {count:>6} images')
    print(f'  TOTAL: {sum(counts.values())} images across {len(counts)} classes')
