"""Create fixed, reproducible, stratified train/val/test CSV manifests."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from inspect_data import EXTENSIONS, scan_dataset


def save_manifest(path: Path, records):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['rel_path', 'label'])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw', help='Directory containing class subfolders')
    parser.add_argument('--split-dir', default='data/splits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    counts = scan_dataset(data_dir)
    if len(counts) < 2:
        raise SystemExit('At least two classes are needed.')
    if any(n < 10 for n in counts.values()):
        raise SystemExit('Each class needs at least 10 images for this 70/15/15 split.')

    records = []
    for label in sorted(counts):
        for p in sorted((data_dir / label).rglob('*')):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                records.append({'rel_path': p.relative_to(data_dir).as_posix(), 'label': label})

    labels = [r['label'] for r in records]
    trainval, test = train_test_split(records, test_size=0.15,
                                    random_state=args.seed, stratify=labels)
    train, val = train_test_split(trainval, test_size=0.15 / 0.85,
                                 random_state=args.seed,
                                 stratify=[r['label'] for r in trainval])

    out = Path(args.split_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, portion in [('train', train), ('val', val), ('test', test)]:
        save_manifest(out / f'{name}.csv', portion)
        print(f'{name:<5}: {len(portion):>5} images | {dict(sorted(Counter(r["label"] for r in portion).items()))}')

    metadata = {'data_dir': str(data_dir), 'classes': sorted(counts),
                'seed': args.seed, 'split_ratio': [0.70, 0.15, 0.15]}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'\nSaved split manifests to {out.resolve()}')
    print('IMPORTANT: Review repeated/near-duplicate images before using the test set.')


if __name__ == '__main__':
    main()
