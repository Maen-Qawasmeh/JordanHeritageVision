"""Audit duplicate images across dataset splits before training.

Reports byte-identical images using SHA-256, and visually similar candidates
using a 64-bit difference hash (dHash). A dHash match is NOT proof of duplication.
No source data, manifest, or split is modified.
"""
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps

SPLIT_DIR = Path('data/splits')
REPORT_DIR = Path('outputs/reports')
NEAR_THRESHOLD = 3  # small Hamming distance; manual inspection required


def digest_file(path):
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def difference_hash(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('L')
        image = image.resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    hash_value = 0
    for y in range(8):
        for x in range(8):
            hash_value = (hash_value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return hash_value


def write_pairs(path, records):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'split_a', 'label_a', 'path_a', 'split_b', 'label_b', 'path_b', 'distance'
        ])
        writer.writeheader()
        writer.writerows(records)


def pair(a, b, distance):
    return {
        'split_a': a['split'], 'label_a': a['label'], 'path_a': a['path'],
        'split_b': b['split'], 'label_b': b['label'], 'path_b': b['path'],
        'distance': distance,
    }


def main():
    meta = json.loads((SPLIT_DIR / 'metadata.json').read_text(encoding='utf-8'))
    root = Path(meta['data_dir'])
    records = []
    for split in ('train', 'val', 'test'):
        with (SPLIT_DIR / f'{split}.csv').open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                path = root / row['rel_path']
                records.append({
                    'split': split, 'label': row['label'],
                    'path': row['rel_path'], 'sha': digest_file(path),
                    'dhash': difference_hash(path),
                })
    print(f'Images scanned: {len(records)}')

    exact = []
    grouped = defaultdict(list)
    for record in records:
        grouped[record['sha']].append(record)
    for items in grouped.values():
        for a, b in combinations(items, 2):
            if a['split'] != b['split']:
                exact.append(pair(a, b, 0))

    # Visual hash similarity is an approximate screening method, not a verdict.
    near = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a['split'] == b['split'] or a['sha'] == b['sha']:
                continue
            distance = (a['dhash'] ^ b['dhash']).bit_count()
            if distance <= NEAR_THRESHOLD:
                near.append(pair(a, b, distance))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_pairs(REPORT_DIR / 'exact_cross_split_duplicates.csv', exact)
    write_pairs(REPORT_DIR / 'possible_near_duplicates.csv', near)
    print(f'Exact duplicate pairs across splits: {len(exact)}')
    print(f'Possible near-duplicate pairs across splits: {len(near)}')
    print(f'Reports saved to: {REPORT_DIR.resolve()}')
    for record in (exact or near)[:5]:
        print(f"  [{record['distance']}] {record['split_a']}:{record['path_a']}  <->  {record['split_b']}:{record['path_b']}")
    if exact:
        print('WARNING: Exact copies cross splits. Fix before trusting model evaluation.')
    if near:
        print('REVIEW: Possible near-duplicates need visual inspection; dHash can produce false positives.')
    if not exact and not near:
        print('No cross-split duplicates detected by these checks (not proof of absence).')


if __name__ == '__main__':
    main()
