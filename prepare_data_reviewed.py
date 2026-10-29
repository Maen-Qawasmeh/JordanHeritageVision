"""Create reproducible, stratified CSV splits after removing *exact* duplicates.

- Raw images are never changed or deleted.
- Old data/splits manifests are never overwritten (new default: data/splits_reviewed).
- Byte-identical files are included at most once across ALL manifests.
- Conflicting exact-image groups cause a hard failure by default.
- Use --exclude-conflicts after manual review to omit ALL files in conflicting groups.
- Potential visual near-duplicates are reported, NOT removed automatically.

Run from project root:
    python prepare_data_reviewed.py --exclude-conflicts
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split

from inspect_data import EXTENSIONS, scan_dataset


CSV_FIELDS = ['rel_path', 'label']
NEAR_FIELDS = ['split_a', 'label_a', 'path_a', 'split_b', 'label_b', 'path_b', 'distance']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dhash_file(path: Path) -> int:
    """64-bit image difference hash; a screening heuristic, NOT definitive evidence."""
    with Image.open(path) as fh:
        image = ImageOps.exif_transpose(fh).convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        pixels = image.tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_preview(pairs: list[dict], root: Path, target: Path, max_pairs: int = 18) -> None:
    """Make a small human-readable contact sheet of the first near-duplicate candidates."""
    sample = pairs[:max_pairs]
    if not sample:
        return
    image_w, image_h, pad, text_h = 245, 165, 10, 42
    cell_w = image_w * 2 + pad * 3
    cell_h = image_h + text_h + pad * 2
    sheet = Image.new('RGB', (cell_w, len(sample) * cell_h + pad), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, pair in enumerate(sample):
        y = i * cell_h + pad
        for j, prefix in enumerate(('a', 'b')):
            with Image.open(root / pair[f'path_{prefix}']) as src:
                image = ImageOps.exif_transpose(src).convert('RGB')
                image.thumbnail((image_w, image_h), Image.Resampling.LANCZOS)
                x = pad + j * (image_w + pad)
                sheet.paste(image, (x + (image_w - image.width) // 2, y))
            label = f"{pair[f'split_{prefix}']} | {pair[f'label_{prefix}']}"
            name = pair[f'path_{prefix}'][-36:]
            draw.text((x, y + image_h + 3), label[:38], fill='black')
            draw.text((x, y + image_h + 20), name, fill='black')
        draw.text((cell_w - 110, y + image_h + 3), f"dHash={pair['distance']}", fill='black')
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--split-dir', default='data/splits_reviewed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--near-threshold', type=int, default=3)
    parser.add_argument('--exclude-conflicts', action='store_true',
                        help='After manual review, exclude every file in exact-image groups with conflicting labels')
    args = parser.parse_args()
    if not 0 <= args.near_threshold <= 64:
        parser.error('--near-threshold must be from 0 to 64')

    root = Path(args.data_dir).expanduser().resolve()
    out = Path(args.split_dir).expanduser().resolve()
    old = (Path('data/splits').resolve())
    if out == old:
        parser.error('Use a NEW --split-dir; do not overwrite the original split files.')
    counts = scan_dataset(root)
    if len(counts) < 2:
        raise SystemExit('At least two class folders are needed.')

    # Collect ALL original images before splitting (not just cross-split pairs).
    by_sha = defaultdict(list)
    for label in sorted(counts):
        for path in sorted((root / label).rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                by_sha[sha256_file(path)].append({
                    'rel_path': path.relative_to(root).as_posix(), 'label': label
                })
    total_original = sum(len(group) for group in by_sha.values())

    # Same binary content assigned to two different classes = conflicting ground truth.
    conflicts = []
    for digest, group in by_sha.items():
        if len({r['label'] for r in group}) > 1:
            for r in group:
                conflicts.append({'sha256': digest, **r})
    conflict_hashes = {r['sha256'] for r in conflicts}
    if conflicts:
        target = out / 'conflicting_exact_labels.csv'
        write_csv(target, ['sha256', *CSV_FIELDS], conflicts)
        if not args.exclude_conflicts:
            raise SystemExit(
                f'STOP: {len(conflict_hashes)} exact-image groups have '
                f'conflicting class labels. Review {target} before splitting. '
                'After reviewing, rerun with --exclude-conflicts to omit the groups.'
            )
        # Exclude the entire contradictory group: not just one of its file paths.
        for digest in conflict_hashes:
            del by_sha[digest]
        print(f'Excluded {len(conflict_hashes)} conflicting SHA-256 groups '
              f'({len(conflicts)} files); no original files were modified.')

    # Keep exactly one path per SHA group; no source files are deleted.
    unique, omitted = [], []
    for digest in sorted(by_sha):
        group = sorted(by_sha[digest], key=lambda r: r['rel_path'])
        kept = group[0]
        unique.append(kept)
        for dropped in group[1:]:
            omitted.append({'kept_path': kept['rel_path'], 'excluded_path': dropped['rel_path'],
                            'label': kept['label'], 'sha256': digest})
    unique.sort(key=lambda r: (r['label'], r['rel_path']))
    class_counts = Counter(r['label'] for r in unique)
    if any(n < 10 for n in class_counts.values()):
        raise SystemExit(f'Too few unique images in a class for 70/15/15: {dict(class_counts)}')

    trainval, test = train_test_split(
        unique, test_size=0.15, random_state=args.seed,
        stratify=[r['label'] for r in unique]
    )
    train, val = train_test_split(
        trainval, test_size=0.15 / 0.85, random_state=args.seed,
        stratify=[r['label'] for r in trainval]
    )
    split_records = {'train': train, 'val': val, 'test': test}
    out.mkdir(parents=True, exist_ok=True)
    for name, records in split_records.items():
        write_csv(out / f'{name}.csv', CSV_FIELDS, records)
        print(f'{name:<5}: {len(records):>5} images | '
              f'{dict(sorted(Counter(r["label"] for r in records).items()))}')
    write_csv(out / 'excluded_exact_duplicates.csv',
              ['kept_path', 'excluded_path', 'label', 'sha256'], omitted)

    metadata = {
        'data_dir': str(root), 'classes': sorted(counts), 'seed': args.seed,
        'split_ratio': [0.70, 0.15, 0.15],
        'original_image_count': total_original,
        'unique_exact_image_count': len(unique),
        'excluded_exact_duplicate_files': len(omitted),
        'excluded_conflict_groups': len(conflict_hashes),
        'excluded_conflict_files': len(conflicts),
        'split_strategy': 'exact SHA-256 deduplication before stratified split',
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # Audit candidate near duplicates that still span multiple splits.
    # These are *possible* visual matches, not necessarily true duplicates.
    entries = []
    for split, rows in split_records.items():
        for row in rows:
            entries.append({**row, 'split': split,
                            'dhash': dhash_file(root / row['rel_path'])})
    near = []
    for idx, a in enumerate(entries):
        for b in entries[idx + 1:]:
            if a['split'] == b['split']:
                continue
            distance = (a['dhash'] ^ b['dhash']).bit_count()
            if distance <= args.near_threshold:
                near.append({
                    'split_a': a['split'], 'label_a': a['label'], 'path_a': a['rel_path'],
                    'split_b': b['split'], 'label_b': b['label'], 'path_b': b['rel_path'],
                    'distance': distance,
                })
    review_csv = out / 'near_duplicate_pairs_TO_REVIEW.csv'
    write_csv(review_csv, NEAR_FIELDS, near)
    preview = out / 'near_duplicate_preview.png'
    make_preview(near, root, preview)

    # Make it explicit that passing exact-copy checks is not final leakage clearance.
    print(f'\nOriginal images: {total_original}')
    print(f'Unique image content (SHA-256): {len(unique)}')
    print(f'Exact duplicate files omitted from CSVs: {len(omitted)}')
    print(f'Conflicting-label groups excluded: {len(conflict_hashes)} ({len(conflicts)} files)')
    print('Exact duplicate files across splits: 0 (by construction)')
    print(f'Possible near-duplicate cross-split pairs: {len(near)}')
    print(f'New manifests: {out}')
    print(f'Exact duplication exclusion report: {out / "excluded_exact_duplicates.csv"}')
    print(f'Potential near-duplicate report: {review_csv}')
    if near:
        print(f'Preview (first {min(18,len(near))} pairs only): {preview}')
        print('STOP: review near-duplicates BEFORE final model evaluation; '
              'do not assume dHash matches are true duplicates.')
    else:
        print('No near pairs detected at this threshold (not proof none exist).')
    print('Raw images and the original data/splits were left unchanged.')


if __name__ == '__main__':
    main()
