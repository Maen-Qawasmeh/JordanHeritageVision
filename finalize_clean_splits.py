"""Conservatively exclude ALL image paths in cross-split dHash candidate pairs.

This script does NOT decide which candidates are true duplicates. It excludes
all candidate endpoints from the manifests as a conservative baseline, without
changing the raw dataset or the source manifests.

Usage:
    python finalize_clean_splits.py
    python finalize_clean_splits.py --source-dir data/splits_reviewed --output-dir data/splits_final
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

FIELDNAMES = ("rel_path", "label")
PAIR_FIELDS = ("split_a", "label_a", "path_a", "split_b", "label_b", "path_b", "distance")
SPLITS = ("train", "val", "test")


def read_rows(path, required):
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not set(required).issubset(reader.fieldnames or []):
            raise SystemExit(f"Incorrect CSV headers in {path}. Expected: {required}")
        return list(reader)


def write_rows(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/splits_reviewed")
    parser.add_argument("--output-dir", default="data/splits_final")
    args = parser.parse_args()
    src = Path(args.source_dir).expanduser().resolve()
    dst = Path(args.output_dir).expanduser().resolve()
    if src == dst:
        parser.error("Source and output directories must be different.")
    if dst.exists() and any(dst.iterdir()):
        parser.error(f"Output directory already contains files: {dst}. Choose a NEW output directory.")

    meta_path = src / "metadata.json"
    if not meta_path.is_file():
        raise SystemExit(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = {s: read_rows(src / f"{s}.csv", FIELDNAMES) for s in SPLITS}
    pairs = read_rows(src / "near_duplicate_pairs_TO_REVIEW.csv", PAIR_FIELDS)
    if not pairs:
        raise SystemExit("No candidate pairs found. No changes required.")

    # Confirm every reviewed path matches an existing manifest row and class.
    location = {}
    for split, items in records.items():
        for item in items:
            rel = item["rel_path"]
            if rel in location:
                raise SystemExit(f"A file path appears in multiple rows: {rel}")
            location[rel] = (split, item["label"])
    excluded = defaultdict(list)
    for idx, pair in enumerate(pairs, start=1):
        for side in ("a", "b"):
            rel = pair[f"path_{side}"]
            observed = (pair[f"split_{side}"], pair[f"label_{side}"])
            if location.get(rel) != observed:
                raise SystemExit(f"Candidate pair #{idx}: {rel} does not match manifests: "
                                 f"report={observed}, current={location.get(rel)}. "
                                 "Regenerate the report before proceeding.")
            excluded[rel].append(idx)

    filtered = {}
    for split, items in records.items():
        filtered[split] = [item for item in items if item["rel_path"] not in excluded]
        if not filtered[split]:
            raise SystemExit(f"Filtering would leave split '{split}' empty. Stopping.")

    dst.mkdir(parents=True, exist_ok=False) if not dst.exists() else None
    for split in SPLITS:
        write_rows(dst / f"{split}.csv", FIELDNAMES, filtered[split])

    removed = [
        {"rel_path": rel, "original_split": location[rel][0],
         "label": location[rel][1], "candidate_pair_ids": ";".join(map(str, pair_ids))}
        for rel, pair_ids in sorted(excluded.items())
    ]
    write_rows(dst / "excluded_near_duplicate_candidates.csv",
               ("rel_path", "original_split", "label", "candidate_pair_ids"), removed)

    original_total = sum(len(items) for items in records.values())
    final_total = sum(len(items) for items in filtered.values())
    meta = dict(meta)
    meta["source_split_dir"] = str(src)
    meta["original_split_ratio_before_near_candidate_filter"] = meta.get("split_ratio")
    meta["split_ratio"] = [len(filtered[s]) / final_total for s in SPLITS]
    meta["split_strategy"] = "SHA-256 exact dedup, conflicted-label exclusion, then conservative dHash candidate removal"
    meta["near_duplicate_candidate_pairs"] = len(pairs)
    meta["excluded_near_duplicate_candidate_files"] = len(excluded)
    meta["final_manifest_image_count"] = final_total
    meta["audit_note"] = "Candidate pairs were excluded, NOT verified as true duplicates. Label and semantic quality need further audits."
    (dst / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Original reviewed records: {original_total}")
    print(f"Cross-split candidate pairs reviewed by filter: {len(pairs)}")
    print(f"Unique candidate image paths excluded: {len(excluded)}")
    for s in SPLITS:
        counts = dict(sorted(Counter(row["label"] for row in filtered[s]).items()))
        print(f"{s:5}: {len(records[s]):4} -> {len(filtered[s]):4} | {counts}")
    print(f"Final number of image paths: {final_total}")
    print(f"New manifests: {dst}")
    print("Source manifests and original image files were NOT modified.")
    print("CAUTION: This filters only candidates from the supplied report, not all possible semantic or near duplicates.")


if __name__ == "__main__":
    main()
