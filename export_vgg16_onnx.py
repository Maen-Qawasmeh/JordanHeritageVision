"""Export JordanHeritageVision's fine-tuned VGG16 for a static browser demo.

Run from the JordanHeritageVision project root (no retraining):
    python export_vgg16_onnx.py --verify-val 8

VGG16 is too large for a single normal GitHub file. This tool:
1) Exports the exact trained PyTorch model to ONNX.
2) Rounds *weight storage* to FP16, then restores FP32 via Cast nodes.
3) Verifies logits/top-1 against the PyTorch checkpoint using ONNX Runtime.
4) Splits the tested ONNX bytes into <50 MiB parts for normal Git.

The split parts are NOT individual ONNX models. The browser must fetch them
IN ORDER and concatenate their bytes before creating an ONNX Runtime session.

This file never modifies the checkpoint, existing docs/, or the website.
Only load checkpoints you personally trained/trust.
"""

import argparse
import csv
import gc
import hashlib
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from heritagevision.checkpoints import load_checkpoint
from heritagevision.data import IMAGENET_MEAN, IMAGENET_STD, build_transform

EXPECTED_CLASSES = (
    "Ajloun", "Jerash", "Petra", "RomanAmphitheater", "UmmQais", "WadiRum"
)


def pack_fp16_weight_storage(source: Path, destination: Path) -> int:
    """Lossy compression: store large FP32 weights in FP16; Cast them back.

    Unlike converting every operator to FP16, convolution and fully connected
    arithmetic still sees FP32 operands, which works with browser WASM.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = onnx.load(str(source))
    graph_inputs = {item.name for item in graph.graph.input}
    casts = []

    for index, initializer in enumerate(graph.graph.initializer):
        if initializer.data_type != TensorProto.FLOAT or initializer.name in graph_inputs:
            continue
        values = numpy_helper.to_array(initializer)
        if values.size < 1024:
            continue  # Leave small vectors (e.g. biases) in FP32.

        original_name = initializer.name
        packed_name = f"__jhv_vgg16_packed_{index}"
        packed = numpy_helper.from_array(
            values.astype(np.float16), name=packed_name
        )
        initializer.CopyFrom(packed)
        casts.append(helper.make_node(
            "Cast", [packed_name], [original_name],
            to=TensorProto.FLOAT, name=f"JHV_VGG16_RestoreFP32_{index}"
        ))

    if not casts:
        raise RuntimeError("No large FP32 tensors found; cannot compress VGG16.")

    for position, node in enumerate(casts):
        graph.graph.node.insert(position, node)

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(graph, str(destination), save_as_external_data=False)
    del graph
    gc.collect()
    onnx.checker.check_model(str(destination))
    return len(casts)


def verify_logits(session, model, images: torch.Tensor, label: str) -> None:
    with torch.inference_mode():
        expected = model(images).detach().cpu().numpy()
    obtained = session.run(["logits"], {"images": images.cpu().numpy()})[0]

    if expected.shape != obtained.shape or expected.shape[-1] != 6:
        raise RuntimeError(f"{label}: incorrect output shapes {expected.shape}, {obtained.shape}")

    max_error = float(np.max(np.abs(expected - obtained)))
    same_top1 = np.array_equal(expected.argmax(axis=1), obtained.argmax(axis=1))
    close = bool(np.allclose(expected, obtained, atol=0.05, rtol=0.01))

    print(f"{label}: max-logit-error={max_error:.6f}; "
          f"same-top1={same_top1}; close-logits={close}", flush=True)

    if not (same_top1 and close):
        raise RuntimeError(f"{label} failed; do NOT publish the exported model.")


def sample_validation(split_dir: Path, count: int):
    manifest = split_dir / "val.csv"
    metadata_file = split_dir / "metadata.json"
    if not manifest.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(f"Need {manifest} and {metadata_file}")

    split_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    if list(split_meta["classes"]) != list(EXPECTED_CLASSES):
        raise RuntimeError("Validation class order differs from the trained model")

    with manifest.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise RuntimeError("Validation manifest is empty")

    if 0 < count < len(rows):
        rows = random.Random(42).sample(rows, count)
    return rows, Path(split_meta["data_dir"])


def verify_validation(session, model, split_dir: Path, count: int, image_size: int):
    rows, data_dir = sample_validation(split_dir, count)
    transform = build_transform(False, image_size)
    agreements = onnx_correct = torch_correct = 0

    for i, row in enumerate(rows, 1):
        with Image.open(data_dir / row["rel_path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            sample = transform(im).unsqueeze(0)

        with torch.inference_mode():
            torch_logits = model(sample).detach().numpy()
        onnx_logits = session.run(["logits"], {"images": sample.numpy()})[0]

        p = int(torch_logits.argmax(axis=1)[0])
        q = int(onnx_logits.argmax(axis=1)[0])
        truth = EXPECTED_CLASSES.index(row["label"])
        agreements += p == q
        torch_correct += p == truth
        onnx_correct += q == truth
        print(f"  Validation {i}/{len(rows)}: PyTorch={p}, ONNX={q}", flush=True)

    print(f"Validation: top-1 agreement={agreements}/{len(rows)}; "
          f"PyTorch accuracy={torch_correct/len(rows):.2%}; "
          f"ONNX accuracy={onnx_correct/len(rows):.2%}")
    if agreements != len(rows):
        raise RuntimeError("Validation has prediction disagreements. Do NOT publish yet.")
    print("Note: sampled validation accuracy is NOT held-out test accuracy.")


def split_onnx(source: Path, out_dir: Path, part_mib: int):
    """Split by raw byte offsets so re-assembly recreates the original ONNX exactly."""
    chunk_bytes = part_mib * 1024 * 1024
    prefix = "vgg16.onnx.part"
    output_names = []
    sha = hashlib.sha256()
    total = 0

    for old in out_dir.glob(prefix + "*"):
        if old.is_file():
            old.unlink()  # Remove stale pieces from a previous run.

    with source.open("rb") as stream:
        index = 0
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            file_name = f"{prefix}{index:03d}"
            (out_dir / file_name).write_bytes(block)
            sha.update(block)
            total += len(block)
            output_names.append(file_name)
            index += 1

    if total != source.stat().st_size or not output_names:
        raise RuntimeError("Failed to split all ONNX model bytes")

    # Re-hash the saved parts (not the original stream), catching short writes.
    check_sha = hashlib.sha256()
    for name in output_names:
        with (out_dir / name).open("rb") as stream:
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                check_sha.update(chunk)

    if check_sha.hexdigest() != sha.hexdigest():
        raise RuntimeError("Saved ONNX pieces failed integrity verification")

    return output_names, total, sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/experiment_06_vgg16/checkpoints/best_model.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("web/vgg16_export"))
    parser.add_argument("--verify-val", type=int, default=0, metavar="N",
                        help="Check N validation images; -1 = all; 0 = skip")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits_final"))
    parser.add_argument("--image", type=Path, default=None,
                        help="Optional local image for real-photo logits comparison")
    parser.add_argument("--part-mib", type=int, default=40,
                        help="Chunk size in MiB; must be 1..49 (default 40)")
    parser.add_argument("--keep-combined", action="store_true",
                        help="Keep the >100 MiB combined ONNX on local disk")
    args = parser.parse_args()

    if not 1 <= args.part_mib <= 49:
        parser.error("--part-mib must be between 1 and 49")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    import onnx
    import onnxruntime as ort

    # CPU-only for deterministic export and portable ONNX verification.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, ckpt, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if ckpt.get("model_name") != "vgg16":
        parser.error(f"Wrong checkpoint model: {ckpt.get('model_name')!r}, expected 'vgg16'")
    classes = list(ckpt["classes"])
    image_size = int(ckpt["image_size"])
    if classes != list(EXPECTED_CLASSES) or image_size != 224:
        parser.error(f"Unexpected classes or image size: {classes}, {image_size}")

    model.eval()
    torch.manual_seed(42)
    sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.out_dir / "vgg16.onnx"

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {classes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("WARNING: VGG16 model download and browser memory requirements are large.")
    print("Exporting original FP32 ONNX locally; this should NOT be committed.", flush=True)

    with tempfile.TemporaryDirectory(prefix="jhv_vgg16_") as temp_dir:
        full_fp32 = Path(temp_dir) / "vgg16_fp32.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model, (sample,), str(full_fp32),
                input_names=["images"], output_names=["logits"],
                opset_version=18, dynamo=False, external_data=False
            )
        onnx.checker.check_model(str(full_fp32))
        full_session = ort.InferenceSession(str(full_fp32), providers=["CPUExecutionProvider"])
        verify_logits(full_session, model, sample, "Original FP32 ONNX")
        del full_session
        gc.collect()

        print("Compressing large tensors to FP16 storage with FP32 Cast nodes...", flush=True)
        packed_count = pack_fp16_weight_storage(full_fp32, output_file)

    packed_session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    verify_logits(packed_session, model, sample, "Compressed ONNX / random input")

    if args.image:
        with Image.open(args.image) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            example = build_transform(False, image_size)(im).unsqueeze(0)
        verify_logits(packed_session, model, example, "Compressed ONNX / real photo")

    if args.verify_val:
        verify_validation(packed_session, model, args.split_dir, args.verify_val, image_size)

    for _ in range(2):
        packed_session.run(["logits"], {"images": sample.numpy()})
    timings = []
    for _ in range(5):
        tic = time.perf_counter()
        packed_session.run(["logits"], {"images": sample.numpy()})
        timings.append((time.perf_counter() - tic) * 1000)

    del packed_session
    gc.collect()
    packed_mib = output_file.stat().st_size / (1024 * 1024)
    print(f"Compressed ONNX: {packed_mib:.2f} MiB; packed tensors={packed_count}")
    print(f"Median local CPU ONNX inference: {statistics.median(timings):.0f} ms")

    print(f"Splitting into parts of at most {args.part_mib} MiB...", flush=True)
    names, size_bytes, digest = split_onnx(output_file, args.out_dir, args.part_mib)
    metadata = {
        "model_name": "vgg16", "classes": classes, "image_size": image_size,
        "resize_short_side": round(image_size * 256 / 224), "crop_type": "center",
        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
        "input_layout": "NCHW", "input_dtype": "float32",
        "onnx_input": "images", "onnx_output": "logits",
        "weight_storage": "float16_with_float32_cast",
        "onnx_parts": names, "onnx_bytes": size_bytes, "onnx_sha256": digest,
    }
    meta_path = args.out_dir / "vgg16_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.keep_combined:
        output_file.unlink()  # Browser rebuilds the identical ONNX from parts.

    print("\nPASS: PyTorch/ONNX verification and byte-for-byte shard integrity passed")
    print(f"Metadata: {meta_path}")
    print(f"Chunks ({len(names)} files; combined {size_bytes / 1048576:.2f} MiB):")
    for name in names:
        print(f"  {name}: {(args.out_dir / name).stat().st_size / 1048576:.2f} MiB")
    print("NEXT: do NOT git push yet. We must add chunk reassembly to app.js,")
    print("      then verify VGG16 on localhost before committing to GitHub.")


if __name__ == "__main__":
    main()
