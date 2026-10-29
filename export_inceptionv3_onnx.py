"""Export fine-tuned JordanHeritageVision InceptionV3 to a browser-sized ONNX model.

Run from your JordanHeritageVision project root:
    python export_inceptionv3_onnx.py
    python export_inceptionv3_onnx.py --image "data/raw/Petra/example.jpg"
    python export_inceptionv3_onnx.py --verify-val 24

Requires: torch, torchvision, onnx, onnxscript, onnxruntime, numpy, Pillow.
Only load a checkpoint YOU trained/trust. Does not modify your checkpoints or website.
"""

import argparse
import csv
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


def compress_fp32_weights(source: Path, destination: Path) -> int:
    """Use FP16 only for weight *storage*, inserting FP32 Cast nodes at runtime."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = onnx.load(str(source))
    cast_nodes = []
    graph_inputs = {item.name for item in graph.graph.input}

    for idx, initializer in enumerate(graph.graph.initializer):
        if initializer.data_type != TensorProto.FLOAT or initializer.name in graph_inputs:
            continue
        weight = numpy_helper.to_array(initializer)
        if weight.size < 1024:
            continue

        original_name = initializer.name
        packed_name = f"__jhv_fp16_weight_{idx}"
        initializer.CopyFrom(
            numpy_helper.from_array(weight.astype(np.float16), name=packed_name)
        )
        cast_nodes.append(helper.make_node(
            "Cast", [packed_name], [original_name],
            to=TensorProto.FLOAT, name=f"JHV_FP32_Cast_{idx}"
        ))

    if not cast_nodes:
        raise RuntimeError("No suitable FP32 weights found for compression")

    for pos, node in enumerate(cast_nodes):
        graph.graph.node.insert(pos, node)

    onnx.checker.check_model(graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(graph, str(destination), save_as_external_data=False)
    return len(cast_nodes)


def verify_logits(session, model, tensor: torch.Tensor, description: str):
    with torch.inference_mode():
        reference = model(tensor).detach().numpy()
    result = session.run(["logits"], {"images": tensor.numpy()})[0]

    if reference.shape != result.shape or reference.ndim != 2 or reference.shape[1] != 6:
        raise RuntimeError(f"{description}: unexpected output shapes {reference.shape}, {result.shape}")
    max_diff = float(np.abs(reference - result).max())
    same_predictions = np.array_equal(reference.argmax(1), result.argmax(1))
    close = np.allclose(reference, result, atol=0.05, rtol=0.01)
    print(f"{description}: max logit difference={max_diff:.6f}; "
          f"same top-1={same_predictions}; close logits={close}")
    if not (same_predictions and close):
        raise RuntimeError(f"{description} failed. Do not publish the exported model.")


def verify_validation(session, model, checkpoint_size: int, split_dir: Path, limit: int) -> None:
    manifest = split_dir / "val.csv"
    metadata_file = split_dir / "metadata.json"
    if not manifest.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(f"Need val.csv and metadata.json in {split_dir}")

    split_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    data_dir = Path(split_meta["data_dir"])
    classes = list(split_meta["classes"])
    with manifest.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError("Validation manifest is empty")
    if limit > 0 and limit < len(rows):
        rows = random.Random(42).sample(rows, limit)

    transform = build_transform(False, checkpoint_size)
    agreements = torch_correct = onnx_correct = 0
    for i, row in enumerate(rows, start=1):
        with Image.open(data_dir / row["rel_path"]) as photo:
            image = ImageOps.exif_transpose(photo).convert("RGB")
            batch = transform(image).unsqueeze(0)
        with torch.inference_mode():
            torch_logits = model(batch).numpy()
        onnx_logits = session.run(["logits"], {"images": batch.numpy()})[0]
        p = int(torch_logits.argmax(1)[0])
        q = int(onnx_logits.argmax(1)[0])
        truth = classes.index(row["label"])
        agreements += p == q
        torch_correct += p == truth
        onnx_correct += q == truth
        if i % 10 == 0:
            print(f"  Checked {i}/{len(rows)} validation images")

    n = len(rows)
    print(f"Validation subset ({n}): top-1 agreement {agreements}/{n}; "
          f"PyTorch accuracy {torch_correct / n:.2%}; ONNX accuracy {onnx_correct / n:.2%}")
    if agreements != n:
        raise RuntimeError("Validation predictions disagree. Review before publishing.")
    print("Note: A sampled subset accuracy is NOT a replacement for the held-out test result.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/experiment_08_inceptionv3/checkpoints/best_model.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("web/inception_v3_export/inception_v3.onnx"))
    parser.add_argument("--image", type=Path, help="Optional real image to compare PyTorch/ONNX")
    parser.add_argument("--verify-val", type=int, default=0, metavar="N",
                        help="Check N validation images; -1 means entire validation set")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits_final"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    import onnx
    import onnxruntime as ort

    # Keep export and verification on CPU so it works even without CUDA.
    model, ckpt, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if ckpt["model_name"] != "inception_v3":
        parser.error(f"Wrong model: {ckpt['model_name']!r} (expected 'inception_v3')")
    classes = list(ckpt["classes"])
    size = int(ckpt["image_size"])
    if len(classes) != 6 or size != 299:
        parser.error(f"Unexpected checkpoint: {len(classes)} classes, image size {size}")
    if getattr(model, "aux_logits", False) or getattr(model, "AuxLogits", None) is not None:
        parser.error("InceptionV3 auxiliary classifier must be disabled for export")

    model.eval()
    torch.manual_seed(42)
    sample = torch.randn(1, 3, size, size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {classes}")
    print(f"Image size: {size} x {size}; transform_input={model.transform_input}")
    print("Exporting FP32 ONNX (temporary, not for GitHub)...")

    with tempfile.TemporaryDirectory(prefix="jhv_inceptionv3_") as tmp:
        fp32 = Path(tmp) / "inception_v3_fp32.onnx"
        torch.onnx.export(
            model, (sample,), str(fp32),
            input_names=["images"], output_names=["logits"],
            opset_version=18, dynamo=True, external_data=False,
        )
        onnx.checker.check_model(str(fp32))
        baseline = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
        verify_logits(baseline, model, sample, "FP32 ONNX")
        del baseline

        print("Compressing weights with FP16 storage and FP32 Cast nodes...")
        compressed_count = compress_fp32_weights(fp32, args.output)

    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    verify_logits(session, model, sample, "Compressed ONNX")

    if args.image:
        with Image.open(args.image) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            real_input = build_transform(False, size)(img).unsqueeze(0)
        verify_logits(session, model, real_input, "Real photo")

    if args.verify_val:
        verify_validation(session, model, size, args.split_dir, args.verify_val)

    # CPU benchmarking on the input shape above; not equivalent to browser latency.
    for _ in range(2):
        session.run(["logits"], {"images": sample.numpy()})
    times = []
    for _ in range(5):
        start = time.perf_counter()
        session.run(["logits"], {"images": sample.numpy()})
        times.append(1000 * (time.perf_counter() - start))

    model_size = args.output.stat().st_size / 1048576
    metadata = {
        "model_name": "inception_v3", "classes": classes, "image_size": size,
        "resize_short_side": round(size * 256 / 224), "crop_type": "center",
        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
        "input_layout": "NCHW", "input_dtype": "float32",
        "onnx_input": "images", "onnx_output": "logits",
        "weight_storage": "float16_with_float32_cast",
        "transform_input": bool(model.transform_input),
    }
    metadata_file = args.output.parent / "inception_v3_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nPASS: compressed {compressed_count} tensors; verified ONNX inference on CPU")
    print(f"ONNX: {args.output} ({model_size:.2f} MiB)")
    print(f"Metadata: {metadata_file}")
    print(f"Median ONNX CPU inference (single image): {statistics.median(times):.1f} ms")
    if model_size >= 100:
        print("WARNING: File exceeds GitHub's 100 MiB standard-Git limit. Do not commit it.")
    elif model_size >= 50:
        print("NOTE: GitHub warns about files over 50 MiB; under 100 MiB can still be pushed.")
    print("NEXT: test in ONNX Runtime Web before publishing; do not modify docs/ yet.")


if __name__ == "__main__":
    main()
