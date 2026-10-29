"""Compare model-only inference speed, checkpoint size, and GPU memory.

Run from the project root after training. Uses synthetic inputs so the benchmark
measures model inference, not image-loading or preprocessing speed.
"""
import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from heritagevision.checkpoints import load_checkpoint


CHECKPOINTS = {
    "resnet101": "outputs/experiment_04_resnet101/checkpoints/best_model.pt",
    "efficientnet_b0": "outputs/experiment_02/checkpoints/best_model.pt",
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(name, path, device, batch_size, warmup, runs):
    model, info, _ = load_checkpoint(path, device)
    if info["model_name"] != name:
        raise ValueError(f"Expected {name}, found {info['model_name']} in {path}")

    x = torch.randn(batch_size, 3, info["image_size"], info["image_size"], device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings_ms = []
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=(device.type == "cuda")
    ):
        for _ in range(warmup):
            model(x)
        synchronize(device)

        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            synchronize(device)  # CUDA operations are asynchronous.
            timings_ms.append((time.perf_counter() - start) * 1000)

    result = {
        "model": name,
        "batch_size": batch_size,
        "image_size": info["image_size"],
        "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "checkpoint_mb": round(path.stat().st_size / (1024 ** 2), 1),
        "median_batch_ms": round(statistics.median(timings_ms), 2),
        "images_per_second": round(batch_size / (statistics.median(timings_ms) / 1000), 1),
        "peak_gpu_allocated_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            if device.type == "cuda" else "N/A"
        ),
    }
    del model, x
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", default="outputs/benchmarks/model_comparison.csv")
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.runs) < 1:
        parser.error("batch-size, warmup, and runs must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Batch size: {args.batch_size} | AMP: {device.type == 'cuda'}")
    results = []
    for name, checkpoint in CHECKPOINTS.items():
        path = Path(checkpoint)
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}")
        print(f"Benchmarking {name} ...")
        result = benchmark(name, path, device, args.batch_size, args.warmup, args.runs)
        results.append(result)
        print(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {output.resolve()}")
    print("Note: Times exclude image loading and preprocessing; rerun for stable measurements.")


if __name__ == "__main__":
    main()
