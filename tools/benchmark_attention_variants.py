"""Benchmark LitePT attention variants on the cached demo cloud."""

import argparse
import copy
import gc
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from tools.profile_global_attention import cuda_point, demo_point
from utils.config import Config


VARIANTS = (
    ("baseline", "configs/scannet/semseg-litept-small-v1m1.py"),
    ("exact-global-s5", "configs/scannet/semseg-litept-small-globalattn-v1m1.py"),
    (
        "exact-global-s45",
        "configs/scannet/semseg-litept-small-globalattn-stage45-v1m1.py",
    ),
    ("relflex-s5", "configs/scannet/semseg-litept-small-relflex-v1m1.py"),
    (
        "relflex-s45",
        "configs/scannet/semseg-litept-small-relflex-stage45-v1m1.py",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def benchmark(name, config_path, point, warmup, iterations):
    cfg = Config.fromfile(config_path)
    torch.manual_seed(0)
    model = build_model(copy.deepcopy(cfg.model.backbone)).cuda().eval()

    with torch.no_grad():
        for _ in range(warmup):
            output = model(cuda_point(point))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    durations = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(cuda_point(point))
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)

    mean_seconds = sum(durations) / len(durations)
    result = {
        "name": name,
        "shape": tuple(output.feat.shape),
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms": mean_seconds * 1000,
        "scenes_s": 1 / mean_seconds,
        "tokens_s": point["feat"].shape[0] / mean_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    del output, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    baseline_cfg = Config.fromfile(VARIANTS[0][1]).model.backbone
    point, raw_count = demo_point(baseline_cfg.in_channels)
    print(
        f"device={torch.cuda.get_device_name(0)} raw_points={raw_count} "
        f"voxelized_tokens={point['feat'].shape[0]} warmup={args.warmup} "
        f"iterations={args.iterations}"
    )
    print(
        "variant\tlatency_ms\tscenes_s\ttokens_s\tpeak_alloc_gib\t"
        "peak_reserved_gib\tparams\tshape"
    )
    for name, config_path in VARIANTS:
        result = benchmark(
            name,
            config_path,
            point,
            args.warmup,
            args.iterations,
        )
        print(
            f"{result['name']}\t{result['latency_ms']:.3f}\t"
            f"{result['scenes_s']:.3f}\t{result['tokens_s']:.1f}\t"
            f"{result['peak_allocated_gib']:.3f}\t"
            f"{result['peak_reserved_gib']:.3f}\t{result['params']}\t"
            f"{result['shape']}"
        )


if __name__ == "__main__":
    main()
