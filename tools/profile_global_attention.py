"""Profile LitePT local versus final-stage global attention on the demo cloud."""

import argparse
import copy
import gc
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.transform import Compose
from litept.model import LitePT as StandaloneLitePT
from models import build_model
from utils.config import Config


GIB = 1024**3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-config",
        default="configs/scannet/semseg-litept-small-v1m1.py",
    )
    parser.add_argument(
        "--global-config",
        default="configs/scannet/semseg-litept-small-globalattn-v1m1.py",
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--standalone-global-pretrained", action="store_true")
    parser.add_argument("--standalone-relflex-pretrained", action="store_true")
    return parser.parse_args()


def demo_point(in_channels):
    lidar_path = hf_hub_download(
        repo_id="prs-eth/LitePT_demo",
        filename="outdoor_sample1.bin",
        repo_type="dataset",
        revision="main",
    )
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    point = Compose(
        [
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
        ]
    )(dict(coord=points[:, :3], strength=points[:, 3:4] / 255))
    point["feat"] = torch.cat((point["coord"].float(), point["strength"].float()), dim=1)
    if point["feat"].shape[1] < in_channels:
        padding = point["feat"].new_zeros(
            point["feat"].shape[0], in_channels - point["feat"].shape[1]
        )
        point["feat"] = torch.cat((point["feat"], padding), dim=1)
    point["feat"] = point["feat"][:, :in_channels]
    return {
        "coord": point["coord"],
        "grid_coord": point["grid_coord"],
        "inverse": point["inverse"],
        "offset": torch.tensor([point["coord"].shape[0]]),
        "feat": point["feat"],
    }, points.shape[0]


def cuda_point(point):
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in point.items()
    }


def add_stage_hooks(model, counts):
    handles = []
    for stage, encoder in enumerate(model.enc.children()):
        block0 = encoder._modules["block0"]

        def record(module, inputs, stage=stage):
            counts.setdefault(stage, int(inputs[0].feat.shape[0]))

        handles.append(block0.register_forward_pre_hook(record))
    return handles


def build_backbone(config_path, seed=0):
    cfg = Config.fromfile(config_path)
    torch.manual_seed(seed)
    model = build_model(copy.deepcopy(cfg.model.backbone)).cuda()
    return model, cfg.model.backbone


def run_forward(config_path, point):
    model, backbone_cfg = build_backbone(config_path)
    model.eval()
    counts = {}
    handles = add_stage_hooks(model, counts)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(cuda_point(point))
    torch.cuda.synchronize()
    result = dict(
        shape=tuple(output.feat.shape),
        counts=counts,
        allocated=torch.cuda.max_memory_allocated(),
        reserved=torch.cuda.max_memory_reserved(),
        params=sum(parameter.numel() for parameter in model.parameters()),
    )
    for handle in handles:
        handle.remove()
    del output, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, backbone_cfg


def run_backward(config_path, point):
    model, _ = build_backbone(config_path)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(cuda_point(point))
        loss = output.feat.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    attention = model.enc._modules["enc4"]._modules["block0"].attn
    grad = attention.qkv.weight.grad
    result = dict(
        loss=float(loss.detach()),
        grad_finite=grad is not None and bool(torch.isfinite(grad).all()),
        grad_norm=float(grad.float().norm()),
        allocated=torch.cuda.max_memory_allocated(),
        reserved=torch.cuda.max_memory_reserved(),
    )
    del output, loss, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def gib(value):
    return value / GIB


def run_standalone_pretrained(attention_type):
    point, raw_count = demo_point(4)
    model = StandaloneLitePT(
        enc_attention_type=attention_type,
        relflex_coord_scale=0.01,
        relflex_eps=0.01,
    ).cuda()
    checkpoint_path = hf_hub_download(
        repo_id="prs-eth/LitePT",
        filename="nuscenes-semseg-litept-small-v1m1/model/model_best.pth",
        repo_type="model",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    prefix = "module.backbone."
    weights = OrderedDict(
        (key[len(prefix) :], value)
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    )
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.no_grad():
        output = model(cuda_point(point))
    torch.cuda.synchronize()
    print(
        f"standalone_{attention_type}_pretrained raw_points={raw_count} "
        f"output_shape={tuple(output.feat.shape)} "
        f"dense_shape={(raw_count, output.feat.shape[1])} strict_load=True"
    )


def main():
    args = parse_args()
    if args.standalone_global_pretrained:
        run_standalone_pretrained("global")
        return
    if args.standalone_relflex_pretrained:
        run_standalone_pretrained("relflex")
        return
    baseline_cfg = Config.fromfile(args.baseline_config).model.backbone
    point, raw_count = demo_point(baseline_cfg.in_channels)
    print(f"raw_points={raw_count} voxelized_tokens={point['feat'].shape[0]}")

    baseline, _ = run_forward(args.baseline_config, point)
    print(
        "baseline "
        f"shape={baseline['shape']} params={baseline['params']} "
        f"peak_allocated_gib={gib(baseline['allocated']):.3f} "
        f"peak_reserved_gib={gib(baseline['reserved']):.3f}"
    )
    print("stage_tokens=" + ",".join(f"{s}:{n}" for s, n in baseline["counts"].items()))
    if args.baseline_only:
        return

    global_cfg = Config.fromfile(args.global_config).model.backbone
    first_global_stage = len(global_cfg.enc_channels) - global_cfg.enc_global_attn_stages
    target_stages = range(first_global_stage, len(global_cfg.enc_channels))
    target_counts = [baseline["counts"][s] for s in target_stages]
    dense_score_bytes = sum(
        n * n * global_cfg.enc_num_head[s] * 2
        for s, n in zip(target_stages, target_counts)
    )
    device_bytes = torch.cuda.get_device_properties(0).total_memory
    print(
        f"preflight targeted_stages={list(target_stages)} "
        f"target_tokens={target_counts} dense_fp16_score_equivalent_gib={gib(dense_score_bytes):.3f} "
        f"device_gib={gib(device_bytes):.3f}"
    )
    if dense_score_bytes + baseline["reserved"] >= device_bytes:
        raise RuntimeError(
            "Global-attention preflight stopped: dense score equivalent plus observed "
            "baseline peak reaches GPU capacity"
        )

    global_result, _ = run_forward(args.global_config, point)
    print(
        "global "
        f"shape={global_result['shape']} params={global_result['params']} "
        f"peak_allocated_gib={gib(global_result['allocated']):.3f} "
        f"peak_reserved_gib={gib(global_result['reserved']):.3f}"
    )
    if global_result["shape"] != baseline["shape"]:
        raise RuntimeError("Baseline and global-attention output shapes differ")
    if global_result["params"] != baseline["params"]:
        raise RuntimeError("Baseline and global-attention parameter counts differ")

    backward = run_backward(args.global_config, point)
    print(
        "global_backward "
        f"loss={backward['loss']:.6g} grad_finite={backward['grad_finite']} "
        f"qkv_grad_norm={backward['grad_norm']:.6g} "
        f"peak_allocated_gib={gib(backward['allocated']):.3f} "
        f"peak_reserved_gib={gib(backward['reserved']):.3f}"
    )
    if not backward["grad_finite"]:
        raise RuntimeError("Global-attention QKV gradient is missing or non-finite")


if __name__ == "__main__":
    main()
