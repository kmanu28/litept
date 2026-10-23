"""Validate the experimental RelFlexformer path on the cached demo cloud."""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litept.relflexformer_attention import RelFlexformerAttention
from tools.profile_global_attention import build_backbone, cuda_point, demo_point


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/scannet/semseg-litept-small-relflex-v1m1.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, backbone_cfg = build_backbone(args.config)
    model.eval()
    point, raw_count = demo_point(backbone_cfg.in_channels)
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, RelFlexformerAttention):
            def report(module, inputs, output, name=name):
                coord = inputs[3]
                print(
                    f"{name}: tokens={coord.shape[0]} "
                    f"coord_min={coord.amin(0).cpu().tolist()} "
                    f"coord_max={coord.amax(0).cpu().tolist()} "
                    f"finite={bool(torch.isfinite(output).all())} "
                    f"absmax={float(output.float().abs().max()):.6g} "
                    f"nan={int(torch.isnan(output).sum())} "
                    f"inf={int(torch.isinf(output).sum())}"
                )

            handles.append(module.register_forward_hook(report))

    with torch.no_grad():
        output = model(cuda_point(point))
    torch.cuda.synchronize()
    print(
        f"raw_points={raw_count} output_shape={tuple(output.feat.shape)} "
        f"finite={bool(torch.isfinite(output.feat).all())} "
        f"absmax={float(output.feat.float().abs().max()):.6g} "
        f"nan={int(torch.isnan(output.feat).sum())} "
        f"inf={int(torch.isinf(output.feat).sum())}"
    )
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
