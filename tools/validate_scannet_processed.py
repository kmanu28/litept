"""Validate processed ScanNet arrays before training."""

import argparse
from pathlib import Path

import numpy as np


REQUIRED = (
    "coord",
    "color",
    "normal",
    "segment20",
    "segment200",
    "instance",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/scannet")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    total_points = 0
    summary = {}
    for split in ("train", "val"):
        scenes = sorted(path for path in (root / split).iterdir() if path.is_dir())
        split_points = 0
        for scene in scenes:
            arrays = {
                key: np.load(scene / f"{key}.npy", mmap_mode="r")
                for key in REQUIRED
            }
            point_count = arrays["coord"].shape[0]
            for key, array in arrays.items():
                if array.shape[0] != point_count:
                    raise ValueError(
                        f"{scene.name}/{key}: {array.shape[0]} != {point_count}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"{scene.name}/{key}: non-finite values")
            if arrays["coord"].shape[1:] != (3,):
                raise ValueError(f"{scene.name}/coord: unexpected shape")
            if arrays["color"].shape[1:] != (3,):
                raise ValueError(f"{scene.name}/color: unexpected shape")
            if arrays["normal"].shape[1:] != (3,):
                raise ValueError(f"{scene.name}/normal: unexpected shape")
            if arrays["segment20"].min() < -1 or arrays["segment20"].max() > 19:
                raise ValueError(f"{scene.name}/segment20: label out of range")
            split_points += point_count
        summary[split] = (len(scenes), split_points)
        total_points += split_points

    for split, (scene_count, point_count) in summary.items():
        print(f"{split}: scenes={scene_count} points={point_count}")
    print(f"PASS: total_scenes={sum(value[0] for value in summary.values())} "
          f"total_points={total_points}")


if __name__ == "__main__":
    main()
