"""Validate processed S3DIS rooms before training."""

import argparse
from pathlib import Path

import numpy as np


AREAS = tuple(f"Area_{index}" for index in range(1, 7))
ASSETS = ("coord", "color", "segment", "instance")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/s3dis")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    total_rooms = 0
    total_points = 0

    for area in AREAS:
        area_path = data_root / area
        if not area_path.is_dir():
            raise FileNotFoundError(f"Missing area directory: {area_path}")
        rooms = sorted(path for path in area_path.iterdir() if path.is_dir())
        if not rooms:
            raise RuntimeError(f"No rooms found in {area_path}")

        area_points = 0
        for room in rooms:
            arrays = {}
            for asset in ASSETS:
                asset_path = room / f"{asset}.npy"
                if not asset_path.is_file():
                    raise FileNotFoundError(f"Missing asset: {asset_path}")
                arrays[asset] = np.load(asset_path, mmap_mode="r")

            point_count = arrays["coord"].shape[0]
            if arrays["coord"].shape != (point_count, 3):
                raise ValueError(f"Invalid coord shape in {room}")
            if arrays["color"].shape != (point_count, 3):
                raise ValueError(f"Invalid color shape in {room}")
            if arrays["segment"].reshape(-1).shape[0] != point_count:
                raise ValueError(f"Segment length mismatch in {room}")
            if arrays["instance"].reshape(-1).shape[0] != point_count:
                raise ValueError(f"Instance length mismatch in {room}")
            if not np.isfinite(arrays["coord"]).all():
                raise ValueError(f"Non-finite coordinates in {room}")
            labels = arrays["segment"].reshape(-1)
            if labels.min() < 0 or labels.max() > 12:
                raise ValueError(f"Semantic label outside [0, 12] in {room}")
            area_points += point_count

        print(f"{area}: rooms={len(rooms)} points={area_points}")
        total_rooms += len(rooms)
        total_points += area_points

    print(f"total: rooms={total_rooms} points={total_points}")


if __name__ == "__main__":
    main()
