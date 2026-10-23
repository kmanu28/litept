"""Profile real S3DIS training steps for an attention configuration."""

import argparse
import copy
import os
import sys
import time
from functools import partial

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset, point_collate_fn
from models import build_model
from utils.config import Config
from utils.optimizer import build_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--active-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def cuda_batch(batch):
    return {
        key: value.cuda(non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_step(model, optimizer, scaler, batch, amp_dtype, clip_grad):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type="cuda", enabled=True, dtype=amp_dtype
    ):
        loss = model(batch)["loss"]
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    scaler.step(optimizer)
    scaler.update()
    return loss


def main():
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cfg = Config.fromfile(args.config)
    cfg.data.train.loop = 1
    batch_size = cfg.batch_size
    dataset = build_dataset(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(point_collate_fn, mix_prob=cfg.mix_prob),
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = build_model(cfg.model).cuda().train()
    optimizer = build_optimizer(
        copy.deepcopy(cfg.optimizer), model, cfg.param_dicts
    )
    scaler = torch.amp.GradScaler("cuda")
    amp_dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16)[
        cfg.amp_dtype
    ]
    iterator = iter(loader)

    for _ in range(args.warmup_steps):
        batch, iterator = next_batch(iterator, loader)
        train_step(
            model,
            optimizer,
            scaler,
            cuda_batch(batch),
            amp_dtype,
            cfg.clip_grad,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    elapsed = []
    point_counts = []
    losses = []
    for _ in range(args.active_steps):
        start = time.perf_counter()
        batch, iterator = next_batch(iterator, loader)
        batch = cuda_batch(batch)
        point_counts.append(int(batch["coord"].shape[0]))
        loss = train_step(
            model, optimizer, scaler, batch, amp_dtype, cfg.clip_grad
        )
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))

    mean_seconds = sum(elapsed) / len(elapsed)
    mean_points = sum(point_counts) / len(point_counts)
    print(f"config={args.config}")
    print(
        f"steps={len(elapsed)} batch_size={batch_size} "
        f"mean_points={mean_points:.1f}"
    )
    print(
        f"mean_step_seconds={mean_seconds:.6f} "
        f"steps_per_second={1 / mean_seconds:.3f} "
        f"points_per_second={mean_points / mean_seconds:.1f}"
    )
    print(
        f"peak_allocated_gib="
        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} "
        f"peak_reserved_gib={torch.cuda.max_memory_reserved() / 1024**3:.3f} "
        f"finite_losses={all(torch.isfinite(torch.tensor(losses)))}"
    )

    if args.profile_steps <= 0:
        return
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for _ in range(args.profile_steps):
            batch, iterator = next_batch(iterator, loader)
            train_step(
                model,
                optimizer,
                scaler,
                cuda_batch(batch),
                amp_dtype,
                cfg.clip_grad,
            )
            profiler.step()
    torch.cuda.synchronize()
    cuda_events = [
        event
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    print(f"cuda_profiler_events={len(cuda_events)}")
    for event in sorted(
        cuda_events,
        key=lambda item: item.self_device_time_total,
        reverse=True,
    )[:12]:
        print(
            f"cuda_kernel_us={event.self_device_time_total:.3f} "
            f"calls={event.count} name={event.name}"
        )
    print(
        profiler.key_averages().table(
            sort_by="self_device_time_total", row_limit=12
        )
    )


if __name__ == "__main__":
    main()
