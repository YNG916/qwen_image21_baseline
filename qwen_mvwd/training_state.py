"""Small resumable state; never serialize the frozen Qwen backbone."""
from __future__ import annotations

import math
import random
from pathlib import Path

from .common import digest, file_hash, read_json, within

SCHEMA = "qwen-mvwd-training-state-v2"


def epoch_indices(size, seed, epoch, world_size, accumulation):
    import torch
    if min(size, world_size, accumulation) < 1 or epoch < 0:
        raise ValueError("Invalid sampler dimensions")
    generator = torch.Generator().manual_seed(int(digest([seed, "epoch", epoch])[:16], 16))
    order = torch.randperm(size, generator=generator).tolist()
    total = math.ceil(size / (world_size * accumulation)) * world_size * accumulation
    return (order * math.ceil(total / size))[:total]


def capture_rng(device, generator):
    import numpy as np
    import torch
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(), "noise_generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device, generator):
    import numpy as np
    import torch
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["noise_generator"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def check_progress(progress, size, world_size, accumulation):
    step, epoch, cursor = (progress[k] for k in ("step", "epoch", "cursor"))
    if any(type(v) is not int or v < 0 for v in (step, epoch, cursor)):
        raise ValueError("Invalid checkpoint progress")
    per_epoch = math.ceil(size / (world_size * accumulation)) * accumulation
    if cursor >= per_epoch or cursor % accumulation or step != epoch * (per_epoch // accumulation) + cursor // accumulation:
        raise ValueError("Checkpoint progress is not an optimizer boundary")
    if progress["global_samples_seen"] != step * accumulation * world_size:
        raise ValueError("Checkpoint sample count mismatch")


def checkpoint_receipt(directory, contract):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    if meta.get("schema") != SCHEMA or not meta.get("complete"):
        raise ValueError("Checkpoint has no complete optimizer/RNG resume state; old LoRA-only checkpoints cannot resume")
    if meta["resume_contract"] != contract:
        raise ValueError("Resume requires identical config, feature caches, implementation, world size and validation protocol")
    expected = {"pytorch_lora_weights.safetensors", *[f"state_rank_{i:04d}.pt" for i in range(contract["world_size"]) ]}
    if set(meta["state_files"]) != expected:
        raise ValueError("Checkpoint state file list mismatch")
    for name, checksum in meta["state_files"].items():
        if file_hash(within(directory, name)) != checksum:
            raise ValueError("Checkpoint state checksum mismatch: " + name)
    check_progress(meta["progress"], contract["sample_count"], contract["world_size"], contract["accumulation"])
    return meta
