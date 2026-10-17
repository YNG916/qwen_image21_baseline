"""Pinned Diffusers 2.1 API; imports heavy dependencies only for GPU work."""
from __future__ import annotations

from pathlib import Path


def require_gpu():
    import os
    import torch
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU commands require an existing Slurm allocation; no job is submitted automatically")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in the allocated GPU shell")


def model_source(config):
    """Resolve a pinned snapshot locally so sharded loaders never call model_info."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    kwargs = {"repo_id": config["model"]["id"], "revision": config["model"]["revision"]}
    try:
        return snapshot_download(**kwargs, local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(**kwargs, allow_patterns=["model_index.json", "transformer/*", "vae/*", "text_encoder/*", "processor/*", "scheduler/*"])


def load_pipeline(config, *, encoding_only=False, offload=False):
    import torch
    from diffusers import QwenImage21Pipeline
    kwargs = {"torch_dtype": torch.bfloat16}
    if encoding_only:
        kwargs["transformer"] = None
    pipe = QwenImage21Pipeline.from_pretrained(model_source(config), **kwargs)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def resized_conditions(pipe, images, resolution):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    result = []
    for image in images:
        w, h, _ = calculate_dimensions(resolution**2, image.width / image.height)
        result.append(pipe.image_processor.resize(image.convert("RGBA"), width=w, height=h))
    return result


def encode_sample(pipe, condition, target, resolution):
    import torch
    images = resized_conditions(pipe, condition["images"], resolution)
    with torch.no_grad():
        embeds, embed_mask, image_mask = pipe.encode_prompt(prompt=condition["prompt"], image=images)
        latents = []
        for image in images:
            pixels = pipe.image_processor.preprocess(image, width=image.width, height=image.height).unsqueeze(2)
            pixels = pixels.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
            latents.append(pipe._encode_vae_image(pixels, generator=None))
        pixels = pipe.image_processor.preprocess(target.convert("RGBA"), width=target.width, height=target.height).unsqueeze(2)
        z = pipe._encode_vae_image(pixels.to(device=pipe.vae.device, dtype=pipe.vae.dtype), generator=None)
    shapes = [(1, latent.shape[-2], latent.shape[-1]) for latent in latents]
    condition_slots = sum(h * w // 4 for _, h, w in shapes)
    if image_mask.shape[0] != 1 or int(image_mask.sum()) != condition_slots:
        raise RuntimeError(f"Vision/latent condition mismatch: {int(image_mask.sum())} vs {condition_slots}")
    return {"target": z.cpu().contiguous(), "conditions": [x.cpu().contiguous() for x in latents],
            "prompt_embeds": embeds.cpu().contiguous(),
            "prompt_mask": None if embed_mask is None else embed_mask.cpu().contiguous(),
            "image_mask": image_mask.cpu().contiguous(), "query": condition["query"]}


def batch_layout(sample):
    """The pinned transformer shares RoPE/image positions across the whole batch."""
    from .common import digest
    return digest({"target": list(sample["target"].shape[1:]),
        "conditions": [list(x.shape[1:]) for x in sample["conditions"]],
        "prompt": list(sample["prompt_embeds"].shape[1:]),
        "image_mask": sample["image_mask"].tolist()})


def collate_samples(samples):
    """Exact layouts only: padding text would move target RoPE positions."""
    import torch
    if not samples or len({batch_layout(s) for s in samples}) != 1:
        raise ValueError("A microbatch requires identical prompt/image layouts")
    return {"target": torch.cat([s["target"] for s in samples]),
        "conditions": [torch.cat([s["conditions"][i] for s in samples]) for i in range(len(samples[0]["conditions"]))],
        "prompt_embeds": torch.cat([s["prompt_embeds"] for s in samples]),
        "prompt_mask": None if all(s["prompt_mask"] is None for s in samples) else torch.cat([
            s["prompt_mask"] if s["prompt_mask"] is not None else torch.ones(s["image_mask"].shape, dtype=torch.bool)
            for s in samples]),
        "image_mask": torch.cat([s["image_mask"] for s in samples])}


def flow_loss(transformer, sample, scheduler, device, *, generator=None):
    import torch
    from diffusers import QwenImage21Pipeline
    dtype = next(transformer.parameters()).dtype
    target = sample["target"].to(device=device, dtype=dtype)
    conds = [x.to(device=device, dtype=dtype) for x in sample["conditions"]]
    noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)
    batch_size = target.shape[0]
    index = torch.randint(len(scheduler.timesteps), (batch_size,), device=device, generator=generator)
    timestep = scheduler.timesteps.to(device)[index]
    sigma = scheduler.sigmas.to(device=device, dtype=dtype)[index].reshape(batch_size, 1, 1, 1, 1)
    noisy = (1 - sigma) * target + sigma * noise

    def pack(z):
        return QwenImage21Pipeline._pack_latents(z, z.shape[0], z.shape[1], z.shape[-2], z.shape[-1])

    packed_target = pack(noisy)
    packed = torch.cat([*[pack(z) for z in conds], packed_target], dim=1)
    shapes = [(1, z.shape[-2], z.shape[-1]) for z in [*conds, target]]
    image_mask = sample["image_mask"].to(device)
    if not torch.all(image_mask.sum(dim=1) == sum(h * w // 4 for _, h, w in shapes[:-1])) or not torch.all(image_mask == image_mask[:1]):
        raise ValueError("Cached image mask does not match condition shapes")
    image_mask = torch.cat([image_mask, torch.ones((batch_size, target.shape[-2] * target.shape[-1] // 4), dtype=image_mask.dtype, device=device)], dim=1)
    prompt_mask = sample["prompt_mask"]
    prediction = transformer(hidden_states=packed,
        encoder_hidden_states=sample["prompt_embeds"].to(device=device, dtype=dtype),
        encoder_hidden_states_mask=None if prompt_mask is None else prompt_mask.to(device),
        timestep=timestep / 1000, img_shapes=[shapes] * batch_size, img_mask=image_mask, return_dict=False)[0]
    prediction = prediction[:, -packed_target.shape[1]:]
    expected = pack(noise - target)
    return (prediction.float() - expected.float()).square().mean()


def train(config, cache_dir, output, *, max_steps=None, resume_from=None,
          validation_cache=None, validate_every=None, validation_seed=137):
    import json
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from .common import digest, file_hash, implementation_hash, read_json, within, write_json
    from .training_state import SCHEMA, capture_rng, restore_rng, epoch_indices, checkpoint_receipt
    require_gpu()
    cache_dir, output = Path(cache_dir), Path(output)
    impl = implementation_hash()

    def read_cache(directory, split):
        receipt = read_json(Path(directory) / "manifest.json")
        if (receipt["config_sha256"] != digest(config) or receipt["dataset"]["split"] != split
                or not receipt["complete"] or not receipt["samples"]):
            raise ValueError(f"Requires a complete nonempty {split} cache with identical config")
        if receipt["implementation_sha256"] != impl:
            raise ValueError("Feature encoding implementation changed; rebuild cache")
        return receipt

    receipt = read_cache(cache_dir, "train")
    val_receipt = read_cache(validation_cache, "val") if validation_cache else None
    if val_receipt and any(val_receipt["dataset"].get(k) != receipt["dataset"].get(k)
                           for k in ("release_id", "robot_assets")):
        raise ValueError("Train/val cache release or robot assets differ")
    if validate_every is not None and (validate_every < 1 or not val_receipt):
        raise ValueError("validate_every must be positive and requires a val cache")
    settings = config["training"]
    interval = (validate_every or settings["checkpoint_every"]) if val_receipt else None
    steps = settings["max_steps"] if max_steps is None else max_steps
    if steps < 1:
        raise ValueError("max_steps must be positive")
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=settings["gradient_accumulation_steps"])
    accumulation, world = settings["gradient_accumulation_steps"], accelerator.num_processes
    micro_batch = settings.get("micro_batch_size", 1)
    layouts = [r["batch_layout"] for r in receipt["samples"]] if micro_batch > 1 else None
    padded_size = len(epoch_indices(len(receipt["samples"]), settings["seed"], 0, world, accumulation, micro_batch, layouts))
    contract = {"config_sha256": digest(config), "cache_sha256": digest(receipt), "implementation_sha256": impl,
        "sample_count": len(receipt["samples"]), "world_size": world, "accumulation": accumulation,
        "micro_batch_size": micro_batch, "padded_epoch_samples": padded_size,
        "sampler": "exact-layout-buckets-shuffled-pad-full-effective-batch-v3",
        "validation": None if val_receipt is None else {"cache_sha256": digest(val_receipt), "every": interval,
            "seed": validation_seed, "probe": "one-fixed-noise-and-timestep-per-query-v1"}}
    resumed = checkpoint_receipt(resume_from, contract) if resume_from else None
    if resumed and steps <= resumed["step"]:
        raise ValueError("max_steps is a total target and must exceed resumed step")
    set_seed(settings["seed"])
    source = model_source(config)
    transformer = QwenImage21Transformer2DModel.from_pretrained(source, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=settings["rank"], lora_alpha=settings["alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights="gaussian"))
    if settings.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    params = [p for p in transformer.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    if resumed:
        state = load_file(str(Path(resume_from) / "pytorch_lora_weights.safetensors"))
        state = {k.removeprefix("transformer."): v for k, v in state.items()}
        if set(state) != set(get_peft_model_state_dict(transformer)):
            raise ValueError("Resume adapter key set differs")
        result = set_peft_model_state_dict(transformer, state)
        if result.unexpected_keys:
            raise ValueError("Unexpected adapter keys")
    optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"], weight_decay=0.01)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(source, subfolder="scheduler")
    if not scheduler.config.use_dynamic_shifting:
        raise ValueError("This implementation expects the base 2.1 dynamic-shift scheduler")
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    noise_generator = torch.Generator(device=accelerator.device).manual_seed(settings["seed"] + accelerator.process_index)
    progress = {"step": 0, "epoch": 0, "cursor": 0, "global_samples_seen": 0}
    if resumed:
        local = torch.load(Path(resume_from) / f"state_rank_{accelerator.process_index:04d}.pt",
                           map_location="cpu", weights_only=True)
        if local["progress"] != resumed["progress"]:
            raise ValueError("Rank/checkpoint progress mismatch")
        optimizer.load_state_dict(local["optimizer"])
        if accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(local["scaler"])
        restore_rng(local["rng"], accelerator.device, noise_generator)
        progress = dict(resumed["progress"])
    transformer.train()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "training_metadata.json", {"schema": SCHEMA, "config": config, "cache": receipt,
            "world_size": world, "effective_batch": world * accumulation * micro_batch,
            "micro_batch_size": micro_batch, "padded_epoch_samples": padded_size, "resume_supported": True,
            "resumed_from": str(Path(resume_from).resolve()) if resumed else None,
            "starting_progress": progress, "resume_contract": contract,
            "target_latents": "deterministic VAE posterior mode",
            "padding": "Exact-layout buckets repeat their prefix to full microbatches; batch order is shuffled, then repeated to a full effective batch; repeated samples count as seen",
            "validation": "Fixed noise/timestep flow-loss proxy; image quality and geometry require generated-view review"})
    accelerator.wait_for_everyone()

    def sample_at(directory, manifest, index):
        row = manifest["samples"][index]
        path = within(directory, row["path"])
        if file_hash(path) != row["sha256"]:
            raise ValueError("Feature cache checksum mismatch")
        return torch.load(path, map_location="cpu", weights_only=True)

    def save(name):
        accelerator.wait_for_everyone()
        dest = output / name
        if accelerator.is_main_process:
            dest.mkdir(exist_ok=False)
            state = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
            QwenImage21Pipeline.save_lora_weights(str(dest), transformer_lora_layers=state, safe_serialization=True)
        accelerator.wait_for_everyone()
        torch.save({"progress": dict(progress), "optimizer": optimizer.state_dict(),
            "rng": capture_rng(accelerator.device, noise_generator),
            "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None},
            dest / f"state_rank_{accelerator.process_index:04d}.pt")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            files = ["pytorch_lora_weights.safetensors", *[f"state_rank_{i:04d}.pt" for i in range(world)]]
            write_json(dest / "metadata.json", {"schema": SCHEMA, "complete": True,
                **progress, "progress": dict(progress), "resume_contract": contract,
                "state_files": {name: file_hash(dest / name) for name in files},
                "config_sha256": digest(config), "cache_identity": receipt["cache_identity"],
                "config": config, "dataset": receipt["dataset"], "trainable_parameters": sum(p.numel() for p in params)})
        accelerator.wait_for_everyone()

    def validate():
        # Dedicated probes plus complete RNG restoration keep validation out of the training noise stream.
        rng = capture_rng(accelerator.device, noise_generator)
        model = accelerator.unwrap_model(transformer)
        model.eval()
        total = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
        try:
            with torch.no_grad(), accelerator.autocast():
                for index in range(accelerator.process_index, len(val_receipt["samples"]), world):
                    seed = int(digest([validation_seed, val_receipt["cache_identity"], val_receipt["samples"][index]["query"]])[:16], 16)
                    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                    loss = flow_loss(model, sample_at(validation_cache, val_receipt, index), scheduler,
                                     accelerator.device, generator=generator)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite validation flow loss")
                    total[0] += loss.double()
                    total[1] += 1
            total = accelerator.reduce(total, reduction="sum")
            row = {"step": progress["step"], "flow_loss": (total[0] / total[1]).item(),
                   "queries": int(total[1].item()), "protocol": contract["validation"]}
            if accelerator.is_main_process:
                with (output / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print({"validation": row}, flush=True)
        finally:
            model.train()
            restore_rng(rng, accelerator.device, noise_generator)

    if val_receipt:
        validate()  # Same fixed probe at the initial/resumed checkpoint.
    import time
    torch.cuda.reset_peak_memory_stats(accelerator.device) if accelerator.device.type == "cuda" else None
    train_started = time.monotonic()
    starting_step = progress["step"]
    loss_sum = torch.zeros((), device=accelerator.device)
    while progress["step"] < steps:
        order = epoch_indices(len(receipt["samples"]), settings["seed"], progress["epoch"], world, accumulation, micro_batch, layouts)
        per_epoch = len(order) // (world * micro_batch)
        while progress["cursor"] < per_epoch and progress["step"] < steps:
            offset = (progress["cursor"] * world + accelerator.process_index) * micro_batch
            sample = collate_samples([sample_at(cache_dir, receipt, i) for i in order[offset:offset + micro_batch]])
            with accelerator.accumulate(transformer):
                with accelerator.autocast():
                    loss = flow_loss(transformer, sample, scheduler, accelerator.device, generator=noise_generator)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite flow loss")
                accelerator.backward(loss)
                loss_sum += loss.detach()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            progress["cursor"] += 1
            progress["global_samples_seen"] += world * micro_batch
            if accelerator.sync_gradients:
                progress["step"] += 1
                mean_loss = accelerator.reduce(loss_sum / accumulation, reduction="mean").item()
                loss_sum.zero_()
                elapsed = time.monotonic() - train_started
                row = {**progress, "elapsed_seconds": elapsed,
                    "samples_per_second": (progress["step"] - starting_step) * world * accumulation * micro_batch / elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(accelerator.device) / 2**30 if accelerator.device.type == "cuda" else 0,
                    "loss": mean_loss, "loss_reduction": "all ranks and all accumulated microbatches"}
                if accelerator.is_main_process:
                    with (output / "loss.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(row, flush=True)
                # Canonical next-sample cursor, including checkpoints exactly at epoch boundaries.
                if progress["cursor"] == per_epoch:
                    progress["epoch"] += 1
                    progress["cursor"] = 0
                if val_receipt and (progress["step"] % interval == 0 or progress["step"] == steps):
                    validate()
                if progress["step"] % settings["checkpoint_every"] == 0:
                    save(f"step_{progress['step']:06d}")
                if progress["cursor"] == 0:
                    break
    save("final")
    accelerator.end_training()
