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


def flow_loss(transformer, sample, scheduler, device, *, generator=None):
    import torch
    from diffusers import QwenImage21Pipeline
    dtype = next(transformer.parameters()).dtype
    target = sample["target"].to(device=device, dtype=dtype)
    conds = [x.to(device=device, dtype=dtype) for x in sample["conditions"]]
    noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)
    index = torch.randint(len(scheduler.timesteps), (1,), device=device, generator=generator)
    timestep = scheduler.timesteps.to(device)[index]
    sigma = scheduler.sigmas.to(device=device, dtype=dtype)[index].reshape(1, 1, 1, 1, 1)
    noisy = (1 - sigma) * target + sigma * noise

    def pack(z):
        return QwenImage21Pipeline._pack_latents(z, z.shape[0], z.shape[1], z.shape[-2], z.shape[-1])

    packed_target = pack(noisy)
    packed = torch.cat([*[pack(z) for z in conds], packed_target], dim=1)
    shapes = [(1, z.shape[-2], z.shape[-1]) for z in [*conds, target]]
    image_mask = sample["image_mask"].to(device)
    if int(image_mask.sum()) != sum(h * w // 4 for _, h, w in shapes[:-1]):
        raise ValueError("Cached image mask does not match condition shapes")
    image_mask = torch.cat([image_mask, torch.ones((1, target.shape[-2] * target.shape[-1] // 4), dtype=image_mask.dtype, device=device)], dim=1)
    prompt_mask = sample["prompt_mask"]
    prediction = transformer(hidden_states=packed,
        encoder_hidden_states=sample["prompt_embeds"].to(device=device, dtype=dtype),
        encoder_hidden_states_mask=None if prompt_mask is None else prompt_mask.to(device),
        timestep=timestep / 1000, img_shapes=[shapes], img_mask=image_mask, return_dict=False)[0]
    prediction = prediction[:, -packed_target.shape[1]:]
    expected = pack(noise - target)
    return (prediction.float() - expected.float()).square().mean()


def train(config, cache_dir, output, *, max_steps=None):
    import json
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from .common import digest, file_hash, implementation_hash, read_json, within, write_json
    require_gpu()
    cache_dir, output = Path(cache_dir), Path(output)
    receipt = read_json(cache_dir / "manifest.json")
    if receipt["config_sha256"] != digest(config) or receipt["dataset"]["split"] != "train" or not receipt["complete"]:
        raise ValueError("Training requires a complete train cache with identical config")
    if receipt["implementation_sha256"] != implementation_hash():
        raise ValueError("Feature encoding implementation changed; rebuild cache")
    if not receipt["samples"]:
        raise ValueError("Empty cache")
    settings = config["training"]
    steps = settings["max_steps"] if max_steps is None else max_steps
    if steps < 1:
        raise ValueError("max_steps must be positive")
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=settings["gradient_accumulation_steps"])
    set_seed(settings["seed"])
    source = model_source(config)
    transformer = QwenImage21Transformer2DModel.from_pretrained(source, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=settings["rank"], lora_alpha=settings["alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights="gaussian"))
    transformer.enable_gradient_checkpointing()
    # FP32 trainable adapter parameters, BF16 frozen backbone, under Accelerator autocast.
    params = [p for p in transformer.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"], weight_decay=0.01)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(source, subfolder="scheduler")
    if not scheduler.config.use_dynamic_shifting:
        raise ValueError("This implementation expects the base 2.1 dynamic-shift scheduler")

    class CacheDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(receipt["samples"])
        def __getitem__(self, index):
            row = receipt["samples"][index]
            path = within(cache_dir, row["path"])
            if file_hash(path) != row["sha256"]:
                raise ValueError("Feature cache checksum mismatch")
            return torch.load(path, map_location="cpu", weights_only=True)

    loader = torch.utils.data.DataLoader(CacheDataset(), batch_size=1, shuffle=True, collate_fn=lambda xs: xs[0], num_workers=0)
    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)
    transformer.train()
    output.mkdir(parents=True, exist_ok=True)
    step, epoch, seen = 0, 0, 0
    if accelerator.is_main_process:
        write_json(output / "training_metadata.json", {"config": config, "cache": receipt,
            "world_size": accelerator.num_processes, "effective_batch": accelerator.num_processes * settings["gradient_accumulation_steps"],
            "resume_supported": False, "target_latents": "deterministic VAE posterior mode"})

    def save(name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            dest = output / name
            dest.mkdir(exist_ok=False)
            state = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
            QwenImage21Pipeline.save_lora_weights(str(dest), transformer_lora_layers=state, safe_serialization=True)
            write_json(dest / "metadata.json", {"step": step, "epoch": epoch, "global_samples_seen": seen,
                "config_sha256": digest(config), "cache_identity": receipt["cache_identity"],
                "config": config, "dataset": receipt["dataset"], "trainable_parameters": sum(p.numel() for p in params)})
        accelerator.wait_for_everyone()

    while step < steps:
        for sample in loader:
            with accelerator.accumulate(transformer):
                with accelerator.autocast():
                    loss = flow_loss(transformer, sample, scheduler, accelerator.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite flow loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            seen += accelerator.num_processes
            if accelerator.sync_gradients:
                step += 1
                mean_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                if accelerator.is_main_process:
                    row = {"step": step, "epoch": epoch, "loss": mean_loss, "global_samples_seen": seen}
                    with (output / "loss.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(row, flush=True)
                if step % settings["checkpoint_every"] == 0:
                    save(f"step_{step:06d}")
                if step >= steps:
                    break
        epoch += 1
    save("final")
    accelerator.end_training()
