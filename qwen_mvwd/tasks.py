from __future__ import annotations

import importlib.metadata
import itertools
import json
from pathlib import Path

import numpy as np

from .common import digest, file_hash, implementation_hash, sample_seed, within, write_json
from .data import Level1Data


def select_data(config, args):
    return Level1Data(config, args.split, scenes=args.scene, max_episodes=args.max_episodes, selection=getattr(args, "selection", None))


def inspect(config, args):
    data = select_data(config, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    query_count = 0
    # Count only metadata. Decode a bounded preview, not the full dataset.
    for query in data.queries():
        if query_count < args.preview_samples:
            condition = data.conditions(query)
            dest = output / f"sample_{query_count:04d}"
            dest.mkdir()
            for i, image in enumerate(condition["images"]):
                image.save(dest / f"condition_{i}.png")
            (dest / "prompt.txt").write_text(condition["prompt"])
            write_json(dest / "geometry.json", {"query": query, "cameras": condition["cameras"],
                "payload": condition["payload"], "world_to_bev": condition["world_to_bev"].tolist()})
            if args.with_targets:
                data.target(query).save(dest / "target.png")
        query_count += 1
    report = {"dataset": data.provenance(), "query_count": query_count,
        "view_count": query_count * (3 if config["mode"] == "joint" else 1),
        "ground_truth_loaded": bool(args.with_targets and args.preview_samples),
        "mode": config["mode"], "config_sha256": digest(config)}
    write_json(output / "inspection.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "dataset"}, indent=2))


def prepare(config, args):
    import torch
    from accelerate import Accelerator
    from .backend import batch_layout, encode_sample, load_pipeline, require_gpu
    from .common import read_json
    require_gpu()
    if args.split not in ("train", "val"):
        raise ValueError("Feature caches accept train/val only; no test supervision during development")
    accelerator = Accelerator(mixed_precision="bf16")
    data = select_data(config, args)
    output = Path(args.output)
    receipt = {"schema": "qwen-mvwd-features-v1", "complete": False,
        "config_sha256": digest(config), "cache_identity": data.cache_identity(),
        "dataset": data.provenance(), "implementation_sha256": implementation_hash(), "samples": []}
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "manifest.json", receipt)
    accelerator.wait_for_everyone()
    rank_dir = output / f"rank_{accelerator.process_index:04d}"
    rank_dir.mkdir(exist_ok=False)
    pipe = load_pipeline(config, encoding_only=True)
    queries = data.queries()
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max_samples must be positive")
        queries = itertools.islice(queries, args.max_samples)
    rows = []
    for i, query in enumerate(queries):
        if i % accelerator.num_processes != accelerator.process_index:
            continue
        condition = data.conditions(query)
        sample = encode_sample(pipe, condition, data.target(query), config["bev_size"])
        path = rank_dir / f"sample_{i:07d}.pt"
        torch.save(sample, path)
        rows.append({"path": str(path.relative_to(output)), "query": query, "sha256": file_hash(path),
                     "batch_layout": batch_layout(sample), "query_index": i})
        if len(rows) % 32 == 0:
            write_json(rank_dir / "manifest.json", {"complete": False, "samples": rows})
            print(f"Rank {accelerator.process_index}: cached {len(rows)} samples", flush=True)
    write_json(rank_dir / "manifest.json", {"complete": True, "samples": rows})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = []
        for rank in range(accelerator.num_processes):
            part = read_json(output / f"rank_{rank:04d}" / "manifest.json")
            if not part["complete"]:
                raise ValueError("Incomplete feature shard")
            merged.extend(part["samples"])
        merged.sort(key=lambda row: row["query_index"])
        if [r["query_index"] for r in merged] != list(range(len(merged))):
            raise ValueError("Feature shards have duplicate/missing query indices")
        receipt["samples"] = merged
        receipt["complete"] = True
        receipt["encoding_processes"] = accelerator.num_processes
        receipt["selection_limited"] = args.max_samples is not None or args.max_episodes is not None or bool(args.scene) or bool(getattr(args, "selection", None))
        write_json(output / "manifest.json", receipt)
        print(f"Completed {len(merged)} cached samples across {accelerator.num_processes} GPUs", flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


def infer(config, args):
    import torch
    from PIL import Image
    from .backend import load_pipeline, require_gpu
    from .common import read_json
    require_gpu()
    data = select_data(config, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pipe = load_pipeline(config, offload=args.cpu_offload)
    lora_hash = None
    if args.lora:
        lora = Path(args.lora)
        trained = read_json(lora / "metadata.json")
        if trained["config_sha256"] != digest(config):
            raise ValueError("LoRA config/conditioning differs from inference config")
        for key in ("release_id", "robot_assets"):
            if trained["dataset"][key] != data.provenance()[key]:
                raise ValueError("LoRA dataset release or asset differs")
        lora_hash = file_hash(lora / "pytorch_lora_weights.safetensors")
        pipe.load_lora_weights(str(lora))
    else:
        if not args.zero_shot:
            raise ValueError("Pass --lora or explicitly choose --zero-shot")
    count = 0
    meta = {"schema": "qwen-mvwd-predictions-v1", "complete": False, "method": "Qwen-Image-2.1-" + config["mode"],
        "config": config, "dataset": data.provenance(), "base_seed": args.seed,
        "ground_truth_loaded": False, "lora_sha256": lora_hash,
        "seed_policy": "independent-query-sha256" if config["mode"] == "independent" else "joint-group-sha256",
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "diffusers", "transformers", "peft")}}
    write_json(output / "metadata.json", meta)
    h, w = config["target_hw"]
    with (output / "predictions.jsonl").open("x") as stream, torch.inference_mode():
        for query in data.queries():
            condition = data.conditions(query)
            seed = sample_seed(args.seed, data.meta["release_id"], query["episode_id"], query["frame_index"], query["robots"], config["mode"])
            result = pipe(prompt=condition["prompt"], image=condition["images"],
                height=h, width=w * len(query["robots"]), output_resolution=config["bev_size"],
                num_inference_steps=config["inference"]["steps"], true_cfg_scale=1.0,
                use_kv_cache=config["inference"]["use_kv_cache"],
                generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
            # Output RGB protocol; alpha composited over white (never discarded blindly).
            rgba = result.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            if rgb.size != (w * len(query["robots"]), h):
                raise ValueError("Pipeline changed requested output size")
            for slot, robot in enumerate(query["robots"]):
                path = Path(query["episode_id"]) / f"robot_{robot:02d}" / f"frame_{query['frame_index']:03d}.png"
                (output / path).parent.mkdir(parents=True, exist_ok=True)
                rgb.crop((slot * w, 0, (slot + 1) * w, h)).save(output / path)
                row = {"episode_id": query["episode_id"], "robot_id": robot, "frame_index": query["frame_index"],
                       "seed": seed, "path": str(path), "group_robots": query["robots"]}
                stream.write(json.dumps(row) + "\n")
                count += 1
            stream.flush()
            print(f"Generated {count} views", flush=True)
    meta.update(complete=True, view_count=count)
    write_json(output / "metadata.json", meta)


def evaluate(config, args):
    """Reuse existing metric mathematics; validate Qwen's own manifest schema."""
    import sys
    from PIL import Image
    from .common import frame_selection, read_json
    sys.path.insert(0, str(Path(args.metrics_repo).resolve()))
    from mvwd_metrics import Level1Evaluator
    from mvwd_metrics.io import load_gt_episode, json_safe
    from adapters.mvwd_raw import MVWDRaw
    from mvwd_metrics.geometry import gt_surface_correspondences
    from itertools import combinations
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    all_rows, metas, identity = {}, [], None
    for directory in map(Path, args.predictions):
        meta = read_json(directory / "metadata.json")
        if meta["schema"] != "qwen-mvwd-predictions-v1" or not meta["complete"] or meta["ground_truth_loaded"]:
            raise ValueError("Incomplete or non-GT-free prediction set")
        if digest(meta["config"]) != digest(config):
            raise ValueError("Prediction config mismatch")
        signature = (meta["lora_sha256"], digest(meta["dataset"]["robot_assets"]), meta["dataset"]["release_id"], digest(meta["versions"]))
        if identity is None:
            identity = signature
        if identity != signature:
            raise ValueError("Mixed model/asset/environment provenance")
        seed = meta["base_seed"]
        if seed in all_rows:
            raise ValueError("Duplicate base seed")
        rows = {}
        for line in (directory / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["episode_id"], row["robot_id"], row["frame_index"])
            if key in rows:
                raise ValueError("Duplicate prediction")
            robots = [key[1]] if config["mode"] == "independent" else [0, 1, 2]
            if row["group_robots"] != robots or row["seed"] != sample_seed(seed, meta["dataset"]["release_id"], key[0], key[2], robots, config["mode"]):
                raise ValueError("Seed/group policy mismatch")
            rows[key] = within(directory, row["path"])
        if len(rows) != meta["view_count"]:
            raise ValueError("Manifest count mismatch")
        all_rows[seed] = rows
        metas.append(meta)
    seeds = sorted(all_rows)
    if not seeds or any(set(all_rows[s]) != set(all_rows[seeds[0]]) for s in seeds):
        raise ValueError("Seed supports differ")
    split = metas[0]["dataset"]["split"]
    data = Level1Data(config, split)
    expected = {(q["episode_id"], robot, q["frame_index"]) for q in data.queries() for robot in q["robots"]}
    keys = set(all_rows[seeds[0]])
    if any(m["dataset"]["split"] != split or m["dataset"]["split_sha256"] != data.split_hash or m["dataset"]["release_id"] != data.meta["release_id"] or m["dataset"]["robot_assets"] != data.provenance()["robot_assets"] for m in metas):
        raise ValueError("Evaluation requires matching pinned split/assets/release")
    official = split == "test" and seeds == [42, 43, 44] and keys == expected and config["target_hw"] == [256, 448] and config["frames_per_episode"] == 2 and config["frame_sample_seed"] == 42 and config["robots"] == [0, 1, 2]
    if not args.preview and not official:
        raise ValueError("Formal evaluation requires full test, fixed 2 frames, 3 robots, 3 seeds, 256x448")
    if not keys <= expected:
        raise ValueError("Queries differ from fixed frame protocol")
    evaluator = Level1Evaluator(seeds=seeds, metrics=args.metrics, device=args.device, weights_dir=args.weights_dir)
    raw = MVWDRaw(config["dataset_root"], split)
    by_episode = {}
    for e, r, t in sorted(keys):
        by_episode.setdefault(e, []).append((r, t))
    geometric = "mv_gtwarp_dino" in args.metrics
    views_file = (output / "per_query.jsonl").open("x")
    pairs_file = (output / "per_pair.jsonl").open("x")
    try:
        for record in raw.records:
            episode = record["id"]
            if episode not in by_episode:
                continue
            units = by_episode[episode]
            # Existing reader handles actual RGB/depth camera registration.
            gt_images, cameras = load_gt_episode(raw, record, units, tuple(config["target_hw"]), geometry=geometric)
            gt = np.stack([gt_images[unit] for unit in units])
            for seed in seeds:
                predicted = []
                for r, t in units:
                    with Image.open(all_rows[seed][(episode, r, t)]) as im:
                        if im.mode != "RGB" or (im.height, im.width) != tuple(config["target_hw"]):
                            raise ValueError("RGB dimensions differ from protocol")
                        predicted.append(np.asarray(im).transpose(2, 0, 1).astype(np.float32) / 255)
                predicted = np.stack(predicted)
                for row in evaluator.update_views(episode, seed, units, predicted, gt):
                    views_file.write(json.dumps(json_safe(row)) + "\n")
                if geometric:
                    for a, b in combinations(range(len(units)), 2):
                        if units[a][1] != units[b][1] or units[a][0] == units[b][0]:
                            continue
                        correspondences = gt_surface_correspondences(cameras[units[a]], cameras[units[b]])
                        if correspondences["ratio"] <= 0.1:
                            continue
                        backend = evaluator.backend("mv_gtwarp_dino")
                        features = backend.features(predicted[[a, b]])
                        row = evaluator.update_gtwarp_pair(episode, seed, (units[a], units[b]), correspondences,
                            features=(features[0], features[1]))
                        pairs_file.write(json.dumps(json_safe(row)) + "\n")
    finally:
        views_file.close()
        pairs_file.close()
    result = evaluator.finish()
    result.update(formal_coverage=official, preview=args.preview,
        benchmark_ready=official and not args.preview and set(args.metrics) == {"psnr", "lpips", "fid", "mv_gtwarp_dino"}
            and all(v.get("mean") is not None for v in result["metrics"].values()),
        ground_truth_used_only_for_evaluation=True, prediction_metadata=metas)
    result["metric_source_sha256"] = {str(p.relative_to(Path(args.metrics_repo))): file_hash(p)
        for p in sorted((Path(args.metrics_repo) / "mvwd_metrics").glob("*.py"))}
    write_json(output / "metrics.json", json_safe(result))
