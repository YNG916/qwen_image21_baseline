"""Fixed-noise camera interventions, separate from benchmark predictions.

Virtual camera interventions keep all robot base states unchanged. They do not
simulate robot motion, validate free space, or provide counterfactual GT.
"""
from __future__ import annotations

import copy
import itertools
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import digest, file_hash, read_json, sample_seed, within, write_json
from .data import Level1Data, check_pose


def with_camera(condition, camera):
    """Replace only the target camera, preserving the original prompt template."""
    result = copy.deepcopy(condition)
    check_pose(np.asarray(camera["camera_to_world"]))
    check_pose(np.asarray(camera["camera_to_local_world"]))
    result["cameras"] = [copy.deepcopy(camera)]
    result["payload"]["cameras"] = [{k: v for k, v in camera.items() if k != "camera_to_world"}]
    prefix, separator, encoded = condition["prompt"].partition("Conditions:\n")
    if not separator or json.loads(encoded) != condition["payload"]:
        raise ValueError("Condition prompt does not match its structured payload")
    result["prompt"] = prefix + separator + json.dumps(result["payload"], separators=(",", ":"))
    from .camera_conditioning import apply_camera_cues
    return apply_camera_cues(result, result.get("camera_cue_variant", "matrix_text"))


def camera_cases(data, episode_id, frame):
    """Recorded views followed by interventions on robot 0's query camera."""
    conditions = [data.conditions({"episode_id": episode_id, "frame_index": frame, "robots": [r]}) for r in (0, 1, 2)]
    reference = conditions[0]
    cases = [{"name": f"recorded_{r}", "condition": condition, "gt_robot": r,
              "intervention": {"kind": "recorded", "robot_id": r}}
             for r, condition in enumerate(conditions)]
    cases.append({"name": "repeat_0", "condition": copy.deepcopy(reference), "gt_robot": 0,
                  "intervention": {"kind": "identical_repeat"}})
    for degrees in (30, -30):
        camera = copy.deepcopy(reference["cameras"][0])
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for key in ("camera_to_world", "camera_to_local_world"):
            T = np.asarray(camera[key], dtype=float)
            T[:3, :3] = rotation @ T[:3, :3]
            camera[key] = T.tolist()
        cases.append({"name": f"yaw_{'plus' if degrees > 0 else 'minus'}30",
                      "condition": with_camera(reference, camera), "gt_robot": None,
                      "intervention": {"kind": "virtual_camera_world_z_yaw", "degrees": degrees}})
    camera = copy.deepcopy(reference["cameras"][0])
    offset = np.asarray(camera["camera_to_world"])[:3, 0] * 0.5
    for key in ("camera_to_world", "camera_to_local_world"):
        T = np.asarray(camera[key], dtype=float)
        T[:3, 3] += offset
        camera[key] = T.tolist()
    cases.append({"name": "right_0p5m", "condition": with_camera(reference, camera), "gt_robot": None,
                  "intervention": {"kind": "virtual_camera_local_right", "meters": 0.5}})
    camera = copy.deepcopy(reference["cameras"][0])
    K = np.asarray(camera["K"], dtype=float)
    K[0, 0] *= 1.25
    K[1, 1] *= 1.25
    camera["K"] = K.tolist()
    cases.append({"name": "focal_1p25x", "condition": with_camera(reference, camera), "gt_robot": None,
                  "intervention": {"kind": "virtual_camera_focal_scale", "factor": 1.25}})
    for case in cases:
        condition = case["condition"]
        for key in ("robot_states", "asset", "local_world_to_bev_pixel"):
            if condition["payload"][key] != reference["payload"][key]:
                raise ValueError(f"Intervention changed shared {key}")
        actual = [condition["images"][0], condition.get("camera_cue_base_image", condition["images"][1])]
        expected = [reference["images"][0], reference.get("camera_cue_base_image", reference["images"][1])]
        if any(a.tobytes() != b.tobytes() for a, b in zip(actual, expected)):
            raise ValueError("Camera diagnostic must preserve BEV and shared robot annotations")
    return cases


def camera_overlay(condition):
    """Target camera rays for human review only; never supplied to Qwen."""
    if condition.get("camera_cue_variant"):
        return condition["images"][1].copy()
    image = condition["images"][1].copy()
    draw = ImageDraw.Draw(image)
    camera = condition["cameras"][0]
    T = np.asarray(camera["camera_to_world"])
    K = np.asarray(camera["K"])
    w2p = condition["world_to_bev"]
    h, w = condition["payload"]["target_view_hw"]
    origin = T[:, 3]
    start = (w2p @ origin)[:2]
    draw.ellipse((*tuple(start - 4), *tuple(start + 4)), fill="magenta")
    for u in (0.5, w / 2, w - 0.5):
        ray = T[:3, :3] @ np.linalg.solve(K, [u, h / 2, 1])
        norm = np.linalg.norm(ray[:2])
        if norm > 1e-8:
            endpoint = origin.copy()
            endpoint[:2] += ray[:2] / norm * 1.5
            draw.line([tuple(start), tuple((w2p @ endpoint)[:2])], fill="magenta", width=2)
    return image


def plan(config, args):
    if config["mode"] != "independent" or config["robots"] != [0, 1, 2]:
        raise ValueError("Camera diagnostics require independent mode and robots [0,1,2]")
    if args.split not in ("train", "val") or args.max_groups < 1:
        raise ValueError("Use a bounded train/val diagnostic with max_groups >= 1")
    if not args.dry_run and not (args.lora or args.zero_shot):
        raise ValueError("Generation requires --lora or --zero-shot")
    data = Level1Data(config, args.split, scenes=args.scene, max_episodes=args.max_episodes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    meta = {"schema": "qwen-mvwd-camera-diagnostic-v1", "complete": False, "dry_run": args.dry_run,
            "config": config, "dataset": data.provenance(), "ground_truth_loaded": False,
            "benchmark_ready": False, "base_seed": args.seed,
            "seed_policy": "anchor robot-0 query seed reused for every case in each group",
            "robot_states_changed": False,
            "camera_conditioning": config.get("camera_conditioning", "matrix_text"),
            "input_image_policy": "shared BEV/robot annotations; optional query-camera cues are recomputed per case",
            "counterfactual_gt_available": False,
            "virtual_camera_free_space_validated": False,
            "lora_path": str(Path(args.lora).resolve()) if args.lora else None,
            "lora_sha256": None, "groups": []}
    # Store all conditions before GPU loading; target() is never called here.
    anchors = (q for q in data.queries() if q["robots"] == [0])
    for index, query in enumerate(itertools.islice(anchors, args.max_groups)):
        seed = sample_seed(args.seed, data.meta["release_id"], query["episode_id"], query["frame_index"], [0], "independent")
        group = {"name": f"group_{index:04d}", "query": query, "seed": seed, "cases": []}
        for case in camera_cases(data, query["episode_id"], query["frame_index"]):
            directory = output / group["name"] / case["name"]
            directory.mkdir(parents=True)
            condition = case["condition"]
            for i, image in enumerate(condition["images"]):
                image.save(directory / f"condition_{i}.png")
            camera_overlay(condition).save(directory / "camera_overlay_review_only.png")
            (directory / "prompt.txt").write_text(condition["prompt"])
            write_json(directory / "geometry.json", {"query": condition["query"], "payload": condition["payload"],
                       "cameras": condition["cameras"], "world_to_bev": condition["world_to_bev"].tolist()})
            group["cases"].append({"name": case["name"], "directory": str(directory.relative_to(output)),
                "gt_robot": case["gt_robot"], "intervention": case["intervention"],
                "prompt_sha256": file_hash(directory / "prompt.txt"),
                "condition_sha256": [file_hash(directory / f"condition_{i}.png") for i in range(2)],
                "geometry_sha256": file_hash(directory / "geometry.json"), "seed": seed})
        meta["groups"].append(group)
    if not meta["groups"]:
        raise ValueError("Empty diagnostic selection")
    write_json(output / "metadata.json", meta)
    if args.dry_run:
        meta["complete"] = True
        write_json(output / "metadata.json", meta)
        print(f"Prepared {len(meta['groups'])} groups; no model or GT loaded", flush=True)
        return
    import torch
    from .backend import load_pipeline, require_gpu
    require_gpu()
    meta["versions"] = {k: importlib.metadata.version(k) for k in ("torch", "diffusers", "transformers", "peft")}
    pipe = load_pipeline(config, offload=args.cpu_offload)
    if args.lora:
        lora = Path(args.lora)
        trained = read_json(lora / "metadata.json")
        if trained["config_sha256"] != digest(config):
            raise ValueError("Diagnostic LoRA/config mismatch")
        if any(trained["dataset"][k] != data.provenance()[k] for k in ("release_id", "robot_assets")):
            raise ValueError("Diagnostic LoRA dataset/assets mismatch")
        meta["lora_sha256"] = file_hash(lora / "pytorch_lora_weights.safetensors")
        pipe.load_lora_weights(str(lora))
    write_json(output / "metadata.json", meta)
    h, w = config["target_hw"]
    with torch.inference_mode():
        for group in meta["groups"]:
            for case in group["cases"]:
                directory = within(output, case["directory"])
                with Image.open(directory / "condition_0.png") as im:
                    clean = im.copy()
                with Image.open(directory / "condition_1.png") as im:
                    annotated = im.copy()
                result = pipe(prompt=(directory / "prompt.txt").read_text(), image=[clean, annotated],
                    height=h, width=w, output_resolution=config["bev_size"],
                    num_inference_steps=config["inference"]["steps"], true_cfg_scale=1.0,
                    use_kv_cache=config["inference"]["use_kv_cache"],
                    generator=torch.Generator(device="cpu").manual_seed(case["seed"])).images[0]
                rgba = result.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                if rgb.size != (w, h):
                    raise ValueError("Pipeline changed diagnostic output size")
                rgb.save(directory / "prediction.png")
                case["prediction_sha256"] = file_hash(directory / "prediction.png")
                write_json(output / "metadata.json", meta)
                print(f"Generated {group['name']}/{case['name']} seed={case['seed']}", flush=True)
    meta["complete"] = True
    write_json(output / "metadata.json", meta)


def review(config, args):
    """Read GT only after completed generation; never score invented poses vs GT."""
    source, output = Path(args.diagnostics).resolve(), Path(args.output)
    meta = read_json(source / "metadata.json")
    if meta["schema"] != "qwen-mvwd-camera-diagnostic-v1" or not meta["complete"] or meta["dry_run"] or meta["ground_truth_loaded"]:
        raise ValueError("Review requires completed GT-free diagnostic generation")
    if digest(meta["config"]) != digest(config):
        raise ValueError("Diagnostic review config mismatch")
    data = Level1Data(config, meta["dataset"]["split"])
    if any(meta["dataset"][k] != data.provenance()[k] for k in ("release_id", "robot_assets", "split_sha256")):
        raise ValueError("Diagnostic review dataset provenance mismatch")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "qwen-mvwd-camera-review-v1", "diagnostics": str(source),
              "diagnostic_metadata_sha256": file_hash(source / "metadata.json"),
              "ground_truth_read_only_after_generation": True, "benchmark_ready": False,
              "note": "Pixel response is not proof of camera adherence; virtual poses have no GT.", "groups": []}
    h, w = config["target_hw"]
    for group in meta["groups"]:
        targets = [data.target({**group["query"], "robots": [r]}) for r in (0, 1, 2)]
        gt_arrays = [np.asarray(im).astype(float) / 255 for im in targets]
        images = {}
        for case in group["cases"]:
            directory = within(source, case["directory"])
            if file_hash(directory / "prediction.png") != case["prediction_sha256"]:
                raise ValueError("Diagnostic prediction checksum mismatch")
            with Image.open(directory / "prediction.png") as im:
                if im.mode != "RGB" or im.size != (w, h):
                    raise ValueError("Diagnostic prediction dimensions differ")
                images[case["name"]] = im.copy()
        baseline = np.asarray(images["recorded_0"]).astype(float) / 255
        rows = []
        canvas = Image.new("RGB", (w * 3, (h + 32) * len(group["cases"])), "white")
        draw = ImageDraw.Draw(canvas)
        for index, case in enumerate(group["cases"]):
            predicted = np.asarray(images[case["name"]]).astype(float) / 255
            row = {"case": case["name"], "seed": case["seed"], "gt_robot": case["gt_robot"],
                   "mean_absolute_pixel_change_vs_recorded_0": float(np.abs(predicted - baseline).mean())}
            if case["gt_robot"] is not None:
                errors = [float(np.square(predicted - gt).mean()) for gt in gt_arrays]
                correct = case["gt_robot"]
                psnr = -10 * np.log10(max(errors[correct], 1e-12))
                row.update(psnr_db=float(psnr), mse_to_each_recorded_gt=errors,
                           own_gt_is_closest=bool(np.argmin(errors) == correct))
            if case["name"] == "repeat_0":
                row["identical_repeat_pixels_equal"] = bool(np.array_equal(predicted, baseline))
            rows.append(row)
            y = index * (h + 32)
            draw.text((4, y + 4), case["name"], fill="black")
            draw.text((w + 4, y + 4), "prediction", fill="black")
            label = f"GT robot {case['gt_robot']}" if case["gt_robot"] is not None else "NO GT: virtual camera only"
            draw.text((2 * w + 4, y + 4), label, fill="black")
            with Image.open(within(source, case["directory"]) / "camera_overlay_review_only.png") as im:
                overlay = im.convert("RGB"); overlay.thumbnail((w, h))
                canvas.paste(overlay, ((w - overlay.width) // 2, y + 32))
            canvas.paste(images[case["name"]], (w, y + 32))
            if case["gt_robot"] is not None:
                canvas.paste(targets[case["gt_robot"]], (2 * w, y + 32))
        canvas.save(output / f"{group['name']}_comparisons.png")
        report["groups"].append({"query": group["query"], "cases": rows})
    write_json(output / "camera_response.json", report)
    print(json.dumps(report, indent=2))
