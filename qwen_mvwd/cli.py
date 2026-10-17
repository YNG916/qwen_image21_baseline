from __future__ import annotations

import argparse
from .common import load_config


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("inspect", "prepare", "train", "infer", "evaluate", "camera_diagnose", "camera_review"):
        p = sub.add_parser(kind)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if kind in ("inspect", "prepare", "infer", "camera_diagnose"):
            p.add_argument("--split", choices=("train", "val", "test"), default="train" if kind != "infer" else "test")
            p.add_argument("--scene", action="append")
            p.add_argument("--max-episodes", type=int)
        if kind == "inspect":
            p.add_argument("--preview-samples", type=int, default=3)
            p.add_argument("--with-targets", action="store_true")
        elif kind == "prepare":
            p.add_argument("--max-samples", type=int)
        elif kind == "train":
            p.add_argument("--cache", required=True)
            p.add_argument("--max-steps", type=int)
        elif kind == "infer":
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--lora")
            group.add_argument("--zero-shot", action="store_true")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--cpu-offload", action="store_true")
        elif kind == "camera_diagnose":
            p.set_defaults(split="train", max_episodes=1)
            p.add_argument("--max-groups", type=int, default=1)
            p.add_argument("--dry-run", action="store_true")
            group = p.add_mutually_exclusive_group()
            group.add_argument("--lora")
            group.add_argument("--zero-shot", action="store_true")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--cpu-offload", action="store_true")
        elif kind == "camera_review":
            p.add_argument("--diagnostics", required=True)
        elif kind == "evaluate":
            p.add_argument("--predictions", nargs="+", required=True)
            p.add_argument("--metrics-repo", required=True)
            p.add_argument("--weights-dir", required=True)
            p.add_argument("--metrics", nargs="+", choices=("psnr", "lpips", "fid", "mv_gtwarp_dino"), default=["psnr", "lpips", "fid", "mv_gtwarp_dino"])
            p.add_argument("--device", default="cuda")
            p.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.kind in ("camera_diagnose", "camera_review"):
        from .camera_diagnostics import plan, review
        (plan if args.kind == "camera_diagnose" else review)(config, args)
    elif args.kind == "train":
        from .backend import train
        train(config, args.cache, args.output, max_steps=args.max_steps)
    else:
        from . import tasks
        getattr(tasks, args.kind)(config, args)


if __name__ == "__main__":
    main()
