"""Full-data recovery, exact support and zero-copy caches, verified without GPUs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from peft import utils as peft_utils
from safetensors.torch import load_file

from qwen_mvwd.backend import batch_layout, train as backend_train
from qwen_mvwd.common import digest, file_hash, implementation_hash, read_json, write_json
from qwen_mvwd.data import Level1Data
from qwen_mvwd.selection import PROTOCOL_KEYS
from tools import full_training as full
from tools import workflow
from test_data import fixture
from test_training_loop import Tiny, Pipeline, Scheduler


class FullTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root / "dataset")
        self.config["training"].update(gradient_accumulation_steps=2, max_steps=6, checkpoint_every=2,
                                       learning_rate=0.01, micro_batch_size=1)
        self.exp = self.root / "experiment"
        for name in ("features", "selections", "steps", "logs", "training", "reports"):
            (self.exp / name).mkdir(parents=True)
        self.world = patch.object(full, "WORLD", 1)
        self.world.start()
        self.plan = {"schema": full.SCHEMA, "implementation_sha256": implementation_hash(), "epochs": 2,
                     "effective_batch": 2, "steps_per_epoch": 3, "max_steps": 6, "validate_every": 3,
                     "validation_seed": 137, "checkpoint_selection": "fixed endpoint", "exposure": {}, "splits": {}}
        self.datasets = {s: Level1Data(self.config, s) for s in ("train", "val")}
        for split, data in self.datasets.items():
            part = {"name": split + "_part000", "selection": f"selections/{split}.json", "queries": 6}
            body = {"schema": "qwen-mvwd-selection-v1", "release_id": data.meta["release_id"], "split": split,
                    "split_sha256": data.split_hash, "query_protocol": {k:self.config[k] for k in PROTOCOL_KEYS},
                    "episodes": [{"id": r["id"], "scene": r["scene"]} for r in data.records]}
            write_json(self.exp / part["selection"], {**body, "selection_sha256": digest(body)})
            self.plan["splits"][split] = {"queries": 6, "episodes": 1, "dataset": data.provenance(),
                                           "cache_identity": data.cache_identity(), "parts": [part]}
            selected = Level1Data(self.config, split, selection=self.exp / part["selection"])
            output = self.exp / "features" / (part["name"] + "_attempt001")
            output.mkdir()
            rows = []
            for i, query in enumerate(selected.queries()):
                sample = {"target": torch.full((1, 2, 1, 4, 4), i / 10),
                          "conditions": [torch.ones(1, 2, 1, 4, 4)],
                          "image_mask": torch.ones((1, 4), dtype=torch.bool),
                          "prompt_embeds": torch.zeros(1, 4, 2), "prompt_mask": None, "query": query}
                path = output / f"sample_{i}.pt"; torch.save(sample, path)
                rows.append({"query": query, "path": path.name, "query_index": i,
                             "batch_layout": batch_layout(sample), "sha256": file_hash(path)})
            write_json(output / "manifest.json", {"complete": True, "dataset": selected.provenance(),
                "config_sha256": digest(self.config), "cache_identity": selected.cache_identity(),
                "implementation_sha256": implementation_hash(), "samples": rows})

    def tearDown(self):
        self.world.stop()
        self.temp.cleanup()

    def assemble(self):
        for split in ("train", "val"):
            full.assemble_cache(self.exp, self.config, self.plan, split)

    def test_full_cache_is_exact_idempotent_and_shares_storage(self):
        self.assemble()
        for split in ("train", "val"):
            directory, manifest = full.full_cache(self.exp, self.config, self.plan, split)
            self.assertEqual([r["query"] for r in manifest["samples"]], full.query_rows(self.datasets[split]))
            original = self.exp / "features" / (split + "_part000_attempt001") / "sample_0.pt"
            linked = directory / manifest["samples"][0]["path"]
            self.assertTrue(original.samefile(linked))
            before = file_hash(directory / "manifest.json")
            full.assemble_cache(self.exp, self.config, self.plan, split)
            self.assertEqual(before, file_hash(directory / "manifest.json"))
            # A distinct conflicting file must not silently replace encoded data.
            linked.unlink(); linked.write_bytes(b"conflict")
            with self.assertRaisesRegex(ValueError, "Conflicting assembled"):
                full.assemble_cache(self.exp, self.config, self.plan, split)

    def test_partial_partition_retries_are_bounded_and_preserve_old_files(self):
        original = self.exp / "features/train_part000_attempt001/manifest.json"
        value = read_json(original); value["complete"] = False; write_json(original, value)
        part = self.plan["splits"]["train"]["parts"][0]
        self.assertIsNone(full.completed_part(self.exp, self.config, "train", part))
        self.assertEqual(full.next_name(self.exp, "train_part000"), "train_part000_attempt002")
        self.assertTrue(original.exists())
        original.write_text('{"complete":')
        self.assertIsNone(full.completed_part(self.exp, self.config, "train", part))

    def test_wrong_query_in_a_complete_partition_is_rejected(self):
        path = self.exp / "features/train_part000_attempt001/manifest.json"
        value = read_json(path); value["samples"][0]["query"]["frame_index"] = 999; write_json(path, value)
        with self.assertRaisesRegex(ValueError, "identity/support"):
            full.assemble_cache(self.exp, self.config, self.plan, "train")

    def test_actual_optimizer_resume_after_interruption_matches_uninterrupted(self):
        self.assemble()
        fake = types.ModuleType("diffusers")
        fake.QwenImage21Pipeline = Pipeline; fake.QwenImage21Transformer2DModel = Tiny
        fake.FlowMatchEulerDiscreteScheduler = Scheduler
        failures = [True]
        def execute(experiment, action, name, extra):
            args = dict(zip(extra[::2], extra[1::2]))
            kwargs = {"max_steps": int(args["--max-steps"]), "validation_cache": args["--validation-cache"],
                      "validate_every": int(args["--validate-every"]), "validation_seed": int(args["--validation-seed"]),
                      "resume_from": args.get("--resume-from")}
            if failures[0]:
                failures[0] = False
                kwargs["max_steps"] = 2
                backend_train(self.config, args["--cache"], self.exp / "training" / name, **kwargs)
                # Simulate another checkpoint writer being killed before commit.
                partial = self.exp / "training" / name / "step_000003/metadata.json"
                partial.parent.mkdir(); partial.write_text('{"complete":')
                raise subprocess.CalledProcessError(143, ["simulated-preemption"])
            backend_train(self.config, args["--cache"], self.exp / "training" / name, **kwargs)
        with patch.dict("sys.modules", {"diffusers": fake}), patch("qwen_mvwd.backend.require_gpu"), \
             patch("qwen_mvwd.backend.model_source", return_value="fixture"), \
             patch.object(peft_utils, "get_peft_model_state_dict", side_effect=lambda model:model.state_dict()), \
             patch.object(peft_utils, "set_peft_model_state_dict", side_effect=lambda model,state:model.load_state_dict(state)), \
             patch.object(full, "execute", side_effect=execute):
            with self.assertRaises(subprocess.CalledProcessError):
                full.train(self.exp, self.config, self.plan, 2, lambda *a,**k:None)
            self.assertEqual(read_json(self.exp / "reports/CHECKPOINTS.json")["latest_step"], 2)
            full.train(self.exp, self.config, self.plan, 2, lambda *a,**k:None)
            backend_train(self.config, self.exp / "features/train_full", self.root / "uninterrupted",
                          validation_cache=self.exp / "features/val_full", validate_every=3)
        index = read_json(self.exp / "reports/CHECKPOINTS.json")
        self.assertEqual(index["latest_step"], 6)
        self.assertEqual(set(index["epochs"]), {"1", "2"})
        actual = load_file(str(self.exp / index["latest"] / "pytorch_lora_weights.safetensors"))
        expected = load_file(str(self.root / "uninterrupted/final/pytorch_lora_weights.safetensors"))
        for key in expected:self.assertTrue(torch.equal(actual[key], expected[key]))
        self.assertTrue((self.exp / "training/epoch_01").is_symlink())
        self.assertTrue((self.exp / "training/final").is_symlink())
        summary = read_json(self.exp / "reports/FULL_TRAINING.json")
        self.assertEqual(summary["logged_sample_visits_including_replayed_attempts"], 12)
        self.assertTrue(summary["completed_budget"])
        # Full-state integrity covers optimizer/RNG files as well as LoRA weights.
        state = self.exp / index["latest"] / "state_rank_0000.pt"
        state.write_bytes(b"corrupted")
        _, a = full.full_cache(self.exp, self.config, self.plan, "train")
        _, b = full.full_cache(self.exp, self.config, self.plan, "val")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            full.checkpoint_index(self.exp, self.plan, full.resume_contract(self.config, self.plan, a, b))

    def test_reference_checks_query_frames_and_assets_without_gt_access(self):
        reference = self.root / "reference"
        directory = reference / "training/train_gpu4_b16"
        directory.mkdir(parents=True)
        metadata = {"datasets":[]}
        for split, data in self.datasets.items():
            row = data.provenance()
            row.update(query_count=6, target_size=self.config["target_hw"], robots=[0,1,2], bev_size=256,
                       frame_sampling={"policy":"stratified_hash", "frames_per_episode":2, "seed":42})
            metadata["datasets"].append(row)
            (directory / (split + ".jsonl")).write_bytes((data.root / "splits" / (split + ".jsonl")).read_bytes())
        write_json(directory / "run_metadata.json", metadata)
        write_json(directory / "training_control.json", {"max_epochs":10})
        write_json(reference / "evaluations/training_plan.json", {"sampling_plan_reference":"frames.jsonl"})
        full.jsonl(reference / "frames.jsonl", full.frame_rows(self.datasets["train"]))
        with patch.object(Level1Data, "target", side_effect=AssertionError("GT read")):
            result = full.verify_reference(self.config, self.datasets, reference)
        self.assertTrue(result["exact_training_frame_plan_matched"])
        value=full.frame_rows(self.datasets["train"]); value[0]["frames"]=[99,100]
        (reference / "frames.jsonl").write_text(json.dumps(value[0])+"\n")
        with self.assertRaisesRegex(ValueError, "query identities differ"):
            full.verify_reference(self.config, self.datasets, reference)

    def test_partial_log_tail_is_preserved_and_interior_corruption_rejected(self):
        log=self.root/"loss.jsonl"; data='{"step":1}\n{"step":'
        log.write_text(data)
        self.assertEqual(full.log_rows(log), [{"step":1}])
        self.assertEqual(log.read_text(),data)
        log.write_text(data+'\n{"step":2}\n')
        with self.assertRaises(json.JSONDecodeError):full.log_rows(log)

    def test_plan_and_gpu_guard_do_not_start_gpu_work(self):
        self.plan["exposure"]={}
        for row in self.plan["splits"].values():row["episodes"]=1
        args=argparse.Namespace(experiment="example",stage="train",through_epoch=1,dry_run=True)
        with patch.object(workflow,"resolve_experiment",return_value=self.exp), \
             patch.object(full,"load_plan",return_value=(self.config,self.plan)), \
             patch.object(torch.cuda,"device_count",side_effect=AssertionError("CUDA touched")), \
             contextlib.redirect_stdout(io.StringIO()):
            full.main_full(args)
            args.dry_run=False
            with patch.dict("os.environ",{},clear=True), self.assertRaisesRegex(RuntimeError,"existing four-H100"):
                full.main_full(args)
        self.assertFalse(list((self.exp/"steps").iterdir()))

    def test_frozen_plan_tampering_is_rejected(self):
        write_json(self.exp/"plans/full_training.json",self.plan)
        write_json(self.exp/"reports/FULL_SETUP.json",{"plan_sha256":"wrong"})
        with self.assertRaisesRegex(ValueError,"plan changed"):
            full.load_plan(self.exp)
