"""CPU-only orchestration checks. No model, GPU process or Slurm step is launched."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from qwen_mvwd.common import read_json, write_json
from tools import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts/mvwd"
        self.experiments = self.artifacts / "experiments"
        self.exp = self.experiments / "example"
        for d in ("steps", "logs", "reports", "predictions", "evaluations", "source"):
            (self.exp / d).mkdir(parents=True)
        write_json(self.exp / "manifest.json", {"purpose": "fixture"})
        write_json(self.exp / "config.json", {"fixture": True})
        self.patches = [patch.object(workflow, "ROOT", self.root), patch.object(workflow, "ARTIFACTS", self.artifacts),
            patch.object(workflow, "EXPERIMENTS", self.experiments)]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def args(self, action="infer", dry=False):
        return argparse.Namespace(action=action, experiment="example", name="train60", dry_run=dry,
                                  review_name=None, processes=1)

    def test_dry_run_has_no_side_effects_and_includes_post_generation_review(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(workflow, "stream_command", side_effect=AssertionError("Unexpected execution")):
            workflow.run_workflow(self.args(dry=True), ["--zero-shot", "--split", "train"])
        self.assertIn("review_predictions.py", out.getvalue())
        self.assertFalse(list((self.exp / "steps").iterdir()))

    def test_infer_failure_blocks_gt_export_and_records_failure(self):
        calls = []
        def fail(command, log):
            calls.append(command)
            raise subprocess.CalledProcessError(7, command)
        with patch.dict("os.environ", {"SLURM_JOB_ID": "fixture"}), \
             patch.object(workflow, "snapshot_tools", return_value={"fixture": True}), \
             patch.object(workflow, "stream_command", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.run_workflow(self.args(), ["--zero-shot"])
        self.assertEqual(len(calls), 1)
        receipt = read_json(self.exp / "steps/train60.workflow.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["exit_code"], 7)
        self.assertTrue((self.exp / "reports/OVERVIEW.md").exists())

    def test_no_gpu_launch_outside_existing_allocation(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(workflow, "stream_command", side_effect=AssertionError("GPU launched")):
            with self.assertRaisesRegex(RuntimeError, "existing GPU allocation"):
                workflow.run_workflow(self.args(), ["--zero-shot"])

    def test_eval_resolves_inference_and_preview_metrics(self):
        calls = []
        with patch.object(workflow, "snapshot_tools", return_value={"fixture": True}), \
             patch.object(workflow, "stream_command", side_effect=lambda command, log:calls.append(command)):
            workflow.run_workflow(self.args("eval"), ["--preview", "--device", "cpu"])
        command = calls[0]
        self.assertIn(str(self.exp / "predictions/train60"), command)
        self.assertIn("train60_eval", command)
        self.assertIn("mv_gtwarp_dino", command)
        self.assertNotIn("fid", command)
        self.assertEqual(read_json(self.exp / "steps/train60_eval.workflow.json")["status"], "completed")

    def test_last_device_flag_controls_gpu_guard(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(workflow, "stream_command", side_effect=AssertionError("GPU launched")):
            with self.assertRaisesRegex(RuntimeError, "existing GPU allocation"):
                workflow.run_workflow(self.args("eval"), ["--preview", "--device", "cpu", "--device=cuda"])
