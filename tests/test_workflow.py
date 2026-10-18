"""CPU-only orchestration checks. No model, GPU process or Slurm step is launched."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import subprocess

from qwen_mvwd.common import read_json, write_json
from tools import workflow


class ExperimentFixture(unittest.TestCase):
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


class WorkflowTests(ExperimentFixture):
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


class TrainingOwnershipTests(ExperimentFixture):
    def test_checkpoint_short_name_resolves_inside_training(self):
        args = self.args(dry=True)
        args.checkpoint = "main2000/final"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            workflow.run_workflow(args, ["--split", "train"])
        self.assertIn(str(self.exp / "training/main2000/final"), out.getvalue())
        self.assertFalse(list((self.exp / "steps").iterdir()))

    def test_cross_training_inputs_rejected_before_any_write_or_execution(self):
        other = self.experiments / "other"
        cases = [
            ("infer", ["--lora", str(other / "training/main/final")]),
            ("train", ["--resume-from=" + str(other / "training/main/final")]),
            ("eval", ["--predictions", str(self.exp / "predictions/a"), str(other / "predictions/b")]),
        ]
        for action, extra in cases:
            with self.subTest(action=action), patch.object(workflow, "stream_command", side_effect=AssertionError("Unexpected execution")):
                with self.assertRaisesRegex(ValueError, "must belong"):
                    workflow.run_workflow(self.args(action, dry=True), extra)
        self.assertFalse(list((self.exp / "steps").iterdir()))

    def test_checkpoint_cannot_escape_or_conflict_with_other_model_flags(self):
        for checkpoint, extra in [("../other/final", []), ("/outside", []),
                                  ("main/final", ["--zero-shot"]), ("main/final", ["--lora=x"])]:
            args = self.args(dry=True)
            args.checkpoint = checkpoint
            with self.subTest(checkpoint=checkpoint, extra=extra), self.assertRaises(ValueError):
                workflow.run_workflow(args, extra)

    def test_legacy_alias_routes_new_inference_into_training_experiment(self):
        alias = self.experiments / "old_diagnostic"
        write_json(alias / "REDIRECT.json", {"experiment": "example"})
        args = self.args(dry=True)
        args.experiment = "old_diagnostic"
        args.checkpoint = "main2000/final"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            workflow.run_workflow(args, [])
        commands = [json.loads(line)["argv"] for line in out.getvalue().splitlines()]
        self.assertEqual(commands[0][3], "example")
        self.assertIn(str(self.exp / "predictions/train60"), commands[1])
        workflow.refresh_index()
        self.assertNotIn("old_diagnostic", (self.artifacts / "EXPERIMENTS.md").read_text())

    def test_legacy_alias_status_refreshes_owner_and_rejects_redirect_cycles(self):
        alias = self.experiments / "old_diagnostic"
        write_json(alias / "REDIRECT.json", {"experiment": "example"})
        with contextlib.redirect_stdout(io.StringIO()):
            workflow.main(["status", "old_diagnostic"])
        self.assertTrue((self.exp / "reports/OVERVIEW.md").exists())
        self.assertFalse((alias / "reports").exists())
        write_json(self.exp / "REDIRECT.json", {"experiment": "old_diagnostic"})
        with self.assertRaisesRegex(ValueError, "redirect cycle"):
            workflow.resolve_experiment("old_diagnostic")


    def test_multiple_evaluations_keep_the_same_inference_and_training(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            workflow.main(["eval", "example", "train60", "--name", "train60_psnr", "--preview", "--metrics", "psnr", "--dry-run"])
        command = json.loads(out.getvalue())["argv"]
        self.assertEqual(command[3], "example")
        self.assertEqual(command[command.index("--name") + 1], "train60_psnr")
        self.assertEqual(command[command.index("--predictions") + 1], str(self.exp / "predictions/train60"))
        self.assertFalse(list((self.exp / "steps").iterdir()))


class ProcessCleanupTests(unittest.TestCase):
    def test_interruption_terminates_the_entire_accelerate_process_group(self):
        def interrupted_output():
            raise KeyboardInterrupt("fixture")
            yield
        child = MagicMock()
        child.pid = 12345
        child.stdout = interrupted_output()
        with patch.object(workflow.subprocess, "Popen", return_value=child) as start, \
             patch.object(workflow.os, "killpg") as stop:
            with self.assertRaises(KeyboardInterrupt):
                workflow.stream_command(["fixture"], io.StringIO())
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with(12345, workflow.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=30)
