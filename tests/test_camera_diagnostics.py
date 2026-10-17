"""Protect camera-only interventions, paired noise, and post-generation GT use."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_data import fixture
from qwen_mvwd.camera_diagnostics import camera_cases, plan, review
from qwen_mvwd.common import read_json
from qwen_mvwd.data import Level1Data
import tempfile


class CameraDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root / "dataset")
        self.args = SimpleNamespace(split="train", scene=None, max_episodes=1, max_groups=1,
            seed=42, dry_run=True, lora=None, zero_shot=False, cpu_offload=False,
            output=str(self.root / "diagnostic"))

    def tearDown(self):
        self.temp.cleanup()

    def test_interventions_preserve_scene_and_robot_states(self):
        data = Level1Data(self.config, "train")
        before = data.conditions(next(data.queries()))
        cases = camera_cases(data, "ep", 0)
        self.assertEqual(len(cases), 8)
        for case in cases:
            condition = case["condition"]
            self.assertEqual(condition["payload"]["robot_states"], before["payload"]["robot_states"])
            self.assertEqual(condition["payload"]["asset"], before["payload"]["asset"])
            for actual, expected in zip(condition["images"], before["images"]):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        self.assertEqual(before["prompt"], data.conditions(next(data.queries()))["prompt"])

    def test_pose_focal_and_gt_semantics(self):
        data = Level1Data(self.config, "train")
        cases = {c["name"]: c for c in camera_cases(data, "ep", 0)}
        camera = lambda name: cases[name]["condition"]["cameras"][0]
        T = lambda name: np.array(camera(name)["camera_to_world"])
        original = T("recorded_0")
        np.testing.assert_allclose(T("yaw_plus30")[:3, 3], original[:3, 3])
        np.testing.assert_allclose(T("yaw_plus30")[:3, :3] @ [1, 0, 0], [np.sqrt(3)/2, .5, 0], atol=1e-12)
        np.testing.assert_allclose(T("right_0p5m")[:3, 3] - original[:3, 3], original[:3, 0] * .5)
        K, scaled = (np.array(camera(n)["K"]) for n in ("recorded_0", "focal_1p25x"))
        self.assertEqual(scaled[0, 0], K[0, 0] * 1.25)
        self.assertEqual(scaled[1, 1], K[1, 1] * 1.25)
        np.testing.assert_array_equal(scaled[:2, 2], K[:2, 2])
        for name in ("yaw_plus30", "yaw_minus30", "right_0p5m", "focal_1p25x"):
            self.assertIsNone(cases[name]["gt_robot"])

    def test_cpu_plan_never_reads_target_and_shares_actual_noise_seed(self):
        with patch.object(Level1Data, "target", side_effect=AssertionError("GT read during planning")):
            plan(self.config, self.args)
        meta = read_json(Path(self.args.output) / "metadata.json")
        self.assertTrue(meta["complete"])
        self.assertFalse(meta["ground_truth_loaded"])
        self.assertFalse(meta["benchmark_ready"])
        cases = meta["groups"][0]["cases"]
        self.assertEqual(len({c["seed"] for c in cases}), 1)
        self.assertEqual(len({tuple(c["condition_sha256"]) for c in cases}), 1)
        self.assertEqual(cases[0]["prompt_sha256"], cases[3]["prompt_sha256"])
        with self.assertRaisesRegex(ValueError, "completed GT-free"):
            review(self.config, SimpleNamespace(diagnostics=self.args.output, output=str(self.root / "review")))

    def test_joint_and_test_split_rejected(self):
        config = copy.deepcopy(self.config)
        config["mode"] = "joint"
        with self.assertRaisesRegex(ValueError, "independent"):
            plan(config, self.args)
        self.args.split = "test"
        with self.assertRaisesRegex(ValueError, "train/val"):
            plan(self.config, self.args)

    def test_generation_and_review_keep_gt_out_of_virtual_pose_scores(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional for data-only checks")
        self.args.dry_run = False
        self.args.zero_shot = True
        seeds = []
        def fake_pipeline(**kwargs):
            import json
            payload = json.loads(kwargs["prompt"].partition("Conditions:\n")[2])
            robot = payload["cameras"][0]["robot_id"]
            color = tuple(255 if i == robot else 0 for i in range(3))
            seeds.append(kwargs["generator"].initial_seed())
            return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), color)])
        with patch("qwen_mvwd.backend.require_gpu"), patch("qwen_mvwd.backend.load_pipeline", return_value=fake_pipeline), \
             patch.object(Level1Data, "target", side_effect=AssertionError("GT read during generation")):
            plan(self.config, self.args)
        self.assertEqual(len(seeds), 8)
        self.assertEqual(len(set(seeds)), 1)
        out = self.root / "review"
        review(self.config, SimpleNamespace(diagnostics=self.args.output, output=str(out)))
        rows = read_json(out / "camera_response.json")["groups"][0]["cases"]
        self.assertTrue(rows[3]["identical_repeat_pixels_equal"])
        for row in rows:
            if row["gt_robot"] is None:
                self.assertNotIn("psnr_db", row)
                self.assertNotIn("mse_to_each_recorded_gt", row)
            else:
                self.assertTrue(row["own_gt_is_closest"])


if __name__ == "__main__":
    unittest.main()
