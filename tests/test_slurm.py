"""CPU-only batch submission and launcher checks; scheduler/GPU calls are mocked."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from qwen_mvwd.common import read_json, write_json
from tools import slurm, workflow

SCRIPT = Path(__file__).resolve().parents[1] / "slurm/full_train_4h100.sbatch"


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exp = self.root / "artifacts/mvwd/experiments/example"
        for name in ("logs", "steps", "source", "plans"):
            (self.exp / name).mkdir(parents=True)
        (self.root / "slurm").mkdir()
        (self.root / "slurm/full_train_4h100.sbatch").write_bytes(SCRIPT.read_bytes())
        self.patches = [patch.object(workflow, "ROOT", self.root),
                        patch.object(workflow, "resolve_experiment", return_value=self.exp),
                        patch.object(workflow, "refresh"),
                        patch("tools.full_training.load_plan", return_value=({}, {})),
                        patch("tools.full_training.full_cache", return_value=(self.exp, {}))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def args(self, **changes):
        values = dict(experiment="example", stage="train", through_epoch=10, after=None, chain=1, dry_run=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_dry_run_builds_afterany_chain_without_writes_or_submission(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        out = io.StringIO()
        with patch.object(slurm.subprocess, "run", side_effect=AssertionError("Submitted a job")), contextlib.redirect_stdout(out):
            slurm.submit(self.args(after="91109", chain=2, dry_run=True))
        rows = [json.loads(s) for s in out.getvalue().splitlines()]
        self.assertIn("--dependency=afterany:91109", rows[0]["argv"])
        self.assertIn("--dependency=afterany:JOB_ID_FROM_SUBMISSION_1", rows[1]["argv"])
        self.assertIn("--output=" + str(self.exp / "logs/slurm-%j.log"), rows[0]["argv"])
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_submission_freezes_script_and_removes_old_allocation_environment(self):
        results = [subprocess.CompletedProcess([], 0, "101\n", ""), subprocess.CompletedProcess([], 0, "102;cluster\n", "")]
        with patch.object(slurm.shutil, "which", return_value="/mock/sbatch"), \
             patch.object(slurm.subprocess, "run", side_effect=results) as submit, \
             patch.dict(os.environ, {"SLURM_JOB_ID":"old", "CUDA_VISIBLE_DEVICES":"2"}), \
             contextlib.redirect_stdout(io.StringIO()):
            slurm.submit(self.args(chain=2))
        second = submit.call_args_list[1]
        self.assertIn("--dependency=afterany:101", second.args[0])
        self.assertNotIn("SLURM_JOB_ID", second.kwargs["env"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", second.kwargs["env"])
        self.assertEqual(second.kwargs["env"]["QWEN_REPO_ROOT"], str(self.root))
        receipt = read_json(next((self.exp / "steps").glob("slurm_submission*.json")))
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual([r["job_id"] for r in receipt["jobs"]], ["101", "102"])
        self.assertEqual((self.exp / receipt["script_snapshot"]).read_bytes(), SCRIPT.read_bytes())

    def test_partial_submission_failure_keeps_submitted_job_ids_and_logs(self):
        results=[subprocess.CompletedProcess([],0,"101\n",""), subprocess.CompletedProcess([],1,"","fixture rejection")]
        with patch.object(slurm.shutil, "which", return_value="/mock/sbatch"), \
             patch.object(slurm.subprocess, "run", side_effect=results), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):slurm.submit(self.args(chain=2))
        receipt=read_json(next((self.exp / "steps").glob("slurm_submission*.json")))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["jobs"][0]["job_id"], "101")
        self.assertIn("fixture rejection", next((self.exp / "logs").glob("*.log")).read_text())

    def test_invalid_dependencies_and_budgets_never_submit(self):
        for change in ({"after":"1;2"}, {"chain":0}, {"chain":9}, {"through_epoch":11}):
            with self.subTest(change=change), self.assertRaises(ValueError), \
                 patch.object(slurm.subprocess,"run",side_effect=AssertionError("Submitted")):
                slurm.submit(self.args(**change))

    def launcher(self, mode):
        write_json(self.exp / "plans/full_training.json", {"fixture":True})
        entry=self.root / "mvwd";entry.write_text("#!/bin/bash\nexit 99\n");entry.chmod(0o755)
        bin_dir=self.root / "bin";bin_dir.mkdir()
        stub=bin_dir / "srun"
        stub.write_text(f'''#!{sys.executable}
import json,os,pathlib,signal,sys,time
pathlib.Path(os.environ["STUB_ARGS"]).write_text(json.dumps(sys.argv[1:]))
if os.environ["STUB_MODE"] == "signal":
 def stop(*args):
  pathlib.Path(os.environ["STUB_STOPPED"]).write_text("received")
  sys.exit(143)
 signal.signal(signal.SIGTERM,stop)
 pathlib.Path(os.environ["STUB_READY"]).write_text("ready")
 while True:time.sleep(0.05)
sys.exit(int(os.environ["STUB_MODE"]))
''');stub.chmod(0o755)
        env={**os.environ,"HOME":str(self.root),"QWEN_REPO_ROOT":str(self.root),"QWEN_PYTHON":sys.executable,
             "SLURM_JOB_ID":"123456","SLURM_RESTART_COUNT":"0","PATH":str(bin_dir)+os.pathsep+os.environ["PATH"],
             "STUB_ARGS":str(self.root / "args.json"),"STUB_MODE":mode,
             "STUB_READY":str(self.root / "ready"),"STUB_STOPPED":str(self.root / "stopped")}
        command=["bash",str(self.root / "slurm/full_train_4h100.sbatch"),"example","train","10"]
        return command,env

    def test_launcher_logs_success_and_starts_only_one_slurm_task(self):
        command,env=self.launcher("0")
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        args=read_json(self.root / "args.json")
        self.assertIn("--ntasks=1",args)
        self.assertEqual(args[-7:],["./mvwd","full","example","--stage","train","--through-epoch","10"])
        receipt=read_json(self.exp / "steps/slurm_job_123456_restart_0.json")
        self.assertEqual(receipt["status"],"completed")
        self.assertEqual(receipt["exit_code"],0)
        self.assertEqual(Path(receipt["script_snapshot"]).read_bytes(),SCRIPT.read_bytes())

    def test_launcher_preserves_nonzero_exit_code(self):
        command,env=self.launcher("7")
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,7)
        receipt=read_json(self.exp / "steps/slurm_job_123456_restart_0.json")
        self.assertEqual(receipt["status"],"failed")
        self.assertEqual(receipt["exit_code"],7)

    def test_launcher_forwards_termination_to_step_and_records_interruption(self):
        command,env=self.launcher("signal")
        with tempfile.TemporaryFile(mode="w+") as output:
            child=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+5
                while not (self.root / "ready").exists() and time.monotonic()<deadline:
                    time.sleep(0.01)
                self.assertTrue((self.root / "ready").exists())
                child.send_signal(signal.SIGTERM)
                self.assertEqual(child.wait(timeout=5),143)
            finally:
                if child.poll() is None:child.kill();child.wait()
        self.assertTrue((self.root / "stopped").exists())
        receipt=read_json(self.exp / "steps/slurm_job_123456_restart_0.json")
        self.assertEqual(receipt["status"],"interrupted")
        self.assertEqual(receipt["exit_code"],143)
