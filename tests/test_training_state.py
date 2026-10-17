import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from qwen_mvwd.training_state import capture_rng, restore_rng, epoch_indices, check_progress, checkpoint_receipt


class TrainingStateTests(unittest.TestCase):
    def test_sharding_padding_and_epoch_boundaries(self):
        order = epoch_indices(7, 42, 0, 2, 3)
        self.assertEqual(len(order), 12)
        self.assertEqual(set(order[:7]), set(range(7)))
        self.assertEqual(order, epoch_indices(7, 42, 0, 2, 3))
        self.assertNotEqual(order, epoch_indices(7, 42, 1, 2, 3))
        check_progress({'step': 2, 'epoch': 1, 'cursor': 0, 'global_samples_seen': 12}, 7, 2, 3)
        with self.assertRaises(ValueError):
            check_progress({'step': 2, 'epoch': 0, 'cursor': 6, 'global_samples_seen': 12}, 7, 2, 3)

    def test_exact_layout_batches_cover_samples_and_resume_boundaries(self):
        layouts = ["a"] * 5 + ["b"] * 3 + ["c"]
        order = epoch_indices(9, 42, 0, 2, 2, 2, layouts)
        self.assertEqual(set(order), set(range(9)))
        self.assertEqual(len(order) % 8, 0)
        for offset in range(0, len(order), 2):
            self.assertEqual(len({layouts[i] for i in order[offset:offset+2]}), 1)
        self.assertEqual(order, epoch_indices(9, 42, 0, 2, 2, 2, layouts))
        self.assertNotEqual(order, epoch_indices(9, 42, 1, 2, 2, 2, layouts))
        updates = len(order) // 8
        check_progress({"step": updates, "epoch": 1, "cursor": 0, "global_samples_seen": len(order)},
                       9, 2, 2, 2, len(order))

    def test_optimizer_rng_and_next_sample_match_uninterrupted(self):
        random.seed(5); np.random.seed(5); torch.manual_seed(5)
        generator = torch.Generator().manual_seed(5)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        def advance(start, end):
            trace = []
            for step in range(start, end):
                order = epoch_indices(7, 42, step // 2, 1, 4)
                for offset in range(4):
                    index = order[(step % 2) * 4 + offset]
                    x = torch.randn(1, 2, generator=generator) + random.random() + np.random.random() + torch.rand(())
                    loss = (model(x) - index).square().mean() / 4
                    loss.backward()
                    trace.append((index, loss.item()))
                optimizer.step(); optimizer.zero_grad()
            return trace

        advance(0, 1)  # Middle-of-epoch optimizer boundary.
        state = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                 'rng': capture_rng('cpu', generator)}
        expected_trace = advance(1, 5)  # Includes several epoch transitions.
        expected = copy.deepcopy(model.state_dict())
        # Exercise safe, on-disk serialization, not only Python in-memory copies.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'; torch.save(state, path)
            loaded = torch.load(path, weights_only=True)
        model.load_state_dict(loaded['model']); optimizer.load_state_dict(loaded['optimizer'])
        restore_rng(loaded['rng'], 'cpu', generator)
        self.assertEqual(expected_trace, advance(1, 5))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))

    def test_old_lora_checkpoint_not_resumable(self):
        from qwen_mvwd.common import write_json
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / 'metadata.json', {'step': 300})
            with self.assertRaisesRegex(ValueError, 'old LoRA-only'):
                checkpoint_receipt(directory, {})
