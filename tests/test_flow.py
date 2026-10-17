"""CPU mathematical contract checks; these do not validate the real Qwen model."""
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is optional for data-only checks")
class FlowTests(unittest.TestCase):
    def sample(self):
        return {"target": torch.zeros(1, 2, 1, 4, 8),
                "conditions": [torch.ones(1, 2, 1, 4, 4) * 9, torch.ones(1, 2, 1, 4, 6) * 7],
                "image_mask": torch.tensor([[True] * 10 + [False] * 2]),
                "prompt_embeds": torch.zeros(1, 12, 4), "prompt_mask": None}

    def transformer(self, scale):
        class Oracle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
            def forward(self, hidden_states, timestep, img_shapes, img_mask, **kwargs):
                self.shapes = img_shapes
                self.mask = img_mask
                # For a zero clean target, the exact velocity is z_sigma / sigma.
                return (hidden_states / timestep[:, None, None] * self.scale,)
        return Oracle()

    def loss(self, model, sample):
        from qwen_mvwd.backend import flow_loss
        fake = types.ModuleType("diffusers")
        fake.QwenImage21Pipeline = types.SimpleNamespace(_pack_latents=lambda z, *args: z.flatten(2).transpose(1, 2))
        scheduler = types.SimpleNamespace(timesteps=torch.tensor([250.0, 500.0, 750.0]), sigmas=torch.tensor([0.25, 0.5, 0.75]))
        with patch.dict(sys.modules, {"diffusers": fake}):
            return flow_loss(model, sample, scheduler, "cpu", generator=torch.Generator().manual_seed(3))

    def test_exact_velocity_and_target_only_loss(self):
        model = self.transformer(1)
        loss = self.loss(model, self.sample())
        self.assertLess(float(loss), 1e-12)
        self.assertEqual(model.shapes, [[(1, 4, 4), (1, 4, 6), (1, 4, 8)]])
        self.assertEqual(int(model.mask.sum()), 18)

    def test_true_batch_exact_layout_and_per_sample_timesteps(self):
        from qwen_mvwd.backend import collate_samples
        samples = [self.sample(), self.sample()]
        batch = collate_samples(samples)
        self.assertEqual(batch["target"].shape[0], 2)
        model = self.transformer(1)
        self.assertLess(float(self.loss(model, batch)), 1e-12)
        self.assertEqual(len(model.shapes), 2)
        self.assertEqual(model.mask.shape[0], 2)
        # Equal-length prompts with different image positions are also incompatible.
        samples[1]["image_mask"] = samples[1]["image_mask"].roll(1, dims=1)
        with self.assertRaisesRegex(ValueError, "identical"):
            collate_samples(samples)

    def test_adapter_receives_gradient(self):
        model = self.transformer(0)
        loss = self.loss(model, self.sample())
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertLess(float(model.scale.grad), 0)

    def test_inconsistent_vision_tokens_rejected(self):
        sample = self.sample()
        sample["image_mask"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "mask"):
            self.loss(self.transformer(1), sample)


if __name__ == "__main__":
    unittest.main()
