"""Exercise the actual backend loop/AdamW/checkpoints on CPU with a tiny backbone."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from peft import utils as peft_utils
from safetensors.torch import load_file, save_file

from qwen_mvwd.backend import train
from qwen_mvwd.common import digest, file_hash, implementation_hash, write_json


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()
    def add_adapter(self, *a, **kw):
        self.scale.requires_grad_(True)
    def enable_gradient_checkpointing(self):
        pass
    def forward(self, hidden_states, **kw):
        return (hidden_states * self.scale,)


class Pipeline:
    _pack_latents = staticmethod(lambda z, *a: z.flatten(2).transpose(1, 2))
    @staticmethod
    def save_lora_weights(dest, transformer_lora_layers, **kw):
        save_file({'transformer.' + k: v.detach().cpu().contiguous() for k, v in transformer_lora_layers.items()},
                  str(Path(dest) / 'pytorch_lora_weights.safetensors'))


class Scheduler:
    config = types.SimpleNamespace(use_dynamic_shifting=True)
    timesteps = torch.tensor([250., 500., 750.])
    sigmas = torch.tensor([.25, .5, .75])
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class TrainingLoopTests(unittest.TestCase):
    def test_actual_loop_resume_and_validation_preserve_training(self):
        fake = types.ModuleType('diffusers')
        fake.QwenImage21Pipeline = Pipeline
        fake.QwenImage21Transformer2DModel = Tiny
        fake.FlowMatchEulerDiscreteScheduler = Scheduler
        config = {'training': {'rank': 1, 'alpha': 1, 'seed': 42, 'max_steps': 4,
                              'gradient_accumulation_steps': 2, 'learning_rate': 0.01, 'checkpoint_every': 2}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, count in (('train', 3), ('val', 2)):
                cache = root / split; cache.mkdir()
                rows = []
                for i in range(count):
                    sample = {'target': torch.full((1, 2, 1, 4, 4), float(i) / 10),
                        'conditions': [torch.ones(1, 2, 1, 4, 4)],
                        'image_mask': torch.ones((1, 4), dtype=torch.bool),
                        'prompt_embeds': torch.zeros(1, 4, 2), 'prompt_mask': None, 'query': {'index': i}}
                    path = cache / f'{i}.pt'; torch.save(sample, path)
                    rows.append({'path': path.name, 'sha256': file_hash(path), 'query': sample['query']})
                write_json(cache / 'manifest.json', {'config_sha256': digest(config), 'cache_identity': split,
                    'dataset': {'split': split}, 'complete': True, 'implementation_sha256': implementation_hash(), 'samples': rows})
            def get_state(model):
                return model.state_dict()
            def set_state(model, state):
                return model.load_state_dict(state)
            with patch.dict('sys.modules', {'diffusers': fake}), patch('qwen_mvwd.backend.require_gpu'), \
                 patch('qwen_mvwd.backend.model_source', return_value='fake'), \
                 patch.object(peft_utils, 'get_peft_model_state_dict', side_effect=get_state), \
                 patch.object(peft_utils, 'set_peft_model_state_dict', side_effect=set_state):
                args = {'validation_cache': root / 'val', 'validate_every': 1}
                train(config, root / 'train', root / 'full', **args)
                train(config, root / 'train', root / 'part', max_steps=1, **args)
                train(config, root / 'train', root / 'resumed', resume_from=root / 'part/final', **args)
                train(config, root / 'train', root / 'no_val')
                for name in ('resumed', 'no_val'):
                    expected = load_file(str(root / 'full/final/pytorch_lora_weights.safetensors'))
                    actual = load_file(str(root / name / 'final/pytorch_lora_weights.safetensors'))
                    for key in expected:
                        self.assertTrue(torch.equal(expected[key], actual[key]), name)
                def rows(name):
                    return [json.loads(s) for s in (root / name / 'loss.jsonl').read_text().splitlines()]
                self.assertEqual(rows('full'), rows('part') + rows('resumed'))
                full = json.loads((root / 'full/final/metadata.json').read_text())
                resumed = json.loads((root / 'resumed/final/metadata.json').read_text())
                self.assertEqual(full['progress'], resumed['progress'])
                self.assertEqual(full['progress'], {'step': 4, 'epoch': 2, 'cursor': 0, 'global_samples_seen': 8})
                # Changed validation protocol cannot silently continue the same run.
                with self.assertRaisesRegex(ValueError, 'identical config'):
                    train(config, root / 'train', root / 'bad', resume_from=root / 'part/final',
                          validation_cache=root / 'val', validate_every=2)
