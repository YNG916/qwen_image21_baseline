"""Measure real LoRA optimizer updates on repeated cached inputs, without modifying the cache."""
import argparse
import gc
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from diffusers import QwenImage21Transformer2DModel, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
from qwen_mvwd.backend import model_source, flow_loss, collate_samples, require_gpu
from qwen_mvwd.common import load_config, file_hash, implementation_hash, write_json

p=argparse.ArgumentParser()
p.add_argument('--config', required=True); p.add_argument('--sample', required=True)
p.add_argument('--output', required=True); p.add_argument('--checkpointing', type=int, choices=[0,1], required=True)
p.add_argument('--batches', nargs='+', type=int, default=[1,2,4,8,16]); p.add_argument('--updates', type=int, default=4)
a=p.parse_args(); require_gpu(); torch.set_num_threads(4)
c=load_config(a.config); source=model_source(c)
m=QwenImage21Transformer2DModel.from_pretrained(source,subfolder='transformer',torch_dtype=torch.bfloat16).to('cuda')
m.requires_grad_(False); m.add_adapter(LoraConfig(r=c['training']['rank'],lora_alpha=c['training']['alpha'],target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights='gaussian'))
if a.checkpointing: m.enable_gradient_checkpointing()
params=[x for x in m.parameters() if x.requires_grad]
for x in params: x.data=x.data.float()
opt=torch.optim.AdamW(params,lr=c['training']['learning_rate'],weight_decay=.01)
scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(source,subfolder='scheduler')
sample=torch.load(a.sample,map_location='cpu',weights_only=True)
g=torch.Generator(device='cuda').manual_seed(42);m.train();rows=[]
report={'kind':'compute-profile-repeated-input-not-quality-experiment','sample_sha256':file_hash(a.sample),'implementation_sha256':implementation_hash(),'gradient_checkpointing':bool(a.checkpointing),'model_parameters':sum(x.numel() for x in m.parameters()),'trainable_parameters':sum(x.numel() for x in params),'prompt_length':sample['prompt_embeds'].shape[1],'rows':rows}
for b in a.batches:
    batch=collate_samples([sample]*b); durations=[]; opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(a.updates+1):
            torch.cuda.synchronize(); start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=flow_loss(m,batch,scheduler,'cuda',generator=g)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite profile loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); elapsed=time.monotonic()-start
            if step: durations.append(elapsed)
        row={'batch':b,'status':'ok','seconds_per_update':sum(durations)/len(durations),'samples_per_second':b*len(durations)/sum(durations),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loss':loss.item()}
    except torch.OutOfMemoryError:
        row={'batch':b,'status':'oom'}
    rows.append(row);write_json(a.output,report); print(json.dumps(row),flush=True)
    if row['status']=='oom': break
