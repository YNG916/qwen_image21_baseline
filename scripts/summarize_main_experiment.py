"""Audit completed training/inference and export all matched validation triplets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from qwen_mvwd.common import read_json, file_hash, write_json, within
from qwen_mvwd.data import Level1Data
from qwen_mvwd.training_state import checkpoint_receipt

p=argparse.ArgumentParser();p.add_argument('--experiment', required=True);a=p.parse_args()
e=Path(a.experiment).resolve(); reports=e/'reports';reports.mkdir(exist_ok=True)
steps={path.stem:read_json(path) for path in sorted((e/'steps').glob('*.json'))}
if any(row.get('status')!='completed' or row.get('exit_code')!=0 for row in steps.values()):
    raise ValueError('Experiment contains unfinished/failed steps; audit explicitly before reporting completion')
health={}
for name in ('ddp_smoke2','main1000','main2000'):
    folder=e/'training'/name/'final';meta=read_json(folder/'metadata.json')
    checkpoint_receipt(folder,meta['resume_contract'])
    state=load_file(str(folder/'pytorch_lora_weights.safetensors'))
    row={'progress':meta['progress'],'tensors':len(state),'parameters':sum(t.numel() for t in state.values()),
         'finite':all(torch.isfinite(t).all().item() for t in state.values()),
         'nonzero_tensors':sum(bool(torch.count_nonzero(t)) for t in state.values()),'state_file_integrity_verified':True,
         'lora_sha256':file_hash(folder/'pytorch_lora_weights.safetensors')}
    if not row['finite'] or row['nonzero_tensors']!=row['tensors']:raise ValueError('Invalid adapter')
    health[name]=row;write_json(reports/(name+'_health.json'),row)
train=read_json(e/'features/encode_train4800/manifest.json');val=read_json(e/'features/encode_val180/manifest.json')
if set(train['dataset']['episode_ids']) & set(val['dataset']['episode_ids']):raise ValueError('Train/val overlap')
mid=read_json(e/'training/main1000/final/metadata.json')
last=read_json(e/'training/main2000/training_metadata.json')
if last['starting_progress']!=mid['progress'] or Path(last['resumed_from']).resolve()!=e/'training/main1000/final':
    raise ValueError('Continuation did not resume the midpoint')
variants=('val_zero180','val1000_180','val2000_180');preds={};metrics={}
for name in variants:
    folder=e/'predictions'/name;meta=read_json(folder/'metadata.json')
    if not meta['complete'] or meta['ground_truth_loaded'] or meta['view_count']!=180:raise ValueError('Invalid predictions')
    preds[name]={}
    for line in (folder/'predictions.jsonl').read_text().splitlines():
        r=json.loads(line);preds[name][(r['episode_id'],r['frame_index'],r['robot_id'])]=r
    result=read_json(e/'evaluations'/(name+'_metrics')/'metrics.json')
    metrics[name]={key:{'mean':v['mean'],'ci95':v['ci95'],'episode_count':v['episode_count'],'status':v['status']}
                   for key,v in result['metrics'].items()}
keys=set(preds[variants[0]])
if any(set(preds[n])!=keys for n in variants):raise ValueError('Different queries')
for k in keys:
    if len({preds[n][k]['seed'] for n in variants})!=1:raise ValueError('Different inference noise seeds')
data=Level1Data(read_json(e/'config.json'),'val')
comps=reports/'comparisons_zero1000_2000';comps.mkdir(exist_ok=False)
h,w=data.config['target_hw'];groups=sorted({(ep,frame) for ep,frame,r in keys})
for number,(episode,frame) in enumerate(groups):
    canvas=Image.new('RGB',(3*w,4*(h+24)),'white');draw=ImageDraw.Draw(canvas)
    for robot in range(3):
        images=[data.target({'episode_id':episode,'frame_index':frame,'robots':[robot]})]
        for name in variants:
            with Image.open(within(e/'predictions'/name,preds[name][(episode,frame,robot)]['path'])) as im:images.append(im.convert('RGB'))
        for row,(label,im) in enumerate(zip(('GT','zero-shot','1000 updates','2000 updates'),images)):
            x,y=robot*w,row*(h+24);draw.text((x+4,y+4),f'{label} / robot {robot}',fill='black');canvas.paste(im,(x,y+24))
    scene=data.by_id[episode]['scene'];canvas.save(comps/f'{number:03d}_{scene}_{episode[3:11]}_frame{frame:03d}.png')
    print('Comparison',number+1,'/',len(groups),flush=True)
loss={};flow={}
for name in ('main1000','main2000'):
    rows=[json.loads(s) for s in (e/'training'/name/'loss.jsonl').read_text().splitlines()]
    loss[name]={'updates':len(rows),'first':rows[0],'last':rows[-1]}
    flow[name]=[json.loads(s) for s in (e/'training'/name/'validation.jsonl').read_text().splitlines()]
summary={'steps':{k:v['exit_code'] for k,v in steps.items()},'health':health,'metrics':metrics,'loss':loss,'validation_flow':flow,
         'matched_validation':{'queries':len(keys),'episodes':len(val['dataset']['episode_ids']),'seed':42,'all_query_seed_pairs_equal':True},
         'training_samples':len(train['samples']),'training_episodes':len(train['dataset']['episode_ids']),
         'effective_batch':last['effective_batch'],'checkpoint_continuation_verified':True,'comparison_grids':len(groups),
         'test_used':False,'formal_benchmark':False,'geometry_pass':'Requires manual review; metrics do not establish camera adherence'}
write_json(reports/'final_summary.json',summary)
lines=['# Four-H100 Level 1 main development run','',
       '4800 training queries, 180 fixed validation queries, effective batch 16. 1000→2000 updates resumes complete optimizer/RNG/sampler state.',
       '','| Model | PSNR ↑ | LPIPS ↓ | MV-GTWarp-DINO ↓ |','|---|---:|---:|---:|']
for name in variants:
    lines.append('| '+name+' | '+' | '.join(f"{metrics[name][k]['mean']:.6f}" for k in ('psnr','lpips','mv_gtwarp_dino'))+' |')
lines+=['','All three variants share queries and actual inference seeds. The preview uses 30 episodes from three held-out validation scenes and one seed; test is untouched.',
        '','Inspect every grid in `comparisons_zero1000_2000/` and `camera_val2000_review/`. Repeated or incorrect layouts and missing robots remain failures even if a metric improves.',
        '','Automated integrity/metric summary: `final_summary.json`. Add the visual assessment after reviewing generated images.']
(reports/'REVIEW.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(metrics,indent=2))
