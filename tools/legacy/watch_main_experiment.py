"""Write live status and audit the finished run; never starts or submits GPU jobs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--timeout-hours',type=float,default=12);a=p.parse_args()
root=Path(__file__).resolve().parents[2];experiment=Path(a.experiment).resolve();end=time.monotonic()+a.timeout_hours*3600
status_path=experiment/'reports/LIVE_STATUS.json'
while time.monotonic()<end:
    steps={}
    try:
        for path in sorted((experiment/'steps').glob('*.json')):
            row=json.loads(path.read_text());steps[path.stem]={k:row.get(k) for k in ('status','exit_code','started_utc','finished_utc')}
    except (OSError,json.JSONDecodeError):
        time.sleep(2);continue
    row={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':steps}
    for name in ('main1000','main2000'):
        path=experiment/'training'/name/'loss.jsonl'
        if path.exists():
            try:row[name+'_last_update']=json.loads(path.read_text().splitlines()[-1])
            except (json.JSONDecodeError,IndexError):pass
    status_path.write_text(json.dumps(row,indent=2)+'\n')
    if any(r.get('status')=='failed' for r in steps.values()):
        raise SystemExit('Experiment step failed; inspect LIVE_STATUS.json and logs; no automatic retry/reconfiguration')
    if (experiment/'reports/DRIVER_COMPLETE.txt').exists():
        with (experiment/'logs/final_audit.log').open('x') as log:
            subprocess.run([sys.executable,str(root/'tools/legacy/summarize_main_experiment.py'),'--experiment',str(experiment)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        (experiment/'reports/AUDIT_COMPLETE.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
        print('Completed final audit, metrics summary and matched-view grids',flush=True)
        break
    time.sleep(30)
else:
    raise SystemExit('Watcher timeout; training was not terminated. Inspect experiment status before restarting the watcher.')
