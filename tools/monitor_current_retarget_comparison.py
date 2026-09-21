#!/usr/bin/env python3
"""Refresh experiment reports and losslessly compress completed NPZ outputs."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from run_current_semantic_retarget_comparison import ROOT,write


def compress(file):
    marker=file.with_suffix('.storage.json')
    if marker.exists():return
    before=file.stat().st_size
    with np.load(file,allow_pickle=False) as payload:
        data={k:payload[k] for k in payload.files}
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()}
    temp=file.with_suffix('.compressed.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**data)
    with np.load(temp,allow_pickle=False) as check:
        assert set(check.files)==set(data)
        for k,v in data.items():
            value=check[k]
            assert value.dtype==v.dtype and value.shape==v.shape
            assert hashlib.sha256(value.tobytes()).hexdigest()==hashes[k]
    temp.replace(file)
    write(marker,{'lossless':True,'original_bytes':before,'compressed_bytes':file.stat().st_size,'array_sha256':hashes})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison')
    args=ap.parse_args();output=args.output.resolve()
    while True:
        for metric in output.glob('*/*/*/metrics.json'):
            row=json.loads(metric.read_text())
            if row.get('status')=='ok':
                file=Path(row['result'])
                assert file.resolve().is_relative_to(output)
                compress(file)
        subprocess.run([sys.executable,str(ROOT/'tools/write_current_retarget_comparison_report.py'),'--output',str(output)],check=True)
        state=json.loads((output/'comparison.json').read_text())
        if state['finished_runs']==state['requested_runs']:
            write(output/'MONITOR_COMPLETE.json',{'finished_at':time.time(),'finished_runs':state['finished_runs'],'lossless_compression_verified':True})
            break
        time.sleep(30)


if __name__=='__main__':main()
