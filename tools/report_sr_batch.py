#!/usr/bin/env python3
"""Summarize completed SR attempts, without interpreting empty output as success."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.batch/'summary.json').read_text())
    rows=[]
    for row in summary['results']:
        rounds=[]
        if row.get('audit'):
            for path in sorted(Path(row['audit']).glob('validation_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
                validation=json.loads(path.read_text())
                rounds.append(dict(round=int(path.stem.split('_')[-1]),
                                   errors=len(validation.get('errors',[])),warnings=len(validation.get('warnings',[])),
                                   unresolved=len(validation.get('unresolved_events',[])),
                                   error_paths=[item['path'] for item in validation.get('errors',[])]))
        rows.append(dict(task=row['task'],status=row['status'],vlm_calls=row.get('vlm_calls',0),
                         reflection_rounds=row.get('reflection_round',0),rounds=rounds,
                         coverage=row.get('coverage'),error=row.get('error')))
    report=dict(finished=summary['finished'],total=summary['total'],done=summary['done'],
                counts=dict(Counter(row['status'] for row in rows)),vlm_calls=sum(row['vlm_calls'] for row in rows),tasks=rows)
    (args.batch/'SR_ROUNDS.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# OMOMO: up to six SR passes','',f"Finished {summary['finished']}/{summary['total']} tasks; {report['vlm_calls']} requests among finished tasks.",'',
           'Initial generation is round 0; reflection rounds are 1–6. Each new task run starts from original inputs. '
           'Error counts alone do not measure improvement: actions may change or disappear. Empty/unresolved candidates are not successful plans.', '',
           '|Task|Outcome|SR passes|Initial errors|Final errors|Final warnings|','|---|---|---:|---:|---:|---:|']
    for row in rows:
        first,last=(row['rounds'][0],row['rounds'][-1]) if row['rounds'] else ({},{})
        lines.append(f"|{row['task']}|{row['status']}|{row['reflection_rounds']}|{first.get('errors','—')}|{last.get('errors','—')}|{last.get('warnings','—')}|")
    (args.batch/'SR_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:report[key] for key in ('finished','total','done','counts','vlm_calls')}))


if __name__=='__main__':main()
