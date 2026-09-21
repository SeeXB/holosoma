#!/usr/bin/env python3
"""Separate Python syntax, supported expressions, and trajectory outcomes."""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
from holosoma_retargeting.semantic_keyframes.event_agents import FUNCTION_SCHEMA
from holosoma_retargeting.semantic_keyframes.python_event_functions import parse_python_functions


def report(batch):
    summary = json.loads((batch/'summary.json').read_text())
    rows = []
    for run in summary['results']:
        audit, output = Path(run['audit']), Path(run['output'])
        validations = [(int(p.stem.split('_')[-1]), json.loads(p.read_text()))
                       for p in audit.glob('validation_*.json')]
        validations.sort()
        event_rounds = [v for _, v in validations if v['agent'] == 'event_agent']
        logic_rounds = [(i, v) for i, v in validations if v['agent'] == 'logic_agent']
        syntax_rounds = [(i, v) for i, v in validations if v['agent'] == 'syntax_repair_agent']
        row = dict(task=run['task'], status=run['status'], vlm_calls=run['vlm_calls'],
                   agent_calls=run.get('agent_calls', {}), sr_used=run.get('reflection_round', 0),
                   syntax_repairs_used=run.get('syntax_repair_calls', 0),
                   syntax_repair_accepted=any(v.get('valid') for _, v in syntax_rounds),
                   stop_reason=run.get('stop_reason'), event_catalog_structure_passed=bool(event_rounds and event_rounds[-1]['valid']),
                   python_syntax_passed=None, python_supported_subset_passed=None,
                   coverage=run.get('coverage'), audit=str(audit), output=str(output),
                   review_only=True, visually_approved=False)
        if logic_rounds:
            i, validation = logic_rounds[-1]
            saved_source = output/'event_functions.py'
            raw = (saved_source if saved_source.exists() else audit/f'response_{i}.txt').read_text()
            source = raw.strip()
            match = re.fullmatch(r'```(?:python|py)?\s*\n([\s\S]*?)\n```', source)
            if match:
                source = match.group(1)
            try:
                ast.parse(source)
                row['python_syntax_passed'] = True
            except (SyntaxError, ValueError):
                row['python_syntax_passed'] = False
            catalog = json.loads((output/'event_catalog.json').read_text())
            with np.load(output/'trajectory_signals.npz') as data:
                _, errors = parse_python_functions(raw, catalog, {k: data[k] for k in data if k != 'fps'}, FUNCTION_SCHEMA)
            row.update(python_supported_subset_passed=not errors,
                       final_errors=validation.get('errors', []), final_warnings=validation.get('warnings', []),
                       per_event=validation.get('per_event', []),
                       python_response=str(saved_source if saved_source.exists() else audit/f'logic_response_{i}.py'))
        else:
            row['final_errors'] = event_rounds[-1]['errors'] if event_rounds else []
        rows.append(row)
    result = dict(batch=str(batch), total=summary['total'], finished=summary['finished'], done=summary['done'],
                  vlm_calls=sum(r['vlm_calls'] for r in rows),
                  event_catalog_structure_passed=sum(r['event_catalog_structure_passed'] for r in rows),
                  python_syntax_passed=sum(r['python_syntax_passed'] is True for r in rows),
                  python_supported_subset_passed=sum(r['python_supported_subset_passed'] is True for r in rows),
                  complete_candidates=sum(r['status'] == 'requires_visual_and_signal_review' for r in rows),
                  tasks=rows)
    (batch/'PYTHON_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Python Logic Agent pilot', '', f"Finished {result['finished']}/{result['total']}; {result['vlm_calls']} API calls.", '',
             '|Task|Event catalog structure|Python syntax|Supported Python subset|Final status|',
             '|---|---|---|---|---|']
    for row in rows:
        lines.append('|'+ '|'.join(str(row[k]) for k in ('task', 'event_catalog_structure_passed', 'python_syntax_passed', 'python_supported_subset_passed', 'status'))+'|')
    lines += ['', 'Syntax acceptance is not function correctness or visual approval. No generated interval is accepted merely to improve success counts.', '']
    for row in rows:
        lines += [f"## {row['task']}", '',
                  f"Agent calls: {row['agent_calls']}; shared SR used: {row['sr_used']}; "
                  f"syntax repairs used: {row['syntax_repairs_used']}; stop: {row['stop_reason']}.", '']
        for error in row.get('final_errors', []):
            location = f" (Python line {error['python_line']}, column {error.get('python_column')})" if error.get('python_line') else ''
            lines.append(f"- `{error['path']}`{location}: {error['reason']}")
        lines += ['', f"Audit: `{row['audit']}`", '']
    (batch/'PYTHON_REPORT.md').write_text('\n'.join(lines)+'\n')
    return {k: v for k, v in result.items() if k != 'tasks'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    print(json.dumps(report(parser.parse_args().batch), indent=2))
