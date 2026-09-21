#!/usr/bin/env python3
"""Stage explicitly reviewed fresh candidates after independent local replay."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from holosoma_retargeting.semantic_keyframes.event_agents import FUNCTION_SCHEMA, assemble_event_program
from holosoma_retargeting.semantic_keyframes.pipeline import validate_semantic_keyframe_json
from holosoma_retargeting.semantic_keyframes.python_event_functions import parse_python_functions
from holosoma_retargeting.semantic_keyframes.trajectory_events import execute_event_program
from pilot_trajectory_event_plans import DATA, file_hash


def stage(row, destination, note):
    if row['status'] != 'requires_visual_and_signal_review' or not note.strip():
        raise ValueError('Only valid nonempty candidates with explicit review notes may be staged')
    output, audit = Path(row['output']), Path(row['audit'])
    manifest = json.loads((audit/'manifest.json').read_text())
    if manifest['generation_policy'] != 'fresh_run_current_run_self_reflection':
        raise ValueError('Expected the user-requested fresh generation with current-run SR')
    for source, digest in manifest['source_sha256'].items():
        if file_hash(Path(source)) != digest:
            raise ValueError(f'Source changed since generation: {source}')
    for source, digest in manifest['input_sha256'].items():
        if file_hash(output/source) != digest:
            raise ValueError(f'Prepared input changed since generation: {source}')
    program = json.loads((output/'event_program.json').read_text())
    candidate = json.loads((output/'semantic_plan.candidate.json').read_text())
    agent_workflows = {
        'visual_event_agent_then_logic_agent_then_trajectory_executor',
        'visual_event_agent_then_logic_agent_then_syntax_repair_agent_then_trajectory_executor',
    }
    if manifest.get('workflow') in agent_workflows:
        catalog = json.loads((output/'event_catalog.json').read_text())
        expected = {event['action']: event for event in catalog['events']}
        actual = {action['action']: {k: action[k] for k in ('action', 'body_parts', 'criticality_level', 'rationale')}
                  for action in program['actions']}
        if expected != actual or len(actual) != len(program['actions']):
            raise ValueError('Program changed the frozen event catalog')
    if manifest.get('workflow') == 'visual_event_agent_then_logic_agent_then_syntax_repair_agent_then_trajectory_executor':
        source_path = output/'event_functions.py'
        expected_hash = candidate['generation_metadata'].get('event_functions_sha256')
        if not source_path.is_file() or not expected_hash or file_hash(source_path) != expected_hash:
            raise ValueError('Saved event function source changed since generation')
        with np.load(output/'trajectory_signals.npz', allow_pickle=False) as data:
            signals = {key: data[key] for key in data if key != 'fps'}
        functions, errors = parse_python_functions(
            source_path.read_text(), catalog, signals, FUNCTION_SCHEMA)
        if errors:
            raise ValueError('Saved event function source no longer parses: ' + json.dumps(errors))
        rebuilt, errors = assemble_event_program(json.dumps(functions), catalog)
        if errors:
            raise ValueError('Saved event function source no longer assembles: ' + json.dumps(errors))
        if program.get('body_part_constraints'):
            rebuilt['body_part_constraints'] = program['body_part_constraints']
        if rebuilt != program:
            raise ValueError('Saved event function source does not reproduce the event program')
    if candidate['generation_metadata'].get('self_reflection_rounds', 0) < 1:
        raise ValueError('Self-reflection did not complete')
    with np.load(output/'trajectory_signals.npz', allow_pickle=False) as data:
        result=execute_event_program(program,{k:data[k] for k in data if k!='fps'},float(data['fps']))
    if not result['events'] or result['unresolved_events']:
        raise ValueError('Empty/unresolved programs cannot enter retargeting')
    for key in ('events','coverage','unresolved_events','recognition_diagnostics'):
        if result[key] != candidate[key]:
            raise ValueError(f'Candidate does not reproduce from its saved trajectory: {key}')
    validate_semantic_keyframe_json(candidate)
    candidate=copy.deepcopy(candidate)
    candidate['generation_metadata'].update(approved_for_retargeting=True, review_only=False,
        active_for_retargeting=True, review_note=note,
        source_candidate_sha256=file_hash(output/'semantic_plan.candidate.json'))
    task_dir=destination/row['dataset']/row['task']
    task_dir.mkdir(parents=True,exist_ok=False)
    (task_dir/'semantic_plan.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (task_dir/'semantic_plan.event_plan.json').write_text(json.dumps(program,indent=2)+'\n')
    return dict(dataset=row['dataset'],task=row['task'],review_note=note,source=str(output),
                candidate_sha256=file_hash(output/'semantic_plan.candidate.json'),coverage=result['coverage'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,required=True,help='JSON mapping dataset/task to human/agent visual review note')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    generation=json.loads(args.generation.read_text())
    reviews=json.loads(args.reviews.read_text())
    rows={r['dataset']+'/'+r['task']:r for r in generation['results']}
    if set(reviews)-set(rows):raise ValueError('Unknown task in reviews')
    args.output.mkdir(parents=True,exist_ok=False)
    rules=json.loads((DATA/'semantic_keyframes/g1_anatomy_v2_20260920/task_body_constraints.json').read_text())
    (args.output/'task_body_constraints.json').write_text(json.dumps(rules,indent=2)+'\n')
    staged=[stage(rows[key],args.output,note) for key,note in reviews.items()]
    (args.output/'REVIEW.json').write_text(json.dumps(staged,indent=2)+'\n')
    print(json.dumps(staged,indent=2))


if __name__=='__main__':main()
