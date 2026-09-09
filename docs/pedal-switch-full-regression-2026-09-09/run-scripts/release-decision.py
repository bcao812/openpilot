import ast
import json
from pathlib import Path

from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
repo = Path('/home/bcao/SourceCode/openpilot-3xl-0817')
summary = json.loads((run / 'evaluation-final/summary.json').read_text())
tests = json.loads((run / 'artifacts/test-accounting.json').read_text())
matrix = json.loads((run / 'artifacts/scenario-matrix-summary.json').read_text())
account = json.loads((run / 'artifacts/final-accounting.json').read_text())
planner = ast.parse((repo / 'openpilot/selfdrive/controls/lib/longitudinal_planner.py').read_text())
initializers = [node for node in ast.walk(planner) if isinstance(node, ast.FunctionDef) and node.name == '__init__']
defaults = {argument.arg: value for node in initializers for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)}
assert isinstance(defaults['experimental_coast_with_lead'], ast.Constant) and defaults['experimental_coast_with_lead'].value is False
calls = [node for node in ast.walk(ast.parse((repo / 'openpilot/selfdrive/controls/plannerd.py').read_text()))
         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'LongitudinalPlanner']
assert len(calls) == 1 and not any(keyword.arg == 'experimental_coast_with_lead' for keyword in calls[0].keywords)
assert account['completed_route_arms'] == 220 and summary['routes'] == 55
assert tests['current_unique_totals']['passed'] == 1117
assert tests['expanded-release-gates']['testcase_counts']['failed'] == 4
assert len(summary['comparisons']['normal/experimental_vs_default']['new_raw_events']) == 1
assert matrix['groups']['steady']['introduced_from_zero'] == 21
result = {
  'decision': 'keep_experimental_coast_with_lead_disabled',
  'production_default_verified_false': True,
  'plannerd_does_not_enable_experiment': True,
  'all_local_readable_routes_evaluated': True,
  'road_release_approved': False,
  'deployment_performed': False,
  'blocking_evidence': {
    'new_normal_raw_bursts': summary['comparisons']['normal/experimental_vs_default']['new_raw_events'],
    'failed_closed_loop_release_assertions': 4,
    'new_steady_closed_loop_oscillating_cases': 21,
    'recorded_source_reconstruction_is_not_frame_exact': True,
  },
  'default_hardening': {
    'new_normal_debounced_events': len(summary['comparisons']['normal/default_vs_baseline']['new_events']),
    'new_normal_raw_events': len(summary['comparisons']['normal/default_vs_baseline']['new_raw_events']),
    'extra_normal_brake_seconds': summary['comparisons']['normal/default_vs_baseline']['extra_brake_seconds'],
    'decision': 'retain_existing_default_hardening_without_claiming_road_validation',
  },
  'evidence_sha256': {relative: sha256(run / relative) for relative in (
    'evaluation-final/summary.json', 'reconstruction/summary.json', 'artifacts/final-accounting.json',
    'artifacts/test-accounting.json', 'artifacts/scenario-matrix-summary.json', 'artifacts/expanded-release-gates.xml')},
}
save_json(run / 'artifacts/release-decision.json', result)
print(result['decision'])
