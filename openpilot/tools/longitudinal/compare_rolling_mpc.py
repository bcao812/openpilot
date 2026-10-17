#!/usr/bin/env python3
"""Recorded-state, native MPC sensitivities for rolling ACC following.

Run from an explicitly committed host snapshot under the workspace resource
runner. Each arm processes the same complete selected prefix with its own
solver warm start and accepted native reference history. No recorded aTarget
is fed back as a candidate's previous request or acceleration trajectory.

This is an MPC boundary experiment, not a full planner/actuator replay. Active
UNIFIED ACC and ordinary E2E motion with external cruise and no stop arbitration
are reconstructed. The real low-speed geometry policy runs over the prefix.
Unsupported frames remain in the outputs and
break the reconstructed history. The recorded independent ceiling is frozen;
min(native reference, ceiling) is a request projection, never a CAN command.

carState/model/radar joins use the plan's explicit input identities. Auxiliary
messages use the last publication at producerMonoTime, since their consumed
identities were not logged. SP is a same-plan policy diagnostic, not a future
sensor input. RadarConsumption and native lead qualification retain the
current source's original evidence, deadlines and withdrawal semantics.
Observed motion is fixed. Predicted gaps use current physical lead prediction
and constant-jerk ego motion, without stopped-equivalence obstacle distances.
"""

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace


NS = 1_000_000_000
MAX_WINDOW_S = 180.0
MAX_LINE_BYTES = 16 * 1024 ** 2
MAX_INPUT_BYTES = 350 * 1024 ** 2
MAX_TOPIC_ROWS = 60_000
TOPICS = ('carState', 'carControl', 'controlsState', 'selfdriveState', 'longitudinalPlan',
          'longitudinalPlanSP', 'radarState', 'vehicleParameters', 'modelV2')
ASYNC_TOPICS = ('controlsState', 'selfdriveState', 'carControl', 'vehicleParameters')
SCOPES = {'primary': (-13.0, -5.0), 'braking_control': (-29.0, -24.0),
          'low_speed_transition': (-24.0, -13.0), 'subsequent_slowing': (-5.0, 3.0),
          'normal_following_control': (3.0, 10.0)}


@dataclass(frozen=True)
class Arm:
  name: str
  comfort_delta_m: float = 0.0
  history_scale: float = 1.0

  def __post_init__(self):
    if not math.isfinite(self.comfort_delta_m) or not -3.0 <= self.comfort_delta_m <= 0.0:
      raise ValueError('Diagnostic comfort delta must be between -3 and 0 m')
    if not math.isfinite(self.history_scale) or not 0.0 <= self.history_scale <= 1.0:
      raise ValueError('Diagnostic history scale must be between 0 and 1')


ARMS = {'baseline': Arm('baseline'), 'geometry': Arm('geometry', -1.0), 'continuity': Arm('continuity', history_scale=0.5)}


class InputRejected(ValueError):
  def __init__(self, reason, *, status='input_rejected'):
    super().__init__(reason)
    self.reason, self.status = reason, status


def sha256(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_value(value):
  """Keep failed/nonfinite numeric results explicit and JSON interoperable."""
  if isinstance(value, dict):
    return {str(key): json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_value(item) for item in value]
  if hasattr(value, 'tolist'):
    return json_value(value.tolist())
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def json_bytes(value):
  return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def value_hash(value):
  return hashlib.sha256(json_bytes(value)).hexdigest()


def save_json(path, value):
  with Path(path).open('x') as stream:
    json.dump(json_value(value), stream, indent=2, allow_nan=False)
    stream.write('\n')


def finite_number(value):
  return type(value) in (int, float) and math.isfinite(value)


def timestamp(value):
  return type(value) is int and 0 < value < 1 << 64


class Series:
  def __init__(self, rows):
    self.rows = rows
    self.times = [row['mono_ns'] for row in rows]
    if any(not timestamp(stamp) for stamp in self.times) or any(b <= a for a, b in zip(self.times, self.times[1:])):
      raise ValueError('Series timestamps must be unique, increasing UInt64 values')
    self.by_time = dict(zip(self.times, rows, strict=True))

  def asof(self, stamp):
    index = bisect_right(self.times, stamp) - 1
    return self.rows[index] if index >= 0 else None


def compact_row(row, topic):
  # Hash the original row, retaining the current model/parking policy inputs.
  if topic == 'modelV2':
    row = {**row, 'data': {key: row['data'][key] for key in ('frameId', 'action', 'meta', 'velocity', 'orientationRate', 'leadsV3')
                         if key in row['data']}}
  return row


def load_inputs(folder, start_ns, end_ns, mark_ns):
  receipt_path = folder / 'receipt.json'
  receipt = json.loads(receipt_path.read_text())
  if (receipt.get('status') != 'PASS' or receipt.get('mark_ns') != mark_ns
      or start_ns < receipt.get('start_ns', start_ns + 1) or end_ns > receipt.get('end_ns', end_ns - 1)):
    raise ValueError('Input receipt does not cover this mark and complete prefix')
  before = {topic: sha256(folder / (topic + ('.json' if topic == 'carParams' else '.jsonl.gz')))
            for topic in (*TOPICS, 'carParams')}
  for topic, digest in before.items():
    if receipt.get('services', {}).get(topic, {}).get('output_sha256') != digest:
      raise ValueError('Input identity differs from receipt: ' + topic)
  decoded_bytes, tables = 0, {}
  for topic in TOPICS:
    rows = []
    previous = None
    with gzip.open(folder / (topic + '.jsonl.gz'), 'rb') as stream:
      while line := stream.readline(MAX_LINE_BYTES + 1):
        decoded_bytes += len(line)
        if len(line) > MAX_LINE_BYTES or decoded_bytes > MAX_INPUT_BYTES:
          raise ValueError('Selected inputs exceed bounded decoder budget')
        row = json.loads(line)
        stamp = row.get('mono_ns')
        if not timestamp(stamp) or previous is not None and stamp <= previous:
          raise ValueError('Duplicate or reversed input timestamp: ' + topic)
        previous = stamp
        if not isinstance(row.get('data'), dict) or type(row.get('valid')) is not bool:
          raise ValueError('Missing original data/validity: ' + topic)
        if start_ns - NS <= stamp <= end_ns + NS:
          row['_sha256'] = hashlib.sha256(line).hexdigest()
          rows.append(compact_row(row, topic))
          if len(rows) > MAX_TOPIC_ROWS:
            raise ValueError('Selected inputs exceed bounded row budget: ' + topic)
    tables[topic] = Series(rows)
  params = json.loads((folder / 'carParams.json').read_text())
  if params.get('valid') is not True or not timestamp(params.get('mono_ns')) or params['mono_ns'] > start_ns:
    raise ValueError('Missing original CarParams before the selected prefix')
  after = {topic: sha256(folder / (topic + ('.json' if topic == 'carParams' else '.jsonl.gz')))
           for topic in before}
  if before != after:
    raise ValueError('Inputs changed while loading')
  plan_sp = {}
  for row in tables['longitudinalPlanSP'].rows:
    key = row['data'].get('longitudinalPlanMonoTime')
    if not timestamp(key) or key in plan_sp:
      raise ValueError('Missing or duplicate SP plan identity')
    plan_sp[key] = row
  tables['sp_by_plan'] = plan_sp
  provenance = {'receipt_sha256': sha256(receipt_path), 'recording_commit': receipt.get('recording_commit'),
                'route': receipt.get('route'), 'sha256': before, 'decoded_bytes': decoded_bytes,
                'selected_segments': [{key: item.get(key) for key in ('segment', 'sha256', 'status')}
                                      for item in receipt.get('selected_segments', [])]}
  return tables, params, provenance


def join_inputs(plan, tables, max_age_ns):
  """Never replace a missing exact consumer identity with a nearby message."""
  data = plan['data']
  reference = data.get('reference', {})
  epoch, producer = reference.get('referenceMonoTime'), reference.get('producerMonoTime')
  car_stamp = reference.get('sourceCarStateMonoTime')
  instance = reference.get('producerInstanceMonoTime')
  if (not all(timestamp(value) for value in (epoch, producer, car_stamp, instance))
      or epoch != car_stamp or not instance <= producer <= plan['mono_ns']
      or not 0 <= producer - epoch <= max_age_ns):
    raise InputRejected('noncausal_or_expired_reference_timing')
  if (plan['valid'] is not True or reference.get('timingVersion') != 1
      or reference.get('timingKnown') is not True or reference.get('initialStateKnown') is not True):
    raise InputRejected('unqualified_recorded_reference')
  result = {}
  exact = {'carState': car_stamp, 'modelV2': data.get('modelMonoTime'), 'radarState': data.get('radarStateInputMonoTime')}
  for topic, stamp in exact.items():
    if not timestamp(stamp):
      raise InputRejected('missing_exact_identity_' + topic)
    row = tables[topic].by_time.get(stamp)
    if row is None:
      raise InputRejected('missing_exact_join_' + topic)
    if not 0 <= producer - stamp <= max_age_ns:
      raise InputRejected('noncausal_or_stale_join_' + topic)
    result[topic] = row
  for topic in ASYNC_TOPICS:
    row = tables[topic].asof(producer)
    if row is None or not 0 <= producer - row['mono_ns'] <= max_age_ns:
      raise InputRejected('missing_or_stale_asof_' + topic)
    result[topic] = row
  sp = tables['sp_by_plan'].get(plan['mono_ns'])
  if sp is None or sp['data'].get('longitudinalPlanMonoTime') != plan['mono_ns']:
    raise InputRejected('missing_exact_planSP')
  if not plan['mono_ns'] <= sp['mono_ns'] <= plan['mono_ns'] + max_age_ns:
    raise InputRejected('noncausal_planSP_publication')
  result['longitudinalPlanSP'] = sp
  for topic, row in result.items():
    if row['valid'] is not True:
      raise InputRejected('invalid_input_' + topic)
  return result


def load_runtime():
  # Deferred imports let --help and pure join tests run without a native build.
  import numpy as np
  from openpilot.cereal import log
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as lm
  from openpilot.selfdrive.controls.lib import longitudinal_planner as lp
  from openpilot.selfdrive.controls.lib import unified_longitudinal_planner as up
  from openpilot.selfdrive.controls.lib import radar_consumption as rc
  from openpilot.selfdrive.controls.lib import reference_timing as timing
  from openpilot.selfdrive.controls.lib.low_speed_stop_distance import PARKING_START_SPEED
  return SimpleNamespace(np=np, log=log, lm=lm, lp=lp, up=up, rc=rc, timing=timing, parking_start_speed=PARKING_START_SPEED)


def namespace(value):
  if isinstance(value, dict):
    return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
  if isinstance(value, list):
    return [namespace(item) for item in value]
  return value


class ReferenceInputs:
  """Run the current upstream reference policies on the common observed prefix."""

  def __init__(self, rt):
    self.rt = rt
    self.clear()

  def clear(self):
    self.model = self.rt.up.ModelReferenceAdapter()
    self.low = self.rt.up.LowSpeedStopDistance()
    self.parking = self.rt.up.ParkingApproach()

  def model_mode(self, model, *, e2e, speed, model_ns, epoch_ns, now_ns):
    if not e2e:
      self.model.clear_rest_continuity()
      return 'acc'
    evidence = self.model.build(model['velocity']['x'], self.rt.lm.T_IDXS, speed,
                                model_mono_time_ns=model_ns, anchor_mono_time_ns=epoch_ns,
                                consumed_mono_time_ns=now_ns, retain_rest_continuity=True)
    unknown = evidence.unknown
    ordinary = (not evidence.has_rest_intent and not evidence.no_forward
                and (unknown is None or unknown.reason == 'model_proxy_after_initial_state'
                     and evidence.raw_motion is not None and evidence.raw_motion.does_not_veto_departure))
    if not ordinary or unknown is not None and unknown.reason not in self.rt.up.COMPATIBLE_MODEL_UNKNOWNS:
      raise InputRejected('nonordinary_model_reference', status='unsupported')
    return 'ordinary_e2e'

  def geometry(self, joined, radar, *, speed, accel, delay, epoch_ns, now_ns, permitted):
    model_ns, radar_ns = joined['modelV2']['mono_ns'], joined['radarState']['mono_ns']
    uncertainty = self.parking.uncertainty_cache.update(
      now_ns=now_ns, model_ns=model_ns, model=namespace(joined['modelV2']['data']),
      radar_ns=radar_ns, radar=radar, valid=True,
    )
    if not uncertainty.qualified:
      self.low.clear(source_barrier_ns=epoch_ns)
    comfort, risk = self.low.update(now_ns=now_ns, source_ns=epoch_ns, radar_ns=radar_ns,
                                    speed=speed, radar=radar, permitted=permitted, allow_activation=True)
    parking = self.parking.update(now_ns=now_ns, source_ns=epoch_ns, radar_ns=radar_ns,
                                  speed=speed, measured_accel=accel, actuator_delay=delay, radar=radar,
                                  qualification=self.low.evidence, permitted=permitted,
                                  allow_activation=True, uncertainty=uncertainty)
    if not parking.allow_compact:
      return self.rt.up.low_speed_follow_distance(speed), (self.rt.lm.STOP_DISTANCE,) * 2, False
    return comfort, risk, True


def build_problem(plan, joined, params, rt, reference_inputs):
  data, ref = plan['data'], plan['data']['reference']
  car, model = joined['carState']['data'], joined['modelV2']['data']
  controls, selfdrive = joined['controlsState']['data'], joined['selfdriveState']['data']
  command, vehicle = joined['carControl']['data'], joined['vehicleParameters']['data']
  sp = joined['longitudinalPlanSP']['data']
  motion, intent = controls.get('motionState', {}), data.get('motionIntent', {})
  if not finite_number(ref.get('initialVEgo')):
    raise InputRejected('invalid_recorded_initial_state')

  def unsupported(condition, reason):
    if condition:
      raise InputRejected(reason, status='unsupported')

  unsupported(params.get('carFingerprint') != 'HONDA_ACCORD' or not params.get('openpilotLongitudinalControl')
              or intent.get('mode') != 2, 'platform_or_planning_mode')
  unsupported(not (selfdrive.get('enabled') and command.get('enabled') and command.get('longActive'))
              or car.get('gasPressed') or car.get('brakePressed') or command.get('cruiseControl', {}).get('override'), 'inactive_or_driver_override')
  unsupported(controls.get('longControlState') != 'pid' or controls.get('forceDecel'), 'control_reset_or_force_decel')
  unsupported(car.get('standstill'), 'standstill_arbitration')
  unsupported(not intent.get('valid') or not intent.get('ready') or intent.get('reasons') != 0 or data.get('shouldStop'),
              'recorded_stop_or_unready_arbitration')
  unsupported(data.get('continuation', {}).get('valid', False), 'recorded_continuation')
  if (motion.get('version') != rt.up.MOTION_INTENT_VERSION or motion.get('planningMode') != 2
      or not motion.get('enabled') or motion.get('degraded') or motion.get('waitingForFreshInput')):
    raise InputRejected('motion_owner_unqualified')
  dec = sp.get('dec', {})
  if dec.get('active') and (dec.get('scopeVersion') != 1 or not dec.get('scopeValid')):
    raise InputRejected('unqualified_dec_scope')
  e2e = bool(selfdrive.get('experimentalMode') and (not dec.get('active') or dec.get('state') == 'blended'))
  unsupported(sp.get('longitudinalPlanSource') != 'cruise', 'noncruise_policy')
  unsupported(sp.get('smartCruiseControl', {}).get('vision', {}).get('active'), 'active_curve_policy')
  unsupported(not e2e and not data.get('allowThrottle'), 'coast_cap_requires_unlogged_pitch_history')
  unsupported(e2e and model.get('action', {}).get('shouldStop'), 'model_zero_cruise_target')
  values = {name: car.get(name) for name in ('vEgo', 'aEgo', 'steeringAngleDeg', 'vCruise')}
  values.update(delay=params.get('longitudinalActuatorDelay'), steer_ratio=params.get('steerRatio'), wheelbase=params.get('wheelbase'),
                angle_offset=vehicle.get('angleOffsetDeg'), ceiling=ref.get('constraintAccel'),
                recorded_reference=ref.get('accel'), recorded_request=data.get('aTarget'),
                model_action=model.get('action', {}).get('desiredAcceleration'))
  if not all(finite_number(value) for value in values.values()):
    raise InputRejected('nonfinite_vehicle_or_reference_input')
  if not 0 <= values['delay'] <= 1 or min(values['steer_ratio'], values['wheelbase']) <= 0:
    raise InputRejected('invalid_vehicle_parameters')
  unsupported(values['vCruise'] == rt.up.V_CRUISE_UNSET, 'unset_cruise')
  cruise = min(values['vCruise'], rt.up.V_CRUISE_MAX) * rt.up.CV.KPH_TO_MS
  unsupported(cruise <= rt.up.STOP_SPEED or not finite_number(sp.get('vTarget'))
              or abs(sp['vTarget'] - cruise) > 1e-4, 'nonexternal_cruise_target')
  for values_array in (model.get('velocity', {}).get('x'), model.get('orientationRate', {}).get('z')):
    if (not isinstance(values_array, list) or len(values_array) != len(rt.up.ModelConstants.T_IDXS)
        or not all(finite_number(value) for value in values_array)):
      raise InputRejected('invalid_model_shape_or_values')
  probabilities = model.get('meta', {}).get('disengagePredictions', {}).get('gasPressProbs', [])
  probability = probabilities[1] if len(probabilities) > 1 else 1.0
  if not finite_number(probability):
    raise InputRejected('nonfinite_throttle_probability')
  speed = max(0.0, values['vEgo'])
  accel = float(rt.np.clip(values['aEgo'], rt.lm.ACCEL_MIN, rt.lm.ACCEL_MAX))
  if speed == 0:
    accel = max(0.0, accel)
  if any(not finite_number(ref.get(key)) or abs(ref[key] - actual) > 1e-6 for key, actual in
         (('sourceVEgo', values['vEgo']), ('sourceAEgo', values['aEgo']), ('initialVEgo', speed), ('initialAEgo', accel))):
    raise InputRejected('recorded_initial_state_mismatch')
  allow_throttle = probability > rt.lp.ALLOW_THROTTLE_THRESHOLD or speed <= rt.lp.MIN_ALLOW_THROTTLE_SPEED
  if allow_throttle != data.get('allowThrottle'):
    raise InputRejected('recorded_throttle_policy_mismatch')
  maximum = rt.up.cruise_accel_limit(SimpleNamespace(**params), speed, values['steeringAngleDeg'] - values['angle_offset'],
                                    e2e, rt.lm.ACCEL_MAX, allow_throttle)
  radar_original = rt.log.RadarState.new_message(**joined['radarState']['data']).as_reader()
  consumer = rt.rc.RadarConsumption()
  radar = radar_original
  if radar_original.fusionActive:
    unsupported(not selfdrive.get('experimentalMode') or not dec.get('enabled'), 'fusion_consumer_scope_unknown')
    radar = consumer.update(radar_original, publication_ns=joined['radarState']['mono_ns'], now_ns=ref['producerMonoTime'], speed=speed)
  if consumer.degraded or getattr(radar, 'fusionDegraded', False) or radar.fusionActive and not radar.fusionHealthy:
    raise InputRejected('degraded_radar_consumption')
  try:
    personality = getattr(rt.log.LongitudinalPersonality, selfdrive['personality'])
  except (KeyError, AttributeError) as error:
    raise InputRejected('unknown_personality') from error
  mode = reference_inputs.model_mode(model, e2e=e2e, speed=speed, model_ns=joined['modelV2']['mono_ns'],
                                     epoch_ns=ref['referenceMonoTime'], now_ns=ref['producerMonoTime'])
  comfort, risk, compact = reference_inputs.geometry(
    joined, radar, speed=speed, accel=values['aEgo'], delay=values['delay'], epoch_ns=ref['referenceMonoTime'],
    now_ns=ref['producerMonoTime'], permitted=bool(params.get('radarUnavailable')),
  )
  reference = rt.lm.MpcReference(cruise_speed=None, accel_min=rt.lm.ACCEL_MIN, accel_max=maximum,
                                  allow_future_rest_reschedule=True, allow_interval_rest=True,
                                  comfort_stop_distance=comfort, lead_risk_stop_distances=risk)
  return SimpleNamespace(plan=plan, joined=joined, reference=reference, radar=radar, original_radar=radar_original,
                         speed=speed, accel=accel, measured_speed=values['vEgo'], measured_accel=values['aEgo'],
                         epoch_ns=ref['referenceMonoTime'], producer_ns=ref['producerMonoTime'],
                         instance_ns=ref['producerInstanceMonoTime'], personality=personality,
                         delay=values['delay'], action_t=values['delay'] + rt.lp.DT_MDL, ceiling=values['ceiling'],
                         model_reference_mode=mode, compact_geometry=compact,
                         expired_identities=consumer.expired_identities, retired_identities=consumer.retired_identities)


def arm_reference(reference, arm):
  """The geometry arm can change only this explicit comfort field."""
  return replace(reference, comfort_stop_distance=max(3.0, reference.comfort_stop_distance + arm.comfort_delta_m))


def history_scale(urgency, minimum_scale, arm):
  return (1.0 - (1.0 - minimum_scale) * urgency) * arm.history_scale


def physical_gap_at(times, trajectory, states, jerks, horizon):
  """Linear physical lead prediction minus exact constant-jerk ego position."""
  if not times[0] <= horizon <= times[-1]:
    raise ValueError('Prediction horizon outside native trajectory')
  stage = min(bisect_right(times, horizon) - 1, len(jerks) - 1)
  duration = horizon - times[stage]
  position, velocity, acceleration = states[stage]
  ego = position + velocity * duration + acceleration * duration ** 2 / 2 + jerks[stage] * duration ** 3 / 6
  lead_duration = times[stage + 1] - times[stage]
  lead = trajectory[stage][0] + (trajectory[stage + 1][0] - trajectory[stage][0]) * duration / lead_duration
  return float(lead - ego)


def installed_problem(mpc, rt):
  return {'x0': mpc.x0.copy(), 'params': mpc.params.copy(), 'weights': mpc.diagnostic_reference_weights.copy(),
          'constraint_weights': mpc._constraint_cost_weights.copy(), 'yref': mpc.yref.copy(),
          'bounds': [value.copy() for value in mpc.diagnostic_reference_bounds],
          'general_constraints': [[value.copy() for value in row] for row in mpc._rest_constraint_buffers],
          'history': json_value(mpc.diagnostic_reference_history),
          't': rt.lm.T_IDXS.copy()}


def invariant_fields(problem, rt):
  """Candidate effects on a_prev/comfort/history weights are the ONLY exclusions."""
  if problem is None:
    return None
  columns = [index for index in range(rt.lm.PARAM_DIM) if index not in (3, rt.lm.COMFORT_STOP_DISTANCE)]
  weights = [index for index in range(rt.lm.COST_DIM) if index != 4]
  return {**{key: problem[key] for key in ('x0', 'constraint_weights', 'yref', 'bounds', 'general_constraints', 't')},
          'risk_and_physical_params': problem['params'][:, columns], 'nonhistory_weights': problem['weights'][:, weights]}


class NativeArm:
  def __init__(self, arm, rt):
    self.arm, self.rt = arm, rt
    self.mpc = rt.lm.LongitudinalMpc(speed_based_following=True)
    self.timing = rt.timing.ReferenceTimingProducer()
    self.initialized = False
    self.previous_request = 0.0
    self.previous_epoch = self.previous_producer = self.instance = None
    self.previous_output = None
    self.continuous_start_ns = None
    self.problem = None
    native_run = self.mpc.run

    def run_with_receipt():
      # Capture before run(): failure recovery may reset native parameters.
      self.problem = installed_problem(self.mpc, rt)
      native_run()

    self.mpc.run = run_with_receipt

  def break_history(self):
    self.mpc.reset()
    self.initialized = False
    self.previous_request = 0.0
    self.previous_epoch = self.previous_producer = self.previous_output = None
    self.continuous_start_ns = None

  def solve(self, inputs):
    rt, mpc = self.rt, self.mpc
    previous_epoch, previous_producer = self.previous_epoch, self.previous_producer
    if self.instance is not None and self.instance != inputs.instance_ns:
      self.break_history()
      self.timing = rt.timing.ReferenceTimingProducer()
    self.instance = inputs.instance_ns
    if self.timing.qualify_source(inputs.epoch_ns, inputs.measured_speed, inputs.measured_accel, inputs.producer_ns) is None:
      raise InputRejected('reordered_or_mutated_source_clock')
    prepared = mpc.prepare_reference_history(inputs.epoch_ns, valid=self.initialized)
    prior_request = self.previous_request
    braking_urgency = rt.lp.get_lead_braking_urgency(inputs.speed, prior_request, inputs.radar, inputs.delay, now_ns=inputs.producer_ns)
    approach_urgency = rt.lp.get_lead_approach_urgency(inputs.speed, inputs.radar, inputs.delay,
                                                   personality=inputs.personality, now_ns=inputs.producer_ns)
    urgency = max(braking_urgency, approach_urgency)
    scale = history_scale(urgency, rt.lp.LEAD_BRAKING_MIN_CHANGE_SCALE, self.arm)
    reference = arm_reference(inputs.reference, self.arm)
    mpc.set_weights(prepared, personality=inputs.personality, a_change_scale=scale)
    mpc.set_cur_state(inputs.speed, inputs.accel)
    self.problem = None
    mpc.update(inputs.radar, personality=inputs.personality, reference=reference,
               now_ns=inputs.producer_ns, reference_ns=inputs.epoch_ns, reference_inputs_valid=True)
    actual_history = self.problem['history'] if self.problem else json_value(mpc.diagnostic_reference_history)
    result = {'status': 'solved' if mpc.solution_status == 0 else 'solver_failed', 'reason': mpc.failure_reason,
              'solver_status': int(mpc.solution_status), 'native_status': mpc._diagnostic_native_return,
              'native_outcome': mpc.outcome.value,
              'solve_time_s': mpc.solve_time, 'history_prepared': prepared,
              'history_enabled': bool(self.problem is not None and rt.np.any(self.problem['weights'][:, 4] > 0)),
              'history_reason': (actual_history or {}).get('reason', 'not_qualified'), 'history_marked': False,
              'previous_source_age_s': None if previous_epoch is None else (inputs.epoch_ns - previous_epoch) / NS,
              'previous_producer_age_s': None if previous_producer is None else (inputs.producer_ns - previous_producer) / NS,
              'previous_projected_request': prior_request, 'braking_urgency': braking_urgency, 'approach_urgency': approach_urgency,
              'a_change_scale': scale, 'comfort_raw_m': inputs.reference.comfort_stop_distance,
              'comfort_candidate_m': reference.comfort_stop_distance,
              'physical_problem_sha256': value_hash(invariant_fields(self.problem, rt)) if self.problem else None}
    node = {'problem': self.problem, 'input_reference': asdict(reference), 'history': actual_history,
            'raw_radar': inputs.original_radar.to_dict(), 'consumed_radar': inputs.radar.to_dict()}
    if mpc.solution_status != 0:
      mpc.mark_reference_history(valid=False)
      self.break_history()
      return result, node
    accel = float(mpc.sample_solution(inputs.action_t)[1])
    if not math.isfinite(accel):
      raise InputRejected('nonfinite_native_reference', status='solver_failed')
    request = float(rt.np.clip(min(accel, inputs.ceiling), rt.lm.ACCEL_MIN, rt.lm.ACCEL_MAX))
    params, states, jerks = self.problem['params'], mpc.x_sol.copy(), mpc.j_solution.copy()
    speed = states[:, 1]
    safe_dynamic = speed ** 2 / (2 * rt.lm.COMFORT_BRAKE) + params[:, 4] * speed
    virtual_gap = params[:, 2] - states[:, 0]
    risk_margin = virtual_gap - params[:, 5] * (safe_dynamic + params[:, rt.lm.RISK_STOP_DISTANCE])
    comfort_error = virtual_gap - safe_dynamic - params[:, rt.lm.COMFORT_STOP_DISTANCE]
    slacks = rt.np.array([mpc.solver.get(stage, 'sl') for stage in range(rt.lm.N)])
    # These ordinary-ACC residuals have the same speed normalization and
    # integration scale as the native objective; raw 3/200/5 weights alone
    # cannot establish which cost dominated a solve.
    residuals = rt.np.column_stack((comfort_error / (speed + 10.0), states[:, 0],
                                   speed - params[:, rt.lm.MODEL_V_REF], states[:, 2] - params[:, rt.lm.MODEL_A_REF],
                                   states[:, 2] - params[:, 3], rt.np.r_[jerks, 0.0]))
    intervals = rt.np.r_[rt.np.diff(rt.lm.T_IDXS), 1.0]
    objective = 0.5 * rt.np.sum(residuals ** 2 * self.problem['weights'] * intervals[:, None], axis=0)
    physical = []
    for index, lead in enumerate((inputs.radar.leadOne, inputs.radar.leadTwo)):
      result[f'lead{index}_present'] = bool(lead.present)
      if not lead.present:
        continue
      trajectory = mpc.process_lead(lead, reference_ns=inputs.epoch_ns)
      floor_offset = mpc.lead_initial_state(lead, reference_ns=inputs.epoch_ns)[0] - trajectory[0, 0]
      trajectory[:, 0] += floor_offset
      physical.append({'slot': index, 'trajectory': trajectory, 'numerical_floor_removed_m': floor_offset})
      for horizon in (1, 2, 3):
        result[f'lead{index}_predicted_gap_{horizon}s_m'] = physical_gap_at(rt.lm.T_IDXS, trajectory, states, jerks, horizon)
      result[f'lead{index}_min_continuous_clearance_m'] = mpc.lead_clearance_minimum(trajectory, states, jerks)
    reference_jerk = request_jerk = None
    if self.previous_output is not None:
      previous_time, previous_accel, previous_request = self.previous_output
      delta_ns = inputs.producer_ns - previous_time
      if 0 < delta_ns <= rt.timing.REFERENCE_MAX_AGE_NS:
        reference_jerk = (accel - previous_accel) * NS / delta_ns
        request_jerk = (request - previous_request) * NS / delta_ns
    if self.continuous_start_ns is None:
      self.continuous_start_ns = inputs.producer_ns
    result.update(reference_accel=accel, projected_request=request, reference_error=accel - inputs.plan['data']['reference']['accel'],
                  projected_request_error=request - inputs.plan['data']['aTarget'],
                  planned_reference_jerk=reference_jerk, projected_request_jerk=request_jerk,
                  native_jerk_abs_max=float(rt.np.max(rt.np.abs(jerks))),
                  risk_margin_min_m=float(rt.np.min(risk_margin[:-1])),
                  risk_margin_first3s_min_m=float(rt.np.min(risk_margin[rt.lm.T_IDXS <= 3.0])),
                  risk_slack_max=float(rt.np.max(slacks[:, 3])),
                  risk_slack_max_m=float(rt.np.max(slacks[:, 3] * (speed[:-1] + 10.0))),
                  objective_distance_cost=float(objective[0]), objective_history_cost=float(objective[4]),
                  objective_jerk_cost=float(objective[5]), native_total_cost=float(mpc.solver.get_cost()),
                  accel_upper_margin_min=float(rt.np.min(params[:-1, 1] - states[:-1, 2])),
                  comfort_applied_min_m=float(rt.np.min(params[:, rt.lm.COMFORT_STOP_DISTANCE])),
                  comfort_applied_max_m=float(rt.np.max(params[:, rt.lm.COMFORT_STOP_DISTANCE])),
                  native_source=str(mpc.source), native_fcw=bool(mpc.crash_cnt > 2),
                  continuous_prefix_s=(inputs.producer_ns - self.continuous_start_ns) / NS)
    # mark_reference_history is deliberately last: update() owns formal support
    # qualification, and only this successful accepted native solve can commit.
    result['history_marked'] = mpc.mark_reference_history(valid=True)
    self.initialized = True
    self.previous_request = request
    self.previous_epoch, self.previous_producer = inputs.epoch_ns, inputs.producer_ns
    self.previous_output = inputs.producer_ns, accel, request
    node.update(states=states, jerks=jerks, physical_leads=physical, risk_margin_m=risk_margin,
                comfort_error_m=comfort_error, native_slacks=slacks, objective_components=objective)
    return result, node


def source_provenance(rt, binding_path):
  root = Path(__file__).resolve().parents[3]
  modules = [Path(__file__), *(Path(module.__file__) for module in (rt.lm, rt.lp, rt.up, rt.rc, rt.timing))]
  native = sys.modules.get(rt.lm.AcadosOcpSolverCython.__module__)
  if native is not None and getattr(native, '__file__', None):
    modules.append(Path(native.__file__))
  result = {'root': str(root), 'loaded_sha256': {str(path.relative_to(root)): sha256(path) for path in modules},
            'source_commit': None, 'binding_sha256': None}
  if (root / '.git').exists():
    result['source_commit'] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
  if binding_path is not None:
    binding = json.loads(binding_path.read_text())
    if result['source_commit'] is not None and binding.get('source_commit') != result['source_commit']:
      raise ValueError('Source binding and checkout commit differ')
    result['source_commit'] = binding.get('source_commit')
    result['binding_sha256'] = sha256(binding_path)
    for field in ('native_artifacts_sha256', 'source_files_sha256'):
      declared = binding.get(field, {})
      if not isinstance(declared, dict):
        raise ValueError('Invalid source binding: ' + field)
      for relative, expected in declared.items():
        if sha256(root / relative) != expected:
          raise ValueError('Changed source binding artifact: ' + relative)
      result[field] = declared
  return result


def statistics(values, rt):
  finite = [value for value in values if finite_number(value)]
  if not finite:
    return {'n': 0}
  array = rt.np.array(finite)
  return {'n': len(finite), 'mean': float(rt.np.mean(array)), 'median': float(rt.np.median(array)),
          'p05': float(rt.np.quantile(array, 0.05)), 'p95': float(rt.np.quantile(array, 0.95)),
          'min': float(array.min()), 'max': float(array.max()), 'abs_max': float(rt.np.abs(array).max())}


def summarize(records, arms, rt):
  metrics = ('reference_accel', 'projected_request', 'reference_error', 'projected_request_error',
             'planned_reference_jerk', 'projected_request_jerk', 'native_jerk_abs_max', 'risk_margin_min_m',
             'risk_margin_first3s_min_m', 'risk_slack_max_m', 'accel_upper_margin_min',
             'objective_distance_cost', 'objective_history_cost', 'objective_jerk_cost',
             'lead0_predicted_gap_1s_m', 'lead0_predicted_gap_2s_m', 'lead0_predicted_gap_3s_m')
  scopes = {'all': (-math.inf, math.inf), **SCOPES}
  result = {}
  for scope, (low, high) in scopes.items():
    selected = [record for record in records if low <= record['relative_s'] <= high]
    by_arm = {}
    for arm in arms:
      rows = [record for record in selected if record['arm'] == arm.name]
      by_arm[arm.name] = {'frames': len(rows), 'statuses': dict(Counter(row['status'] for row in rows)),
                          'reasons': dict(Counter(row['reason'] for row in rows if row['status'] != 'solved')),
                          'history_enabled_frames': sum(row.get('history_enabled', False) for row in rows),
                          'invariant_violations': sum(row.get('invariants_match_baseline') is False for row in rows),
                          'metrics': {metric: statistics([row.get(metric) for row in rows], rt) for metric in metrics}}
    result[scope] = by_arm
  return result


def common_frame(plan, mark_ns, joined=None, inputs=None):
  data, ref = plan['data'], plan['data'].get('reference', {})
  result = {'plan_mono_ns': plan['mono_ns'], 'relative_s': (plan['mono_ns'] - mark_ns) / NS,
            'producer_ns': ref.get('producerMonoTime'), 'source_ns': ref.get('referenceMonoTime'),
            'recorded_solver_status': data.get('solverStatus'), 'recorded_reference_accel': ref.get('accel'),
            'recorded_a_target': data.get('aTarget'), 'recorded_independent_ceiling': ref.get('constraintAccel'),
            'recorded_constraint_source': ref.get('constraintSource'), 'recorded_source': ref.get('source'),
            'recorded_a_ego': ref.get('sourceAEgo'), 'recorded_v_ego': ref.get('sourceVEgo')}
  if joined:
    for topic, row in joined.items():
      result[topic + '_mono_ns'] = row['mono_ns']
    result['auxiliary_join_kind'] = 'asof_producer_no_consumed_identity'
    result['exact_join_sha256'] = value_hash({key: (row['mono_ns'], row.get('_sha256')) for key, row in joined.items()})
    for index, key in enumerate(('leadOne', 'leadTwo')):
      lead = joined['radarState']['data'].get(key, {})
      evidence = lead.get('evidence', {})
      result[f'lead{index}_observed_present'] = lead.get('present', False)
      for output, field in (('observed_gap_m', 'dRel'), ('observed_speed_mps', 'vLead'), ('observed_v_rel', 'vRel')):
        result[f'lead{index}_{output}'] = lead.get(field) if lead.get('present') else None
      result[f'lead{index}_identity'] = json.dumps([evidence.get('sessionId'), evidence.get('trackId')])
      result[f'lead{index}_measurement_ns'] = evidence.get('positionMonoTime')
      result[f'lead{index}_measurement_age_s'] = ((ref['producerMonoTime'] - evidence['positionMonoTime']) / NS
                                                 if timestamp(evidence.get('positionMonoTime')) else None)
  if inputs:
    result.update(action_horizon_s=inputs.action_t, raw_comfort_m=inputs.reference.comfort_stop_distance,
                  maximum_accel=inputs.reference.accel_max, native_cruise_external=True,
                  model_reference_mode=inputs.model_reference_mode, compact_geometry=inputs.compact_geometry,
                  consumed_radar_sha256=value_hash(inputs.radar.to_dict()),
                  radar_expired_identities=json.dumps(inputs.expired_identities), radar_retired_identities=json.dumps(inputs.retired_identities))
  return result


def run(args):
  import csv

  if (not math.isfinite(args.start_offset) or not math.isfinite(args.end_offset)
      or not 0 < args.end_offset - args.start_offset <= MAX_WINDOW_S or not timestamp(args.mark_ns)):
    raise ValueError('Select a finite, positive window no longer than 180 seconds and a UInt64 mark')
  arm_choices = {**ARMS, 'geometry': Arm('geometry', args.geometry_delta), 'continuity': Arm('continuity', history_scale=args.continuity_scale)}
  names = args.arms.split(',')
  if not names or names[0] != 'baseline' or len(set(names)) != len(names) or any(name not in arm_choices for name in names):
    raise ValueError('Arms must start with baseline and contain unique supported names')
  if args.output.exists():
    raise ValueError('Use a new output directory')
  started = time.monotonic()
  start_ns, end_ns = (args.mark_ns + round(offset * NS) for offset in (args.start_offset, args.end_offset))
  tables, params, input_proof = load_inputs(args.input_dir, start_ns, end_ns, args.mark_ns)
  rt = load_runtime()
  source_proof = source_provenance(rt, args.source_binding)
  reference_inputs = ReferenceInputs(rt)
  arms = [arm_choices[name] for name in names]
  runners = [NativeArm(arm, rt) for arm in arms]
  plans = [row for row in tables['longitudinalPlan'].rows if start_ns <= row['mono_ns'] <= end_ns]
  if not plans or plans[0]['mono_ns'] > start_ns + rt.timing.REFERENCE_MAX_AGE_NS or plans[-1]['mono_ns'] < end_ns - rt.timing.REFERENCE_MAX_AGE_NS:
    raise ValueError('Incomplete selected plan coverage')
  args.output.mkdir(parents=True, exist_ok=False)
  records = []
  with gzip.open(args.output / 'nodes.jsonl.gz', 'xb', compresslevel=1) as nodes:
    for plan in plans:
      joined = inputs = rejection = None
      try:
        joined = join_inputs(plan, tables, rt.timing.REFERENCE_MAX_AGE_NS)
        inputs = build_problem(plan, joined, params['data'], rt, reference_inputs)
      except InputRejected as error:
        reference_inputs.clear()
        rejection = error
      common = common_frame(plan, args.mark_ns, joined, inputs)
      baseline_invariants = None
      for runner in runners:
        node = {}
        try:
          if rejection is not None:
            raise rejection
          result, node = runner.solve(inputs)
          invariants = invariant_fields(runner.problem, rt)
          if runner.arm.name == 'baseline':
            baseline_invariants = invariants
            result['invariants_match_baseline'] = True if invariants is not None else None
          elif invariants is not None and baseline_invariants is not None:
            changed = [key for key in invariants if value_hash(invariants[key]) != value_hash(baseline_invariants[key])]
            result['invariants_match_baseline'] = not changed
            result['invariant_differences'] = ','.join(changed)
          else:
            result['invariants_match_baseline'] = None
        except (InputRejected, RuntimeError, ValueError) as error:
          runner.break_history()
          result = {'status': error.status if isinstance(error, InputRejected) else 'native_exception',
                    'reason': error.reason if isinstance(error, InputRejected) else f'{type(error).__name__}: {error}',
                    'history_enabled': False, 'history_marked': False, 'invariants_match_baseline': None}
        record = json_value({**common, 'arm': runner.arm.name, **result})
        records.append(record)
        nodes.write(json_bytes({**record, **node}) + b'\n')
  with (args.output / 'frames.csv').open('x', newline='') as stream:
    columns = list(dict.fromkeys(key for record in records for key in record))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)
  result = {'schema_version': 1, 'status': 'DIAGNOSTIC_COMPLETE', 'source': source_proof, 'inputs': input_proof,
            'mark_ns': args.mark_ns, 'start_offset_s': args.start_offset, 'end_offset_s': args.end_offset,
            'arms': [asdict(arm) for arm in arms], 'plans_in_selected_prefix': len(plans),
            'scope': 'Recorded-state native MPC sensitivity; not planner/CAN/vehicle acceptance',
            'window_clock': 'longitudinalPlan Event mono_ns', 'planned_jerk_clock': 'actual producerMonoTime',
            'history': 'Own accepted native references; reset at every unsupported/rejected frame; no logged aTarget feedback',
            'risk_comparison': 'Installed constraints are compared per frame; unchanged constraints do not prove unchanged risk margins',
            'independent_ceiling': 'Frozen reference.constraintAccel; projected request is clip(min(native sample, ceiling))',
            'physical_gap': 'Current lead prediction with numerical distance floor removed, linearly sampled; exact piecewise-jerk ego x',
            'limitations': ['Measured distance and acceleration cannot improve in this recorded-state experiment',
                            'Auxiliary as-of joins do not prove their historical consumed identities',
                            'Frozen independent ceiling and no full planner/actuator feedback',
                            'Unsupported frames break native history; inspect continuous_prefix_s before interpreting the primary window',
                            'No closed-loop response, braking safety or deployment acceptance is inferred'],
            'results': summarize(records, arms, rt), 'elapsed_s': time.monotonic() - started,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'outputs_sha256': {name: sha256(args.output / name) for name in ('frames.csv', 'nodes.jsonl.gz')}}
  save_json(args.output / 'summary.json', result)
  primary = {name: {'statuses': value['statuses'], 'reference_error': value['metrics']['reference_error'],
                    'invariant_violations': value['invariant_violations']}
             for name, value in result['results']['primary'].items()}
  print(json.dumps({'status': result['status'], 'output': str(args.output), 'plans': len(plans),
                    'elapsed_s': result['elapsed_s'], 'primary': primary}, allow_nan=False), flush=True)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input-dir', type=Path, required=True)
  parser.add_argument('--mark-ns', type=int, required=True)
  parser.add_argument('--start-offset', type=float, default=-60.0)
  parser.add_argument('--end-offset', type=float, default=10.0)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--arms', default='baseline,geometry,continuity')
  parser.add_argument('--geometry-delta', type=float, default=-1.0, help='Diagnostic comfort-only delta in metres')
  parser.add_argument('--continuity-scale', type=float, default=0.5, help='Diagnostic multiplier of current a_change_scale')
  parser.add_argument('--source-binding', type=Path, help='Explicit host snapshot source/native artifact binding')
  run(parser.parse_args())


if __name__ == '__main__':
  main()
