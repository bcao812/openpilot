"""Measured extent provenance, plus its actual fusion producer lifecycle."""
import copy
import math
from pathlib import Path

import capnp
import pytest

from openpilot.cereal import custom
from openpilot.selfdrive.car.honda_radar_body import BODY_MAX_AGE_NS, RadarBody, measured_radar_body
from openpilot.selfdrive.ui.tests.test_honda_radar_boundaries import START, observation, with_aux


BODY_DEFAULTS = {
  'bodyWidth': 0.0, 'bodyLeftX': 0.0, 'bodyLeftY': 0.0, 'bodyRightX': 0.0, 'bodyRightY': 0.0,
  'bodyMonoTime': 0, 'bodyAvailableMonoTime': 0, 'bodyValid': False,
}


@pytest.fixture(scope='module')
def legacy_schema():
  # A separate parser allows the actual old and new definitions of the same
  # type IDs to coexist, without mutating cereal's global schema registry.
  parser = capnp.SchemaParser()
  path = Path(__file__).with_name('fixtures') / 'honda_radar_observations_86f3dc03.capnp'
  legacy = parser.load(str(path)).HondaRadarObservations
  assert legacy.FusionTrack.schema.node.id == custom.HondaRadarObservations.FusionTrack.schema.node.id
  assert not set(BODY_DEFAULTS).intersection(legacy.FusionTrack.schema.fields)
  yield legacy


def legacy_wire_track(index):
  # Every pre-body field is explicit; distinct values across 16 elements also
  # detect shifted inline-composite entries after the struct's data area grows.
  sample = index + 1
  wide = (1 << 40) + 100 * sample
  return {
    'trackId': wide + 1, 'oemTrackId': sample, 'slot': index,
    'positionSeq': wide + 2, 'positionMonoTime': wide + 3, 'velocityMonoTime': wide + 4,
    'dRel': 20.125 + sample, 'yRel': -.375 - sample, 'groundSpeed': 10.625 + sample,
    'lateralSpeed': .25 + sample / 8, 'positionVariance': .125 + sample / 16,
    'velocityVariance': .25 + sample / 16, 'observations': 10 + sample,
    'mature': True, 'velocityGood': True, 'coasting': bool(index % 2), 'pathState': 'entering',
    'pathOffset': .5 + sample, 'entryTime': .125 + sample / 8, 'motion': 'moving', 'u10Raw': 100 + sample,
    'active': True, 'generation': wide + 5, 'birthReason': f'legacy-track-{sample}-birth',
    'lastInvalidReason': f'legacy-track-{sample}-invalid', 'lastInvalidMonoTime': wide + 6,
    'frameIndex': sample, 'lifecycleRaw': 200 + sample, 'previousPositionSeq': wide + 7,
    'previousPositionMonoTime': wide + 8, 'previousDRel': 30.375 + sample, 'previousYRel': -1.625 - sample,
    'acceleration': -.125 - sample, 'accelerationGood': True, 'lateralPositionVariance': .375 + sample / 16,
    'lateralVelocityVariance': .625 + sample / 16, 'lateralVelocityValid': True,
  }


def test_old_schema_tracks_read_with_invalid_body_defaults(legacy_schema):
  tracks = [legacy_wire_track(index) for index in range(16)]
  assert set(tracks[0]) == set(legacy_schema.FusionTrack.schema.fields)
  old_message = legacy_schema.new_message(sessionId='before-body-fields', fusion={'tracks': tracks})
  with custom.HondaRadarObservations.from_bytes(old_message.to_bytes()) as reader:
    assert reader.sessionId == 'before-body-fields'
    assert len(reader.fusion.tracks) == len(tracks)
    for track, expected in zip(reader.fusion.tracks, tracks, strict=True):
      actual = track.to_dict()
      assert {name: actual[name] for name in expected} == expected
      assert {name: actual[name] for name in BODY_DEFAULTS} == BODY_DEFAULTS


def test_new_schema_tracks_preserve_old_fields_for_old_readers(legacy_schema):
  tracks = [legacy_wire_track(index) for index in range(16)]
  extended = []
  for index, track in enumerate(tracks):
    extended.append({**track, 'bodyWidth': 1.375 + index / 8, 'bodyLeftX': 100.125 + index,
                     'bodyLeftY': 2.375 + index / 8, 'bodyRightX': 100.625 + index, 'bodyRightY': -1.875 - index / 8,
                     'bodyMonoTime': (1 << 41) + 2 * index, 'bodyAvailableMonoTime': (1 << 41) + 2 * index + 1,
                     'bodyValid': True})
  message = custom.HondaRadarObservations.new_message(sessionId='with-body-fields', fusion={'tracks': extended})
  with legacy_schema.from_bytes(message.to_bytes()) as reader:
    assert reader.sessionId == 'with-body-fields'
    assert [track.to_dict() for track in reader.fusion.tracks] == tracks


def test_old_schema_cannot_create_body_cut_in_braking(legacy_schema):
  from openpilot.selfdrive.car.honda_radar_control import CUT_IN_REASON, HondaRadarControl
  from openpilot.selfdrive.car.tests.test_honda_radar_body_cutin import body_track
  from openpilot.selfdrive.car.tests.test_honda_radar_control import present_leads, scene
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead

  old_control, body_control = HondaRadarControl(), HondaRadarControl()
  for frame in range(3):
    target = body_track(frame)
    inputs = scene(frame, tracks=[target], ego=20.92)
    # Positive control proves that this geometry would actually use body
    # braking when the newly added evidence is present.
    body_result = body_control.update(**inputs)
    old_observation = copy.deepcopy(inputs['observation'])
    old_observation['fusion']['tracks'] = [{name: value for name, value in target.items() if name not in BODY_DEFAULTS}]
    message = legacy_schema.new_message(**old_observation)
    with custom.HondaRadarObservations.from_bytes(message.to_bytes()) as reader:
      assert not reader.fusion.tracks[0].bodyValid
      old_result = old_control.update(**{**inputs, 'observation': reader})
      assert old_result.healthy and not old_result.degraded
      assert not present_leads(old_result)
  lead = present_leads(body_result)[0]
  assert CUT_IN_REASON in lead['evidence']['reason']
  assert qualified_lead(lead, 'brake', now_ns=inputs['now_ns'])


def test_asymmetric_edges_do_not_center_the_body_on_the_reflection():
  sample = observation()
  body = measured_radar_body(sample, START + 1)
  assert body is not None and body.position_ns == body.available_ns == START
  assert body.left != pytest.approx(-body.right)
  left, right = body.endpoints(sample['dRel'], sample['yRel'], 0.0)
  assert left[1] == pytest.approx(sample['dRel'] * math.tan(76 / 2048.0))
  assert right[1] == pytest.approx(sample['dRel'] * math.tan(-4 / 2048.0))
  assert body.width == pytest.approx(left[1] - right[1])


def test_width_rotates_with_the_target_without_changing_its_size():
  body = RadarBody(1.4, -.2, START, START)
  left, right = body.endpoints(20.0, 2.0, math.pi / 2)
  assert left == pytest.approx((18.6, 2.0))
  assert right == pytest.approx((20.2, 2.0))
  assert math.dist(left, right) == pytest.approx(body.width)


def test_late_aux_does_not_refresh_the_body_or_its_availability():
  sample = observation()
  original = measured_radar_body(sample, START + 1)
  aux = with_aux(sample, START + 100_000_000)
  assert measured_radar_body(aux, START + 100_000_001) == original
  assert measured_radar_body(aux, START + BODY_MAX_AGE_NS + 1) is None


@pytest.mark.parametrize('fault', ['missing-edge', 'degenerate', 'not-contained', 'wrong-frame', 'wrong-range',
                                  'missing-frame', 'malformed-frames', 'future-frame', 'invalid-position'])
def test_invalid_extent_has_no_nominal_or_angular_width_fallback(fault):
  sample = observation()
  sample['raw']['angularWidthRaw'] = 256
  if fault == 'missing-edge':
    del sample['raw']['edgeARaw']
  elif fault == 'degenerate':
    sample['raw']['edgeARaw'] = sample['raw']['edgeBRaw']
  elif fault == 'not-contained':
    sample['raw'].update(edgeARaw=1100, edgeBRaw=1200)
  elif fault == 'wrong-frame':
    sample['frames'][3]['address'] += 4
  elif fault == 'wrong-range':
    sample['dRel'] += 1.0
  elif fault == 'missing-frame':
    sample['frames'].pop()
  elif fault == 'malformed-frames':
    sample['frames'] = 42
  elif fault == 'future-frame':
    sample['frames'][3]['monoTime'] = START + 2
  else:
    sample['positionValid'] = False
  assert measured_radar_body(sample, START + 1) is None


def _producer_sample(frame):
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import STEP, observation as fusion_observation
  sample = fusion_observation(frame)
  geometry = with_aux(observation(sequence=sample['positionSeq'], stamp=START + frame * STEP, range_raw=403),
                      START + frame * STEP)
  sample.update(dRel=geometry['dRel'], yRel=geometry['yRel'], frames=geometry['frames'])
  sample['raw'].update(geometry['raw'])
  return sample


def test_fusion_carries_width_from_the_accepted_position_and_withdraws_missing_edges():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import STEP, advance
  fusion = HondaRadarFusion()
  sample = _producer_sample(0)
  track = advance(fusion, 0, [sample])['tracks'][0]
  assert track['bodyValid'] and track['bodyMonoTime'] == START
  assert track['bodyWidth'] == pytest.approx(measured_radar_body(sample, START + 1).width)
  assert track['bodyLeftY'] > track['yRel'] > track['bodyRightY']
  aux = copy.deepcopy(sample)
  aux.update(velocitySeq=20, velocityMonoTime=START + STEP)
  aux['frames'][-1]['monoTime'] = START + STEP
  track = advance(fusion, 1, [aux])['tracks'][0]
  assert track['bodyMonoTime'] == track['bodyAvailableMonoTime'] == START
  replacement = _producer_sample(2)
  del replacement['raw']['edgeARaw']
  track = advance(fusion, 2, [replacement])['tracks'][0]
  assert track['positionMonoTime'] == START + 2 * STEP
  assert not track['bodyValid'] and track['bodyMonoTime'] == 0


def test_fusion_invalidated_track_cannot_publish_its_old_body():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import STEP, advance
  fusion = HondaRadarFusion()
  sample = _producer_sample(0)
  assert advance(fusion, 0, [sample])['tracks'][0]['bodyValid']
  invalid = copy.deepcopy(sample)
  invalid.update(positionValid=False, velocityValid=False, reason='withdrawn', positionMonoTime=START + STEP)
  result = advance(fusion, 1, [invalid])
  assert all(not track['bodyValid'] for track in result['tracks'])


def test_fusion_body_fields_survive_the_wire_roundtrip():
  from opendbc.can import CANPacker
  from openpilot.cereal import log
  from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
  from openpilot.selfdrive.car.tests.test_honda_radar_observer import SETTINGS, can_message, configuration, scan_frames
  from openpilot.selfdrive.car.tests.test_honda_radar_shadow import fusion_state

  # Exercise the production decoder/fusion/message boundary with real cereal
  # model values. The math-only advance fixture carries NumPy path scalars;
  # converting its output here would skip the actual serialization contract.
  frames = scan_frames(distance=403, azimuth=1054, velocity=864)
  frames[3] = CANPacker('honda_bosch_a_radar').make_can_msg(frames[3][0], 2, {
    'FRAME_IDX': 0, 'TRACK_ID': 1, 'AZIMUTH_EDGE_A_RAW': 1020, 'AZIMUTH_EDGE_B_RAW': 1100,
  })
  sm = fusion_state(START)
  message = HondaRadarObserver(*configuration(), session_id='body-wire').update(
    [can_message(START, frames)], sm, SETTINGS, START + 1, sm)
  assert message is not None and message.hondaRadarObservations.fusion.active
  original, = message.hondaRadarObservations.fusion.tracks
  expected = {name: getattr(original, name) for name in BODY_DEFAULTS}
  with log.Event.from_bytes(message.to_bytes()) as reader:
    assert reader.valid and reader.hondaRadarObservations.fusion.active
    track, = reader.hondaRadarObservations.fusion.tracks
    assert {name: getattr(track, name) for name in BODY_DEFAULTS} == expected
    assert track.bodyValid and track.bodyMonoTime == track.bodyAvailableMonoTime == track.positionMonoTime == START
    assert track.bodyWidth == pytest.approx(track.bodyLeftY - track.bodyRightY)
