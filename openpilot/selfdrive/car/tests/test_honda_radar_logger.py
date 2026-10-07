"""Native logger smoke test; build loggerd first. No vehicle IPC or log directory."""
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from openpilot.cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import CONTROL, ObserverSettings, SERVICE
from openpilot.selfdrive.car.tests.test_honda_radar_observer import can_message, configuration, scan_frames
from openpilot.selfdrive.car.tests.test_honda_radar_shadow import fusion_state
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_CONTROL_OBJECTS
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
from openpilot.tools.lib.logreader import LogReader, decompress_stream


def synthetic_track_diagnostics(index):
  # Deliberately artificial wire-test values, not a valid fusion-state fixture.
  sample = index + 1
  wide = (1 << 40) + 100 * sample
  return {
    'active': True,
    'generation': wide + 1,
    'birthReason': f'wire-track-{sample}-birth',
    'lastInvalidReason': f'wire-track-{sample}-invalid',
    'lastInvalidMonoTime': wide + 2,
    'frameIndex': sample,
    'lifecycleRaw': 100 + sample,
    'previousPositionSeq': wide + 3,
    'previousPositionMonoTime': wide + 4,
    'previousDRel': 100 + sample + 0.125,
    'previousYRel': -(100 + sample + 0.375),
    'acceleration': -(sample + 0.125),
    'accelerationGood': True,
    'lateralPositionVariance': sample + 0.375,
    'lateralVelocityVariance': sample + 1.625,
    'lateralVelocityValid': True,
  }


def synthetic_lead_diagnostics(index):
  sample = index + 1
  wide = (1 << 42) + 100 * sample
  value = 1000 + 100 * sample
  return {
    'identityState': f'wire-lead-{sample}-state',
    'identityReason': f'wire-lead-{sample}-reason',
    'candidateTrackId': wide + 1,
    'bindingStartMonoTime': wide + 2,
    'bindingSamples': 200 + sample,
    'recoverySamples': 300 + sample,
    'independentPair': True,
    'residualX': value + 0.125,
    'residualY': value + 0.25,
    'residualV': value + 0.375,
    'residualDeltaX': value + 0.5,
    'residualDeltaY': value + 0.625,
    'bindingAnchorX': value + 0.75,
    'bindingAnchorY': value + 0.875,
    'rejectionMonoTime': wide + 3,
    'gateX': value + 1.125,
    'gateY': value + 1.25,
    'visionXStd': value + 1.375,
    'visionYStd': value + 1.5,
    'visionVStd': value + 1.625,
    'blend': value + 1.75,
    'coastWeight': value + 1.875,
    'velocityGate': value + 2.125,
  }


def synthetic_control_lead(index, stamp, source='radar'):
  # These source-specific fixtures exercise the wire contract, not the
  # producer's association, calibration or departure-confirmation behavior.
  sample = index + 1
  physical = source != 'vision'
  predicted = source == 'predicted'
  entering = index == 3 and not predicted
  edge = index == 4 and physical and not predicted
  physical_conflict = physical and not predicted and not edge
  evidence = {
    'version': 1, 'source': source, 'sessionId': 'native-logger-control' if physical else '',
    'trackId': (1 << 40) + sample if physical else 0, 'generation': (1 << 41) + sample if physical else 0,
    'positionMonoTime': stamp - 30_000_000 - sample, 'velocityMonoTime': stamp - 20_000_000 - sample,
    'estimateMonoTime': stamp - 10_000_000, 'modelMonoTime': stamp - 5_000_000,
    'positionStd': .25 + sample / 64, 'velocityStd': .125 + sample / 128,
    'positionValid': True, 'velocityValid': True, 'accelerationValid': not predicted,
    'followQualified': not (entering or predicted or edge), 'brakeQualified': not edge,
    'stopQualified': not (entering or predicted or edge), 'departureQualified': not (entering or predicted or edge),
    'fcwQualified': not (predicted or edge), 'coasting': predicted, 'cautionQualified': edge,
    # Non-Float32 fractions catch truncation of the independent Float64
    # physical interval through native logging and the carControlSP mirror.
    'physicalConflictValid': physical_conflict,
    'physicalConflictStart': .275123456789123 if physical_conflict and entering else 0.0,
    'physicalConflictEnd': 1.23456789012345 + sample / 128 if physical_conflict else 0.0,
    'pathState': 'edge' if edge else 'entering' if entering else 'inPath',
    'pathOffset': 2.125 if edge else 2.25 if entering else -.125,
    'conflictStart': .125 if entering else 0.0, 'conflictEnd': .25 if predicted else 1.75 if entering else 10.0,
    'lateralSpeed': -.75 if entering else .125, 'pathSource': 'modelPath' if entering else 'laneLines',
    'visionIndex': index % 2 if source in ('vision', 'fused') else -1,
    'reason': f'wire-control-{sample}-{source}',
  }
  return {
    'present': True, 'dRel': 15.125 + 2 * index, 'yRel': -.125 * sample,
    'vRel': .5 + index / 8, 'vLead': 5.0 + index / 4, 'vLeadK': 5.125 + index / 4,
    'aLeadK': 0.0 if predicted else -.125 * sample, 'aLeadTau': 1.5,
    'modelProb': .96875 if source == 'vision' else .75 if source == 'fused' else 0.0,
    'radar': physical, 'radarTrackId': sample if physical else -1, 'evidence': evidence,
  }


def synthetic_control_states(now):
  states = []
  for index, state in enumerate(('active', 'degraded', 'edge', 'cleared', 'fallback')):
    stamp = now - 100_000_000 + index * 20_000_000
    visions = [synthetic_control_lead(MAX_CONTROL_OBJECTS - 2 + slot, stamp, 'vision') for slot in range(2)]
    # radard preserves the original visual messages before adding formal
    # evidence. Their probabilities and motion values remain independent.
    fallback = [{field: value for field, value in lead.items() if field != 'evidence'} for lead in visions]
    if state == 'active':
      objects = [synthetic_control_lead(track, stamp) for track in range(MAX_CONTROL_OBJECTS - 2)] + visions
      displayed = (objects[0], objects[-1])
    elif state == 'degraded':
      objects = [synthetic_control_lead(0, stamp, 'predicted'), synthetic_control_lead(1, stamp, 'fused'), visions[1]]
      displayed = (objects[1], objects[0])
    elif state == 'edge':
      objects = [synthetic_control_lead(4, stamp), visions[0]]
      displayed = (objects[0], objects[1])
    else:
      objects = []
      displayed = fallback if state == 'fallback' else ({'present': False}, {'present': False})
      if state == 'cleared':
        fallback = [{'present': False}, {'present': False}]
    event = messaging.new_message('radarState', valid=True)
    event.logMonoTime = stamp
    event.radarState = {
      'mdMonoTime': stamp - 5_000_000, 'leadOne': displayed[0], 'leadTwo': displayed[1],
      'radarObjects': objects, 'fusionActive': state != 'fallback',
      'fusionHealthy': state in ('active', 'edge', 'cleared'), 'fusionDegraded': state == 'degraded',
      'fusionReason': state, 'visionLeadOne': fallback[0], 'visionLeadTwo': fallback[1],
      'fusionClearedTracks': [(1 << 40) + 1, (1 << 40) + 2] if state == 'cleared' else [],
      'fusionSessionId': 'native-logger-control' if state != 'fallback' else '',
    }
    states.append((state, event))
  return states


def synthetic_object_influence_events(now):
  # Artificial wire values only: no solver, owner acceptance, measurement
  # quality or causal contribution is established by these attribution rows.
  model_ns = now - 5_000_000
  events = []
  for index, service in enumerate(('longitudinalPlan', 'controlsState')):
    influences = [
      {'version': 1, 'sessionId': 'native-logger-control', 'trackId': (1 << 40) + 1,
       'modelMonoTime': 0, 'visionIndex': -1, 'kind': 'braking', 'strength': .875 - index * .25},
      {'version': 1, 'sessionId': '', 'trackId': 0, 'modelMonoTime': model_ns,
       'visionIndex': 1, 'kind': 'hold', 'strength': 1.0},
      {'version': 1, 'sessionId': 'native-logger-control', 'trackId': (1 << 40) + 5,
       'modelMonoTime': 0, 'visionIndex': -1, 'kind': 'caution', 'strength': .375 - index * .25},
    ]
    event = messaging.new_message(service, valid=True)
    event.logMonoTime = now + 2 + index
    if service == 'longitudinalPlan':
      event.longitudinalPlan.modelMonoTime = model_ns
      event.longitudinalPlan.objectInfluences = influences
    else:
      motion = event.controlsState.motionState
      motion.version, motion.enabled, motion.phase = 1, True, 'track'
      motion.planMonoTime, motion.modelMonoTime = events[0].logMonoTime, model_ns
      motion.objectInfluences = influences
    events.append(event)
  return events


class TestHondaRadarLogger(unittest.TestCase):
  def test_native_logger_records_radar_and_vision_without_radar_qlog_duplication(self):
    # LOG_ROOT must be explicit: on a C3, OpenpilotPrefix alone does not move the
    # road-log directory. The private params path is checked before loggerd runs.
    with TemporaryDirectory(prefix='honda-radar-logger-') as directory, \
         patch.dict(os.environ, {'LOG_ROOT': directory}), OpenpilotPrefix() as prefix:
      self.assertIn(prefix.prefix, Params().get_param_path())
      services = (SERVICE, 'modelV2', 'radarState', 'carControlSP', 'longitudinalPlan', 'controlsState')
      pm = messaging.PubMaster(list(services))
      logger_dir = Path(BASEDIR) / 'openpilot/system/loggerd'
      # Test-only dependency selection keeps host tools separate from tracked ARM binaries.
      logger_executable = os.environ.get('HONDA_RADAR_TEST_LOGGER', str(logger_dir / 'loggerd'))
      process = subprocess.Popen([logger_executable], cwd=logger_dir,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
      try:
        for service in services:
          self.assertTrue(pm.wait_for_readers_to_update(service, timeout=5), f'loggerd did not subscribe to {service}')
        now = time.monotonic_ns()
        frames = [frame for slot in range(16) for frame in scan_frames(slot=slot)]
        sm = fusion_state(now)
        observation = HondaRadarObserver(*configuration(), session_id='native-logger').update(
          [can_message(now, frames)], sm, ObserverSettings(CONTROL, True, True, True), now + 1, sm)
        self.assertEqual(observation.hondaRadarObservations.mode, 4)
        self.assertTrue(observation.hondaRadarObservations.fusion.active)
        self.assertEqual(len(observation.hondaRadarObservations.fusion.tracks), 16)
        self.assertEqual(len(observation.hondaRadarObservations.fusion.leads), 3)
        # Exercise enlarged inline-composite lists and fields unknown to older
        # native loggers. Distinct non-default numeric/Text values catch lost or
        # mis-strided elements; True is the only non-default Bool value.
        # These replacements test wire preservation, not real fusion behavior.
        diagnostics = {}
        for name, fixture in (('tracks', synthetic_track_diagnostics), ('leads', synthetic_lead_diagnostics)):
          diagnostics[name] = []
          for index, item in enumerate(getattr(observation.hondaRadarObservations.fusion, name)):
            fields = fixture(index)
            for field, value in fields.items():
              setattr(item, field, value)
            diagnostics[name].append(fields)
        expected = observation.hondaRadarObservations.to_dict()
        expected_wire = observation.to_bytes()
        model = messaging.new_message('modelV2', valid=True)
        model.modelV2 = sm['modelV2']
        model.modelV2.frameId = 712
        pm.send(SERVICE, expected_wire)
        pm.send('modelV2', model)
        for service in (SERVICE, 'modelV2'):
          self.assertTrue(pm.wait_for_readers_to_update(service, timeout=5))
        control_expected = []
        for state, event in synthetic_control_states(now):
          # Exercise the actual SP mirror conversion, whose presence field
          # is named status rather than the formal LeadData.present.
          mirror = messaging.new_message('carControlSP', valid=True)
          mirror.logMonoTime = event.logMonoTime + 1
          for name in ('leadOne', 'leadTwo'):
            src = getattr(event.radarState.as_reader(), name)
            dst = getattr(mirror.carControlSP, name)
            ControlsExt.get_lead_data(dst, src)
            self.assertEqual(dst.status, src.present)
            self.assertEqual(dst.evidence.to_dict(), src.evidence.to_dict())
          for service, message in (('radarState', event), ('carControlSP', mirror)):
            expected_fields = getattr(message, service).to_dict()
            wire = message.to_bytes()
            control_expected.append((state, service, message.logMonoTime, expected_fields, wire))
            pm.send(service, wire)
            self.assertTrue(pm.wait_for_readers_to_update(service, timeout=5))
        for event in synthetic_object_influence_events(now):
          service = event.which()
          expected_fields = getattr(event, service).to_dict()
          wire = event.to_bytes()
          control_expected.append(('objectInfluences', service, event.logMonoTime, expected_fields, wire))
          pm.send(service, wire)
          self.assertTrue(pm.wait_for_readers_to_update(service, timeout=5))
      finally:
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=10)
      self.assertEqual(process.returncode, 0, output.decode(errors='replace'))
      rlogs = list(Path(directory).glob('*/rlog.zst'))
      qlogs = list(Path(directory).glob('*/qlog.zst'))
      self.assertEqual((len(rlogs), len(qlogs)), (1, 1))
      # Compare the bytes actually sent through native IPC with the decompressed
      # rlog, without rebuilding a reader (which could change Cap'n Proto layout).
      rlog_bytes = decompress_stream(rlogs[0].read_bytes())
      self.assertEqual(rlog_bytes.count(expected_wire), 1,
                       'loggerd did not preserve the complete radar event wire bytes exactly once')
      messages = list(LogReader(str(rlogs[0])))
      radar = [event for event in messages if event.which() == SERVICE]
      self.assertEqual(len(radar), 1)
      self.assertTrue(radar[0].valid)
      self.assertEqual(radar[0].hondaRadarObservations.mode, 4)
      self.assertEqual(radar[0].hondaRadarObservations.to_dict(), expected)
      for name, expected_items in diagnostics.items():
        actual_items = getattr(radar[0].hondaRadarObservations.fusion, name)
        self.assertEqual(len(actual_items), len(expected_items))
        for index, fields in enumerate(expected_items):
          for field, value in fields.items():
            with self.subTest(fusion_list=name, index=index, field=field):
              self.assertEqual(getattr(actual_items[index], field), value)
      for state, service, timestamp, fields, wire in control_expected:
        with self.subTest(state=state, service=service):
          self.assertEqual(rlog_bytes.count(wire), 1, 'loggerd changed or dropped a complete control event')
          recorded = [event for event in messages if event.which() == service and event.logMonoTime == timestamp]
          self.assertEqual(len(recorded), 1)
          self.assertTrue(recorded[0].valid)
          self.assertEqual(getattr(recorded[0], service).to_dict(), fields)
      formal = [event.radarState for event in messages if event.which() == 'radarState']
      self.assertEqual([state.fusionReason for state in formal], ['active', 'degraded', 'edge', 'cleared', 'fallback'])
      active, degraded, edge, cleared, fallback = formal
      self.assertEqual(len(active.radarObjects), MAX_CONTROL_OBJECTS)
      self.assertEqual(len({(lead.evidence.sessionId, lead.evidence.trackId)
                            for index, lead in enumerate(active.radarObjects) if index < MAX_CONTROL_OBJECTS - 2}),
                       MAX_CONTROL_OBJECTS - 2)
      self.assertEqual(active.leadOne.modelProb, 0.0)
      self.assertTrue(active.leadOne.evidence.brakeQualified)
      self.assertFalse(active.visionLeadOne.radar)
      self.assertTrue(degraded.fusionDegraded)
      self.assertFalse(degraded.fusionHealthy)
      self.assertTrue(degraded.leadTwo.evidence.coasting)
      self.assertEqual(str(degraded.leadOne.evidence.source), 'fused')
      self.assertEqual(str(edge.leadOne.evidence.pathState), 'edge')
      self.assertTrue(edge.leadOne.evidence.cautionQualified)
      self.assertFalse(edge.leadOne.evidence.brakeQualified or edge.leadOne.evidence.stopQualified)
      mirrors = [event.carControlSP for event in messages if event.which() == 'carControlSP']
      self.assertEqual(len(mirrors), len(formal))
      self.assertEqual(str(mirrors[2].leadOne.evidence.pathState), 'edge')
      self.assertTrue(mirrors[2].leadOne.evidence.cautionQualified)
      self.assertTrue(cleared.fusionHealthy)
      self.assertEqual(list(cleared.fusionClearedTracks), [(1 << 40) + 1, (1 << 40) + 2])
      self.assertEqual(len(cleared.radarObjects), 0)
      self.assertFalse(cleared.leadOne.present or cleared.leadTwo.present)
      self.assertFalse(fallback.fusionActive or fallback.fusionDegraded)
      self.assertEqual(fallback.leadOne.to_dict(), fallback.visionLeadOne.to_dict())
      self.assertEqual(fallback.leadTwo.to_dict(), fallback.visionLeadTwo.to_dict())
      influence_lists = {}
      for service in ('longitudinalPlan', 'controlsState'):
        recorded = [event for event in messages if event.which() == service]
        self.assertEqual(len(recorded), 1)
        value = getattr(recorded[0], service)
        if service == 'controlsState':
          value = value.motionState
        influences = value.objectInfluences
        influence_lists[service] = influences
        self.assertEqual([str(item.kind) for item in influences], ['braking', 'hold', 'caution'])
        self.assertTrue(all(item.version == 1 for item in influences))
        self.assertEqual(influences[0].sessionId, 'native-logger-control')
        self.assertEqual(influences[0].trackId, (1 << 40) + 1)
        self.assertEqual((influences[1].sessionId, influences[1].trackId, influences[1].visionIndex), ('', 0, 1))
        self.assertEqual(influences[1].modelMonoTime, now - 5_000_000)
        self.assertEqual(influences[2].trackId, (1 << 40) + 5)
      self.assertEqual(influence_lists['longitudinalPlan'][0].strength, .875)
      self.assertEqual(influence_lists['controlsState'][0].strength, .625)
      self.assertEqual(influence_lists['longitudinalPlan'][2].strength, .375)
      self.assertEqual(influence_lists['controlsState'][2].strength, .125)
      self.assertTrue(any(event.which() == 'modelV2' and event.modelV2.frameId == 712 for event in messages))
      self.assertFalse(any(event.which() == SERVICE for event in LogReader(str(qlogs[0]))))
      del pm


if __name__ == '__main__':
  unittest.main()
