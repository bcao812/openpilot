"""Optional Accord telemetry; no control or sendcan publisher.

Supplies the formal radar publisher's separately qualified control objects.
This process never publishes a control command or a formal lead.
"""
import time
import uuid

from opendbc.car.honda.bosch_a_radar import BoschARadarObserver, MAX_AGE_NS
from opendbc.car.honda.hondacan import CanBus
from openpilot.cereal import messaging
from openpilot.common.realtime import drop_realtime, set_core_affinity
from openpilot.selfdrive.car.honda_radar_observer_config import CONTROL, SERVICE, ObserverSettings, scope_state

PUBLISH_PERIOD_NS = 50_000_000
HEARTBEAT_NS = 1_000_000_000
MAX_ENVELOPES = 64
MAX_FRAMES_PER_ENVELOPE = 512
DECODER_VERSION = 'bosch-a-observe-2'


def configure_observer_process():
  # Diagnostic work stays in the ordinary small-core pool. CPU3 serves CAN/SPI,
  # CPU4 the 100 Hz control chain; neither is a decoder worker. These helpers
  # are no-ops on PC, preserving the host resource runner's CPU reservation.
  drop_realtime()
  set_core_affinity([0, 1, 2])


class HondaRadarObserver:
  def __init__(self, cp, cp_sp, session_id=None):
    self.cp, self.cp_sp = cp, cp_sp
    bus = CanBus(cp).camera
    if not 0 <= bus <= 255:
      raise ValueError('Invalid CAN mapping: CarParams must include safetyConfigs')
    self.decoder = BoschARadarObserver(bus)
    self.session_id = session_id or uuid.uuid4().hex
    self.active = False
    self.sequence = 0
    self.last_publish_ns = 0
    self.rejected_envelopes = 0
    self.backlog_count = 0
    self.processing_ns = 0
    self.fusion = None
    self.last_fusion_error_ns = 0
    self.fusion_input_error = False

  def _fusion_failed(self, now_ns):
    if self.fusion is not None:
      self.fusion.reset(clear_ego=True)
    if now_ns - self.last_fusion_error_ns >= 5_000_000_000:
      from openpilot.common.swaglog import cloudlog
      cloudlog.exception('Honda radar shadow fusion unavailable')
      self.last_fusion_error_ns = now_ns

  def update(self, envelopes, sm, settings, now_ns, fusion_sm=None):
    if not settings.radar_signal:
      self.decoder.reset()
      self.fusion = None
      self.active = False
      self.fusion_input_error = False
      return None
    started_ns = time.monotonic_ns()
    active, reason, scope = scope_state(self.cp, self.cp_sp, settings, sm)
    if active != self.active:
      self.decoder.reset()
    self.active = active
    if self.fusion is None:
      try:
        from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
        self.fusion = HondaRadarFusion()
      except Exception:
        self._fusion_failed(now_ns)
        self.fusion_input_error = True
    if self.fusion is not None:
      try:
        self.fusion.record_inputs(fusion_sm, now_ns)
      except Exception:
        self._fusion_failed(now_ns)
        self.fusion_input_error = True
    if active:
      for envelope in envelopes:
        if not envelope.valid or not 0 < envelope.logMonoTime <= now_ns or len(envelope.can) > MAX_FRAMES_PER_ENVELOPE:
          self.rejected_envelopes += 1
          continue
        self.decoder.feed(envelope.logMonoTime, ((f.address, f.dat, f.src) for f in envelope.can))
      self.decoder.advance(now_ns)
      if self.decoder.last_frame_ns == 0:
        reason = 'noRadarFrames'
      elif now_ns - self.decoder.last_frame_ns > MAX_AGE_NS:
        reason = 'radarSilence'
    self.processing_ns += time.monotonic_ns() - started_ns
    if now_ns - self.last_publish_ns < PUBLISH_PERIOD_NS:
      return None
    self.sequence += 1
    msg = messaging.new_message(SERVICE)
    msg.valid = True  # telemetry validity; sensor/scope validity is carried inside
    msg.logMonoTime = now_ns
    body = getattr(msg, SERVICE)
    body.schemaVersion = 1
    body.mode = CONTROL
    body.decoderVersion = DECODER_VERSION
    body.sessionId = self.session_id
    body.sequence = self.sequence
    body.active = active
    body.reason = reason
    body.radarBus = self.decoder.bus
    body.lastRadarFrameMonoTime = self.decoder.last_frame_ns
    body.lastPositionMonoTime = self.decoder.last_position_ns
    observations = self.decoder.drain()
    body.observations = observations
    body.counters = self.decoder.counts
    body.scope = scope
    body.calibrated = False
    body.processingTimeNs = self.processing_ns
    body.rejectedEnvelopes = self.rejected_envelopes
    body.backlogCount = self.backlog_count
    try:
      if self.fusion_input_error or self.fusion is None:
        raise ValueError('Invalid fusion input since the previous publication')
      # Reuse this call's decoder events after the schema validation above.
      # Their measurements are Float64; headers retain the wire-normalized
      # values. No native observation fields need to be read back for fusion.
      fusion_body = {'schemaVersion': body.schemaVersion, 'mode': body.mode, 'sessionId': body.sessionId, 'sequence': body.sequence,
                     'active': body.active, 'counters': body.counters, 'scope': body.scope, 'observations': observations}
      body.fusion = self.fusion.update(fusion_body, fusion_sm, now_ns)
    except Exception:
      # A fusion calculation failure must not interrupt raw observation or
      # leave a previously computed candidate apparently current.
      self._fusion_failed(now_ns)
      body.fusion = {'version': 1, 'enabled': True, 'active': False, 'calibrated': False, 'controlAllowed': False,
                    'reason': 'fusionError', 'tracks': [], 'leads': []}
    self.fusion_input_error = False
    self.processing_ns = 0
    self.last_publish_ns = now_ns
    return msg


def main():
  configure_observer_process()
  # Imports with runtime side effects belong only in the managed process.
  from opendbc.car.structs import car
  from openpilot.cereal import custom
  from openpilot.common.params import Params

  params = Params()
  settings = ObserverSettings.read(params)
  if not settings.radar_signal:
    return
  with car.CarParams.from_bytes(params.get('CarParams', block=True)) as cp, \
       custom.CarParamsSP.from_bytes(params.get('CarParamsSP', block=True)) as cp_sp:
    observer = HondaRadarObserver(cp, cp_sp)
    sm = messaging.SubMaster(['selfdriveState', 'longitudinalPlanSP', 'carControl'])
    fusion_sm = messaging.SubMaster(['carState', 'modelV2'])
    pm = messaging.PubMaster([SERVICE])
    can_sock = messaging.sub_sock('can', timeout=20)
    next_settings_ns = 0
    while True:
      now_ns = time.monotonic_ns()
      if now_ns >= next_settings_ns:
        settings = ObserverSettings.read(params)
        next_settings_ns = now_ns + HEARTBEAT_NS
      if not settings.radar_signal:
        return
      # Bound each drain: a busy CAN producer cannot starve scope checks or expiry.
      envelopes = []
      first = can_sock.receive()
      if first is not None:
        envelopes.append(messaging.log_from_bytes(first))
        for _ in range(MAX_ENVELOPES - 1):
          raw = can_sock.receive(non_blocking=True)
          if raw is None:
            break
          envelopes.append(messaging.log_from_bytes(raw))
        if len(envelopes) == MAX_ENVELOPES:
          observer.backlog_count += 1
      sm.update(0)
      fusion_sm.update(0)
      # Model/carState may have arrived during the settings/subscriber reads.
      # Validate their source times against a clock taken after those reads.
      now_ns = time.monotonic_ns()
      msg = observer.update(envelopes, sm, settings, now_ns, fusion_sm)
      if msg is not None:
        pm.send(SERVICE, msg)


if __name__ == '__main__':
  main()
