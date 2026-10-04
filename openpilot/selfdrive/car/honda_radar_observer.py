"""Optional Accord telemetry. Intentionally has no control or sendcan publisher."""
import time
import uuid

from opendbc.car.honda.bosch_a_radar import BoschARadarObserver, MAX_AGE_NS
from opendbc.car.honda.hondacan import CanBus
from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE, ObserverSettings, scope_state

PUBLISH_PERIOD_NS = 50_000_000
HEARTBEAT_NS = 1_000_000_000
MAX_ENVELOPES = 64
MAX_FRAMES_PER_ENVELOPE = 512
DECODER_VERSION = 'bosch-a-observe-1'


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
    self.last_reason = ''
    self.rejected_envelopes = 0
    self.backlog_count = 0
    self.processing_ns = 0

  def update(self, envelopes, sm, settings, now_ns):
    started_ns = time.monotonic_ns()
    active, reason, scope = scope_state(self.cp, self.cp_sp, settings, sm, now_ns)
    if active != self.active:
      self.decoder.reset()
    self.active = active
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
    if not self.decoder.events and reason == self.last_reason and now_ns - self.last_publish_ns < HEARTBEAT_NS:
      return None
    self.sequence += 1
    msg = messaging.new_message(SERVICE)
    msg.logMonoTime = now_ns
    body = getattr(msg, SERVICE)
    body.schemaVersion = 1
    body.decoderVersion = DECODER_VERSION
    body.sessionId = self.session_id
    body.sequence = self.sequence
    body.active = active
    body.reason = reason
    body.radarBus = self.decoder.bus
    body.lastRadarFrameMonoTime = self.decoder.last_frame_ns
    body.lastPositionMonoTime = self.decoder.last_position_ns
    body.observations = self.decoder.drain()
    body.counters = self.decoder.counts
    body.scope = scope
    body.calibrated = False
    body.processingTimeNs = self.processing_ns
    body.rejectedEnvelopes = self.rejected_envelopes
    body.backlogCount = self.backlog_count
    self.processing_ns = 0
    self.last_publish_ns, self.last_reason = now_ns, reason
    return msg


def main():
  # Imports with runtime side effects belong only in the managed process.
  from opendbc.car.structs import car
  from openpilot.cereal import custom
  from openpilot.common.params import Params

  params = Params()
  with car.CarParams.from_bytes(params.get('CarParams', block=True)) as cp, \
       custom.CarParamsSP.from_bytes(params.get('CarParamsSP', block=True)) as cp_sp:
    observer = HondaRadarObserver(cp, cp_sp)
    sm = messaging.SubMaster(['selfdriveState', 'longitudinalPlanSP', 'carControl'])
    pm = messaging.PubMaster([SERVICE])
    can_sock = messaging.sub_sock('can', timeout=20)
    settings = ObserverSettings.read(params)
    next_settings_ns = 0
    while True:
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
      now_ns = time.monotonic_ns()
      if now_ns >= next_settings_ns:
        settings = ObserverSettings.read(params)
        next_settings_ns = now_ns + HEARTBEAT_NS
      msg = observer.update(envelopes, sm, settings, now_ns)
      if msg is not None:
        pm.send(SERVICE, msg)


if __name__ == '__main__':
  main()
