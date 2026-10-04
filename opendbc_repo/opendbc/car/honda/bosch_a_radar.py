"""Receive-only Bosch-A observations. No RadarPoint, vehicle configuration or CAN TX.

Bit layout/conversion reference: 627c28695b14bfb4e91c9bf31ee7d6c567624a3c.
These are uncalibrated candidates, NOT qualified measurements for control. OEM
quality, lifecycle and velocity endpoint meanings still need vehicle evidence.
Raw CAN remains in rlog; frame references identify every byte used here.
"""
from collections import deque
from dataclasses import dataclass, field
import math

from opendbc.can.dbc import DBC
from opendbc.can.parser import get_raw_value


DBC_NAME = "honda_bosch_a_radar"
NUM_SLOTS = 16
# Initial observation/display bounds, not sensor specifications or control limits.
MAX_AGE_NS = 200_000_000
MAX_PENDING_PER_SLOT = 3
MAX_EVENTS = 64
MAIN_IDS = tuple(tuple((0x280 + 4 * s if s < 4 else 0x2D0 + 4 * (s - 4)) + f for f in range(4)) for s in range(NUM_SLOTS))
AUX_IDS = tuple(0x2C8 + s if s < 8 else 0x290 + s - 8 for s in range(NUM_SLOTS))
ADDRESS_MAP = {a: (s, f) for s in range(NUM_SLOTS) for f, a in enumerate((*MAIN_IDS[s], AUX_IDS[s]))}

# Canonical raw aliases only; other/unknown bits remain accessible in the CAN log.
RAW_SIGNALS = (
  {"statusRaw": "STATUS", "rangeRaw": "RANGE_RAW", "azimuthRaw": "AZIMUTH_RAW", "rangeSigmaRaw": "RANGE_SIGMA_RAW"},
  {"existenceRaw": "OBJECT_EXISTENCE_PROBABILITY_RAW", "angularWidthRaw": "ANGULAR_WIDTH_RAW"},
  {"lifecycleRaw": "LIFECYCLE_RAW", "normalizedClosingRaw": "NORMALIZED_CLOSING_RAW", "closingSigmaRaw": "NORMALIZED_CLOSING_SIGMA_RAW"},
  {"oemTrackId": "TRACK_ID", "edgeARaw": "AZIMUTH_EDGE_A_RAW", "edgeBRaw": "AZIMUTH_EDGE_B_RAW", "edgeSigmaARaw": "AZIMUTH_EDGE_SIGMA_A_RAW"},
  {"u11Raw": "REL_VELOCITY_RAW", "u10Raw": "REL_VELOCITY_UNCERTAINTY_RAW", "edgeSigmaBRaw": "AZIMUTH_EDGE_SIGMA_B_RAW", "ratioRaw": "RANGE_RATIO_RAW"},
)


@dataclass(frozen=True)
class RadarFrame:
  time_ns: int
  index: int
  address: int
  data: bytes
  values: dict


@dataclass
class Scan:
  index: int
  first_ns: int
  frames: dict[int, RadarFrame] = field(default_factory=dict)
  position_seq: int = 0
  velocity_seq: int = 0
  generation: int = 0
  rejected: bool = False


@dataclass
class Slot:
  scans: dict[int, Scan] = field(default_factory=dict)
  last_index: int | None = None
  last_life: int | None = None
  last_id: int | None = None
  last_position_ns: int = 0
  generation: int = 0
  expired: bool = True
  last_signature: tuple | None = None


class BoschARadarObserver:
  def __init__(self, bus: int):
    self.bus = bus
    self.slots = [Slot() for _ in range(NUM_SLOTS)]
    dbc = DBC(DBC_NAME)
    self.signals = {addr: (dbc.msgs[addr].sigs['FRAME_IDX'],
                          [(key, dbc.msgs[addr].sigs[name]) for key, name in RAW_SIGNALS[part].items()])
                    for addr, (_, part) in ADDRESS_MAP.items()}
    self.events: deque[dict] = deque()
    self.sequence = 0
    self.generation = 0
    self.last_frame_ns = 0
    self.last_position_ns = 0
    self.counts = dict(frames=0, badLength=0, duplicate=0, late=0, conflict=0, incomplete=0, expired=0, dropped=0)

  def _emit(self, event):
    if len(self.events) >= MAX_EVENTS:
      self.events.popleft()
      self.counts['dropped'] += 1
    self.events.append(event)

  def reset(self):
    """A scope/session discontinuity cannot inherit a previous track."""
    self.slots = [Slot() for _ in range(NUM_SLOTS)]
    self.events.clear()
    self.last_frame_ns = self.last_position_ns = 0

  def _event(self, slot, scan, reason):
    raw = {}
    for part in sorted(scan.frames):
      raw.update(scan.frames[part].values)
    frames = [dict(part=p, address=f.address, monoTime=f.time_ns, frameInEnvelope=f.index) for p, f in sorted(scan.frames.items())]
    return dict(slot=slot, frameIndex=scan.index, frameMask=sum(1 << p for p in scan.frames), generation=scan.generation,
                positionSeq=scan.position_seq, velocitySeq=scan.velocity_seq, positionMonoTime=0, velocityMonoTime=0,
                dRel=0.0, yRel=0.0, vRel=0.0, positionValid=False, velocityValid=False,
                reason=reason, raw=raw, frames=frames)

  def _publish_scan(self, slot_num, scan):
    """A late AUX may add velocity, but never creates a second position sample."""
    slot = self.slots[slot_num]
    if scan.rejected or not all(p in scan.frames for p in range(4)):
      return
    raw = {k: v for p in range(4) for k, v in scan.frames[p].values.items()}
    t_pos = scan.frames[0].time_ns
    if not scan.position_seq:
      signature = tuple(scan.frames[p].data for p in range(4))
      if signature == slot.last_signature:
        self.counts['duplicate'] += 1
        scan.rejected = True
        self._emit(self._event(slot_num, scan, 'duplicateScan'))
        return
      if slot.last_index is not None and not slot.expired:
        delta = (scan.index - slot.last_index) & 15
        if delta == 0 or delta > 8 or t_pos <= slot.last_position_ns:
          self.counts['late'] += 1
          scan.rejected = True
          self._emit(self._event(slot_num, scan, 'oldScan'))
          return
      valid = (raw['statusRaw'] != 15 and raw['rangeRaw'] != 4095 and raw['azimuthRaw'] != 2047
               and raw['lifecycleRaw'] != 4095 and 1 <= raw['oemTrackId'] <= 63 and raw['rangeRaw'] * 0.05712 - 3.0 > 0)
      if not valid:
        scan.rejected = True
        slot.expired = True
        self._emit(self._event(slot_num, scan, 'invalidGeometry'))
        return
      delta = (scan.index - slot.last_index) & 15 if slot.last_index is not None else 0
      continuous = (not slot.expired and slot.last_id == raw['oemTrackId'] and slot.last_life is not None
                    and ((raw['lifecycleRaw'] - slot.last_life) & 4095) == 2 * delta)
      if not continuous:
        self.generation += 1
        slot.generation = self.generation
      self.sequence += 1
      scan.position_seq = self.sequence
      scan.generation = slot.generation
      slot.last_index, slot.last_life, slot.last_id = scan.index, raw['lifecycleRaw'], raw['oemTrackId']
      slot.last_position_ns = t_pos
      slot.last_signature = signature
      slot.expired = False
      self.last_position_ns = max(self.last_position_ns, t_pos)
    event = self._event(slot_num, scan, 'positionOnly')
    event['positionMonoTime'] = t_pos
    event['positionValid'] = True
    event['dRel'] = raw['rangeRaw'] * 0.05712 - 3.0
    event['yRel'] = event['dRel'] * math.tan((raw['azimuthRaw'] - 1024) / 2048.0)
    if 4 in scan.frames:
      aux = scan.frames[4]
      event['velocityMonoTime'] = aux.time_ns
      # No ratio fallback; endpoints are retained but explicitly unqualified.
      candidate_valid = 0 < aux.values['u11Raw'] < 1728 and aux.values['u10Raw'] <= 511
      event['reason'] = 'uncalibrated' if candidate_valid else 'invalidVelocity'
      if candidate_valid:
        if not scan.velocity_seq:
          self.sequence += 1
          scan.velocity_seq = self.sequence
        event['velocitySeq'] = scan.velocity_seq
        event['velocityValid'] = True
        event['vRel'] = (aux.values['u11Raw'] - 864) / 64.0
    self._emit(event)

  def feed(self, time_ns: int, frames):
    """Process every occurrence in original envelope order; optional addresses only."""
    self.advance(time_ns)
    for index, (address, data, bus) in enumerate(frames):
      if bus != self.bus or address not in ADDRESS_MAP:
        continue
      self.counts['frames'] += 1
      if len(data) != 8:
        self.counts['badLength'] += 1
        continue
      if time_ns <= 0 or time_ns < self.last_frame_ns:
        self.counts['late'] += 1
        continue
      self.last_frame_ns = time_ns
      slot_num, part = ADDRESS_MAP[address]
      slot = self.slots[slot_num]
      idx_sig, signals = self.signals[address]
      idx = get_raw_value(data, idx_sig)
      scan = slot.scans.get(idx)
      if scan is not None and time_ns - scan.first_ns > MAX_AGE_NS:
        self._retire(slot_num, idx)
        scan = None
      if scan is None:
        if len(slot.scans) >= MAX_PENDING_PER_SLOT:
          self._retire(slot_num, min(slot.scans, key=lambda i: slot.scans[i].first_ns))
        scan = slot.scans[idx] = Scan(idx, time_ns)
      previous = scan.frames.get(part)
      incoming = RadarFrame(time_ns, index, address, bytes(data), {k: get_raw_value(data, s) for k, s in signals})
      if previous is not None:
        if previous.data == data:
          self.counts['duplicate'] += 1
        else:
          self.counts['conflict'] += 1
          scan.rejected = True
          scan.frames[part] = incoming
          self._emit(self._event(slot_num, scan, 'conflictingFrame'))
        continue
      scan.frames[part] = incoming
      if scan.position_seq and part != 4:
        continue
      self._publish_scan(slot_num, scan)

  def _retire(self, slot_num, idx):
    scan = self.slots[slot_num].scans.pop(idx)
    if not scan.position_seq and not scan.rejected:
      self.counts['incomplete'] += 1
      self._emit(self._event(slot_num, scan, 'incomplete'))

  def advance(self, now_ns):
    """Must be called on idle ticks too: absence of CAN cannot freeze validity."""
    for s, slot in enumerate(self.slots):
      for idx in list(slot.scans):
        if now_ns - slot.scans[idx].first_ns > MAX_AGE_NS:
          self._retire(s, idx)
      if not slot.expired and now_ns - slot.last_position_ns > MAX_AGE_NS:
        slot.expired = True
        self.counts['expired'] += 1
        self._emit(dict(slot=s, generation=slot.generation, positionMonoTime=slot.last_position_ns,
                        positionValid=False, velocityValid=False, reason='expired'))

  def drain(self):
    events = list(self.events)
    self.events.clear()
    return events
