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
# Reuse the shadow tracker's existing position-innovation hard caps for the
# additional raw-position check needed by the 4094 plateau exception. These
# are coarse discontinuity bounds, not calibrated noise or sensor limits.
CONTINUITY_MAX_DREL = 8.0
CONTINUITY_MAX_YREL = 3.0
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


def radar_scan_continuous(previous_index: int, previous_life: int, previous_ns: int,
                          index: int, life: int, stamp: int) -> bool:
  """Necessary sequence/time condition; callers still qualify object identity.

  Observed 4094 plateaus can continue across a progressing frame counter. This
  is compatibility with the recorded field behavior, not an OEM lifecycle
  definition. Same identity, active state, compatible geometry and absence of
  simultaneous slot conflicts must be checked separately by each caller.
  Repeated positions/AUX are never independent observations under this rule.
  """
  if not (0 <= previous_index < 16 and 0 <= index < 16 and 0 <= previous_life < 4095 and 0 <= life < 4095
          and 0 < previous_ns < stamp and stamp - previous_ns <= MAX_AGE_NS):
    return False
  delta = (index - previous_index) & 15
  return 1 <= delta <= 8 and (((life - previous_life) & 4095) == 2 * delta or previous_life == life == 4094)


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
  position_ns: int = 0
  velocity_seq: int = 0
  generation: int = 0
  rejection: str = ''
  reported_frames: int = 0


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
  last_position: tuple[float, float] | None = None
  identity_conflict: bool = False


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
    scan.reported_frames = len(scan.frames)
    # Keep the original position time even if a conflicting F0 replaces its raw
    # bytes. A late conflict invalidates that observation, not a newer scan.
    t_pos = scan.position_ns or (scan.frames[0].time_ns if 0 in scan.frames else 0)
    t_vel = scan.frames[4].time_ns if 4 in scan.frames else 0
    return dict(slot=slot, frameIndex=scan.index, frameMask=sum(1 << p for p in scan.frames), generation=scan.generation,
                positionSeq=scan.position_seq, velocitySeq=scan.velocity_seq, positionMonoTime=t_pos, velocityMonoTime=t_vel,
                dRel=0.0, yRel=0.0, vRel=0.0, positionValid=False, velocityValid=False,
                reason=reason, raw=raw, frames=frames)

  def _publish_scan(self, slot_num, scan):
    """A late AUX may add velocity, but never creates a second position sample."""
    slot = self.slots[slot_num]
    if not all(p in scan.frames for p in range(4)):
      return
    if scan.rejection:
      # Empty/invalid slots carry useful AUX quality and sentinel evidence too.
      # They stay invalid; only the raw snapshot and frame references are filled.
      self._emit(self._event(slot_num, scan, scan.rejection))
      return
    raw = {k: v for p in range(4) for k, v in scan.frames[p].values.items()}
    t_pos = scan.frames[0].time_ns
    if not scan.position_seq:
      signature = tuple(scan.frames[p].data for p in range(4))
      if signature == slot.last_signature:
        self.counts['duplicate'] += 1
        scan.rejection = 'duplicateScan'
        self._emit(self._event(slot_num, scan, 'duplicateScan'))
        return
      if slot.last_index is not None and not slot.expired:
        delta = (scan.index - slot.last_index) & 15
        if delta == 0 or delta > 8 or t_pos <= slot.last_position_ns:
          self.counts['late'] += 1
          scan.rejection = 'oldScan'
          self._emit(self._event(slot_num, scan, 'oldScan'))
          return
      distance = raw['rangeRaw'] * 0.05712 - 3.0
      valid = (raw['statusRaw'] != 15 and raw['rangeRaw'] != 4095 and raw['azimuthRaw'] != 2047
               and raw['lifecycleRaw'] != 4095 and 1 <= raw['oemTrackId'] <= 63 and distance > 0)
      if not valid:
        scan.rejection = 'invalidGeometry'
        slot.expired = True
        self._emit(self._event(slot_num, scan, 'invalidGeometry'))
        return
      lateral = distance * math.tan((raw['azimuthRaw'] - 1024) / 2048.0)
      # Preserve both raw observations. Two live slots with this OEM ID in the
      # same frame cannot establish identity continuity. The preceding frame
      # in another slot alone is a normal migration, not a simultaneous claim.
      identity_conflict = False
      for other in self.slots:
        if (other is not slot and not other.expired and other.last_id == raw['oemTrackId']
            and other.last_index == scan.index and abs(t_pos - other.last_position_ns) <= MAX_AGE_NS):
          identity_conflict = other.identity_conflict = True
      plateau = slot.last_life == raw['lifecycleRaw'] == 4094
      geometry_continuous = (not plateau or (slot.last_position is not None
                             and abs(distance - slot.last_position[0]) <= CONTINUITY_MAX_DREL
                             and abs(lateral - slot.last_position[1]) <= CONTINUITY_MAX_YREL))
      continuous = (not slot.expired and not slot.identity_conflict and not identity_conflict
                    and slot.last_id == raw['oemTrackId'] and slot.last_index is not None and slot.last_life is not None
                    and geometry_continuous and radar_scan_continuous(slot.last_index, slot.last_life, slot.last_position_ns,
                                                                     scan.index, raw['lifecycleRaw'], t_pos))
      if not continuous:
        self.generation += 1
        slot.generation = self.generation
      self.sequence += 1
      scan.position_seq = self.sequence
      scan.position_ns = t_pos
      scan.generation = slot.generation
      slot.last_index, slot.last_life, slot.last_id = scan.index, raw['lifecycleRaw'], raw['oemTrackId']
      slot.last_position_ns = t_pos
      slot.last_signature = signature
      slot.last_position = distance, lateral
      slot.identity_conflict = identity_conflict
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
          scan.rejection = 'conflictingFrame'
          scan.frames[part] = incoming
          if scan.position_seq and scan.position_ns == slot.last_position_ns:
            slot.expired = True
          self._emit(self._event(slot_num, scan, 'conflictingFrame'))
        continue
      scan.frames[part] = incoming
      if scan.position_seq and part != 4:
        continue
      self._publish_scan(slot_num, scan)

  def _retire(self, slot_num, idx):
    scan = self.slots[slot_num].scans.pop(idx)
    if not scan.position_seq and not scan.rejection:
      self.counts['incomplete'] += 1
      self._emit(self._event(slot_num, scan, 'incomplete'))
    elif scan.rejection and len(scan.frames) > scan.reported_frames:
      self._emit(self._event(slot_num, scan, scan.rejection))

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
