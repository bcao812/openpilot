#!/usr/bin/env python3
import math
import time
import numpy as np
from collections import deque
from typing import Any

import capnp
from openpilot.cereal import messaging, log, custom
from opendbc.car.structs import car
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings, SERVICE as HONDA_RADAR_SERVICE, SCOPE_MAX_AGE_NS
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumption

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead

    # computed velocity and accelerations
    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    # Learn if constant acceleration
    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "present": True,
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than 0.75, are almost always glitches on toyota radars
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track]):
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This isn't exactly right, but it's a good heuristic
    return prob_d * prob_y * prob_v

  track = max(tracks.values(), key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 10) or (v_ego + track.vRel > 3)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float, lead_prob: float):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "modelProb": float(lead_prob),
    "present": True,
    "radar": False,
    "radarTrackId": -1,
  }


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, lead_prob: float, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
             low_speed_override: bool = True) -> dict[str, Any]:
  # Object membership follows the current model hypothesis. A historical
  # confidence must not keep an absent visual target or confirm a radar match.
  vision_qualified = ready and math.isfinite(lead_prob) and .5 < lead_prob <= 1.0
  if len(tracks) > 0 and vision_qualified:
    track = match_vision_to_track(v_ego, lead_msg, tracks)
  else:
    track = None

  lead_dict = {'present': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_prob)
    lead_dict = get_custom_yrel(CP, CP_SP, lead_dict, lead_msg)
  elif vision_qualified:
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego, lead_prob)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead_dict['present']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  return lead_dict


def get_custom_yrel(CP: structs.CarParams, CP_SP: structs.CarParamsSP, lead_dict: dict[str, Any],
                    lead_msg: capnp._DynamicStructReader) -> dict[str, Any]:
  if CP.brand == "hyundai" and (CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC or
                                CP.flags & (HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC)):
    lead_dict['yRel'] = float(-lead_msg.y[0])

  return lead_dict


class RadarD:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParams, delay: float = 0.0, *, params=None):
    self.CP = CP
    self.CP_SP = CP_SP

    self.current_time = 0.0
    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False
    self.radar_qualified_ns = None
    self.publication_expiry = RadarConsumption()

    self.ready = False
    self.params = params if params is not None else Params()
    self.radar_settings = ObserverSettings()
    self.settings_next_ns = 0
    self.fusion_control = None
    self.fusion_state = None
    self.control_scope_unavailable = False

  def control_enabled(self, sm, now_ns):
    # Unavailable feedback revokes current execution scope, but does not
    # establish a driver/configuration exit or clear an old obstacle.
    self.control_scope_unavailable = True
    if now_ns >= self.settings_next_ns:
      self.radar_settings = ObserverSettings.read(self.params)
      self.settings_next_ns = now_ns + 1_000_000_000
    if not self.radar_settings.available:
      return False
    if not self.radar_settings.control_requested(self.CP, self.CP_SP):
      self.control_scope_unavailable = False
      return False
    for service in ('selfdriveState', 'longitudinalPlanSP'):
      stamp = sm.logMonoTime.get(service, 0)
      if (not getattr(sm, 'seen', {}).get(service, False)
          or not 0 < stamp <= now_ns or now_ns - stamp > SCOPE_MAX_AGE_NS):
        return False
      if service == 'selfdriveState':
        if not sm.valid.get(service, False):
          return False
        if not sm[service].experimentalMode:
          self.control_scope_unavailable = False
          return False
    dec = sm['longitudinalPlanSP'].dec
    scope_version = int(getattr(dec, 'scopeVersion', 0))
    if scope_version == 0:
      # Older producers have no independent scope receipt.
      if not sm.valid.get('longitudinalPlanSP', False):
        return False
    elif scope_version == 1:
      source_ns = int(dec.scopeSourceMonoTime)
      if (not dec.scopeValid or not 0 < source_ns <= sm.logMonoTime['longitudinalPlanSP']
          or now_ns - source_ns > SCOPE_MAX_AGE_NS):
        return False
    else:
      return False
    self.control_scope_unavailable = False
    return bool(sm['selfdriveState'].experimentalMode and dec.enabled and dec.active)

  def update(self, sm: messaging.SubMaster, rr: car.RadarData, *, now_ns=None):
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    self.radar_qualified_ns = now_ns
    self.ready = sm.seen['modelV2']

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead)

    # *** publish radarState ***
    # Optional Honda inputs do not create a circular primary-health dependency.
    # The original vision path keeps publishing while perception/DEC starts.
    self.radar_state_valid = sm.all_checks(['modelV2', 'carState', 'radarTracks'])
    self.radar_state = log.RadarState.new_message()
    self.radar_state.fusionRiskUnresolved = False
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      self.radar_state.leadOne = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, leads_v3[0].prob,
                                          self.CP, self.CP_SP, low_speed_override=True)
      self.radar_state.leadTwo = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, leads_v3[1].prob,
                                          self.CP, self.CP_SP, low_speed_override=False)
    self.radar_state.visionLeadOne = self.radar_state.leadOne
    self.radar_state.visionLeadTwo = self.radar_state.leadTwo
    enabled = self.control_enabled(sm, now_ns)
    if enabled and self.fusion_control is None:
      from openpilot.selfdrive.car.honda_radar_control import HondaRadarControl
      self.fusion_control = HondaRadarControl()
    if self.fusion_control is not None:
      # Perception confirmation follows current sensor identity and geometry.
      # Driver actuation transitions are enforced by the planner and owner.
      if not enabled and self.control_scope_unavailable:
        # A missing/invalid scope receipt must not reset perception history
        # or masquerade as clear-road evidence. New SP producers certify
        # scope separately from MPC success; legacy producers still use
        # Event.valid. Retain history across unavailable input in either
        # protocol until fresh scope or an explicit configuration boundary.
        # Publish the original visual leads during this scope gap. These
        # diagnostics preserve perception history; they do not establish that
        # an object ever controlled the car or grant any new radar authority.
        # Existing obstacle-specific braking/HOLD belongs to its consumer.
        self.radar_state.fusionActive = False
        self.radar_state.fusionHealthy = False
        self.radar_state.fusionDegraded = True
        self.radar_state.fusionSessionId = self.fusion_control._session
        self.radar_state.fusionReason = 'controlScopeUnavailable'
        return
      observation_ns = sm.logMonoTime.get(HONDA_RADAR_SERVICE, 0)
      observation_valid = bool(getattr(sm, 'seen', {}).get(HONDA_RADAR_SERVICE, False)
                               and sm.valid.get(HONDA_RADAR_SERVICE, False))
      observation = sm[HONDA_RADAR_SERVICE] if observation_ns else None
      if observation is not None:
        # Record the actual invocation input before calling the optional
        # adapter. A rejected/disabled/failed invocation is still traceable;
        # this metadata grants no authority and does not replace its result.
        self.radar_state.fusionInputAttempted = True
        try:
          input_session = getattr(observation, 'sessionId', '')
          input_sequence = getattr(observation, 'sequence', 0)
          self.radar_state.fusionInputSessionId = input_session
          self.radar_state.fusionInputSequence = input_sequence
          self.radar_state.fusionInputMonoTime = observation_ns
        except Exception:
          # Optional diagnostic reads cannot bypass the adapter's own input
          # handling. Zero IDs mean this attempted input could not be recorded.
          self.radar_state.fusionInputMonoTime = 0
          self.radar_state.fusionInputSessionId = ''
          self.radar_state.fusionInputSequence = 0
      try:
        result = self.fusion_control.update(
          now_ns=now_ns, enabled=enabled, observation=observation, observation_ns=observation_ns,
          observation_valid=observation_valid, model=sm['modelV2'], model_ns=sm.logMonoTime['modelV2'], v_ego=self.v_ego,
          vision_leads=(self.radar_state.leadOne.to_dict(), self.radar_state.leadTwo.to_dict()),
        )
      except Exception:
        # Keep the original vision publication usable while exposing the
        # optional producer failure. UNIFIED retains its previous HOLD and
        # prevents a sensor exception from certifying clear-road acceleration.
        self.fusion_control = None
        self.radar_state.fusionHealthy = False
        self.radar_state.fusionDegraded = enabled
        self.radar_state.fusionReason = 'controlAdapterError'
        cloudlog.exception('Honda radar control adapter failed')
        return
      self.radar_state.leadOne, self.radar_state.leadTwo = result.leads
      # Ordinary fusion publishes only leadOne/leadTwo. Keep the historical
      # list field readable without exposing extra control candidates.
      self.radar_state.radarObjects = []
      self.radar_state.fusionActive = enabled
      self.radar_state.fusionHealthy = result.healthy
      self.radar_state.fusionDegraded = result.degraded
      self.radar_state.fusionReason = result.reason
      self.radar_state.fusionClearedTracks = getattr(result, 'cleared', ())
      self.radar_state.fusionSessionId = getattr(result, 'session_id', '')
      state = enabled, result.healthy, result.degraded, result.reason
      if state != self.fusion_state:
        cloudlog.event('honda_radar_control', enabled=enabled, healthy=result.healthy, degraded=result.degraded,
                       reason=result.reason, leads=sum(lead.present for lead in (self.radar_state.leadOne, self.radar_state.leadTwo)))
        self.fusion_state = state

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    radar_msg.radarState.radarObjects = []
    # Preserve the schema field for older readers without publishing old-ID risk.
    radar_msg.radarState.fusionRiskUnresolved = False
    view = self.publication_expiry.finalize_publication(
      self.radar_state, qualified_ns=self.radar_qualified_ns, publication_ns=radar_msg.logMonoTime, speed=self.v_ego,
    )
    if self.publication_expiry.expired_identities or self.publication_expiry.retired_identities:
      radar_msg.radarState.leadOne, radar_msg.radarState.leadTwo = view.leadOne.to_dict(), view.leadTwo.to_dict()
    if self.publication_expiry.expired_identities:
      cloudlog.event('honda_radar_publication_expiry', identities=self.publication_expiry.expired_identities,
                     qualifiedMonoTime=self.radar_qualified_ns, publicationMonoTime=radar_msg.logMonoTime,
                     degraded=self.publication_expiry.degraded)
    if self.publication_expiry.retired_identities:
      cloudlog.event('honda_radar_publication_retirement', identities=self.publication_expiry.retired_identities,
                     reason='futureEntryDisabled', qualifiedMonoTime=self.radar_qualified_ns,
                     publicationMonoTime=radar_msg.logMonoTime)
    if self.publication_expiry.degraded:
      radar_msg.radarState.fusionDegraded = True
      if radar_msg.radarState.fusionReason in ('active', 'noQualifiedObjects'):
        radar_msg.radarState.fusionReason = 'objectDegraded'
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  # Core 7, with modeld -- same reasoning as plannerd.py, which carries the full note.
  # radard also polls modelV2, so it moves for the same reason and must not be left behind on
  # core 5: radarState showed the same p95 overrun of its 50 ms period (61.98-69.57 ms p99)
  # from the same UI preemption.
  config_realtime_process(7, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("radard got CarParamsSP")

  # *** setup messaging
  optional_services = [HONDA_RADAR_SERVICE, 'selfdriveState', 'longitudinalPlanSP', 'carControl']
  sm = messaging.SubMaster(['modelV2', 'carState', 'radarTracks', *optional_services], poll='modelV2',
                          ignore_alive=optional_services, ignore_avg_freq=optional_services, ignore_valid=optional_services)
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP, CP_SP, CP.radarDelay, params=params)

  while 1:
    sm.update()

    RD.update(sm, sm['radarTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
