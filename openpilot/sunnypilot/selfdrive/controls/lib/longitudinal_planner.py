"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from dataclasses import dataclass
import math

from openpilot.cereal import messaging, custom
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.car.honda_radar_observer_config import SCOPE_MAX_AGE_NS
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


@dataclass(frozen=True)
class CruiseTarget:
  source: object
  speed: float
  acceleration: float


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP, mpc):
    self.events_sp = EventsSP()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl(hold_entry_speed=resolve_longitudinal_mode(CP, CP_SP) == LongitudinalMode.UNIFIED)
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.
    self.cruise_target = CruiseTarget(self.source, 0.0, 0.0)

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> CruiseTarget:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(sm)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.events_sp)

    targets = {
      LongitudinalPlanSource.cruise: (v_cruise, a_ego),
      LongitudinalPlanSource.sccVision: (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      LongitudinalPlanSource.speedLimitAssist: (self.sla.output_v_target, self.sla.output_a_target),
    }

    # An active curve guard must not silently become unrestricted cruise
    # because comparisons with NaN happen to select another dictionary item.
    # UNIFIED explicitly rejects this invalid policy at its existing boundary.
    if (self.scc.vision.hold_entry_speed and self.scc.vision.is_active
        and not all(math.isfinite(value) for value in targets[LongitudinalPlanSource.sccVision])):
      self.source = LongitudinalPlanSource.sccVision
      self.cruise_target = CruiseTarget(self.source, float('nan'), float('nan'))
      self.output_v_target = float('nan')
      return self.cruise_target

    self.source = min(targets, key=lambda k: targets[k][0])
    speed, acceleration = targets[self.source]
    self.cruise_target = CruiseTarget(self.source, float(speed), float(acceleration))
    self.output_v_target = speed
    return self.cruise_target

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    self.dec.update(sm)
    self.e2e_alerts_helper.update(sm, self.events_sp)

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster, parent_plan_mono_time=0,
                                  *, legacy_acceleration: float | None = None) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = (sm.all_checks(service_list=['carState', 'controlsState'])
                         and (not getattr(self, 'unified', False) or self.motion_intent.valid or self.reference_planner.standby))

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanMonoTime = parent_plan_mono_time
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    # UNIFIED publishes its policy target; LEGACY explicitly supplies the final
    # reference acceleration used by this field before mode convergence.
    longitudinalPlanSP.aTarget = float(self.cruise_target.acceleration if legacy_acceleration is None else legacy_acceleration)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control
    dec = longitudinalPlanSP.dec
    dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
    dec.enabled = self.dec.enabled()
    dec.active = self.dec.active()
    # The radar execution scope depends on healthy DEC inputs, not whether
    # the preceding MPC found a solution. Keep Event.valid's plan contract
    # above, and carry the oldest input epoch so re-publication cannot extend
    # this independent receipt's lifetime at the radar consumer.
    scope_services = ['carState', 'controlsState', 'selfdriveState']
    scope_times = {service: sm.logMonoTime.get(service, 0) for service in scope_services}
    dec.scopeVersion = 1
    dec.scopeSourceMonoTime = min(scope_times.values())
    dec.scopeValid = bool(sm.all_checks(service_list=scope_services) and all(
      getattr(sm, 'seen', {}).get(service, False) and sm.valid.get(service, False)
      and 0 < stamp <= plan_sp_send.logMonoTime and plan_sp_send.logMonoTime - stamp <= SCOPE_MAX_AGE_NS
      for service, stamp in scope_times.items()
    ))

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = self.scc.vision.state
    sccVision.vTarget = float(self.scc.vision.output_v_target)
    sccVision.aTarget = float(self.scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(self.scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(self.scc.vision.max_pred_lat_acc)
    sccVision.enabled = self.scc.vision.is_enabled
    sccVision.active = self.scc.vision.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(self.resolver.speed_limit)
    resolver.speedLimitLast = float(self.resolver.speed_limit_last)
    resolver.speedLimitFinal = float(self.resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(self.resolver.speed_limit_final_last)
    resolver.speedLimitValid = self.resolver.speed_limit_valid
    resolver.speedLimitLastValid = self.resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(self.resolver.speed_limit_offset)
    resolver.source = self.resolver.source
    assist = speedLimit.assist
    assist.state = self.sla.state
    assist.enabled = self.sla.is_enabled
    assist.active = self.sla.is_active
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    pm.send('longitudinalPlanSP', plan_sp_send)
