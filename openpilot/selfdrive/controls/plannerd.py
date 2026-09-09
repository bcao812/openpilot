#!/usr/bin/env python3
from openpilot.cereal import custom
from opendbc.car.structs import car
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.sunnypilot.common.phase_timer import PhaseTimer
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace, TRACE_MARKER
import openpilot.cereal.messaging as messaging


# plannerd's loop. sm.update() blocks on modelV2 at 20 Hz, so wall there is the model's cadence.
# long_update is the MPC; it is the phase this process exists for.
PHASE_TIMER_MARKER = "/data/plannerd_profile"
PHASE_TIMER_EVENT = "plannerd.phase_timing"
PHASE_TIMER_PERIOD = 600         # 30 s at 20 Hz
PHASES = ("sm_update", "sla_buttons", "long_update", "long_publish", "ldw_publish")


def main():
  # Core 7, with modeld, not core 5 with the UI.
  #
  # plannerd polls modelV2 (below), which modeld publishes from core 7, so this puts the wake on
  # the core that already holds the message -- no cross-core IPI and no cold cache per frame.
  # modeld runs FIFO 54 against CTRL_LOW's 51, so it still cannot be delayed by this, and it
  # leaves the core free while it waits on the GPU: measured 19.8% CPU for a 35-42 ms inference.
  #
  # What this moves AWAY from matters more. On core 5 the UI sits at CTRL_HIGH (ui.py), above
  # CTRL_LOW, so every UI frame preempted the longitudinal planner. Measured over 11 routes,
  # longitudinalPlan missed its 50 ms period at p95 (61.7-67.0 ms p99) while modelV2 did not
  # (56.9 ms p99) -- the delay was scheduling, not the model.
  #
  # Upstream comma puts the UI above plannerd on purpose (`# above plannerd and radard` in
  # ui.py). This fork takes the other side of that trade. Lowering the UI instead was the
  # obvious alternative and is NOT safe here: the UI has ~4 ms of margin against a 16.67 ms
  # vblank and falls to a latched 30 FPS when it misses, so starving it has a visible cliff.
  # Moving the planner costs the UI nothing.
  config_realtime_process(7, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  cloudlog.info("plannerd is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("plannerd got CarParamsSP")

  gps_location_service = get_gps_location_service(params)
  ignore_services = ["liveMapDataSP", "carStateSP", "selfdriveStateSP", gps_location_service]

  ldw = LaneDepartureWarning()
  trace = LongitudinalTrace.create(CP)
  if trace is not None:
    cloudlog.warning("plannerd: longitudinal trace ENABLED by %s", TRACE_MARKER)
  longitudinal_planner = LongitudinalPlanner(CP, CP_SP, trace=trace)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'longitudinalPlanSP'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'vehicleParameters', 'radarState', 'modelV2', 'selfdriveState',
                            'liveMapDataSP', 'carStateSP', 'selfdriveStateSP', gps_location_service],
                           poll='modelV2', ignore_alive=ignore_services, ignore_avg_freq=ignore_services, ignore_valid=ignore_services)

  # None unless /data/plannerd_profile exists at process start. plannerd is FIFO 51 on core 7
  # with modeld; the writer drops to SCHED_OTHER and moves to the little cluster.
  prof = PhaseTimer.create(PHASE_TIMER_MARKER, PHASE_TIMER_EVENT, PHASES, PHASE_TIMER_PERIOD)
  if prof:
    prof.start()
    cloudlog.warning("plannerd: phase timing ENABLED by %s", PHASE_TIMER_MARKER)

  while True:
    if prof:
      prof.begin()
    sm.update()
    if prof:
      prof.mark("sm_update")
    longitudinal_planner.sla.update_buttons(sm['selfdriveStateSP'].buttonsReleaseToggle)
    if prof:
      prof.mark("sla_buttons")
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm)
      if prof:
        prof.mark("long_update")
      longitudinal_planner.publish(sm, pm)
      if prof:
        prof.mark("long_publish")

      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks()
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)
    # ldw and the driverAssistance publish are one phase: both are gated on modelV2 and the
    # publish is three field writes. loop_rest also absorbs the frames where modelV2 did not
    # arrive, which is 2 of every 3 at 60 Hz sm.update -- those are the CONDITIONAL frames.
    if prof:
      prof.mark("ldw_publish")
      prof.end_frame()


if __name__ == "__main__":
  main()
