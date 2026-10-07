"""Shared loss relevance for published radar objects and their consumers."""
import math


HORIZON = 10.0


def remaining_occupancy_window(start, end, elapsed, *, horizon=HORIZON):
  """Age a known exit without turning the forecast boundary into an exit.

  Callers separately enforce the measurement and path evidence deadlines.
  An endpoint exactly at the forecast horizon means that no clearing event
  was established; keeping that meaning does not renew any evidence clock.
  A genuinely finite endpoint still expires, even if other quality flags
  describe it as limited.
  """
  return max(0.0, start - elapsed), horizon if end == horizon else end - elapsed


def loss_relevant(*, distance, ground_speed, ego_speed, path_state, conflict_start, conflict_end,
                  lateral_speed, horizon=HORIZON):
  """Use an already qualified hypothesis to decide whether its loss matters.

  This is the producer's loss-handling domain, not a sensor detection cut or
  permission to discard a currently qualified obstacle from the joint MPC.
  """
  near = distance <= max(15.0, 2.0 * ego_speed + ego_speed * ego_speed / 6.0)
  closing = ego_speed - ground_speed
  eta = distance / closing if closing > .1 else math.inf
  crossing = path_state == 'entering' or conflict_end < horizon or abs(lateral_speed) >= .5
  conflict = crossing and conflict_start <= eta + .5 and eta - .5 <= conflict_end
  return near and path_state == 'inPath' or conflict and path_state in ('inPath', 'entering')
