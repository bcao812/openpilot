"""Feedback-state contracts; these tests do not claim vehicle-response fidelity."""

import math
from collections import deque

import pytest

from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.launch_response import LaunchResponse, ResponseDomain


INSTANCE_NS = 1_000_000_000
START_NS = 2_000_000_000
STEP_NS = 10_000_000


class FeedbackHarness:
  def __init__(self):
    self.pid = PIDController(0.0, 0.0, neg_limit=-3.5, pos_limit=2.0)
    self.response = LaunchResponse(self.pid, max_gap_ns=100_000_000)
    self.now_ns = START_NS

  def release(self, *, arm=True):
    self.now_ns += STEP_NS
    self.response.freeze_release(INSTANCE_NS, self.now_ns, arm=arm)

  def step(self, *, base=0.25, target=0.25, measured=0.0, upper=2.0, forward=True, final=None, elapsed_ns=STEP_NS,
           rolling=False, positive_allowed=False, response_domain=ResponseDomain.MODEL):
    self.now_ns += elapsed_ns
    proposed = self.response.propose(base, target, measured, upper, producer_instance_ns=INSTANCE_NS,
                                     now_ns=self.now_ns, forward=forward, rolling=rolling, positive_allowed=positive_allowed,
                                     response_domain=response_domain)
    self.response.confirm_output(proposed if final is None else final)
    return proposed

  def approach(self, *, base=-0.5, upper=2.0, elapsed_ns=STEP_NS):
    self.now_ns += elapsed_ns
    output = self.response.freeze_braking(base, upper, producer_instance_ns=INSTANCE_NS, now_ns=self.now_ns)
    self.response.confirm_output(output)
    return output


@pytest.mark.parametrize('forward', [False, True])
def test_tracking_reference_cannot_create_a_launch_session(forward):
  harness = FeedbackHarness()
  for _ in range(100):
    assert harness.step(forward=forward) == 0.25
  assert not harness.response.armed
  assert harness.response.bias == 0.0


def test_repeated_release_does_not_learn_from_stationary_acceleration():
  harness = FeedbackHarness()
  for _ in range(100):
    harness.release()
    assert harness.response.armed
    assert harness.response.bias == 0.0
  harness.step()
  assert 0.0 < harness.response.bias < 0.01


def test_matching_acceleration_has_no_extra_effort_and_excess_acceleration_unwinds():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    assert harness.step(measured=0.25) == 0.25
  for _ in range(50):
    harness.step()
  assert harness.response.bias > 0.1
  previous = harness.response.bias
  for _ in range(100):
    harness.step(measured=0.75)
    assert 0.0 <= harness.response.bias <= previous
    previous = harness.response.bias
  assert harness.response.bias == 0.0


def test_elapsed_time_not_number_of_plan_or_control_calls_sets_error_accumulation():
  fast, slow = FeedbackHarness(), FeedbackHarness()
  fast.release()
  slow.release()
  for _ in range(50):
    fast.step()
  for _ in range(10):
    slow.step(elapsed_ns=5 * STEP_NS)
  assert fast.response.bias == pytest.approx(slow.response.bias)


def test_upper_saturation_does_not_store_future_effort_when_limit_is_restored():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(500):
    harness.step(upper=0.3)
    assert harness.response.bias <= 0.05 + 1e-12
  before = harness.response.bias
  restored = harness.step(upper=2.0)
  assert 0.25 + before <= restored <= 0.25 + before + 0.003


def test_final_rate_limited_output_is_used_even_when_requested_limit_allows_more():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(500):
    assert harness.step(measured=-4.0, final=0.251) > 0.251
    assert harness.response.bias == pytest.approx(0.001)
  restored = harness.step(measured=0.25)
  assert restored == pytest.approx(0.251)


def test_falling_external_limit_removes_stored_bias_before_recovery():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(100):
    harness.step()
  assert harness.response.bias > 0.2
  harness.step(final=0.1)
  assert harness.response.bias == 0.0
  assert harness.step(measured=0.25) == 0.25


def test_moving_handoff_retains_then_unwinds_but_cannot_ratchet_up():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    harness.step()
  learned = harness.response.bias
  for _ in range(100):
    assert harness.step(forward=False, measured=0.0) == pytest.approx(0.25 + learned)
    assert harness.response.bias == pytest.approx(learned)
  for _ in range(100):
    harness.step(forward=False, measured=0.75)
  assert harness.response.bias == 0.0
  for _ in range(100):
    assert harness.step(forward=False) == 0.25


def test_unconfirmed_proposal_and_duplicate_confirmation_do_not_accumulate():
  harness = FeedbackHarness()
  harness.release()
  proposed = harness.response.propose(0.25, 0.25, 0.0, 2.0, producer_instance_ns=INSTANCE_NS,
                                      now_ns=harness.now_ns + STEP_NS, forward=True, response_domain=ResponseDomain.MODEL)
  assert proposed > 0.25
  assert harness.response.bias == 0.0
  harness.response.confirm_output(0.25)
  harness.response.confirm_output(proposed)
  assert harness.response.bias == 0.0


@pytest.mark.parametrize('clock', ['duplicate', 'backward', 'gap', 'noninteger'])
def test_bad_clock_discards_session_before_extra_effort(clock):
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    harness.step()
  now = {'duplicate': harness.now_ns, 'backward': harness.now_ns - 1, 'gap': harness.now_ns + 100_000_001,
         'noninteger': float(harness.now_ns + STEP_NS)}[clock]
  assert harness.response.propose(0.25, 0.25, 0.0, 2.0, producer_instance_ns=INSTANCE_NS, now_ns=now,
                                  forward=True, response_domain=ResponseDomain.MODEL) == 0.25
  assert not harness.response.armed and harness.response.bias == 0.0


@pytest.mark.parametrize('instance', [0, True, INSTANCE_NS - 1, INSTANCE_NS + 1])
def test_another_instance_cannot_inherit_tracking_correction(instance):
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    harness.step()
  assert harness.response.propose(0.25, 0.25, 0.0, 2.0, producer_instance_ns=instance,
                                  now_ns=harness.now_ns + STEP_NS, forward=True, response_domain=ResponseDomain.MODEL) == 0.25
  assert not harness.response.armed and harness.response.bias == 0.0


@pytest.mark.parametrize('measured', [math.nan, math.inf, -math.inf, True])
def test_bad_measurement_cannot_increase_effort_or_poison_later_sessions(measured):
  harness = FeedbackHarness()
  harness.release()
  assert harness.step(measured=measured) == 0.25
  assert not harness.response.armed and harness.response.bias == 0.0
  harness.release()
  assert harness.step() > 0.25


def test_negative_target_or_invalid_final_output_removes_positive_correction():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    harness.step()
  assert harness.step(base=-0.5, target=-0.5) == -0.5
  assert not harness.response.armed and harness.response.bias == 0.0
  harness.release()
  harness.step(final=math.nan)
  assert not harness.response.armed and harness.response.bias == 0.0


@pytest.mark.parametrize('forward', [False, True])
def test_rolling_without_departure_permission_cannot_learn_positive_correction(forward):
  harness = FeedbackHarness()
  for _ in range(100):
    assert harness.step(rolling=True, forward=forward) == 0.25
    assert harness.pid.i == 0.0
  # Model motion is a preference, so it cannot acquire braking correction
  # from excess measured acceleration either.
  for _ in range(100):
    assert harness.step(base=0.0, target=0.0, measured=0.25, rolling=True, forward=forward) == 0.0
  assert harness.pid.i == 0.0


@pytest.mark.parametrize('measured', [-0.25, 0.25])
def test_missing_response_domain_cannot_infer_acc_from_measurement(measured):
  harness = FeedbackHarness()
  for _ in range(100):
    assert harness.step(base=0.0, target=0.0, measured=measured, rolling=True, response_domain=ResponseDomain.NONE) == 0.0
  assert harness.pid.i == 0.0


@pytest.mark.parametrize('measured', [-0.25, 0.25])
def test_explicit_absolute_acc_learns_both_signs_without_departure_permission(measured):
  harness = FeedbackHarness()
  for _ in range(100):
    harness.step(base=0.0, target=0.0, measured=measured, rolling=True, forward=False, response_domain=ResponseDomain.ABSOLUTE_ACC)
  assert harness.pid.i * measured < 0.0
  assert harness.step(base=0.0, target=0.0, measured=measured, rolling=True, forward=False) == 0.0
  assert harness.pid.i == 0.0  # model preference inherits neither ACC sign


@pytest.mark.parametrize('response_domain', [ResponseDomain.ABSOLUTE_ACC, ResponseDomain.LEAD_BRAKING])
def test_explicit_rolling_domains_do_not_create_feedback_near_rest(response_domain):
  harness = FeedbackHarness()
  for _ in range(100):
    assert harness.step(base=-0.25, target=-0.25, measured=0.0, response_domain=response_domain) == -0.25
  assert harness.pid.i == 0.0


def test_lead_domain_retires_positive_effort_and_ends_with_its_negative_target():
  harness = FeedbackHarness()
  for _ in range(100):
    harness.step(rolling=True, response_domain=ResponseDomain.ABSOLUTE_ACC)
  assert harness.pid.i > 0.0
  for _ in range(100):
    assert harness.step(base=-0.5, target=-0.5, rolling=True, response_domain=ResponseDomain.LEAD_BRAKING) < -0.5
    assert harness.pid.i < 0.0
  for target in (0.0, 0.25):
    assert harness.step(base=target, target=target, rolling=True, response_domain=ResponseDomain.LEAD_BRAKING) == target
    assert harness.pid.i == 0.0


def test_departure_permission_is_independent_of_reference_kind_and_confirmed_output():
  harness = FeedbackHarness()
  for _ in range(100):
    assert harness.step(rolling=True, forward=False, positive_allowed=True) == 0.25
  for _ in range(100):
    harness.step(rolling=True, positive_allowed=True)
  learned = harness.pid.i
  assert learned > 0.0
  # A continuing FORWARD may learn; MOVING may keep/unwind, never ratchet up.
  harness.step(rolling=True, positive_allowed=False)
  assert harness.pid.i > learned
  learned = harness.pid.i
  for _ in range(100):
    harness.step(rolling=True, forward=False)
    assert harness.pid.i == pytest.approx(learned)


def test_launch_to_rolling_changes_gain_once_without_jumping_the_single_integral():
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    harness.step()
  learned = harness.pid.i
  assert harness.response.armed
  assert harness.step(rolling=True, measured=0.25) == pytest.approx(0.25 + learned)
  assert not harness.response.armed
  assert harness.pid.i == pytest.approx(learned)
  for rolling in (False, True, False, True):
    harness.step(rolling=rolling, forward=False)
    assert harness.pid.i == pytest.approx(learned)
  harness.step(rolling=True)
  assert harness.pid.i - learned == pytest.approx(0.001)
  assert harness.response.bias == harness.pid.i


@pytest.mark.parametrize('base', [-4.0, 3.0])
def test_raw_saturated_feedforward_cannot_hide_or_manufacture_correction(base):
  harness = FeedbackHarness()
  for _ in range(200):
    harness.step(base=0.0, target=0.0, measured=math.copysign(0.3, -base), rolling=True, positive_allowed=True,
                 response_domain=ResponseDomain.ABSOLUTE_ACC)
  assert harness.pid.i != 0.0
  output = harness.step(base=base, target=base, measured=base, rolling=True, response_domain=ResponseDomain.ABSOLUTE_ACC)
  assert output == (-3.5 if base < 0 else 2.0)
  assert harness.pid.i == 0.0
  assert harness.step(base=0.0, target=0.0, measured=0.0, rolling=True) == 0.0


def braking_feedback():
  harness = FeedbackHarness()
  for _ in range(1000):
    harness.step(base=0.0, target=0.0, measured=0.6, rolling=True, forward=False, response_domain=ResponseDomain.ABSOLUTE_ACC)
  assert harness.pid.i < -2.0
  return harness


def test_approach_freezes_only_delivered_braking_and_keeps_its_clock_current():
  harness = braking_feedback()
  assert harness.approach(base=-3.0) == -3.5
  assert harness.pid.i == pytest.approx(-0.5)
  for _ in range(40):
    assert harness.approach(elapsed_ns=50_000_000) == pytest.approx(-1.0)
    assert harness.pid.i == pytest.approx(-0.5)
  harness.step(base=0.0, target=0.0, measured=0.25, rolling=True, response_domain=ResponseDomain.ABSOLUTE_ACC)
  assert harness.pid.i == pytest.approx(-0.501)  # only this 10 ms TRACK sample
  harness.pid.neg_limit = -0.2
  assert harness.approach(base=0.0) == -0.2
  harness.pid.neg_limit = -3.5
  assert harness.approach(base=0.0) == -0.2


@pytest.mark.parametrize('fault', ['instance', 'clock', 'backward', 'gap'])
def test_approach_cannot_freeze_braking_across_invalid_identity_or_time(fault):
  harness = braking_feedback()
  instance = INSTANCE_NS + 1 if fault == 'instance' else INSTANCE_NS
  timestamp = {'instance': harness.now_ns + STEP_NS, 'clock': harness.now_ns,
               'backward': harness.now_ns - 1, 'gap': harness.now_ns + 100_000_001}[fault]
  output = harness.response.freeze_braking(-0.5, 2.0, producer_instance_ns=instance, now_ns=timestamp)
  harness.response.confirm_output(output)
  assert output == -0.5
  assert harness.pid.i == 0.0
  assert not harness.response.armed


@pytest.mark.parametrize('delay,response_time', [(0.3, 0.2), (0.5, 0.4), (1.0, 0.8)])
@pytest.mark.parametrize('offset', [-0.25, 0.0, 0.25])
@pytest.mark.parametrize('response_domain', [ResponseDomain.MODEL, ResponseDomain.ABSOLUTE_ACC, ResponseDomain.LEAD_BRAKING])
def test_fixed_reference_delayed_response_respects_domain_through_disturbance_reversal(delay, response_time, offset, response_domain):
  """One fixed permission domain; unsupported correction signs stay forbidden.

  Offsets, delay/lag, reversal and convergence tolerance are unchanged. This
  does not claim either one-sided domain rejects both signs of disturbance.
  """
  harness = FeedbackHarness()
  dt = STEP_NS / 1e9
  commands = deque([0.0] * round(delay / dt))
  measured = 0.0
  target = -0.5 if response_domain == ResponseDomain.LEAD_BRAKING else 0.0
  for disturbance in (offset, 0.0, -offset):
    tail = deque(maxlen=100)
    for _ in range(2500):
      command = harness.step(base=target, target=target, measured=measured, rolling=True, forward=True,
                             positive_allowed=True, response_domain=response_domain)
      commands.append(command)
      measured += (commands.popleft() + disturbance - measured) * (1.0 - math.exp(-dt / response_time))
      tail.append(measured)
      assert -3.5 <= command <= 2.0
      assert harness.response.bias == harness.pid.i
      if response_domain == ResponseDomain.MODEL:
        assert harness.pid.i >= 0.0
      elif response_domain == ResponseDomain.LEAD_BRAKING:
        assert harness.pid.i <= 0.0
    expected_correction = -disturbance
    if response_domain == ResponseDomain.MODEL:
      expected_correction = max(expected_correction, 0.0)
    elif response_domain == ResponseDomain.LEAD_BRAKING:
      expected_correction = min(expected_correction, 0.0)
    assert max(abs(value - target - disturbance - expected_correction) for value in tail) < 0.005
    assert harness.pid.i == pytest.approx(expected_correction, abs=0.005)
