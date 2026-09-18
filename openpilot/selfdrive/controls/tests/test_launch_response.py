"""Feedback-state contracts; these tests do not claim vehicle-response fidelity."""

import math

import pytest

from openpilot.selfdrive.controls.lib.launch_response import LaunchResponse


INSTANCE_NS = 1_000_000_000
START_NS = 2_000_000_000
STEP_NS = 10_000_000


class FeedbackHarness:
  def __init__(self):
    self.response = LaunchResponse(max_gap_ns=100_000_000)
    self.now_ns = START_NS

  def release(self, *, arm=True):
    self.now_ns += STEP_NS
    self.response.freeze_release(INSTANCE_NS, self.now_ns, arm=arm)

  def step(self, *, base=0.25, target=0.25, measured=0.0, upper=2.0, forward=True, final=None, elapsed_ns=STEP_NS):
    self.now_ns += elapsed_ns
    proposed = self.response.propose(base, target, measured, upper, producer_instance_ns=INSTANCE_NS,
                                     now_ns=self.now_ns, forward=forward)
    self.response.confirm_output(proposed if final is None else final)
    return proposed


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
                                      now_ns=harness.now_ns + STEP_NS, forward=True)
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
  assert harness.response.propose(0.25, 0.25, 0.0, 2.0, producer_instance_ns=INSTANCE_NS, now_ns=now, forward=True) == 0.25
  assert not harness.response.armed and harness.response.bias == 0.0


@pytest.mark.parametrize('instance', [0, True, INSTANCE_NS - 1, INSTANCE_NS + 1])
def test_another_instance_cannot_inherit_tracking_correction(instance):
  harness = FeedbackHarness()
  harness.release()
  for _ in range(50):
    harness.step()
  assert harness.response.propose(0.25, 0.25, 0.0, 2.0, producer_instance_ns=instance,
                                  now_ns=harness.now_ns + STEP_NS, forward=True) == 0.25
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
