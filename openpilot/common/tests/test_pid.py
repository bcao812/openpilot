import math
from fractions import Fraction

import numpy as np
import pytest

from openpilot.common.pid import PIDController


def test_pid_terms_and_frozen_integrator():
  pid = PIDController(2., 1., .5, rate=10, pos_limit=10., neg_limit=-10.)
  assert pid.update(.3, error_rate=-.2, feedforward=.4) == pytest.approx(.93)
  assert (pid.p, pid.i, pid.d, pid.f) == pytest.approx((.6, .03, -.1, .4))
  assert pid.update(-.1, error_rate=.2, feedforward=.4, freeze_integrator=True) == pytest.approx(.33)
  assert pid.i == pytest.approx(.03)


def test_windup_recovery_and_dynamic_limits():
  pid = PIDController(0., 1., rate=10, pos_limit=1., neg_limit=-1.)
  for _ in range(100):
    assert pid.update(1., feedforward=1.) == 1.
  assert pid.i == 0.
  assert pid.update(-1., feedforward=1.) == pytest.approx(.9)
  assert pid.i == pytest.approx(-.1)
  pid.set_limits(.05, -.05)
  assert pid.update(5., freeze_integrator=True) == -.05
  assert pid.i == pytest.approx(-.1)
  pid.reset()
  assert (pid.p, pid.i, pid.d, pid.f, pid.control) == (0., 0., 0., 0., 0.)


def test_mutable_gain_tables_apply_immediately():
  proportional = ([0.], [1.])
  pid = PIDController(proportional, 0.)
  assert pid.update(.2, speed=5.) == pytest.approx(.2)
  proportional[1][0] = 2.
  assert pid.update(.2, speed=5.) == pytest.approx(.4)
  proportional[0].append(10.)
  proportional[1].append(4.)
  assert pid.update(.2, speed=5.) == pytest.approx(.6)
  pid._k_p = ([0.], [.5])
  assert pid.update(.2, speed=5.) == pytest.approx(.1)


@pytest.mark.parametrize("scalar", [float, np.float32, np.float64])
def test_special_values_and_limit_contract(scalar):
  pid = PIDController(1., 0., pos_limit=scalar(1.), neg_limit=scalar(-1.))
  assert pid.update(scalar(math.inf), freeze_integrator=True) == 1.
  assert pid.update(scalar(-math.inf), freeze_integrator=True) == -1.
  assert math.isnan(pid.update(scalar(math.nan), freeze_integrator=True))
  pid.set_limits(scalar(-.5), scalar(.5))
  assert pid.update(scalar(0.), freeze_integrator=True) == -.5
  pid.set_limits(scalar(-0.), scalar(-1.))
  assert math.copysign(1., pid.update(scalar(0.), freeze_integrator=True)) == 1.
  pid.set_limits(scalar(math.nan), scalar(-1.))
  assert math.isnan(pid.update(scalar(.5), freeze_integrator=True))


@pytest.mark.parametrize("kind", ["fraction", "object_scalar"])
@pytest.mark.parametrize("field", ["error", "feedforward"])
@pytest.mark.parametrize("freeze", [False, True])
def test_generic_scalar_input_contract(kind, field, freeze):
  value = Fraction(1, 4) if kind == "fraction" else np.array(.25, dtype=object)
  inputs = {"error": .25, "error_rate": .125, "feedforward": .25, "freeze_integrator": freeze}
  inputs[field] = value
  pid = PIDController(1., 1., 1.)
  output = pid.update(**inputs)
  integral = 0. if freeze else .0025
  assert output == pytest.approx(.625 + integral)
  assert (pid.p, pid.i, pid.d, pid.control) == pytest.approx((.25, integral, .125, .625 + integral))
  assert type(output) is np.float64
  assert type(pid.i) is (float if freeze else np.float64)
  assert pid.f is inputs["feedforward"]


def test_custom_limit_array_dispatch_is_preserved():
  class ClipLimit(np.ndarray):
    def __array_function__(self, function, types, args, kwargs):
      if function is np.clip:
        return np.float64(.125)
      return super().__array_function__(function, types, args, kwargs)

  upper = np.asarray(1.).view(ClipLimit)
  pid = PIDController(1., 0., pos_limit=upper, neg_limit=-1.)
  assert pid.update(.25, freeze_integrator=True) == .125
