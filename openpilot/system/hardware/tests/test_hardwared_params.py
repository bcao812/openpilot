from unittest.mock import Mock, call

import pytest

from openpilot.common import params as params_module
from openpilot.common.params import Params
from openpilot.system.hardware.hardwared import BoolParamPublisher


@pytest.fixture(params=['GithubRunnerSufficientVoltage', 'NetworkMetered'])
def key(request):
  return request.param


@pytest.fixture
def params(tmp_path):
  return Params(str(tmp_path))


@pytest.fixture
def publisher(params):
  return BoolParamPublisher(params)


@pytest.fixture
def writer(monkeypatch, params):
  real_put_bool = params.put_bool
  # Complete normal writes immediately; the pending-write case below controls its queue.
  writer = Mock(side_effect=lambda key, value: real_put_bool(key, value, block=True))
  monkeypatch.setattr(params, 'put_bool', writer)
  return writer


@pytest.mark.parametrize('value', [False, True])
@pytest.mark.parametrize('existing', [None, False, True])
def test_first_publish_and_stable_values(params, publisher, writer, key, value, existing):
  if existing is not None:
    params.put(key, existing, block=True)
  for _ in range(120):
    publisher.put(key, value)
  writer.assert_called_once_with(key, value)
  assert params.get(key) is value


def test_state_changes_and_keys_are_independent(params, publisher, writer):
  keys = ('GithubRunnerSufficientVoltage', 'NetworkMetered')
  for value in (False, False, True, True, False):
    for key in keys:
      publisher.put(key, value)
      assert params.get(key) is value
  assert writer.call_args_list == [call(key, value) for value in (False, True, False) for key in keys]


@pytest.mark.parametrize('value', [False, True])
@pytest.mark.parametrize('clear', [False, True])
def test_external_changes_are_repaired(params, publisher, writer, key, value, clear):
  publisher.put(key, value)
  if clear:
    params.remove(key)
  else:
    params.put(key, not value, block=True)
  publisher.put(key, value)
  publisher.put(key, value)
  assert params.get(key) is value
  assert writer.call_args_list == [call(key, value)] * 2


@pytest.mark.parametrize('value', [False, True])
def test_silent_write_failure_is_retried(params, publisher, writer, key, value, monkeypatch):
  params.put(key, not value, block=True)
  real_put_bool = params_module.params_put_bool
  failed_put = Mock(return_value=-1)
  monkeypatch.setattr(params_module, 'params_put_bool', failed_put)
  for _ in range(2):
    publisher.put(key, value)
    assert params.get(key) is not value
  assert failed_put.call_count == 2

  monkeypatch.setattr(params_module, 'params_put_bool', real_put_bool)
  publisher.put(key, value)
  publisher.put(key, value)
  assert params.get(key) is value
  assert writer.call_args_list == [call(key, value)] * 3


def test_pending_writes_preserve_reversals(params, publisher, writer, key):
  params.put(key, False, block=True)
  pending = []
  writer.side_effect = lambda key, value: pending.append((key, value))
  for value in (False, True, False):
    publisher.put(key, value)
  # The last False must be queued even though storage is still False from before True.
  assert pending == [(key, False), (key, True), (key, False)]
  for queued_key, value in pending:
    params.put(queued_key, value, block=True)
  publisher.put(key, False)
  assert writer.call_count == 3
  assert params.get(key) is False
