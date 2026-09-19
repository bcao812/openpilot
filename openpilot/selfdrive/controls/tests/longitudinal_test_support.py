"""Shared fixtures, enabled only by explicit import in the requesting test module."""

import pytest

from openpilot.common.prefix import OpenpilotPrefix


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield
