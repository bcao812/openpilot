"""Qualify the optional lane correction at the actual lateral consumer."""

import math
import time


MODEL_MAX_AGE_NS = 250_000_000
CAR_STATE_MAX_AGE_NS = 100_000_000
MIN_SPEED = 5.0
MIN_LINE_PROB = 0.3


class BlendfulConsumer:
  def __init__(self):
    self.epoch = 0
    self.model_mono_time = 0
    self.curvature = 0.0
    self._last_epoch = 0
    self._eligible = False
    self._input_outage = False
    self._model_ns = 0
    self._capture_ns = 0
    self._capture_high_water = 0
    self._frame_qualified = False

  def _new_epoch(self, now_ns: int) -> None:
    self.epoch = max(now_ns, self._last_epoch + 1)
    self._last_epoch = self.epoch
    self._model_ns = self._capture_ns = self._capture_high_water = 0
    self._frame_qualified = False

  @staticmethod
  def _healthy(sm, service: str, now_ns: int, max_age_ns: int) -> bool:
    try:
      if not all(getattr(sm, field, {}).get(service, False) for field in ('seen', 'valid', 'alive', 'freq_ok')):
        return False
      stamp = int(sm.logMonoTime[service])
      return 0 < stamp <= now_ns and now_ns - stamp <= max_age_ns
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
      return False

  def update(self, sm, *, enabled: bool, lat_active: bool, model_path: bool = True, now_ns: int | None = None) -> float:
    """Return the pre-clip correction; the nominal model message is never modified."""
    self.model_mono_time = 0
    self.curvature = 0.0
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    try:
      cs = sm['carState']
      eligible = (enabled and lat_active and model_path and now_ns > 0 and cs.canValid and
                  math.isfinite(cs.vEgo) and cs.vEgo >= MIN_SPEED and
                  self._healthy(sm, 'carState', now_ns, CAR_STATE_MAX_AGE_NS))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
      eligible = False
    if not eligible:
      self._eligible = False
      self._input_outage = False
      self.epoch = 0
      return 0.0

    new_session = not self._eligible
    if new_session:
      self._new_epoch(now_ns)
    self._eligible = True

    try:
      model = sm['modelV2']
      capture_ns = int(model.timestampEof)
      inputs_healthy = (all(self._healthy(sm, service, now_ns, MODEL_MAX_AGE_NS) for service in ('modelV2', 'modelDataV2SP')) and
                        0 < capture_ns <= now_ns and now_ns - capture_ns <= MODEL_MAX_AGE_NS)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
      inputs_healthy = False
    if not inputs_healthy:
      # One reset per complete outage. Pairing gaps, an old capture before
      # this epoch, and a producer still awaiting our epoch do not renew it.
      if not self._input_outage and not new_session:
        self._new_epoch(now_ns)
      self._input_outage = True
      return 0.0
    self._input_outage = False

    try:
      model_ns = int(sm.logMonoTime['modelV2'])
      if model_ns > self._model_ns:
        self._frame_qualified = capture_ns >= self.epoch and capture_ns > self._capture_high_water
        self._model_ns, self._capture_ns = model_ns, capture_ns
        self._capture_high_water = max(self._capture_high_water, capture_ns)
      elif model_ns == self._model_ns and capture_ns != self._capture_ns:
        # A changed payload at one Event identity cannot create a new frame.
        self._frame_qualified = False
      if model_ns != self._model_ns or not self._frame_qualified:
        return 0.0

      output = sm['modelDataV2SP']
      if (model_ns != int(sm.logMonoTime['modelDataV2SP']) or
          int(output.blendfulControlEpoch) != self.epoch):
        return 0.0

      correction, trust, left_prob, right_prob, lane_width, model_curvature = (
        float(output.blendfulCurvature), float(output.blendfulTrust), float(output.blendfulLeftProb),
        float(output.blendfulRightProb), float(output.blendfulLaneWidth), float(output.modelCurvature))
      if (not all(math.isfinite(value) for value in (correction, trust, left_prob, right_prob, lane_width, model_curvature)) or
          not 0.0 < trust <= 1.0 or not 0.0 <= left_prob <= 1.0 or not 0.0 <= right_prob <= 1.0 or
          left_prob + right_prob <= 0.0 or lane_width <= 0.0):
        return 0.0
      # The paired producer validates current geometry and does the fit at
      # model rate. A filtered probability cannot authorize a missing side.
      for index, probability in ((1, left_prob), (2, right_prob)):
        if probability > 0.0:
          raw_probability = float(model.laneLineProbs[index])
          if not math.isfinite(raw_probability) or not MIN_LINE_PROB <= raw_probability <= 1.0:
            return 0.0

      self.model_mono_time = model_ns
      self.curvature = correction
      return correction
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
      # Optional/old schemas and malformed diagnostics have no authority. Keep
      # the requested epoch stable so a compatible producer can acknowledge it.
      return 0.0
