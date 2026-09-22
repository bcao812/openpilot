import numpy as np
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.parse_model_outputs import Parser as StockParser
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser, _infer_mhp, sigmoid, softmax


class TestParseModelOutputs(OpenpilotTestCase):
  @staticmethod
  def tagged_lead_predictions(n_hypotheses=3):
    batch, slot, step, column = np.indices((2, n_hypotheses, 6, 4), dtype=np.float32)
    # Every batch, trajectory, timestep and x/y/v/a column has its own label.
    means = 0.25 + 1000 * batch + 100 * slot + 10 * step + column
    stds = 0.25 + 2 * batch + 0.2 * slot + 0.02 * step + 0.003 * column
    return means, stds

  def test_infer_mhp_lead(self):
    in_hypotheses, out_selections = _infer_mhp(102, 24)
    assert in_hypotheses == 2
    assert out_selections == 3

  def test_infer_mhp_plan(self):
    in_hypotheses, out_selections = _infer_mhp(4955, 495)
    assert in_hypotheses == 5
    assert out_selections == 1

  def test_infer_mhp_non_mdn(self):
    in_hypotheses, out_selections = _infer_mhp(48, 24)
    assert in_hypotheses == 1
    assert out_selections == 0

  def test_check_missing_raises(self):
    parser = Parser(ignore_missing=False)
    with self.assertRaises(ValueError):
      parser.check_missing({}, "missing_key")

  def test_check_missing_ignored(self):
    parser = Parser(ignore_missing=True)
    assert parser.check_missing({}, "missing_key") is True

  def test_binary_crossentropy(self):
    parser = Parser()
    raw_logits = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)
    outs = {"meta": raw_logits.copy()}
    parser.parse_binary_crossentropy("meta", outs)
    expected_probabilities = sigmoid(raw_logits)
    np.testing.assert_allclose(outs["meta"], expected_probabilities, rtol=1e-5, atol=1e-6)

  def test_categorical_crossentropy(self):
    parser = Parser()
    raw_logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    outs = {"desire_state": raw_logits.copy()}
    parser.parse_categorical_crossentropy("desire_state", outs)
    expected_probabilities = softmax(raw_logits)
    np.testing.assert_allclose(outs["desire_state"], expected_probabilities, rtol=1e-5, atol=1e-6)

  def test_parse_vision_outputs(self):
    parser = Parser()
    pose_raw = np.zeros((1, ModelConstants.POSE_WIDTH * 2), dtype=np.float32)
    road_transform_raw = np.zeros((1, ModelConstants.POSE_WIDTH * 2), dtype=np.float32)
    lead_raw = np.zeros((1, 102), dtype=np.float32)
    meta_raw = np.zeros((1, 55), dtype=np.float32)
    vision_outputs = {"pose": pose_raw, "road_transform": road_transform_raw, "lead": lead_raw, "meta": meta_raw}
    parsed = parser.parse_vision_outputs(vision_outputs)
    assert "pose" in parsed
    assert "road_transform" in parsed
    assert "lead" in parsed
    assert "meta" in parsed
    assert parsed["pose"].shape == (1, ModelConstants.POSE_WIDTH)
    assert parsed["lead"].shape == (1, ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH)

  def test_parse_policy_outputs(self):
    parser = Parser()
    plan_raw = np.zeros((1, 4955), dtype=np.float32)
    desire_state_raw = np.zeros((1, ModelConstants.DESIRE_PRED_WIDTH), dtype=np.float32)
    action_raw = np.zeros((1, ModelConstants.ACTION_WIDTH * 2), dtype=np.float32)
    policy_outputs = {"plan": plan_raw, "desire_state": desire_state_raw, "action": action_raw}
    parsed = parser.parse_policy_outputs(policy_outputs)
    assert parsed["plan"].shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert parsed["action"].shape == (1, ModelConstants.ACTION_WIDTH)
    assert parsed["desire_state"].shape == (1, ModelConstants.DESIRE_PRED_WIDTH)

  def test_parse_outputs_combined(self):
    parser = Parser()
    outputs = {"plan": np.zeros((1, 4955), dtype=np.float32), "pose": np.zeros((1, ModelConstants.POSE_WIDTH * 2),
               dtype=np.float32), "meta": np.zeros((1, 55), dtype=np.float32)}
    parsed = parser.parse_outputs(outputs)
    assert parsed["plan"].shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert parsed["pose"].shape == (1, ModelConstants.POSE_WIDTH)
    assert parsed["meta"].shape == (1, 55)

  def test_non_mdn_lead_layout_matches_stock(self):
    means, stds = self.tagged_lead_predictions()
    raw = np.concatenate((means.reshape(2, 72), np.log(stds).reshape(2, 72)), axis=1)
    logits = np.array([[-2.4, 0.7, 2.1], [1.2, -1.7, 0.35]], dtype=np.float32)
    probabilities = 1 / (1 + np.exp(-logits))

    for parser in (StockParser(ignore_missing=True), Parser()):
      with self.subTest(parser=type(parser).__module__):
        parsed = parser.parse_vision_outputs({"lead": raw.copy(), "lead_prob": logits.copy()})
        # Compare both implementations with independent, labelled values: a
        # shape-only check would miss the original mean/std and slot mix-up.
        np.testing.assert_array_equal(parsed["lead"], means)
        np.testing.assert_allclose(parsed["lead_stds"], stds, rtol=1e-6)
        np.testing.assert_allclose(parsed["lead_prob"], probabilities, rtol=1e-6)
        assert "lead_weights" not in parsed
        assert "lead_hypotheses" not in parsed

  def test_non_mdn_lead_split_and_combined_with_other_heads(self):
    lead_means, lead_stds = self.tagged_lead_predictions()
    vision = {"lead": np.concatenate((lead_means.reshape(2, 72), np.log(lead_stds).reshape(2, 72)), axis=1)}
    policy = {}
    expected = {"lead": lead_means, "lead_stds": lead_stds}
    head_shapes = {
      "pose": (6,), "road_transform": (6,), "wide_from_device_euler": (3,),
      "lane_lines": (4, 33, 2), "road_edges": (2, 33, 2),
      "plan": (33, 15), "action": (2,), "desired_curvature": (1,),
    }
    for index, (name, shape) in enumerate(head_shapes.items()):
      values = np.arange(2 * int(np.prod(shape)), dtype=np.float32).reshape((2,) + shape)
      means, stds = index + 0.25 + values / 100, 0.5 + values / 2000
      expected[name], expected[name + "_stds"] = means, stds
      target = policy if name in ("plan", "action", "desired_curvature") else vision
      target[name] = np.concatenate((means.reshape(2, -1), np.log(stds).reshape(2, -1)), axis=1)

    for combined in (False, True):
      with self.subTest(combined=combined):
        parser = Parser()
        vision_inputs = {name: value.copy() for name, value in vision.items()}
        policy_inputs = {name: value.copy() for name, value in policy.items()}
        if combined:
          parsed = parser.parse_outputs({**vision_inputs, **policy_inputs})
        else:
          parsed = parser.parse_vision_outputs(vision_inputs)
          parsed.update(parser.parse_policy_outputs(policy_inputs))
        for name, value in expected.items():
          np.testing.assert_allclose(parsed[name], value, rtol=1e-6, err_msg=name)

  def test_weighted_lead_layouts_preserve_selections(self):
    cases = (
      (2, [[[3.0, -2.0, 1.5], [-0.5, 2.5, 0.3]],
           [[-1.0, 2.0, -0.8], [1.2, -0.4, 2.8]]], [[0, 1, 0], [1, 0, 1]]),
      (5, [[[0.5, 4.0, -1.0], [-2.0, 0.2, 0.0], [1.0, -1.0, 1.5], [3.0, 0.8, -0.2], [-0.7, 1.1, 4.2]],
           [[-1.0, 0.2, 0.5], [4.2, 0.5, -0.4], [0.3, -2.0, 3.1], [1.0, 0.7, 1.2], [2.0, 4.0, 1.8]]],
       [[3, 0, 4], [1, 4, 2]]),
    )
    for n_hypotheses, weight_logits, winners in cases:
      means, stds = self.tagged_lead_predictions(n_hypotheses)
      weight_logits = np.array(weight_logits, dtype=np.float32)
      raw = np.concatenate((means.reshape(2, n_hypotheses, 24), np.log(stds).reshape(2, n_hypotheses, 24),
                            weight_logits), axis=2).reshape(2, -1)
      expected_weights = np.exp(weight_logits) / np.exp(weight_logits).sum(axis=1, keepdims=True)
      selected = np.arange(2)[:, None], np.array(winners)
      lead_logits = np.array([[-2.4, 0.7, 2.1], [1.2, -1.7, 0.35]], dtype=np.float32)
      for entrypoint in ("parse_vision_outputs", "parse_outputs"):
        with self.subTest(slice_size=raw.shape[1], entrypoint=entrypoint):
          parsed = getattr(Parser(), entrypoint)({"lead": raw.copy(), "lead_prob": lead_logits.copy()})
          np.testing.assert_array_equal(parsed["lead"], means[selected])
          np.testing.assert_allclose(parsed["lead_stds"], stds[selected], rtol=1e-6)
          np.testing.assert_array_equal(parsed["lead_hypotheses"], means)
          np.testing.assert_allclose(parsed["lead_stds_hypotheses"], stds, rtol=1e-6)
          np.testing.assert_allclose(parsed["lead_weights"], expected_weights, rtol=1e-6)
          np.testing.assert_allclose(parsed["lead_prob"], 1 / (1 + np.exp(-lead_logits)), rtol=1e-6)

  def test_non_mdn_lead_publishing_and_vision_radar_fallback(self):
    from opendbc.car.structs import CarParams, CarParamsSP
    from openpilot.cereal import log, messaging
    from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA, get_lead
    from openpilot.sunnypilot.modeld_v2.constants import Meta
    from openpilot.sunnypilot.modeld_v2.fill_model_msg import PublishState, fill_model_msg

    slot, step = np.indices((3, 6), dtype=np.float32)
    means = np.stack((40 + 15 * slot + 1.3 * step, -0.25 + 0.3 * slot + 0.02 * step,
                      18 + 3 * slot + 0.15 * step, -0.5 - 0.2 * slot - 0.03 * step), axis=-1)[None]
    _, all_stds = self.tagged_lead_predictions()
    stds = all_stds[:1]
    raw_outputs = {"lead": np.concatenate((means.reshape(1, 72), np.log(stds).reshape(1, 72)), axis=1)}

    # Supply the other outputs required by the real publisher, with a monotonic
    # plan so lane/edge time interpolation also follows its normal path.
    plan = np.zeros((1, 33, 15), dtype=np.float32)
    model_v_ego, v_ego = 22.0, 24.0
    plan[0, :, 0] = model_v_ego * np.array(ModelConstants.T_IDXS)
    plan[0, :, 3] = model_v_ego
    for name, values in (("plan", plan), ("lane_lines", np.zeros((1, 4, 33, 2), dtype=np.float32)),
                         ("road_edges", np.zeros((1, 2, 33, 2), dtype=np.float32))):
      flat = values.reshape(1, -1)
      raw_outputs[name] = np.concatenate((flat, np.full_like(flat, np.log(0.5))), axis=1)
    for name, width in (("lane_lines_prob", 8), ("meta", 55), ("desire_state", 8), ("desire_pred", 32)):
      raw_outputs[name] = np.zeros((1, width), dtype=np.float32)
    raw_outputs["raw_pred"] = np.zeros((1, 1), dtype=np.float32)
    CP = CarParams.new_message(brand="honda", radarUnavailable=True, openpilotLongitudinalControl=True)
    CP_SP = CarParamsSP()

    for probabilities in ((0.92, 0.73, 0.12), (0.11, 0.87, 0.66), (0.83, 0.17, 0.58)):
      with self.subTest(probabilities=probabilities):
        probabilities = np.array(probabilities, dtype=np.float32)
        inputs = {name: values.copy() for name, values in raw_outputs.items()}
        inputs["lead_prob"] = np.log(probabilities / (1 - probabilities))[None]
        parsed = Parser().parse_outputs(inputs)
        base = messaging.new_message("drivingModelData")
        extended = messaging.new_message("modelV2")
        action = log.ModelDataV2.Action.new_message(desiredCurvature=0.001, desiredAcceleration=0.25, shouldStop=False)
        fill_model_msg(base, extended, parsed, action, PublishState(), vipc_frame_id=40, vipc_frame_id_extra=40,
                       frame_id=40, frame_drop=0.0, timestamp_eof=1, model_execution_time=0.01, valid=True, model_meta=Meta)
        assert base.valid and extended.valid

        # Exercise actual schema serialization and the consumer's reader type;
        # no copy of the publisher's lead-assignment code is used here.
        with log.Event.from_bytes(extended.to_bytes()) as event:
          leads = event.modelV2.leadsV3
          assert len(leads) == 3
          for slot, lead in enumerate(leads):
            for column, field in enumerate(("x", "y", "v", "a")):
              np.testing.assert_array_equal(getattr(lead, field), means[0, slot, :, column])
              np.testing.assert_allclose(getattr(lead, field + "Std"), stds[0, slot, :, column], rtol=1e-6)
            np.testing.assert_array_equal(lead.t, [0, 2, 4, 6, 8, 10])
            self.assertAlmostEqual(lead.prob, probabilities[slot], places=6)
            assert lead.probTime == (0, 2, 4)[slot]

          # RadarD consumes the first two slots. Empty tracks must preserve the
          # vision probability/ready gates independently for each slot.
          for slot in range(2):
            lead = leads[slot]
            for ready in (False, True):
              state = get_lead(v_ego, ready, {}, lead, model_v_ego, lead.prob, CP, CP_SP)
              if not ready or probabilities[slot] <= 0.5:
                assert state == {"present": False}
                continue
              assert state["present"] and not state["radar"]
              assert state["radarTrackId"] == -1
              x, y, velocity, acceleration = means[0, slot, 0]
              expected = {"dRel": x - RADAR_TO_CAMERA, "yRel": -y, "vRel": velocity - model_v_ego,
                          "vLead": v_ego + velocity - model_v_ego, "vLeadK": v_ego + velocity - model_v_ego,
                          "aLeadK": acceleration, "aLeadTau": 0.3, "modelProb": probabilities[slot]}
              for field, value in expected.items():
                self.assertAlmostEqual(state[field], value, places=5, msg=f"slot {slot}: {field}")
