"""Wire and native-parameter contracts used by Blendful's model/UI boundary."""

from openpilot.cereal import custom
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.common.test import OpenpilotTestCase


class TestBlendfulContract(OpenpilotTestCase):
  def test_default_off_and_preference_survives_process_and_drive_transitions(self):
    params = Params()
    self.assertEqual(params.get_type('Blendful'), ParamKeyType.BOOL)
    self.assertIs(params.get('Blendful', return_default=True), False)
    self.assertIn(b'Blendful', params.all_keys(ParamKeyFlag.BACKUP))
    self.assertIn(b'Blendful', params.all_keys(ParamKeyFlag.PERSISTENT))
    for enabled in (True, False):
      params.put_bool('Blendful', enabled, block=True)
      for transition in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                         ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
        params.clear_all(transition)
        self.assertIs(Params().get_bool('Blendful'), enabled)

  def test_diagnostics_preserve_existing_ordinals_and_roundtrip(self):
    fields = ('laneTurnDirection', 'leftLaneChangeEdgeBlock', 'rightLaneChangeEdgeBlock',
              'blendfulCurvature', 'blendfulTrust', 'modelCurvature', 'blendfulGoingAround',
              'blendfulLeftProb', 'blendfulRightProb', 'blendfulLaneWidth')
    for ordinal, name in enumerate(fields):
      self.assertEqual(custom.ModelDataV2SP.schema.fields[name].proto.ordinal.explicit, ordinal)
    values = dict(laneTurnDirection='turnLeft', leftLaneChangeEdgeBlock=True, rightLaneChangeEdgeBlock=False,
                  blendfulCurvature=0.000125, blendfulTrust=0.75, modelCurvature=-0.002,
                  blendfulGoingAround=True, blendfulLeftProb=0.875, blendfulRightProb=0.625, blendfulLaneWidth=3.5)
    message = custom.ModelDataV2SP.new_message(**values)
    with custom.ModelDataV2SP.from_bytes(message.to_bytes()) as decoded:
      self.assertEqual(decoded.to_dict(), message.to_dict())
    legacy = custom.ModelDataV2SP.new_message(laneTurnDirection='turnRight', rightLaneChangeEdgeBlock=True)
    with custom.ModelDataV2SP.from_bytes(legacy.to_bytes()) as decoded:
      self.assertEqual(decoded.laneTurnDirection, 'turnRight')
      self.assertTrue(decoded.rightLaneChangeEdgeBlock)
      self.assertEqual(decoded.blendfulCurvature, 0.0)
      self.assertEqual(decoded.blendfulTrust, 0.0)

  def test_diagnostics_are_logged_in_full_and_decimated_logs(self):
    service = SERVICE_LIST['modelDataV2SP']
    self.assertTrue(service.should_log)
    self.assertEqual(service.frequency, 20.0)
    self.assertEqual(service.decimation, 10)
