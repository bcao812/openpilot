import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.tests import test_blendful_integration as frame_loop_fixture


class TestModelStartupScheduler(unittest.TestCase):
  def setUp(self):
    self.events = []
    self.enterContext(mock.patch.object(modeld.gc, 'disable', side_effect=lambda: self.events.append(('gc',))))
    self.enterContext(mock.patch.object(modeld, 'drop_realtime', side_effect=lambda: self.events.append(('ordinary',))))
    self.enterContext(mock.patch.object(modeld, 'set_core_affinity', side_effect=lambda cores: self.events.append(('affinity', cores))))
    self.enterContext(mock.patch.object(modeld, 'config_realtime_process',
                                       side_effect=lambda core, priority: self.events.append(('fifo', core, priority))))
    self.scheduler = modeld._ModelStartupScheduler()
    self.model = SimpleNamespace(name='model')

  def test_initialization_keeps_placement_and_gc_without_fifo(self):
    self.assertEqual(self.events, [('gc',), ('ordinary',), ('affinity', [7])])
    self.assertFalse(self.scheduler._realtime_enabled)

  def test_preparation_only_outputs_do_not_enable_fifo(self):
    for _ in range(3):
      self.scheduler.prepare(self.model)
      self.scheduler.complete(self.model, None)
    self.assertEqual(self.events, [('gc',), ('ordinary',), ('affinity', [7])])
    self.assertEqual(self.scheduler._ready_models, [])

  def test_first_output_enables_fifo_once_and_preserves_output(self):
    output = {'position': np.arange(6, dtype=np.float32)}
    original = output['position'].copy()
    for _ in range(3):
      self.scheduler.prepare(self.model)
      self.scheduler.complete(self.model, output)
    self.assertEqual(self.events, [('gc',), ('ordinary',), ('affinity', [7]), ('fifo', 7, 54)])
    self.assertEqual(self.scheduler._ready_models, [self.model])
    self.assertEqual(set(output), {'position'})
    np.testing.assert_array_equal(output['position'], original)

  def test_cold_fallback_stays_ordinary_until_its_own_output(self):
    self.scheduler.prepare(self.model)
    self.scheduler.complete(self.model, {})
    self.events.clear()
    fallback = SimpleNamespace(name='fallback')
    self.scheduler.prepare(fallback)
    self.scheduler.complete(fallback, None)
    self.scheduler.prepare(fallback)
    self.assertEqual(self.events, [('ordinary',)])
    self.assertFalse(self.scheduler._realtime_enabled)
    self.scheduler.complete(fallback, {})
    self.scheduler.prepare(fallback)
    self.scheduler.complete(fallback, {})
    self.assertEqual(self.events, [('ordinary',), ('fifo', 7, 54)])

  def test_ready_instance_can_be_reselected_without_cold_initialization(self):
    self.scheduler.prepare(self.model)
    self.scheduler.complete(self.model, {})
    self.events.clear()
    self.scheduler.prepare(SimpleNamespace(name='cold'))
    self.scheduler.prepare(self.model)
    self.assertEqual(self.events, [('ordinary',), ('fifo', 7, 54)])
    self.assertTrue(self.scheduler._realtime_enabled)

  def test_readiness_belongs_to_the_model_instance(self):
    self.scheduler.prepare(self.model)
    self.scheduler.complete(self.model, {})
    same_attributes = SimpleNamespace(name='model')
    self.assertEqual(self.model, same_attributes)
    self.events.clear()
    self.scheduler.prepare(same_attributes)
    self.scheduler.complete(same_attributes, None)
    self.assertEqual(self.events, [('ordinary',)])


class TestModelStartupSchedulingInMain(OpenpilotTestCase):
  run_frames = frame_loop_fixture.TestBlendfulFrameLoop.run_frames

  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.object(modeld.gc, 'disable'))
    self.enterContext(mock.patch.object(modeld, 'drop_realtime'))
    self.enterContext(mock.patch.object(modeld, 'set_core_affinity'))
    self.events = []
    scheduler_type = modeld._ModelStartupScheduler

    def create_scheduler():
      scheduler = scheduler_type()
      prepare, complete = scheduler.prepare, scheduler.complete

      def record_prepare(model):
        before = scheduler._realtime_enabled
        prepare(model)
        self.events.append(('prepare', model, before, scheduler._realtime_enabled))

      def record_complete(model, output):
        before = scheduler._realtime_enabled
        complete(model, output)
        self.events.append(('complete', model, output, before, scheduler._realtime_enabled))

      scheduler.prepare, scheduler.complete = record_prepare, record_complete
      return scheduler

    self.enterContext(mock.patch.object(modeld, '_ModelStartupScheduler', side_effect=create_scheduler))

  def test_main_waits_for_real_output_then_keeps_steady_state(self):
    published, outputs, _, _, small = self.run_frames([{'missing_output': True}, {}, {}])
    prepares = [row for row in self.events if row[0] == 'prepare']
    completed = [row for row in self.events if row[0] == 'complete']
    self.assertEqual([row[2:] for row in prepares], [(False, False), (False, False), (True, True)])
    self.assertTrue(all(row[1] is small for row in completed))
    self.assertEqual([row[3:] for row in completed], [(False, False), (False, True), (True, True)])
    self.assertIsNone(completed[0][2])
    self.assertIs(completed[1][2], outputs[1])
    self.assertIs(completed[2][2], outputs[2])
    self.assertEqual(set(published), {1, 2})

  def test_main_drops_fifo_for_a_cold_fallback_after_an_exception(self):
    published, outputs, _, large, small = self.run_frames([{}, {'fail_big': True}, {'missing_output': True}, {}], chestnut=True)
    prepared_small = [row for row in self.events if row[0] == 'prepare' and row[1] is small]
    self.assertEqual([row[2:] for row in prepared_small], [(True, False), (False, False), (False, False)])
    completed = [row for row in self.events if row[0] == 'complete']
    self.assertIs(completed[0][1], large)
    self.assertIs(completed[0][2], outputs[0])
    self.assertEqual(completed[0][3:], (False, True))
    self.assertTrue(all(row[1] is small for row in completed[1:]))
    self.assertEqual([row[3:] for row in completed[1:]], [(False, False), (False, False), (False, True)])
    self.assertIsNone(completed[1][2])
    self.assertIsNone(completed[2][2])
    self.assertIs(completed[3][2], outputs[3])
    self.assertEqual(set(published), {0, 3})
