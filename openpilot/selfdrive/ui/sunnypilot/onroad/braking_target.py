"""Show object constraints explicitly acknowledged by the execution owner.

Strength is the owner's qualitative, normalized accepted request, not an
individual object's share of braking, confidence, or measured deceleration.
Other braking sources can be active while this object-only panel is absent.
"""
import time

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_control_influence import accepted_influences, influence_rgb
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached


class BrakingTargetRenderer:
  MAX_ROWS = 2
  ROW_HEIGHT = 76

  def __init__(self):
    self._font = gui_app.font(FontWeight.SEMI_BOLD)

  def render(self, rect: rl.Rectangle, sm) -> None:
    # Revalidate every draw, including when all detection geometry is cached.
    # Exact identities alone deduplicate receipts; nearby targets remain distinct.
    targets = {}
    for item in accepted_influences(sm, time.monotonic_ns()):
      if item.kind not in ('braking', 'hold'):
        continue
      identity = ('radar', item.session_id, item.track_id) if item.session_id else ('vision', item.model_ns, item.vision_index)
      prior = targets.get(identity)
      if prior is None or (item.strength, item.kind == 'hold') > (prior.strength, prior.kind == 'hold'):
        targets[identity] = item
    if not targets:
      return

    rows = tuple(targets.values())[:self.MAX_ROWS]
    x, y, width = rect.x + 60, rect.y + 320, min(500, rect.width - 120)
    panel = rl.Rectangle(x, y, width, 60 + self.ROW_HEIGHT * len(rows) + 10)
    rl.draw_rectangle_rounded(panel, 0.12, 10, rl.Color(12, 16, 20, 225))
    rl.draw_rectangle_rounded_lines_ex(panel, 0.12, 10, 2, rl.Color(255, 255, 255, 100))
    self._draw_text(tr("Target constraint"), x + 18, y + 14, 28)
    overflow = len(targets) - len(rows)
    if overflow:
      text = tr("+{count} more").format(count=overflow)
      self._draw_text(text, x + width - 18 - measure_text_cached(self._font, text, 24).x, y + 16, 24)

    for index, item in enumerate(rows):
      row_y = y + 60 + index * self.ROW_HEIGHT
      if item.session_id:
        if item.model_ns > 0 and item.vision_index in (0, 1):
          source = tr("RADAR #{track} + V{index}").format(track=item.track_id, index=item.vision_index + 1)
        else:
          source = tr("RADAR #{track}").format(track=item.track_id)
      else:
        source = tr("VISION {index}").format(index=item.vision_index + 1)
      action = tr("Hold") if item.kind == 'hold' else tr("Braking")
      action_width = measure_text_cached(self._font, action, 28).x
      source_width = measure_text_cached(self._font, source, 32).x
      source_size = min(32, 32 * (width - 56 - action_width) / max(source_width, 1))
      self._draw_text(source, x + 18, row_y, source_size)
      self._draw_text(action, x + width - 18 - action_width, row_y + 2, 28)

      # The scale may be shared by multiple constraints on the same plan. It is
      # neither an independent per-target acceleration cap nor physical force.
      percent = f"{round(item.strength * 100)}%"
      percent_width = measure_text_cached(self._font, percent, 26).x
      bar_width = width - 36 - percent_width - 18
      rl.draw_rectangle_rec(rl.Rectangle(x + 18, row_y + 47, bar_width, 10), rl.Color(255, 255, 255, 65))
      rl.draw_rectangle_rec(rl.Rectangle(x + 18, row_y + 47, bar_width * item.strength, 10), rl.Color(*influence_rgb(item.strength), 255))
      self._draw_text(percent, x + width - 18 - percent_width, row_y + 37, 26)

  def _draw_text(self, text: str, x: float, y: float, size: float) -> None:
    rl.draw_text_ex(self._font, text, rl.Vector2(x, y), size, 0, rl.WHITE)
