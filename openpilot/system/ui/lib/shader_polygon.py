import pyray as rl
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, cast
from openpilot.system.ui.lib.application import gui_app, GL_VERSION

MAX_GRADIENT_COLORS = 20  # includes stops as well


@dataclass
class Gradient:
  start: tuple[float, float]
  end: tuple[float, float]
  colors: list[rl.Color]
  stops: list[float]

  def __post_init__(self):
    if len(self.colors) > MAX_GRADIENT_COLORS:
      self.colors = self.colors[:MAX_GRADIENT_COLORS]
      print(f"Warning: Gradient colors truncated to {MAX_GRADIENT_COLORS} entries")

    if len(self.stops) > MAX_GRADIENT_COLORS:
      self.stops = self.stops[:MAX_GRADIENT_COLORS]
      print(f"Warning: Gradient stops truncated to {MAX_GRADIENT_COLORS} entries")

    if not len(self.stops):
      color_count = min(len(self.colors), MAX_GRADIENT_COLORS)
      self.stops = [i / max(1, color_count - 1) for i in range(color_count)]


class PreparedGradient:
  """Independent snapshot of a gradient's normalized shader inputs.

  Use this only when the caller replaces the snapshot when its current gradient
  changes. Mutable Gradient callers keep their per-draw behavior. Screen-space
  endpoints are still calculated from the live origin rectangle on every draw.
  """
  def __init__(self, gradient: Gradient):
    self.start = tuple(gradient.start)
    self.end = tuple(gradient.end)
    self.color_count = len(gradient.colors)
    self.stop_count = len(gradient.stops)
    if max(self.color_count, self.stop_count) > MAX_GRADIENT_COLORS:
      raise ValueError("PreparedGradient exceeds the shader's color/stop capacity")
    self._colors = rl.ffi.new("float[]", [channel / 255.0 for color in gradient.colors
                                        for channel in (color.r, color.g, color.b, color.a)])
    stops = [float(stop) for stop in gradient.stops]
    self._stops = rl.ffi.new("float[]", [0.0 if stop < 0.0 else 1.0 if stop > 1.0 else stop for stop in stops])


def prepare_gradient(gradient: Gradient) -> Gradient | PreparedGradient:
  # Packing a short, frequently replaced gradient costs more than its small
  # normalization loop. The measured 8/16-color paths benefit from reuse;
  # leave shorter gradients on the existing mutable path without allocations.
  return PreparedGradient(gradient) if len(gradient.colors) >= 8 else gradient


FRAGMENT_SHADER = GL_VERSION + """
in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec4 fillColor;

// Gradient line defined in *screen pixels*
uniform int useGradient;
uniform vec2 gradientStart;  // e.g. vec2(0, 0)
uniform vec2 gradientEnd;    // e.g. vec2(0, screenHeight)
uniform vec4 gradientColors[20];
uniform float gradientStops[20];
uniform int gradientColorCount;

vec4 getGradientColor(vec2 p) {
  // Compute t from screen-space position
  vec2 d = gradientStart - gradientEnd;
  float len2 = max(dot(d, d), 1e-6);
  float t = clamp(dot(p - gradientEnd, d) / len2, 0.0, 1.0);

  // Clamp to range
  float t0 = gradientStops[0];
  float tn = gradientStops[gradientColorCount-1];
  if (t <= t0) return gradientColors[0];
  if (t >= tn) return gradientColors[gradientColorCount-1];

  for (int i = 0; i < gradientColorCount - 1; i++) {
    float a = gradientStops[i];
    float b = gradientStops[i+1];
    if (t >= a && t <= b) {
      float k = (t - a) / max(b - a, 1e-6);
      return mix(gradientColors[i], gradientColors[i+1], k);
    }
  }

  return gradientColors[gradientColorCount-1];
}

void main() {
  // TODO: do proper antialiasing
  finalColor = useGradient == 1 ? getGradientColor(gl_FragCoord.xy) : fillColor;
}
"""

# Default vertex shader
VERTEX_SHADER = GL_VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
out vec2 fragTexCoord;
uniform mat4 mvp;

void main() {
  fragTexCoord = vertexTexCoord;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

UNIFORM_INT = rl.ShaderUniformDataType.SHADER_UNIFORM_INT
UNIFORM_FLOAT = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT
UNIFORM_VEC2 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2
UNIFORM_VEC4 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4


class ShaderState:
  _instance: Any = None

  @classmethod
  def get_instance(cls):
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def __init__(self):
    if ShaderState._instance is not None:
      raise Exception("This class is a singleton. Use get_instance() instead.")

    self.initialized = False
    self.shader = None

    # Shader uniform locations
    self.locations = {
      'fillColor': None,
      'useGradient': None,
      'gradientStart': None,
      'gradientEnd': None,
      'gradientColors': None,
      'gradientStops': None,
      'gradientColorCount': None,
      'mvp': None,
    }

    # Pre-allocated FFI objects
    self.fill_color_ptr = rl.ffi.new("float[]", [0.0, 0.0, 0.0, 0.0])
    self.use_gradient_ptr = rl.ffi.new("int[]", [0])
    self.color_count_ptr = rl.ffi.new("int[]", [0])
    self.gradient_colors_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS * 4)
    self.gradient_stops_ptr = rl.ffi.new("float[]", MAX_GRADIENT_COLORS)

  def initialize(self):
    if self.initialized:
      return

    self.shader = rl.load_shader_from_memory(VERTEX_SHADER, FRAGMENT_SHADER)

    # Cache all uniform locations
    for uniform in self.locations.keys():
      self.locations[uniform] = rl.get_shader_location(self.shader, uniform)

    # Orthographic MVP (origin top-left)
    proj = rl.matrix_ortho(0, gui_app.width, gui_app.height, 0, -1, 1)
    rl.set_shader_value_matrix(self.shader, self.locations['mvp'], proj)

    self.initialized = True

  def cleanup(self):
    if not self.initialized:
      return
    if self.shader:
      rl.unload_shader(self.shader)
      self.shader = None

    self.initialized = False


def _configure_shader_color(state: ShaderState, color: Optional[rl.Color],  # noqa: UP045  # rl.Color is a function, so `rl.Color | None` fails
                            gradient: Gradient | PreparedGradient | None, origin_rect: rl.Rectangle):
  assert (color is not None) != (gradient is not None), "Either color or gradient must be provided"

  if gradient is None:
    color_count = 0
    prepared = False
  else:
    prepared = isinstance(gradient, PreparedGradient)
    color_count = gradient.color_count if prepared else len(gradient.colors)
  use_gradient = 1 if color_count else 0
  state.use_gradient_ptr[0] = use_gradient
  rl.set_shader_value(state.shader, state.locations['useGradient'], state.use_gradient_ptr, UNIFORM_INT)

  if use_gradient:
    gradient = cast("Gradient | PreparedGradient", gradient)
    if prepared:
      colors_ptr, stops_ptr, stop_count = gradient._colors, gradient._stops, gradient.stop_count
    else:
      # Ordinary gradients remain mutable, including in-place color and stop edits.
      colors_ptr, stops_ptr, stop_count = state.gradient_colors_ptr, state.gradient_stops_ptr, len(gradient.stops)
      for i in range(len(gradient.colors)):
        c = gradient.colors[i]
        base = i * 4
        colors_ptr[base:base + 4] = [c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0]
      for i in range(len(gradient.stops)):
        s = float(gradient.stops[i])
        stops_ptr[i] = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s

    # Other polygons share this shader. Re-upload even when the CPU buffers are reused.
    state.color_count_ptr[0] = color_count
    rl.set_shader_value_v(state.shader, state.locations['gradientColors'], colors_ptr, UNIFORM_VEC4, color_count)
    rl.set_shader_value_v(state.shader, state.locations['gradientStops'], stops_ptr, UNIFORM_FLOAT, stop_count)
    rl.set_shader_value(state.shader, state.locations['gradientColorCount'], state.color_count_ptr, UNIFORM_INT)

    # Map normalized start/end to screen pixels
    start_vec = rl.Vector2(origin_rect.x + gradient.start[0] * origin_rect.width, origin_rect.y + gradient.start[1] * origin_rect.height)
    end_vec = rl.Vector2(origin_rect.x + gradient.end[0] * origin_rect.width, origin_rect.y + gradient.end[1] * origin_rect.height)
    rl.set_shader_value(state.shader, state.locations['gradientStart'], start_vec, UNIFORM_VEC2)
    rl.set_shader_value(state.shader, state.locations['gradientEnd'], end_vec, UNIFORM_VEC2)
  else:
    color = color or rl.WHITE
    state.fill_color_ptr[0:4] = [color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0]
    rl.set_shader_value(state.shader, state.locations['fillColor'], state.fill_color_ptr, UNIFORM_VEC4)


def triangulate(pts: np.ndarray) -> list[tuple[float, float]]:
  """Only supports simple polygons with two chains (ribbon).

  Interleaves the two chains into a triangle strip: [L0, R0, L1, R1, ...] where L is the first
  half and R is the second half reversed. Two slice assignments, not a Python loop.

  WHY THIS IS NOT THE OBVIOUS LOOP ANY MORE, WITH THE NUMBERS

  It was `for i in range(n): append(pts[i]); append(pts[-i-1])` followed by
  `np.array(list).tolist()`. That is three expensive things stacked:

    * n indexings into a numpy array, each allocating a (2,) ndarray object
    * np.array() over a list of 2n tiny ndarrays -- the slowest way to build an array, since
      numpy has to inspect every element to find a common dtype and shape
    * .tolist() on the result

  Measured on an x86 host with _logs/torque_bar_cost.py's sizes:

      N=66  (a modelV2 lane line or path)   19.68 us -> 3.84 us
      N=226 (a torque bar arc)              67.91 us -> 12.76 us

  The big onroad UI calls draw_polygon about nine times a frame -- four lane lines, two road
  edges, one or two path fills, two torque bar arcs -- so this is 273.6 us -> 52.4 us per frame
  on the host. At the 7.0x host->A75 factor measured in
  _arch/c3xl-dev/01-runtime-resource-map.md §4, that is roughly 1.5 ms per frame on the device,
  against a measured UI frame of 9.7 ms while driving.

  The output is byte-identical -- torque_bar_cost.py asserts equality against the old
  implementation at every size before reporting a timing.

  This compatibility path retains .tolist(): rl.draw_triangle_strip accepts a Python
  sequence or CFFI pointer, but not a bare ndarray. PreparedPolygon instead exposes the
  independent strip array through a retained CFFI buffer.
  """
  return cast(list, _triangle_strip_array(pts).tolist())


def _triangle_strip_array(pts: np.ndarray) -> np.ndarray:
  # TODO: consider deduping close screenspace points
  if len(pts) % 2 != 0:
    pts = pts[:-1]

  n = len(pts) // 2
  tri_strip = np.empty((2 * n, 2), dtype=pts.dtype)
  tri_strip[0::2] = pts[:n]
  tri_strip[1::2] = pts[n:][::-1]

  return tri_strip


class PreparedPolygon:
  """Snapshot of a ribbon's float32 vertices, ready for repeated draws.

  Keep one snapshot per current geometry in the caller. Colors and gradient coordinates
  are deliberately not stored here. The original array can change without changing this
  snapshot; callers must prepare a replacement when their geometry changes.
  """
  def __init__(self, points: np.ndarray):
    self.point_count = len(points)
    self.vertex_count = 0
    self._buffer = None
    self.vertices = rl.ffi.NULL
    if self.point_count < 3:
      return

    pts = np.ascontiguousarray(points, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 2, "points must be (N,2)"
    strip = _triangle_strip_array(pts)
    # Vector2 is two consecutive C floats. from_buffer retains this independent,
    # contiguous float32 array without a tolist() round trip or another vertex copy.
    self._buffer = rl.ffi.from_buffer("Vector2[]", strip)
    # pyray treats a bare CFFI array as a struct and takes its address. Supply the
    # pointer expected by DrawTriangleStrip, while retaining the owning array above.
    self.vertices = rl.ffi.cast("Vector2 *", self._buffer)
    self.vertex_count = len(strip)

  def __len__(self):
    return self.point_count


def draw_polygon(origin_rect: rl.Rectangle, points: np.ndarray | PreparedPolygon,
                 color: Optional[rl.Color] = None, gradient: Gradient | PreparedGradient | None = None):  # noqa: UP045  # rl.Color is a function, so `rl.Color | None` fails

  """
  Draw a ribbon polygon (two chains) with a triangle strip and gradient.
  - Input must be [L0..Lk-1, Rk-1..R0], even count, no crossings/holes.
  """
  if len(points) < 3:
    return

  # Initialize shader on-demand
  state = ShaderState.get_instance()
  state.initialize()

  if isinstance(points, PreparedPolygon):
    tri_strip, strip_count = points.vertices, points.vertex_count
  else:
    # Mutable callers retain the uncached behavior, including in-place changes.
    pts = np.ascontiguousarray(points, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 2, "points must be (N,2)"
    tri_strip = triangulate(pts)
    strip_count = len(tri_strip)

  # Configure gradient shader
  _configure_shader_color(state, color, gradient, origin_rect)

  # Draw strip, color here doesn't matter
  rl.begin_shader_mode(state.shader)
  rl.draw_triangle_strip(tri_strip, strip_count, rl.WHITE)
  rl.end_shader_mode()


def cleanup_shader_resources():
  state = ShaderState.get_instance()
  state.cleanup()
