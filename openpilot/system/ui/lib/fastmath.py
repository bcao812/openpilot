"""Scalar replacements for the numpy calls the render path uses on single floats.

WHY THIS EXISTS

np.interp and np.clip are array operations. Called on one Python float they still pay the whole
dispatch -- argument parsing, dtype resolution, an output array allocation -- for one element.
Measured on an x86 host:

    np.interp(x, [a, b], [c, d])    1.451 us      lerp(x, a, b, c, d)     0.073 us     20x
    np.clip(x, lo, hi)              2.833 us      clamp(x, lo, hi)        0.068 us     42x

The big onroad UI makes about thirty-five such calls per frame -- two per _map_line_to_polygon
across seven lines, one per lane line and road edge for alpha, five in the torque bar, two for
the camera offset, and the lead-vehicle sizing. That is ~70 us a frame on a host, and at the
7.0x host->A75 factor measured in _arch/c3xl-dev/01-runtime-resource-map.md §4, roughly half a
millisecond a frame on the device against a 9.7 ms driven frame.

THESE ARE NOT APPROXIMATIONS

lerp reproduces np.interp exactly for a TWO-POINT increasing table, including both saturating
ends and the degenerate a == b case, where numpy returns d for x >= b. The branch order below
is what makes that last one match, and _logs/ui_fastmath_check.py asserts equality over the
whole range rather than trusting it.

They are NOT drop-in for the general cases: lerp takes a two-point table only -- np.interp with
three or more points still needs numpy -- and clamp assumes lo <= hi, which np.clip does not
require.
"""


def lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
  """np.interp(x, [x0, x1], [y0, y1]) for scalars, with numpy's saturating ends.

  Branch order matters and is not stylistic: `x >= x1` is tested FIRST so that a degenerate
  table (x0 == x1) returns y1 at x == x0, which is what np.interp does. Testing `x <= x0` first
  would return y0 there and silently disagree in exactly the case nobody writes a test for.
  """
  if x >= x1:
    return y1
  if x <= x0:
    return y0
  # slope * (x - x0) + y0, in THAT order, because that is the order numpy's compiled_interp
  # evaluates it in. Writing the algebraically identical y0 + (y1-y0)*((x-x0)/(x1-x0)) differs
  # in the last bit -- 8.9e-16 on the tables here -- and these values reach int(255 * alpha),
  # where one ULP at a boundary is one wrong colour channel.
  return ((y1 - y0) / (x1 - x0)) * (x - x0) + y0


def clamp(x: float, lo: float, hi: float) -> float:
  """np.clip(x, lo, hi) for scalars. Assumes lo <= hi, which every call site here satisfies."""
  return lo if x < lo else (hi if x > hi else x)
