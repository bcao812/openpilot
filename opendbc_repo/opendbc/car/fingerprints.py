from opendbc.car.body.fingerprints import FINGERPRINTS as BODY_FINGERPRINTS
from opendbc.car.gm.fingerprints import FINGERPRINTS as GM_FINGERPRINTS


# These are the only brands retaining CAN-only fingerprints. The completeness
# test compares this table with all brand modules so a future addition cannot
# silently disappear. Loading every firmware database here is unnecessary.
_FINGERPRINTS = BODY_FINGERPRINTS | GM_FINGERPRINTS
_DEBUG_ADDRESS = {1880: 8}   # reserved for debug purposes


def __getattr__(name):
  # Keep the historical import API for tools and route migrations; neither
  # table is needed to sample the live car's CAN configuration.
  if name == 'FW_VERSIONS':
    from opendbc.car.interfaces import get_interface_attr
    value = get_interface_attr('FW_VERSIONS', combine_brands=True, ignore_none=True)
  elif name == 'MIGRATION':
    from opendbc.car.fingerprint_migration import MIGRATION
    value = MIGRATION
  else:
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
  globals()[name] = value
  return value


def is_valid_for_fingerprint(msg, car_fingerprint: dict[int, int]):
  adr = msg.address
  # ignore addresses that are more than 11 bits
  return (adr in car_fingerprint and car_fingerprint[adr] == len(msg.dat)) or adr >= 0x800


def eliminate_incompatible_cars(msg, candidate_cars):
  """Removes cars that could not have sent msg."""
  compatible_cars = []
  for car_name in candidate_cars:
    for fingerprint in _FINGERPRINTS[car_name]:
      # add alien debug address
      if is_valid_for_fingerprint(msg, fingerprint | _DEBUG_ADDRESS):
        compatible_cars.append(car_name)
        break
  return compatible_cars


def all_legacy_fingerprint_cars():
  """Returns a list of all known car strings, FPv1 only."""
  return list(_FINGERPRINTS.keys())
