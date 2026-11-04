import capnp
from typing import Any

from openpilot.cereal import custom
from opendbc.car import structs

_FIELDS = '__dataclass_fields__'  # copy of dataclasses._FIELDS


def is_dataclass(obj):
  """Similar to dataclasses.is_dataclass without instance type check checking"""
  return hasattr(obj, _FIELDS)


def _asdictref_inner(obj) -> dict[str, Any] | Any:
  if is_dataclass(obj):
    ret = {}
    for field in getattr(obj, _FIELDS):  # similar to dataclasses.fields()
      ret[field] = _asdictref_inner(getattr(obj, field))
    return ret
  elif isinstance(obj, (tuple, list)):
    return type(obj)(_asdictref_inner(v) for v in obj)
  else:
    return obj


def asdictref(obj) -> dict[str, Any]:
  """
  Similar to dataclasses.asdict without recursive type checking and copy.deepcopy
  Note that the resulting dict will contain references to the original struct as a result
  """
  if not is_dataclass(obj):
    raise TypeError("asdictref() should be called on dataclass instances")

  return _asdictref_inner(obj)


def convert_to_capnp(struct: structs.CarParamsSP | structs.CarStateSP) -> capnp.lib.capnp._DynamicStructBuilder:
  struct_dict = asdictref(struct)

  if isinstance(struct, structs.CarParamsSP):
    struct_capnp = custom.CarParamsSP.new_message(**struct_dict)
  elif isinstance(struct, structs.CarStateSP):
    struct_capnp = custom.CarStateSP.new_message(**struct_dict)
  else:
    raise ValueError(f"Unsupported struct type: {type(struct)}")

  return struct_capnp


def convert_carControlSP(struct: capnp.lib.capnp._DynamicStructReader) -> structs.CarControlSP:
  # TODO: recursively handle any car struct as needed
  def remove_deprecated(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.endswith('DEPRECATED')}

  struct_dict = struct.to_dict()
  struct_dataclass = structs.CarControlSP(**remove_deprecated({k: v for k, v in struct_dict.items() if not isinstance(k, dict)}))

  struct_dataclass.mads = structs.ModularAssistiveDrivingSystem(**remove_deprecated(struct_dict.get('mads', {})))
  # struct_dataclass.params = [structs.CarControlSP.Param(**remove_deprecated(p)) for p in struct_dict.get('params', [])]
  struct_dataclass.leadOne = structs.LeadData(**remove_deprecated(struct_dict.get('leadOne', {})))
  struct_dataclass.leadTwo = structs.LeadData(**remove_deprecated(struct_dict.get('leadTwo', {})))
  struct_dataclass.intelligentCruiseButtonManagement = structs.IntelligentCruiseButtonManagement(
    **remove_deprecated(struct_dict.get('intelligentCruiseButtonManagement', {}))
  )

  return struct_dataclass


_CAR_CONTROL_SP_FIELDS = frozenset(('mads', 'params', 'leadOne', 'leadTwo', 'intelligentCruiseButtonManagement'))
_CAR_CONTROL_SP_CACHE_SUPPORTED = frozenset(
  name for name in custom.CarControlSP.schema.fields if not name.endswith('DEPRECATED')
) == _CAR_CONTROL_SP_FIELDS
_LEAD_DEPRECATED_FIELDS = tuple(name for name in custom.LeadData.schema.fields if name.endswith('DEPRECATED'))
_MADS_DEPRECATED_FIELDS = tuple(name for name in custom.ModularAssistiveDrivingSystem.schema.fields if name.endswith('DEPRECATED'))
_ICBM_DEPRECATED_FIELDS = tuple(name for name in custom.IntelligentCruiseButtonManagement.schema.fields if name.endswith('DEPRECATED'))


def _without_deprecated(values, names):
  # to_dict() returned a fresh dictionary owned by this conversion.
  for name in names:
    values.pop(name, None)
  return values


def _flat_lead_template(values):
  # The current evidence is a flat dictionary of immutable wire scalars. Fall
  # back to a fresh decode if a future schema adds another mutable container.
  for name, value in values.items():
    if name == 'evidence':
      if not isinstance(value, dict) or any(isinstance(item, (dict, list)) for item in value.values()):
        return False
    elif isinstance(value, (dict, list)):
      return False
  return True


def _fresh_lead(values):
  copied = values.copy()
  if 'evidence' in copied:
    copied['evidence'] = copied['evidence'].copy()
  return structs.LeadData(**copied)


class CarControlSPConverter:
  """Keep two private lead templates, keyed by complete current wire contents.

  Radar leads commonly repeat between their 20 Hz publications while card
  consumes carControlSP at 100 Hz. The key is an owned serialization of the
  current lead, not its timestamp or a potentially mutable reader's identity.
  Every result is independently mutable; missing and changed leads replace the
  template immediately. No health, freshness or actuation decision is cached.
  """

  def __init__(self):
    self._leads = [None, None]

  def _lead(self, message, name, index):
    if not message._has(name):
      self._leads[index] = None
      return structs.LeadData()
    lead = getattr(message, name)
    payload = lead.as_builder().to_bytes()
    cached = self._leads[index]
    if cached is not None and payload == cached[0]:
      return _fresh_lead(cached[1])
    values = _without_deprecated(lead.to_dict(), _LEAD_DEPRECATED_FIELDS)
    if _flat_lead_template(values):
      self._leads[index] = (payload, values)
      return _fresh_lead(values)
    self._leads[index] = None
    return structs.LeadData(**values)

  def __call__(self, message):
    if not _CAR_CONTROL_SP_CACHE_SUPPORTED:
      self._leads = [None, None]
      return convert_carControlSP(message)
    if isinstance(message, capnp.lib.capnp._DynamicStructBuilder):
      message = message.as_reader()
    mads = _without_deprecated(message.mads.to_dict(), _MADS_DEPRECATED_FIELDS) if message._has('mads') else {}
    icbm = (_without_deprecated(message.intelligentCruiseButtonManagement.to_dict(), _ICBM_DEPRECATED_FIELDS)
            if message._has('intelligentCruiseButtonManagement') else {})
    return structs.CarControlSP(
      mads=structs.ModularAssistiveDrivingSystem(**mads),
      params=[param.to_dict() for param in message.params],
      leadOne=self._lead(message, 'leadOne', 0),
      leadTwo=self._lead(message, 'leadTwo', 1),
      intelligentCruiseButtonManagement=structs.IntelligentCruiseButtonManagement(**icbm),
    )
