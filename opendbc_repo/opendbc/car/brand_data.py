from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Any


class BrandData(Mapping[str, Any]):
  """Load one brand's data on lookup, all brands only when enumerated."""

  def __init__(self, attr: str):
    self.attr = attr
    self._loaded: dict[str, Any] = {}
    self._complete = False

  def __getitem__(self, brand: str):
    if brand not in self._loaded:
      if self._complete or not isinstance(brand, str) or not brand.isidentifier():
        raise KeyError(brand)
      module_name = 'fingerprints' if self.attr in ('FW_VERSIONS', 'FINGERPRINTS') else 'values'
      try:
        module = import_module(f'opendbc.car.{brand}.{module_name}')
        self._loaded[brand] = getattr(module, self.attr)
      except (ImportError, AttributeError, OSError) as exc:
        raise KeyError(brand) from exc
    return self._loaded[brand]

  def _all(self):
    if not self._complete:
      from opendbc.car.interfaces import get_interface_attr
      self._loaded = get_interface_attr(self.attr, ignore_none=True)
      self._complete = True
    return self._loaded

  def __iter__(self) -> Iterator[str]:
    return iter(self._all())

  def __len__(self) -> int:
    return len(self._all())

  def copy(self) -> dict[str, Any]:
    return self._all().copy()
