from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
import secrets
import struct
import sys
import time


LOGGING_VERSION = 1
ANALYSIS_VERSION = 1
SOURCE_MANIFEST_VERSION = 2
UINT64_MAX = (1 << 64) - 1
CHECK_FLAGS = {"valid": 1, "alive": 2, "freq_ok": 4, "seen": 8, "updated": 16}
PLANNER_INPUTS = (
  "carState", "radarState", "modelV2", "controlsState", "carControl", "selfdriveState", "vehicleParameters",
  "selfdriveStateSP", "carStateSP", "gpsLocationExternal", "gpsLocation",
)
CONTROL_INPUTS = (
  "carState", "longitudinalPlan", "modelV2", "selfdriveState", "onroadEvents", "carOutput", "longitudinalPlanSP",
)
SOURCE_PATHS = (
  "openpilot/common/longitudinal_logging.py",
  "openpilot/common/params.py",
  "openpilot/common/params.h",
  "openpilot/common/params.cc",
  "openpilot/common/params_c.cc",
  "openpilot/common/params_keys.h",
  "openpilot/common/constants.py",
  "openpilot/common/filter_simple.py",
  "openpilot/common/pid.py",
  "openpilot/common/realtime.py",
  "openpilot/cereal/log.capnp",
  "openpilot/cereal/custom.capnp",
  "opendbc_repo/opendbc/car/car.capnp",
  "openpilot/selfdrive/controls/plannerd.py",
  "openpilot/selfdrive/controls/controlsd.py",
  "openpilot/selfdrive/controls/lib/control_observation.py",
  "openpilot/selfdrive/controls/lib/longitudinal_observation.py",
  "openpilot/selfdrive/controls/lib/longitudinal_planner.py",
  "openpilot/selfdrive/controls/lib/longitudinal_mode.py",
  "openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py",
  "openpilot/selfdrive/controls/lib/drive_helpers.py",
  "openpilot/selfdrive/controls/lib/cruise_response.py",
  "openpilot/selfdrive/controls/lib/launch_response.py",
  "openpilot/selfdrive/controls/lib/model_reference.py",
  "openpilot/selfdrive/controls/lib/reference_timing.py",
  "openpilot/selfdrive/controls/lib/longcontrol.py",
  "openpilot/selfdrive/controls/lib/motion_supervisor.py",
  "openpilot/selfdrive/controls/lib/longitudinal_trace.py",
  "openpilot/selfdrive/selfdrived/longitudinal_observation.py",
  "openpilot/selfdrive/selfdrived/selfdrived.py",
  "openpilot/selfdrive/selfdrived/state.py",
  "openpilot/selfdrive/selfdrived/events.py",
  "openpilot/selfdrive/locationd/helpers.py",
  "openpilot/selfdrive/modeld/constants.py",
  "openpilot/sunnypilot/selfdrive/selfdrived/events.py",
  "openpilot/sunnypilot/selfdrive/selfdrived/events_base.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/dec/dec.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/smart_cruise_control.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_assist.py",
  "openpilot/selfdrive/car/actuation_observation.py",
  "openpilot/selfdrive/car/card.py",
  "openpilot/selfdrive/car/cruise.py",
  "openpilot/selfdrive/car/longitudinal_authority.py",
  "openpilot/selfdrive/pandad/pandad_api_impl.py",
  "opendbc_repo/opendbc/__init__.py",
  "opendbc_repo/opendbc/can/__init__.py",
  "opendbc_repo/opendbc/can/dbc.py",
  "opendbc_repo/opendbc/can/packer.py",
  "opendbc_repo/opendbc/can/parser.py",
  "opendbc_repo/opendbc/car/__init__.py",
  "opendbc_repo/opendbc/car/structs.py",
  "opendbc_repo/opendbc/car/interfaces.py",
  "opendbc_repo/opendbc/car/common/conversions.py",
  "opendbc_repo/opendbc/car/common/simple_kalman.py",
  "opendbc_repo/opendbc/car/honda/carcontroller.py",
  "opendbc_repo/opendbc/car/honda/carstate.py",
  "opendbc_repo/opendbc/car/honda/hondacan.py",
  "opendbc_repo/opendbc/car/honda/interface.py",
  "opendbc_repo/opendbc/car/honda/values.py",
  "opendbc_repo/opendbc/sunnypilot/car/interfaces.py",
  "opendbc_repo/opendbc/sunnypilot/car/honda/carstate_ext.py",
  "opendbc_repo/opendbc/sunnypilot/car/honda/icbm.py",
  "opendbc_repo/opendbc/sunnypilot/car/honda/mads.py",
  "opendbc_repo/opendbc/sunnypilot/car/honda/values_ext.py",
  "opendbc_repo/opendbc/sunnypilot/car/intelligent_cruise_button_management_interface_base.py",
  # Accord's PT DBC is generated in memory before dbc.py considers a generated
  # file on disk. Bind the generator, template and its actual imported inputs.
  "opendbc_repo/opendbc/dbc/generator/generator.py",
  "opendbc_repo/opendbc/dbc/generator/honda/honda_civic_hatchback_ex_2017_can.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_honda_common.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_bosch_2018.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_bosch_radar_acc.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_lkas_hud_5byte.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_bosch_standstill.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_steering_sensors_a.dbc",
  "opendbc_repo/opendbc/dbc/generator/honda/_gearbox_common.dbc",
)
# CDLL and ELF dependencies are not all represented by Python modules. These
# are disk identities only: hashing them does not prove what is mapped/loaded.
EXPLICIT_NATIVE_PATHS = (
  "openpilot/common/libparams_c.so",
  "msgq_repo/msgq/ipc_pyx.so",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/libacados_ocp_solver_long.so",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/libacados.so",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/libblasfeo.so",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/libhpipm.so",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/libqpOASES_e.so.3.1",
)


def logging_supported(params, params_sp):
  try:
    from opendbc.car.honda.values import CAR
    return (params.carFingerprint == CAR.HONDA_ACCORD and params.openpilotLongitudinalControl
            and not params_sp.enableGasInterceptor)
  except Exception:
    return False


def is_uint64(value):
  if type(value) is int:
    return 0 <= value <= UINT64_MAX
  if value is None:
    return False
  return isinstance(value, Integral) and not isinstance(value, bool) and 0 <= value <= UINT64_MAX


def file_digest(path):
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def source_manifest():
  root = Path(__file__).resolve().parents[2]

  def selected_files(paths):
    files, missing = {}, []
    for relative in paths:
      try:
        path = root / relative
        if not path.resolve().is_relative_to(root):
          raise ValueError("File escapes the selected tree")
        files[relative] = file_digest(path)
      except (OSError, ValueError):
        missing.append(relative)
    return files, missing

  sources, missing = selected_files(SOURCE_PATHS)
  explicit_native, missing_native = selected_files(EXPLICIT_NATIVE_PATHS)
  native = {}
  for name, module in tuple(sys.modules.items()):
    if not name.startswith(("openpilot.", "opendbc.", "msgq", "capnp.lib.")):
      continue
    filename = getattr(module, "__file__", None)
    if filename is None or not str(filename).endswith((".so", ".pyd")):
      continue
    try:
      native[name] = file_digest(Path(filename))
    except OSError:
      native[name] = None
  return {
    "version": SOURCE_MANIFEST_VERSION,
    "scope": "selected source, explicit native library and imported native module files on disk at observer startup; not loaded-code or firmware attestation",
    "source_files": sources,
    "missing_source_files": missing,
    "explicit_native_library_files": explicit_native,
    "missing_explicit_native_library_files": missing_native,
    "imported_native_module_files": native,
    "python": platform.python_version(),
    "machine": platform.machine(),
  }


def configuration_digest(params):
  try:
    encoding = configuration_encoding(params)
    if encoding == "dataclass-json-v1":
      payload = {
        "type": f"{type(params).__module__}.{type(params).__qualname__}",
        "fields": asdict(params),
      }
      serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    else:
      reader = params.as_reader() if callable(getattr(params, "as_reader", None)) else params
      serialized = reader.as_builder().to_bytes()
    return hashlib.sha256(encoding.encode() + b"\0" + serialized).hexdigest()
  except Exception:
    return None


def configuration_encoding(params):
  return "dataclass-json-v1" if is_dataclass(params) and not isinstance(params, type) else "capnp-copy-bytes-v1"


def capture_inputs(state, services):
  result = []
  identities = getattr(state, "logMonoTime", {})
  health_sources = None
  for service in services:
    if service not in identities:
      continue
    if health_sources is None:
      health_sources = tuple((getattr(state, name, {}).get, flag) for name, flag in CHECK_FLAGS.items())
    mono = identities[service]
    valid_id = is_uint64(mono)
    known_flags = flags = 0
    for get_health, flag in health_sources:
      value = get_health(service)
      # Missing and malformed values both leave the health bit unknown.
      # Never coerce a value or invoke its __bool__ during observation.
      if value is True:
        known_flags |= flag
        flags |= flag
      elif value is False:
        known_flags |= flag
    known_id = valid_id and (mono > 0 or bool(flags & CHECK_FLAGS["seen"]))
    result.append({
      "service": service, "monoTime": int(mono) if valid_id else 0, "idKnown": known_id,
      "knownFlags": known_flags, "flags": flags, "kind": "consumerSnapshot",
    })
  return result


def local_input(service, mono, valid):
  return {
    "service": service, "monoTime": int(mono) if is_uint64(mono) else 0, "idKnown": is_uint64(mono),
    "knownFlags": CHECK_FLAGS["valid"], "flags": CHECK_FLAGS["valid"] if valid else 0, "kind": "localPublished",
  }


def can_digest(frames):
  digest = hashlib.sha256()
  for address, payload, bus in frames:
    payload = bytes(payload)
    digest.update(struct.pack("<IHH", address, bus, len(payload)))
    digest.update(payload)
  return digest.hexdigest()


class ObservationRecorder:
  def __init__(self, producer, params=None, params_sp=None, *, enabled=None, clock=None, manifest_interval=1000):
    self.enabled = logging_supported(params, params_sp) if enabled is None else bool(enabled)
    self.producer = producer
    self.clock = time.monotonic_ns if clock is None else clock
    self.sequence = 0
    self.errors = 0
    self.session = 0
    self.identity = ""
    self.manifest = ""
    self.manifest_interval = max(1, int(manifest_interval))
    if self.enabled:
      try:
        self.session = secrets.randbits(64) or 1
        source = source_manifest()
        configuration = {"carParams": configuration_digest(params), "carParamsSP": configuration_digest(params_sp)}
        encodings = {
          name: configuration_encoding(value) if configuration[name] is not None else None
          for name, value in (("carParams", params), ("carParamsSP", params_sp))
        }
        payload = {
          "schema": LOGGING_VERSION, "producer": producer, "source": source, "configuration": configuration,
          "configuration_encoding": encodings,
          "configuration_scope": "startup CarParams and CarParamsSP representation only; domain-separated hashes, not all dynamic Params reads",
          "complete": not source["missing_source_files"] and all(configuration.values())
                      and not source["missing_explicit_native_library_files"]
                      and all(value is not None for value in source["imported_native_module_files"].values()),
        }
        self.manifest = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.identity = hashlib.sha256(self.manifest.encode()).hexdigest()
      except Exception:
        self.record_error()

  def record_error(self):
    self.errors = min(self.errors + 1, (1 << 32) - 1)

  def begin(self, frame=None):
    if not self.enabled:
      return None
    try:
      self.sequence += 1
      started = self.clock()
      if not is_uint64(started) or not is_uint64(self.sequence):
        raise ValueError("Invalid observation clock or sequence")
      return {
        "sequence": self.sequence, "startMonoTime": int(started), "endMonoTime": 0,
        "frame": int(frame) if is_uint64(frame) else 0, "frameKnown": is_uint64(frame),
      }
    except Exception:
      self.record_error()
      return None

  def finish(self, sample):
    if sample is None:
      return None
    try:
      finished = self.clock()
      if not is_uint64(finished) or finished < sample["startMonoTime"]:
        raise ValueError("Invalid observation completion clock")
      return dict(sample, endMonoTime=int(finished))
    except Exception:
      self.record_error()
      return None

  def context(self, sample):
    return {
      "version": LOGGING_VERSION, "producer": self.producer, "session": self.session,
      "errors": self.errors, "identity": self.identity,
      "manifest": self.manifest if sample["sequence"] == 1 or sample["sequence"] % self.manifest_interval == 0 else "",
      **sample,
    }


def attach_diagnostics(message, field, factory, recorder, sample, values):
  if recorder is None or not recorder.enabled or sample is None:
    return False
  try:
    diagnostics = factory.new_message()
    for name, value in values.items():
      setattr(diagnostics, name, value)
    diagnostics.context = recorder.context(sample)
    setattr(message, field, diagnostics)
    return True
  except Exception:
    recorder.record_error()
    return False


def observe(observer, method, *arguments, **keywords):
  if observer is None:
    return None
  recorder = None
  try:
    recorder = getattr(observer, "recorder", observer)
    if not getattr(recorder, "enabled", False):
      return None
    return getattr(observer, method)(*arguments, **keywords)
  except Exception:
    try:
      recorder.record_error()
    except Exception:
      pass
    return None
