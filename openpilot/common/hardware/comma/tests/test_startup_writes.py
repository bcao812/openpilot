import subprocess
from types import SimpleNamespace

import pytest

from openpilot.common.hardware.comma import hardware as hardware_module


@pytest.fixture
def hardware(monkeypatch):
  result = SimpleNamespace(
    trace=[], groups=[], values={}, online=set(), missing=set(),
    irqs={"msm_vidc": ["21"], "i2c_geni": ["22"], "fts_ts": ["23"],
          "msm_drm": ["24"], "spi_geni": ["25"], "kgsl-3d0": ["26"], "a5": ["27", "28"]},
  )

  def write(value, path):
    if path in result.missing:
      raise FileNotFoundError(path)
    if path.endswith("/online"):
      if value == "1":
        result.online.add(path)
      else:
        result.online.discard(path)
    if "/policy4/" in path:
      # Simulate policy4 appearing only after the big cluster was brought up.
      assert result.online == {f"/sys/devices/system/cpu/cpu{i}/online" for i in range(4, 8)}
    result.values[path] = value
    result.trace.append(("write", value, path))

  def write_many(writes):
    result.groups.append(list(writes))
    for value, path in writes:
      write(value, path)

  def irqs(action):
    result.trace.append(("irqs", action))
    return result.irqs.get(action, [])

  def run(command, **kwargs):
    result.trace.append(("run", command, kwargs))
    return SimpleNamespace(returncode=0)

  def call(command):
    result.trace.append(("call", command))
    return 0

  def check_output(command, **kwargs):
    result.trace.append(("check_output", command, kwargs))
    return "42\n"

  monkeypatch.setattr(hardware_module, "sudo_write", write)
  monkeypatch.setattr(hardware_module, "sudo_write_many", write_many)
  monkeypatch.setattr(hardware_module, "get_irqs_for_action", irqs)
  monkeypatch.setattr(hardware_module, "gpio_init", lambda *args: result.trace.append(("gpio_init", *args)))
  monkeypatch.setattr(hardware_module, "gpio_set", lambda *args: result.trace.append(("gpio_set", *args)))
  monkeypatch.setattr(hardware_module, "subprocess", SimpleNamespace(
    run=run, call=call, check_output=check_output, CalledProcessError=subprocess.CalledProcessError,
  ))
  result.device = hardware_module.HardwareComma()
  result.device.amplifier = None
  return result


def test_onroad_then_offroad_keeps_cpu_settings_and_refreshes_irqs(hardware):
  hardware.device.set_power_save(False)
  assert hardware.groups[:2] == [
    [("1", f"/sys/devices/system/cpu/cpu{i}/online") for i in range(4, 8)],
    [("performance", "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"),
     ("1689600", "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"),
     ("performance", "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor"),
     ("1689600", "/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq")],
  ]
  assert hardware.values["/proc/irq/26/smp_affinity_list"] == "7"
  assert hardware.values["/proc/irq/27/smp_affinity_list"] == "6"
  assert hardware.values["/proc/irq/28/smp_affinity_list"] == "6"

  hardware.irqs["kgsl-3d0"] = ["76"]
  hardware.irqs["a5"] = ["77"]
  hardware.trace.clear()
  hardware.groups.clear()
  hardware.device.set_power_save(True)
  assert hardware.groups[:2] == [
    [("0", f"/sys/devices/system/cpu/cpu{i}/online") for i in range(4, 8)],
    [("ondemand", "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor")],
  ]
  assert hardware.online == set()
  assert not any(item[0] == "write" and "/policy4/" in item[2] for item in hardware.trace)
  assert hardware.values["/proc/irq/76/smp_affinity_list"] == "7"
  assert hardware.values["/proc/irq/77/smp_affinity_list"] == "6"
  assert [item[1] for item in hardware.trace if item[0] == "irqs"] == [
    "kgsl-3d0", "a5", "cci", "cpas_camnoc", "cpas-cdm", "csid", "ife", "csid-lite", "ife-lite",
  ]


def test_initialization_keeps_fan_gpu_encoder_irq_and_scheduler_configuration(hardware):
  hardware.device.initialize_hardware()
  assert hardware.trace[:3] == [
    ("run", "sudo chmod a+w /dev/kmsg", {"shell": True}),
    ("gpio_init", hardware_module.GPIO.SOM_ST_IO, True),
    ("gpio_set", hardware_module.GPIO.SOM_ST_IO, True),
  ]
  assert [item[1:] for item in hardware.trace if item[0] == "write"] == [
    ("f", "/proc/irq/default_smp_affinity"),
    ("1", "/proc/irq/21/smp_affinity_list"),
    ("1", "/proc/irq/22/smp_affinity_list"),
    ("5", "/proc/irq/23/smp_affinity_list"),
    ("5", "/proc/irq/24/smp_affinity_list"),
    ("1", "/sys/class/kgsl/kgsl-3d0/min_pwrlevel"),
    ("1", "/sys/class/kgsl/kgsl-3d0/max_pwrlevel"),
    ("1", "/sys/class/kgsl/kgsl-3d0/force_bus_on"),
    ("1", "/sys/class/kgsl/kgsl-3d0/force_clk_on"),
    ("1", "/sys/class/kgsl/kgsl-3d0/force_rail_on"),
    ("1000", "/sys/class/kgsl/kgsl-3d0/idle_timer"),
    ("performance", "/sys/class/kgsl/kgsl-3d0/devfreq/governor"),
    ("710", "/sys/class/kgsl/kgsl-3d0/max_clock_mhz"),
    ("performance", "/sys/class/devfreq/soc:qcom,cpubw/governor"),
    ("performance", "/sys/class/devfreq/soc:qcom,memlat-cpu0/governor"),
    ("performance", "/sys/class/devfreq/soc:qcom,memlat-cpu4/governor"),
    ("N", "/sys/kernel/debug/msm_vidc/clock_scaling"),
    ("Y", "/sys/kernel/debug/msm_vidc/disable_thermal_mitigation"),
    ("3", "/proc/irq/25/smp_affinity_list"),
  ]
  assert hardware.trace[-3:] == [
    ("check_output", ["pgrep", "-f", "spi0"], {"encoding": "utf8"}),
    ("call", ["sudo", "chrt", "-f", "-p", "1", "42"]),
    ("call", ["sudo", "taskset", "-pc", "3", "42"]),
  ]


def test_missing_cpu_node_stops_before_policy_or_irq_changes(hardware):
  hardware.missing.add("/sys/devices/system/cpu/cpu6/online")
  with pytest.raises(FileNotFoundError, match="cpu6"):
    hardware.device.set_power_save(False)
  assert hardware.trace == [
    ("write", "1", "/sys/devices/system/cpu/cpu4/online"),
    ("write", "1", "/sys/devices/system/cpu/cpu5/online"),
  ]


def test_missing_gpu_node_stops_before_encoder_and_panda_thread_changes(hardware):
  hardware.missing.add("/sys/class/kgsl/kgsl-3d0/force_bus_on")
  with pytest.raises(FileNotFoundError, match="force_bus_on"):
    hardware.device.initialize_hardware()
  assert hardware.trace[-1] == ("write", "1", "/sys/class/kgsl/kgsl-3d0/max_pwrlevel")
  assert not any(item[0] == "irqs" and item[1] == "spi_geni" for item in hardware.trace)
  assert not any(item[0] == "check_output" for item in hardware.trace)


def test_missing_irq_keeps_warning_without_writes(hardware, capsys):
  hardware_module.affine_irq(6, "absent-device")
  assert hardware.trace == [("irqs", "absent-device")]
  assert hardware.groups == []
  assert "No IRQs found for 'absent-device'" in capsys.readouterr().out
