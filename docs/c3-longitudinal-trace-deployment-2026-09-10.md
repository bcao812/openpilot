# C3 诊断部署记录：2026-09-09/10

## 实际部署范围

用户于 2026-09-09 提供在线 C3 地址 `192.168.137.206` 并批准继续。
操作跨过午夜，最终 C3 `c3xl-dev` 为 **`153ccd592e4cceea51f17bf07721c3ecd561f4b4`**，
父提交为 `1115e7a94a2c4bb6e80545f14c608ee0ab4e18e1`，不是主机的 `dfab4a5b1`。

**只部署诊断，不部署新的抗抖控制策略。** 三个运行文件：

- `openpilot/selfdrive/controls/lib/longitudinal_trace.py`
- `openpilot/selfdrive/controls/lib/longitudinal_planner.py`
- `openpilot/selfdrive/controls/plannerd.py`

trace 与 plannerd 为 `df4e0fee1` 的原样诊断实现/接线；planner 的接线单独回移到设备旧基线。
去掉 trace 接线后的 planner AST 与设备原基线完全一致。
没有携带主机 planner 的默认健康硬化、lead coast、brake dwell 或 median 实现；
设备没有新增 `cruise_smoothing.py`。没有更新 PID、MPC 权重、Panda safety、参数表或任何二进制。

`/data/longitudinal_trace` 已创建。它只请求诊断在后续自然进程启动时采集，不是抗抖优化开关。
未重启 openpilot/设备，未查询 onroad/IsOffroad、实时车速或要求用户重启，遵守 c3-sync 的既有约束。
**本次不声称正在运行的 plannerd 已采到新诊断，也不声称道路放行。**

## 为什么没有整批同步主机 HEAD

C3 初始 HEAD 只有 `1115e7a94`，并有 **40 个既存未提交改动**：构建二进制/生成代码、模型文件、
Panda 编译产物及 beep.py。beep 内容指纹匹配主机已知 `824987a6f`，但没有重新部署它。
其余生成/二进制文件只按实际指纹保全，没有将“看起来像本机编译”当作来源证明。
需要的实时、日志、MPC 等 Python 依赖指纹与主机一致，两个原设备 planner/plannerd 则匹配旧提交。

因此普通 c3-sync 的“快进主机整批 SHA、status 为空”流程不适用：
快进会夹带控制变化或把未同步的文件伪装成最新版本；清理 dirty 更会覆盖现有设备构建。
采用原分支上的独立诊断回移提交，**只 stage 这三个文件**，其余 40 个改动不入提交，前后 SHA-256 完全不变。
设备仍 dirty 是保留原状，不是同步遗漏。设备 `prebuilt` 仍在，原有 `is_dirty()` 的短路语义未修改；
分析后续日志必须结合本记录，不能因日志 dirty=False 就推断所有原生文件等于 Git blob。

**设备现在与主机主线分叉。后续同步必须以 `153ccd592` 重新比较历史，不能把它当成主机 HEAD 的祖先直接覆盖。**
回移提交通过 Git bundle 保存在本机；不强制移动主机分支、不把设备编译产物并入主线。
归档补丁为零上下文格式，核验/应用需要 `git apply --unidiff-zero`；提供补丁不构成再次部署授权。

## 部署前验证

在 `/tmp/c3-trace-sync-20260909-uQ9YHN/` 两份 x86 隔离副本中，用旧设备控制基线重新验证，
没有直接把前一轮新主线测试结果当成此次回移的验证：

| 验证 | 结果 |
|---|---|
| 移除 trace 接线后的 planner AST | 与 `1115e7a94` 相同 |
| 固定 21 路 / 41 窗 | 全部逐字段相同，含 19 个无原始抖动对照窗 |
| 计划 / 控制 / CAN 覆盖 | 25,941 / 129,188 / 64,610 帧 |
| 诊断实际采样 | 25,941；队列丢失 0、异常 0 |
| 六类闭环 × 三种延迟 | 18/18 完全相同，覆盖制动、切入、override、闪烁、无 lead |
| 队列饱和闭环 | 丢弃 15,084 个诊断样本，控制输出仍与基线相同 |
| 诊断单元与提取测试 | 40 passed；六项旧签名闭环测试不直接运行，由上述 18 组替代 |
| 主机车辆树 ARM 共享对象 | 14 个哈希不变，x86 泄漏 0 |

这是诊断增量的中性验证，不证明旧基线停车间距或车辆实际舒适性安全。
此前否决的控制实验放行结论没有改变，未重复计算或累计到本轮通过数。

## C3 原机隔离自测

在 `/usr/local/venv/bin/python` 下以低优先级运行 60 秒合成快照测试。
没有启动 planner 控制循环；没有发布 `longitudinalPlan`、`carControl` 或 CAN；
使用独立 `OPENPILOT_PREFIX` 的 ZMQ 接收端，不把合成数据写进真实驾驶日志。

| 指标 | 观测 |
|---|---:|
| 生成/收到 | 1,200 / 1,200 |
| 本地丢弃/异常 | 0 / 0 |
| 采样及入队 P50 | 185.27 µs |
| P95 | 240.62 µs |
| P99 | **261.47 µs** |
| 最大 | 480.98 µs |
| writer 策略 | SCHED_OTHER |
| writer CPU affinity | 0,1,2 |

隔离自测满足拟定的 P99 ≤ 1 ms 门槛，才创建真实诊断 marker。
它只验证原机 Python/消息读取/后台序列化及独立 socket 路径；
**没有验证真实 plannerd 与 modeld 共核运行时开销、真实 logmessaged→loggerd→rlog 送达或热降频影响**。
需在后续自然记录中检查 `longitudinal.trace` 序号、精确 plan_mono 联结、现有 plannerd_profile 和计划延迟。
实际日志明显丢样、调度退化或异常时不得将诊断作为可靠归因依据，更不能开启控制实验。

两次环境预检失败均如实保留：第一次使用系统 Python，缺 numpy，三文件自动恢复，HEAD 未改；
确认 `/etc/profile` 的 `/usr/local/venv` 后才重试成功。另一次合成基准固定 core 7 时收到 EINVAL，
尚未初始化采样器；改为从可用 affinity 选择测试 CPU 后完成。没有修改 CPU 在线配置或网络防火墙。

## 数据与回退保全

部署前保全 2026-09-09 **116 段 / 1,225,388,378 字节** rlog：
`00000057--d1a59a0182` 0–64，`00000058--bee2818cef` 0–50。
只拉已完成 rlog，不含视频、不删除设备数据；本地大小与 manifest 相符。
原始日志已落在工作区 `_logs/`，本轮不分析其 GPS 日期、控制表现或抖动变化。

设备备份：`/data/diagnostics/longitudinal-trace-20260910-0008/`，包含原有 40 文件及两个旧运行文件、
原 index、状态/指纹、回移提交 bundle、自测和部署结果。
主机持久副本：`_logs/c3-trace-deployment-2026-09-10/`；该路径被根仓库忽略，不等于已远程备份。

关闭采样只需后续安全操作中移除 marker；不在运行中重启。
如需回退源代码，必须重新核验设备 live HEAD、dirty 指纹和本次三文件 diff，
只撤销诊断提交/接线，不能 `reset --hard` 或恢复全部构建文件。
本报告不是自动执行回退、删 marker、重启或道路试验的授权。

后续优先保全自然启动后的 rlog，检查真实诊断覆盖及正常/抖动对照，再决定控制优化方向；
没有额外承诺定时监控，也未绕过用户的“不重启、不检查 onroad”规则。

## 证据

- [设备部署结果与开关](c3-longitudinal-trace-deployment-2026-09-10/enabled.json)
- [初始设备指纹](c3-longitudinal-trace-deployment-2026-09-10/device-before.json)
- [旧基线回放等价](c3-longitudinal-trace-deployment-2026-09-10/port-evaluation.json)
- [设备隔离自测](c3-longitudinal-trace-deployment-2026-09-10/device-selftest.json)
- [日志选择清单](c3-longitudinal-trace-deployment-2026-09-10/log-pull-manifest.txt)
- [设备三文件完整补丁](c3-longitudinal-trace-deployment-2026-09-10/diagnostic-backport.patch)
- [备份与本机验证归属](c3-longitudinal-trace-deployment-2026-09-10/provenance.json)
- [上一轮定位结论及限制](pedal-switch-following-diagnosis-2026-09-09.md)
