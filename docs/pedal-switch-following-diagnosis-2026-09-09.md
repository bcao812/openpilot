# 跟车油门/刹车交替：定位结果与下一步

本报告记录 2026-09-09 的一次分析/验证，不代表未来 checkout 或道路放行。
仓库 `openpilot-3xl-0817`，分支 `c3xl-dev`，起点 `9d95ca380705d3537ba625d2a578aa10e2944379`。
车辆范围：2019 Accord Hybrid、openpilot 纵向、`radarUnavailable=True`；不是 `BOSCH_RADARLESS` 平台。

## 结论与决策

**不继续通过加速度域滞回、全模式最小驻留或泛化低通掩盖切换。**
本轮最重要的进展是把主要跟车事件定位到计划及其上游，而不是声称又找到一种已验证的抗抖修复。
只提交控制输出中性的链路诊断与验证工具；所有既有实验优化保持原默认值，未同步 C3、未接触设备。
这是为下一次数据采集补齐因果证据，不是改善了车辆抖动。

提交按类型拆分：`df4e0fee1` 为诊断代码及其单元/闭环测试；`c5ba3d3ae` 为短窗等价回归、提取器及工具测试；
本报告与证据另作独立文档提交。均为本地提交，未 push。

- 275 个原始跟车事件中，233 个在 0.2–0.5 次切换/秒；不能全部视为高频执行器抖动。
- 16 个代表窗口中，116 个原始控制切换都能关联到先发生的计划切换；114 个原始 CAN 切换都能关联到上游控制切换。
- 三种**不可部署**的单输入去噪探针未消除主要去抖事件，不支持直接给视觉前车速度加滤波。
- 下一优先项是精确对齐 MPC 的实际输入、状态递推、候选与车辆响应；证据不足时不修改制动权限/时序。

## 全体跟车事件筛查

复用此前 55 路、2,180 分段的指纹化本地缓存；本轮不是再跑 55 路控制回放。
从固定原始审计事件中取 275 个 `lead_following`，核验所引用 native 审计文件的哈希。
这些分类基于时间窗统计，不保证每一帧都由 lead MPC 主导。

| 切换频率（次/秒） | 小于 5 秒 | 5–15 秒 | 15–30 秒 | 至少 30 秒 |
|---|---:|---:|---:|---:|
| 0.2–0.5 | 0 | 201 | 31 | 1 |
| 0.5–1 | 7 | 21 | 0 | 0 |
| 1–2 | 11 | 2 | 0 | 0 |
| 至少 2 | 1 | 0 | 0 | 0 |

该频率是相邻油门/刹车模式转换频率，不是完整振荡周期频率。
Honda `ACC_CONTROL` 的油门/刹车请求分界是 **−0.2 m/s²**，不是零加速度。
中性相位跨越容限 0.1 秒；去抖要求相位持续至少 0.15 秒。

用于粗筛的每个原始事件前后额外加 1 秒，得到以下汇总；有重叠可能，**不能把这些数当成原始事件切换总数**。

| 环节 | raw | 去抖 |
|---|---:|---:|
| CAN 请求 | 1,341 | 1,307 |
| carControl 加速度 | 1,345 | 1,309 |
| longitudinalPlan.aTarget | 1,343 | 1,295 |
| 从记录轨迹重算 MPC 候选 | 1,263 | 1,225 |
| model action 加速度 | 709 | 671 |

246/275 个窗口中，CAN、命令、计划、重算 MPC 的去抖切换**数量**相同。
数量相同不等于逐次对应，因此继续做了下面的时间戳关联。

## 时间戳关联与输入探针

从原始事件选 6 个快速纯跟车、3 个快速混合来源、4 个慢速纯跟车、3 个慢速混合来源窗口。
此处计数在原始事件前后各 0.5 秒内，保留健康 engaged 掩码。
使用记录的 `controlsState.longitudinalPlanMonoTime` 精确关联控制使用的计划；
CAN 链路使用同方向、非负且不超过 150 ms 的一对一时序匹配，**CAN 本身没有计划引用 ID**。

| 关联 | 上游/下游 raw 数量 | 匹配 | 延迟中位数 / P95 / 最大 |
|---|---:|---:|---:|
| 计划 → 控制 | 116 / 116 | 116 | 6.87 / 12.28 / 22.07 ms |
| 控制 → CAN | 116 / 114 | 114 | 15.06 / 31.62 / 115.71 ms |

116 个 raw 控制切换、106 个去抖控制切换对应的精确计划引用，油门/刹车侧均一致。
去抖后不是完全一一匹配：计划→控制 104/106，控制→CAN 105/106。
150 ms 相位边界及采样率会改变去抖计数，不能隐去这些未匹配项。
上述证据支持“所选窗口的 CAN 并未独立产生主要切换”，不能证明实车执行器不会放大上游振荡。

纯跟车快速窗中，MPC 与最终目标同样各产生 4 个去抖切换，巡航候选均为 0；
混合窗口 `006` 则为最终目标 7 个、MPC 1 个，主要来自巡航链路。
因此不能把所有事件交给同一个末端去抖器。

视觉路径直接提供 `dRel`、`vLead = vEgo + modelLeadV − modelEgoV`、`aLeadK`，
并使用 `aLeadTau=0.3`；这些窗口不是实体雷达 KF 的输入。
MPC 障碍预测包含前车速度平方项，确实对前车速度变化敏感。
但帧间速度变化与目标变化相关，不能区分真实交通运动与感知误差。

对 10 个固定纯跟车窗口分别只替换 `dRel`、`vRel` 或 `aLeadK`：11 点、二阶、中心平滑，
只在连续存在的视觉 lead 块内处理；改 `vRel` 同步调整 `vLead/vLeadK`。
**中心滤波使用未来数据，只是诊断干预，不能放进实时车辆代码；也不是闭环消噪验证。**

| 探针 | 快速 6 窗 CAN raw / 去抖 | 快速计划去抖 | 慢速 4 窗 CAN 去抖 |
|---|---:|---:|---:|
| 原输入 | 32 / 24 | 24 | 31 |
| 距离单字段平滑 | 34 / 24 | 24 | 31 |
| 相对速度单字段平滑 | 25 / 23 | 26 | 31 |
| 前车加速度单字段平滑 | 30 / 24 | 24 | 31 |

速度探针减少了部分极短 raw 穿越，却把某一窗口的计划去抖切换 4→6；
不能依据 CAN 总数只减 1 就把相对速度滤波视作有效方案。
输入探针没有进入可部署候选，故不为其重复正常窗/道路放行测试。

另将命令与延后 0.5 秒的 `aEgo` 对照：快速 6 窗的中位残差从 −0.161 到 +0.066 m/s²，
同路线正常窗也有正负残差。未发现足以支持单一固定补偿的共同方向。
此项混有坡度、测量滤波、交通运动和假设延迟，**不是 Honda 执行器标定**。

## 本轮代码改动

`longitudinal_trace.py` 只读采样，`longitudinal_planner.py` 三处调用，`plannerd.py` 在启动时创建记录器。
仅 Accord openpilot 纵向且启动时存在 `/data/longitudinal_trace` 普通文件才启用。
不轮询磁盘，不新增优化开关，不更改 MPC 权重、PID、模式驻留、油门/刹车边界或 Panda safety。

每个 20 Hz 计划快照包含：

- 实际 SubMaster 的 model/radar/carState/carControl/controlsState 等消息单调时间戳及 radar 的模型引用。
- MPC 本次初始速度/加速度、上次巡航状态、修正后的巡航目标、coast limit、两条 lead 输入。
- MPC 来源/加速度、原始及整形后巡航候选、model 候选、e2e 资格、仲裁结果、停车/FCW/健康门控。
- 最终计划 `logMonoTime` 与下一帧递推速度；发布在两条控制计划消息之后。

单生产者/单消费者 CPython deque，最多排队 40 样本，后台最多另持有 10 样本；满时丢新样本并累计计数。
后台先退出实时调度、移到小核，再用**独立 logger/socket handler**，避免共享日志 handler 锁阻塞实时线程。
每 10 帧一批，走 `zmq.NOBLOCK`、DEBUG `logMessage`，不在实时循环 JSON 编码或写文件，也不重复写 INFO 级 swaglog 文件。
快照/队列/后台异常会停采，不改变控制计算；线程启动失败则不挂载采样器。

限制：CPython GIL、内存分配仍非硬实时；socket 满时可能静默丢日志；排队尾样本在进程退出时可能丢失。
序号/`plan_mono` 可以检测缺口，但 `dropped` 只计本地队列丢弃，不代表端到端送达。
当前 MPC 自己在求解失败后 reset status；采样的 `solver_status` 是**复位后的可见值**，不能据此认定从未求解失败。
实际输入还不含所有 solver warm start、完整 cost 向量和车辆物理真值，采样不是万能的根因证明。

## 回归与风险评估

固定选择 **21 路、41 个短窗**：16 抖动、12 同路线正常跟车、7 原本无抖动 engaged 路线、6 风险窗。
正常窗先用原始 raw/去抖事件前后 10 秒排除法选定，不用候选输出反选。
计量区间合计 **21.62 分钟**；通常预热 120 秒，两个起步较早的正常窗用 60 秒以尊重参数到达时间。
这不是 21.62 分钟总解码工作量，更不是完整行程回放。

开/关诊断逐字段严格一致：**25,941 计划、129,188 控制、64,610 CAN 帧**；41/41 窗覆盖完整。
19 个正常窗相较关闭诊断基线无新增刹车帧、油门/刹车切换或停车/控制状态变化；风险窗没有削弱强制动。
这是诊断开/关增量为零，不保证历史记录的“正常”在当前默认控制重建中仍完全没有交替。
关闭诊断的结果还与本轮起点保存的 41 窗输出逐字段一致。
开启后确实采到 25,941 样本，本地丢弃/错误均为 0，不是“代码未触发所以通过”。
回放使用内存 JSON writer 验证后台，不把该数字冒充设备 logMessage 端到端送达。

六种闭环场景（稳定跟车、前车制动、切入、人工 override、lead 闪烁、无 lead）中，
实 MPC→LongControl→Honda CAN 链路的诊断开/关结果完全一致，队列满也不影响输出。
原有 plant 是延迟+一阶执行器的简化加速度模型，不是标定过的 Honda 再生/液压制动模型。
这些闭环证明的是诊断中性，不是抗抖效果或道路安全。

测试汇总与逐项身份见 [test-accounting.json](pedal-switch-following-diagnosis-2026-09-09/test-accounting.json)。
最终去重 **1,302 passed、619 skipped、6 xfailed**，其中新增诊断/提取/回归工具测试 59 项；不累计重跑和 953 个 subtests。
Honda C safety 为 760 passed、581 skipped。额外 6 项功能 skip 来自本轮扩大巡航速度测试范围，
其余 skip 主要为参数化基类和不适用车型模式；保留一条已有多线程 fork 弃用警告。
显式 `--runxfail -k release_gate` **6 failed**：旧加速度整形/中值候选和长执行器延迟停车间距仍不满足放行要求。
未改成通过，未为实验新增豁免。

宿主微基准：采样+入队 P95 约 285 µs、P99 421 µs、最大 853 µs；
10 样本 JSON 编码 P95 303 µs、最大 495 µs。关闭分支 P95 约 0.24 µs。
这是并发宿主负载下的观测，**不是 C3 时限保证**；未覆盖设备 ZMQ、压缩/落盘与热降频。
实际窗口 JSON 载荷约 18.5 KB/s、67 MB/小时未压缩，另有 logMessage 元数据；需限制采集时长和保全日志。

全程在 `/tmp` 原生宿主副本运行测试；车辆源码树 14 个跟踪 ARM 共享对象哈希不变，x86 泄漏为 0。
Ruff、格式与差异空白检查通过；未运行全仓 UI/摄像头/所有车型或硬件/道路认证。

## C3 下一步：只采证，不打开抗抖实验

**需要后续设备操作批准并通过同步前检查，本轮没有执行以下操作。**
同步前重新核验 C3/主机分支、提交、差异及 ARM 二进制；仅同步诊断代码及其 planner/plannerd 接线，
不要把测试脚本、文档或 `/tmp` x86 副本发到设备。已存在的默认硬化修复另按实际设备版本核对，不能由本报告推断缺失。

1. 停车状态、正常控制设置不变，创建 `/data/longitudinal_trace`，下次进程启动生效；所有抗抖实验保持 off。
2. 先做停车/安全静态采样，核验 `longitudinal.trace` 存在、时间戳可联结、无异常、无日志断流或调度退化。
3. 仅在该 gate 通过后，另行批准的受控验证中采集短时正常跟车/自然复现场景；不主动制造紧急切入/制动。
4. 同一路况同时保留无抖动对照，立即按 log-pull 保全 rlog；之后移除 marker 并在安全状态重启进程退出采样。
5. 根据确切输入链路，优先区分：视觉 lead 残差、MPC 状态递推误差、候选仲裁切换、Honda 实际响应非线性。

若诊断开销 P99 超过拟定 1 ms 预算、计划延迟相较无采样基线退化、线程降级失败或日志明显丢样，
不进入带车运动的采集；先缩小采样范围/把 writer 移出进程。1 ms 是待实测门槛，不是本轮测出的 C3 能力。

**后续优化优先级：**
首先研究“递推状态/执行器实效一致性”，在正负残差分场景辨识后才考虑有界状态校正或分段标定；
风险是坡度/延迟被误学、低速停车制动力改变，因此不能套用一个固定偏置。
只有精确数据证明某种 lead 输入存在短时孤立误差，才研究“带可信度/轨迹一致性约束的输入处理”；
风险是错把前车真实制动/切入当噪声，必须以即时风险响应作为硬否决。
全模式最小驻留会阻止必要的模式转换，且覆盖不到本轮绝大部分慢交替；暂不推进。

## 复现与证据

固定窗口/指纹：[selection.json](pedal-switch-following-diagnosis-2026-09-09/selection.json)。
结论数字：[summary.json](pedal-switch-following-diagnosis-2026-09-09/summary.json)、
逐窗回归：[evaluation.json](pedal-switch-following-diagnosis-2026-09-09/evaluation.json)、
原始事件摘要：[following-screen.json](pedal-switch-following-diagnosis-2026-09-09/following-screen.json)、
逐转换关联：[transition-alignment.json](pedal-switch-following-diagnosis-2026-09-09/transition-alignment.json)、
输入干预：[probe-evaluation.json](pedal-switch-following-diagnosis-2026-09-09/probe-evaluation.json)、
响应筛查：[response-screen.json](pedal-switch-following-diagnosis-2026-09-09/response-screen.json)。
源代码、缓存和分析脚本归属：[provenance.json](pedal-switch-following-diagnosis-2026-09-09/provenance.json)。
归档脚本与本次实际运行源码的归一化 AST 对照：[analysis-provenance.json](pedal-switch-following-diagnosis-2026-09-09/analysis-provenance.json)。

在按 host-build 隔离的宿主副本、正确 PYTHONPATH 下执行；不得在 ARM 车辆树直接加载原生测试：

```bash
python -m openpilot.selfdrive.controls.tests.replay_longitudinal_trace run \
  --selection docs/pedal-switch-following-diagnosis-2026-09-09/selection.json \
  --output /tmp/longitudinal-trace-recheck --workers 3
python -m openpilot.selfdrive.controls.tests.replay_longitudinal_trace evaluate \
  --selection docs/pedal-switch-following-diagnosis-2026-09-09/selection.json \
  --output /tmp/longitudinal-trace-recheck
python -m openpilot.selfdrive.controls.tests.extract_longitudinal_trace \
  /path/to/rlog.zst --output /tmp/longitudinal-trace.jsonl
```

窗口清单引用本机已有的指纹化 `/tmp` corpus；迁移时调整路径并重新核验指纹。
历史一次性分析脚本随证据归档，只读原始缓存、写指定分析目录；中心滤波探针不用于车辆运行。
旧实验结论继续保存在 [temporal 报告](pedal-switch-temporal-regression-2026-09-09.md)，本报告不重写其历史测量。
