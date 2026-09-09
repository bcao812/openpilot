# Honda 巡航油刹切换：修改、验证与放行评估

后续全本地日志、独立正常对照及扩展放行评估见
[全行程回归补充报告](pedal-switch-full-regression-2026-09-09.md)；本报告保留事件窗口测量的历史范围。

## 结论与适用范围

本次在 `openpilot-3xl-0817`、`c3xl-dev`、起始提交
`824987a6fd134b76ed6d3f04dea4eadff9b8271f` 上完成离线修改和评估。
平台为 `HONDA_ACCORD`、Bosch、有雷达硬件但 `radarUnavailable=True`，
使用 openpilot 纵向控制；不是 `BOSCH_RADARLESS`。

**不能把回放改善当成上车放行。最终默认只启用安全门槛加固；有前车巡航扩展默认关闭。**
有前车扩展在实际日志回放中明显减少切换，但闭环压力试验发现新的振荡反例。
释放限速、静态滞回、速度渐退组合也分别被反例否决，没有合入车辆控制逻辑。
未改变 Honda CAN 映射、制动位、安全限值、MPC 权重、DEC、停车或起步常数；未访问设备、推送或部署。

原始全量审计保留在工作区 `_logs/PEDAL-SWITCH-AUDIT-2026-09-09.md`，不是本仓库的跟踪文件。
本次复核的本地驾驶日志全部早于 9/8 改动；没有本次修改后的道路数据。

## 按类型拆分

| 本地提交 | 内容 |
|---|---|
| `54a2ffe87` | 默认巡航参与状态、健康和风险门槛 |
| `f43ff86` | 原生节拍回放与比较工具 |
| `0226083` | 默认关闭的前车实验和闭环放行反例 |
| 本报告所在提交 | 评估、数据指纹和测试结果 |

### 1. 默认启用：巡航整形的参与状态与风险门槛

`openpilot/selfdrive/controls/lib/longitudinal_planner.py`：

- 同时要求 `selfdriveState.enabled`、`carControl.enabled`、`longActive`，保留人工踏板、override 和 reset 旁路。
- `all_checks` 增加 `carControl`、`selfdriveState`，不只检查规划输入。
- FCW、MPC/model 停车请求、非有限候选和 model 请求低于 −0.5 m/s² 时不进行舒适性整形。
- 最终仍取 MPC/cruise/有效 e2e 候选的 minimum；ACC 模式不会因为本改动强制切成 e2e。
- 该修改是**收紧舒适性功能权限**，不是全局 NaN 修复，也不能修复整条链路所有无效输入问题。

风险：返回原始 cruise 候选可能比整形值更强或更弱；门槛切换也可能重新暴露短脉冲。
不得为了降低切换计数而覆盖 FCW/model 风险信号。

### 2. 默认关闭：远且不约束巡航的前车扩展

仅构造参数 `experimental_coast_with_lead=True` 或离线回放显式参数可以启用；
`plannerd` 没有启用此参数，没有 UI/Params 开关，也不随 experimentalMode 自动开启。

除上述门槛外，还要求：

- `HONDA_ACCORD`、openpilot 纵向、ACC 模式、SP cruise 来源、速度 ≥10 m/s，距设定速度 ≤1 km/h。
- 原始 cruise 比 MPC 至少更小 0.1 m/s²；只整形 cruise 候选，不修改 MPC/e2e。
- 两个 present lead 都满足检查：数值有限、非负前车速度、正的加速度衰减时间常数。
- 实际距离大于 personality 对应 MPC 静态安全距离再加 5 m，并大于闭合速度乘 6 s。
- 任一前车 `aLeadK < −0.5 m/s²`、切入到近距离、MPC 更强制动、model stop/brake 或 FCW 立即旁路。

这些阈值是保守的实验资格门槛，不是已经标定的碰撞安全保证。
视觉前车误关联、延迟、速度估计错误仍可能通过门槛；反馈也会影响下一帧 MPC。

### 3. 可复现验证与失败用例

- `replay_pedal_switches.py`：压缩本地回放语料、校验输入 SHA256、真实 planner/MPC/LongControl/CarController/CAN。
- `evaluate_pedal_switches.py`：严格对齐比较、保留录制版本重建误差、分组指标和每事件结果。
- `test_longitudinal_cruise_safety.py`、`test_longitudinal_cruise_leads.py`：默认关闭、两前车、风险、手动及健康门槛。
- `test_longitudinal_cruise_scenarios.py`：20 Hz planner、100 Hz LongControl/CarController、50 Hz 真实 Honda CAN 的简化闭环。
- `test_pedal_switch_replay.py`：0.15 s 离线去抖、短脉冲、中性间断、人工退出和时间缺口。

## 回放范围与方法

从原审计 55 routes / 2180 rlogs 中，全部 **378 个主事件**位于 33 routes、12 个录制提交。
这些事件加六个既有近停/接管风险窗口合并成 **240 个窗口**，读取602个不同 rlog，
计量区间共150.54分钟，另给每窗60秒预热。不是重放全部22.05小时行驶。

每臂处理180,649个规划周期、899,667个控制周期、449,820个原生 CAN 周期；
其中178,343个规划周期满足回放自动控制与有效输入口径。
另将73个 raw-only 短脉冲事件构成72个窗口，进行当前HEAD与最终默认版本对比。

关键限制与修正：

1. 使用原审计 `t0_mono` 和 native CAN 的实际 segment ID，不能用 `t // 60` 推断分段；允许缺segment0的route。
2. 参数来自各段录制的 `CarParams`/`CarParamsSP`；`initData` 校验提交且禁止 dirty/mixed 录制。
3. `longitudinalPlan.modelMonoTime` 对齐 model 快照；其他输入只使用该快照之前的消息。
   plannerd 的实际 SubMaster 接收批次未被完整记录，这仍是规划重建误差来源。
4. SP target/DEC 决策固定为日志值；cruise 使用本臂的递归加速度状态，其他 SP 来源使用录制 target。
   不模拟反事实 DEC/SCC 决策或感知，不声称完整进程重放。
5. SP 消息可能晚于 controls 消费同一 plan；先配对 SP 决策，再在原 plan 时间重放。
   `controlsState.longitudinalPlanMonoTime` 明确选出100 Hz控制使用的plan。
6. CarController 使用 carState 发布前可见的 carControl，保持100/50 Hz奇偶节拍，经过Float32和真实DBC。
   仅比较 ACC_CONTROL，不把未重建的 stock HUD/MADS 数据当作逐字节全车回放。
7. 原始指令、控制激活及时间缺口不混接；0.15秒是**离线计量门槛，不是刹车延时**。
8. open-loop中的速度、跟距、实测jerk和驾驶员接管固定不变；不得拿它们宣布停车距离或实际舒适性改善。

### 重建基线可信度

各录制提交分别由 `git archive` 导出；使用相同MPC源码与隔离host构建依赖，未在真实ARM工作树构建。

- 全事件录制版本重建的 `aTarget` 绝对误差：median 0.00108、P95 0.01082、max 0.62456 m/s²。
  **全体逐帧精确等价不成立**，尤其旧低速状态和异步输入，不用全局数字证明所有事件根因。
- 41个巡航事件9,488帧：P95 0.00536 m/s²；最大0.09093来自route4e t629.92短事件。
  该事件单独标为低可信度；没有用它宣称收益。
- 对16个代表窗口固定录制的 `aTarget/shouldStop`，仅重放真实控制链：
  66,726个LongControl输出完全匹配；33,229个有效自动CAN请求的accel/gas/brake完全匹配。
  证明控制调度与编码可信，不意味着planner所有输入已精确重建。
- route4a t1227.33、route4f t1200.54、route51 t2646.47录制版本的去抖计数分别72/62/52，与native完全一致。

## 默认版本结果

相对起始HEAD：

| 指标 | 结果 |
|---|---:|
| 378个主事件的切换计数变化 | 全部不变 |
| 规划输出变化 | 65帧 / 180,649帧 |
| 自动有效帧最大更强/更弱请求 | −0.01685 / +0.03921 m/s² |
| 低速<10 m/s输出变化 | 0帧 |
| shouldStop变化 | 0帧 |
| MPC求解失败 / 同时正油门与刹车请求 | 0 / 0 |
| 六个既有风险区间 | 请求和shouldStop均不变 |

因此默认版本**不是“已经修复有前车抖动”**；只是避免舒适性整形越过参与状态和风险门槛。

短脉冲补充集：15,508个规划周期、77,223个控制周期、38,606个CAN周期；
16个规划帧变化，最大0.03943 m/s²；无shouldStop变化/求解失败。
0.15秒去抖计数144→144，但raw计数328→330：
route48 t7118.39–7119.04的model请求低于−0.5，门槛恢复原始cruise，产生额外两个短边沿。
这是明确保留的舒适性代价，不能描述为“所有抖动指标无回归”。

## 实验扩展结果：不作为默认收益

相对最终默认版，**显式开启实验参数**后的相同事件窗口统计：

| 原事件分组 | 个数 | 默认切换 | 实验切换 | 改善/不变/恶化事件 |
|---|---:|---:|---:|---:|
| 有前车但cruise主导 | 29 | 461 | 137 | 24 / 5 / 0 |
| 基本无前车巡航 | 12 | 15 | 12 | 2 / 10 / 0 |
| 普通跟车 | 275 | 1207 | 1207 | 0 / 275 / 0 |
| 低速停走 | 52 | 238 | 238 | 0 / 52 / 0 |

41个巡航事件合计476→149，约68.7%指令切换减少；这是**条件化open-loop模拟**，不是实车改善百分比。
无前车组的微小变化来自同一窗口上下文的lead出现/消失及递归状态，不意味着无lead逻辑另行放宽。

| 代表事件 | 时长 | native/录制重建 | 默认→实验 | 默认→实验切换频率 |
|---|---:|---:|---:|---:|
| route4a t1227.326 | 34.51s | 72 / 72 | 2→2 | 0.029→0.029 次/s |
| route4f t1200.537 | 34.94s | 62 / 62 | 56→6 | 1.574→0.143 次/s |
| route51 t2646.468 | 37.28s | 52 / 52 | 50→4 | 1.314→0.080 次/s |

最大额外更强/更弱请求为−0.03997/+0.05998 m/s²；shouldStop不变、六个风险区间不变、无求解失败和CAN冲突。
38个低速帧有递归残差，最大仅0.00000352 m/s²，**不是严格逐位不变**。
新增lead时间常数检查对语料872,881个present-lead快照均通过；因此最终额外检查不改变已测回放资格。
不能据此保证未回放路况没有幽灵制动；全行程反事实ghost/h门槛没有完成。

## 闭环压力试验与拒绝的方案

简化车辆：纯延迟队列加一阶响应，常值加速度偏置模拟未辨识路负载；噪声为确定性正弦。
此模型没有真实轮端力、混动再生、坡度辨识或制动液压，不能把仿真间距换成实车距离。

### 实验功能的放行反例

- `offset=0.254545 m/s²`、速度噪声0.02 m/s、纯延迟0.8s、一阶响应0.4s：
  默认0次去抖切换，实验26次；稳态速度范围0.007→0.336 km/h。
  证实新增下肩部和1 km/h硬门槛可能形成闭环振荡，故实验功能默认关闭。
- `offset=0.25`、lead presence以2 Hz更新切换、纯延迟0.5s：12→28次。
  单帧“前车不约束”测试通过不代表真实动态资格边界稳定。
- 前车−3 m/s²减速、纯延迟0.5s再加0.4s响应：默认和实验最小仿真间距分别−1.865/−1.930m，
  都越过碰撞门槛；不是本次引入的全部问题，但进一步禁止上车放行。
- 在较快的0.3s纯延迟+0.2s响应模型中，该减速例最小间距3.036/3.022m；−1 m/s²请求时刻均3.25s。
  这是另一组模型条件，不替代失败条件。切入场景、九种巡航组合另有断言。

两个放行反例作为 `strict=True` 的xfail回归保留，理由写在测试名和结果中。
**xfail不是安全通过**；`pytest --runxfail` 可恢复红灯验证，修复前不得开启实验功能。

### 未合入原型

| 原型 | 观测 | 决定 |
|---|---|---|
| 仅释放限速0.2 m/s³，额外减速度上限0.04 | 29个lead巡航137→157，6事件恶化；长延迟26→33次 | 拒绝 |
| −0.21制动保持、−0.23释放的静态滞回 | 可减lead闪烁，但长延迟仍24次，不能消除边界振荡 | 不合入 |
| 速度渐退0.2…0.2778 m/s+制动保持 | 长延迟仿真0次，但无lead日志15→124，11/12事件恶化 | 拒绝 |
| 仅下肩部速度渐退0.25…0.2778+滞回 | 代表日志较好，长延迟仍29次 | 不合入 |

不通过改BRAKE_REQUEST位、扩大油刹死区、全局限jerk、锁DEC或调停车常数来粉饰计数。
下一步需要实际执行器辨识、带路负载估计的巡航控制设计及独立安全复核，不是继续调单个滞回阈值。

## 测试与数据产物

临时运行目录：`/tmp/c3-pedal-fixes-20260909-t4YwSk`。
持久化摘要、输入指纹、分组和测试结果位于附件目录 `pedal-switch-validation-2026-09-09/`；
大体积缓存和逐帧结果不提交，原始rlog不修改。

| 验证 | 最终结果 |
|---|---|
| controls单元/闭环+Honda车辆测试+纵向maneuvers | 241 passed、2 skipped、2 xfailed、69 subtests passed |
| Honda safety C测试（UBSan构建） | 760 passed、581 skipped、884 subtests passed |
| 第1类安全加固独立中间态 | 127 passed |
| Ruff、`git diff --check` | 通过 |
| 真实工作树ARM二进制 | 14个已跟踪共享库hash不变，0个x86泄漏 |

合计1001个常规测试通过；跳过项是既有抽象/不适用配置，2个xfail是明确的放行失败，不计入通过。
后续提交只包含Python源码、测试与报告，不包含设备二进制、CAN/safety或固件修改。

### 可复现命令

必须先按工作区 `host-build` 流程建立**隔离host副本**，设置其根目录及vendored依赖的`PYTHONPATH`，
并把`HOME`重定向到可写临时目录。不要在携带已跟踪ARM二进制的车辆工作树执行构建或测试。
以下命令在该副本内执行：

```bash
python openpilot/selfdrive/controls/tests/replay_pedal_switches.py prepare \
  --logs /home/bcao/SourceCode/_logs \
  --audit /home/bcao/SourceCode/_logs/pedal-switch-2026-09-09 \
  --all-events --workers 3 --output /tmp/pedal-corpus
python openpilot/selfdrive/controls/tests/replay_pedal_switches.py run \
  --corpus /tmp/pedal-corpus --output /tmp/pedal-default --workers 3
python openpilot/selfdrive/controls/tests/replay_pedal_switches.py run \
  --corpus /tmp/pedal-corpus --output /tmp/pedal-experimental --workers 3 \
  --experimental-coast-with-lead
python openpilot/selfdrive/controls/tests/evaluate_pedal_switches.py \
  --corpus /tmp/pedal-corpus --recorded /tmp/pedal-recorded \
  --baseline /tmp/pedal-default --candidate /tmp/pedal-experimental \
  --output /tmp/pedal-evaluation.json
```

`/tmp/pedal-recorded` 必须逐录制版本导出源码并用同一回放工具执行`run --commit <SHA前缀>`；
该参数只筛选窗口，**不会替换当前导入的源码**。先对照附件源码hash，不得拿当前代码冒充录制版。
控制链独立核对加`--recorded-plan`，不能把这种模式作为完整planner收益比较臂。
短脉冲语料加`prepare --events-file raw-only-events.json`。

```bash
python -m pytest -q openpilot/selfdrive/controls/tests \
  --ignore=openpilot/selfdrive/controls/tests/test_leads.py \
  opendbc_repo/opendbc/car/honda/tests/test_honda.py \
  openpilot/selfdrive/test/longitudinal_maneuvers/test_longitudinal.py
python -m pytest -q opendbc_repo/opendbc/safety/tests/test_honda.py
python -m pytest --runxfail -q \
  openpilot/selfdrive/controls/tests/test_longitudinal_cruise_scenarios.py \
  -k 'release_gate'
```

最后一条应显示两个已记录的放行失败，不应为发布目的改成通过。
`test_leads.py`的完整card进程重放、整机全量构建、全仓库测试、设备/道路验证不在本次通过声明内。
