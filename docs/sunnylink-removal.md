# c3xl-dev：移除 sunnylink

## 范围与结论

仓库：`openpilot-3xl-0817`；分支：`c3xl-dev`。
基线：`be2ee5fe98a8c8f8d1cc81e68ae9dab071293930`，开始工作时工作树干净。
候选为本次提交的源码；操作范围是本地修改、测试和提交，不包含推送或设备部署。

基线中的 sunnylink 是可选的云端运维通道，并非本地辅助驾驶的必要依赖。
除了远程修改设置和上传日志，还包括加密配置备份/恢复、设备及账户配对、
赞助角色查询、额外统计上报，以及复用 Athena 的远程诊断、SSH 代理、
导航目的地、视频/文件等 RPC。这里描述的是客户端实现能力，不是对线上网站
当前开放功能的认证。它可以辅助远程支持，但不执行感知、规划、转向或制动。

模型目录获取和模型文件下载使用独立通道。车辆识别、车型选择、地图、OTA、
本地日志记录及控制算法不依赖 sunnylink。本次不调整驾驶参数，也不宣称
移除云服务已经带来经过测量的帧率、内存或辅助驾驶效果提升。

## 改动

- 删除 `openpilot/sunnypilot/sunnylink/` 的客户端、远程设置描述/生成工具及专用测试。
- 删除五个进程定义：`manage_sunnylinkd`、`sunnylink_registration_manager`、
  `sunnylink_uploader`、`statsd_sp`、`backup_manager`，包括相关运行条件。
- 删除大屏和 mici 两套界面的菜单、配对/备份页面、专用同意页、侧栏状态、
  状态查询线程及 `backupManagerSP` 订阅，同时修正界面回放脚本。
- 移除 13 个翻译目录文件中的 sunnylink 专用消息及失效引用，保留共享译文。
- 移除车辆初始化中的云统计旁路、统计的双重发送和 Sentry 的 sunnylink 身份标签。
- Athena 保留原来非 sunnylink 的消息格式、队列优先级与文件处理，删除专用压缩分支。
- 保留本地 `car_list.json` 和旧车型迁移；未识别车辆的提示改为本地车型菜单。

静态逐项比较确认，其余 **51 个进程定义的 AST 完全不变**。车辆接口中
扭矩/NNLC/ICBM 初始化和不支持参数清理保持原样。comma Athena、常规 uploader、
普通统计、Sentry、本地 SSH、模型选择/下载、驾驶条款与培训均保留。
因此这不是关闭所有云连接或所有远程功能。

## 兼容与设备边界

保留 `params_keys.h` 中历史 sunnylink/备份键、共享 `BACKUP` 标记、
`cereal/services.py`/生成头文件中的惰性服务定义，以及历史 Cap'n Proto 字段。
这些声明没有 sunnylink 功能消费者或后台进程；通用 Params 默认值初始化仍可能
写入旧键，但旧 `SunnylinkEnabled=1` 也不能恢复已删除的服务。

保留这些声明是为了不破坏历史日志和本分支已跟踪的 ARM 原生制品，
也避免在 `prebuilt` 模式下制造原生源码/二进制不一致。此次没有 C/C++、
schema、模型文件、panda 固件或任何原生制品改动，功能移除不依赖替换设备二进制。
宿主测试生成的 x86 文件仅存在于隔离目录，不能部署到 C3。

没有删除设备凭据、参数目录、本地路线日志、模型缓存或共享的
`AthenadUploadQueue`。旧上传任务没有可靠的 sunnylink 所有者标记，不能为此
清空普通 Athena 的队列；已有队列仍可能由保留的通道处理。

**后续单独授权部署时，需要完整重启设备。** `DaemonProcess.stop()` 本来就是
空操作，旧守护进程可以跨 manager 重启继续存活；仅重启 manager 不能证明
旧 sunnylink 已停止。本次没有连接或修改设备。

## 测试与证据

原生依赖按工作区 `host-build` 程序构建，未在 ARM 源码工作树内构建：

- 基线：`/tmp/sunnylink-removal-baseline-20260913`
- 候选：`/tmp/sunnylink-removal-candidate-20260913`
- 两侧均构建 `msgq_repo/`、`openpilot/common/`、`openpilot/cereal/` 和纵向 MPC，
  使用 `--minimal --cache-disable -j4`。
- 隔离目录不含 Git 元数据；测试用 `build.json` 从实际源仓库元数据和快照
  HEAD/dirty 收据生成。测试 HOME 指向临时目录；HTTP 测试只使用本机服务。
- 构建结束后，仅修正了 UI 测试的无显示环境夹具，并将同一测试文件应用到快照。
  产品代码及原生依赖不变，未改写原始快照收据。
  测试文件 SHA256：
  `6e883de06ed1201ad97d2d3bed2368df2a278af7ae12b90154b635c5535d31e2`。

| 检查 | 结果与范围 |
| --- | --- |
| 后台/公共功能测试 | **PASS：137 项**，含进程策略、缺少 sunnylink 时的真实导入、隔离的 manager 初始化、单路统计、Sentry 身份、Athena RPC/日志/统计/上传、本地车型迁移及模型下载/缓存 |
| 界面及翻译测试 | **PASS：60 项**，其中 12 项覆盖两套菜单/条款/培训/状态/侧栏，48 项覆盖译文；UI 子用例 252 项 |
| 选择器/比较器自测 | **PASS：50 项**，固定套件分析器与失败关闭比较器的合成测试 |
| 固定输入快速检查 | **PASS**：core 的 327 例、449 段大小/mtime 检查；不是全量 SHA256 认证 |
| smoke 输入身份 | **PASS**：21 例、55 段、563,593,382 字节的 SHA256 检查，未刷新套件 |
| A/A 可重复性 | **PASS：18/18 任务**，同代码同输入逐项一致 |
| 基线/候选 smoke 等价性 | **PASS：18/18 任务**；每侧 3,613 条 planner、17,993 条 control、9,001 条 CAN 记录，最大绝对差值 **0.0** |
| 静态检查 | **PASS**：22 个变更 Python 文件解析，无新增 Ruff 问题；shell/JSON 和 `git diff --check` 通过 |
| ARM 保全 | **PASS**：14 个跟踪共享库均为 aarch64，全部原生源码/制品无 Git 差异 |
| 独立代码审查 | 无阻塞发现；没有新增控制权、平台门控或固件修改 |

最终合并执行结果：**197 passed、2 skipped、2 deselected，285 个子用例通过**。
另有原有 `test_get_message` 的多线程进程使用 `fork()` 弃用警告；没有测试失败。

测试保留的已知缺口：

- **SKIPPED（2）**：原有 manager 完整进程退出测试标记为 flaky；
  在线模型清单可用性测试要求外网。
- **DESELECTED（2）**：原有 `test_unregistered` 在基线的 HTTP 402 无限重试中
  挂起，已中断；`test_set_params_with_default_value` 依赖无 Git 快照缺少的
  构建信息和未选择构建的原生 bootlog，最初在基线失败。本次不修改这两项
  无关逻辑，另增了不联网、不运行 ARM bootlog 的 manager 初始化回归。
- 首轮综合测试受到本机 socket 沙箱限制，并在 GUI 自动探测显示器时崩溃。
  后台测试在允许本机 socket 的环境重跑；UI 夹具通过局部 `SCALE=1` 和
  禁止真实窗口创建的保护修复，验证运行后环境恢复。它们不是被计入通过的失败。
- 大屏侧栏原有 Ruff `F401`、`UP032`、`B905` 与基线一致，没有顺带修改。
- **NOT RUN**：真实窗口绘制/设备实测、资源占用测量、录制版 fidelity、
  额外预热收敛和新代码道路效果。
- **N/A**：本次云服务/UI 移除的完整驾驶 core/rotation、模型重新推理、
  新控制行为验收以及固件重编译/刷写。smoke 是补充的未改行为检查，
  不替代这些层面的测试或授权。

回放使用冻结的 `accord-v1`：
`manifest.json` SHA256 为
`22ff92653bcee0ebe842101083eee45f51983a665d0a15eb6fa264e9ffd83d61`；
smoke 计划 SHA256 为
`3b93d30cab903e00b7300f68e2efb6c4ede497f7a80f6c7f6757bd4173a26f89`。
浮点容差仍为 `1e-7`，CAN/离散字段精确比较。感知、radar、DEC/SP、
横向和车辆运动保持录制输入，不据此宣称重新验证整个驾驶系统。

本机原始结果保留在以下位置；它们不属于此 Git 提交，也不是路线数据备份：

- `/tmp/sunnylink-removal-final-tests.xml`
- `/tmp/sunnylink-removal-candidate-backend-tests.xml`
- `/tmp/sunnylink-ui-tests-with-translations.log`
- `/tmp/sunnylink-ui-tests-overlay.json`
- `/tmp/sunnylink-removal-static-checks.json`
- `/tmp/sunnylink-removal-regression-20260913/baseline-repeatability-summary.json`
- `/tmp/sunnylink-removal-regression-20260913/candidate-equivalence-summary.json`
- `/tmp/sunnylink-removal-regression-20260913/equivalence/summary.json`

重跑时使用新的隔离宿主目录和临时 HOME；原来的套件、阈值、评分窗口及失败记录
不得静默刷新。所有候选源码应来自本提交，不能只复制菜单文件而留下旧后台模块。
