# 本地修改与优化说明

本文仅列出 `c3xl-dev` 从本地开发起点 `b6c8ff1c6` 到源码版本 `796e5921e` 之间，由本地开发新增且当前仍生效的修改与优化，供安装前比较。不包含上游仓库的同步、cherry-pick 或直接移植；在上游功能上额外完成的本地修正，只说明本地增加的部分。按首次本地修改的先后顺序排列，同一功能的后续修订合并为当前行为；未列入已撤回或仅用于离线试验的方案。

安装前请先了解适用条件：

- **当前包含专用硬件及开发配置。** 驾驶员摄像头监控停用，使用 C3XL 的 GPIO 蜂鸣器；未系安全带事件检查也在代码中关闭，源码将其标为台架／开发用途。这些是当前版本的实际行为，不能当成通用设备配置。
- **统一纵向控制限定雅阁平台。** 仅在 `HONDA_ACCORD`、启用 openpilot 纵向控制且没有油门拦截器时启用。下文的统一纵向跟车、停车、起步及故障恢复均指这一组合；使用原厂 ACC 或其他车型时，不会自动获得这些优化。
- **雅阁雷达融合还有额外条件。** 需要开启 Alpha Longitudinal、Experimental Mode、DEC 和 Radar Signal，并满足当前输入及控制条件；雷达叠加显示还需要单独打开显示开关。
- **Blendful 是默认关闭的上游可选功能。** 在 Steering／转向设置中开启后，用于 v2 模型。它保留滤波车道置信度、已学习宽度和历史拟合；新画面不再检测到车道线时，仍可能使用此前有效观测形成的拟合，没有固定硬超时。低速、转弯或变道等条件下会渐退，损坏／失效输入会清空修正状态，关闭功能后不再输出该修正；模型状态也可以在横向控制未接通时积累。其曲率修正既影响转向，也可能经自适应 SCC 改变加减速，并非只影响显示；不另行取得启用或重新启用控制的权限，也没有独立改动 FCW 开关，经曲率和 SCC 产生的间接影响未在本次发布中单独验证。

1. **调整停车、低速拖行和起步衔接。** 增加防拖行判断、停车保持和起步输出渐变。当前防拖行辅助必须有仍然有效的停车依据；新的有效观测撤销停车原因后，行驶中的车辆可以继续跟随，避免仅凭旧停车状态反复制动。

2. **调整雅阁低速转向响应。** 在使用对应 PID 横向调校时，低速降低控制增益，随车速升高逐步恢复，针对低速方向盘来回修正的问题。其他横向控制器不直接使用这组增益。

3. **增加安全带事件的代码级开关。** 当前开关为关闭，未系安全带状态不会单独产生 openpilot 的拒绝启用或退出事件。它不是设置页中的用户开关，当前配置保留了源码注明的台架／开发用途限制。

4. **恢复部分原先隐藏的设置入口。** 恢复独立开关面板、开发者设置，以及踩油门退出、驾驶偏好、录音录像等现存选项，便于按使用习惯配置。

5. **修正滑行判断中的俯仰偏置。** 估计缓慢变化的安装偏差，减少把摄像头安装角当成道路坡度而产生的不必要减速限制；这不等同于新增对所有坡道、所有模式通用的补偿。

6. **明确驾驶员摄像头配置并恢复输入健康检查。** 对当前座舱摄像头不可用的硬件配置明确停用该输入，避免等待不存在的摄像头影响道路摄像头启动；恢复相关进程、通信、传感器和 GPS 异常提示。驾驶员摄像头监控在当前配置下仍然停用。

7. **移除局域网录像网页和 C3 Director 远程命令服务。** 不再提供原来的 LAN dashcam 网页与 C3 Director 服务。该项不代表关闭 SSH、所有后台连接或所有数据上传。

8. **降低启动、控制和界面的运行开销。** 为当前设备指定 QCOM 后端，减少启动时不必要的后端探测与模块加载，调整后台进程的 CPU 分配；复用控制消息、雷达解码和路径投影结果，减少重复校验与内存分配。界面减少重复布局和文字测量，IP 地址查询移到后台。学习器对已确认无效的窗口提前退出，三个姿态消费者只转换所需角速度；规划器跳过空目标归因和已安装的相同默认约束，界面在同一更新内共享归因解码，并减少两个硬件状态参数的重复写入。补充 CPU4 消息处理与 CAN 解码优化，按需执行启动依赖检查；模型冷启动完成首次有效推理后才启用实时调度，避免冷初始化过早占用实时线程。状态变化、输入有效期和失败恢复仍逐次检查。仅归纳当前保留的优化，不承诺固定的帧率、温度或耗时收益。

9. **完善 C3XL 蜂鸣提示。** 补上紧急接管和重复提醒原先没有声音的问题，区分紧急与普通提醒；使用稳定 GPIO 电平驱动，移除软件 PWM 循环，保留各提示的名义鸣响时长。适用于当前 GPIO 蜂鸣器配置，实际是否播放还受 Sound 选项控制。

10. **调整雅阁巡航滑行与轻制动衔接。** 在符合条件的小幅巡航减速区间使用滑行映射，并保持已经开始的轻制动连续；当前巡航请求由单一调速逻辑执行，避免重复叠加反馈。

11. **调整接近前车时的制动响应。** 对已确认且接近速度较大的前车，调整舒适性平滑与当前制动需求的协调，完善近距离减速和停车衔接。

12. **修复模型文件加载与内置依赖识别。** 加强模型文件、分片数量和完整性的校验，统一显式指定模型文件时的判断；按随代码提供的 tinygrad 实际版本识别依赖，减少导入整合后的加载和版本判断问题。

13. **修正哈弗 H6 与 MG 的接口兼容和控制边界。** 修复控制参数传递及消息对象读写兼容，避免直接改动输入消息；哈弗 H6 的取消命令和踩刹车优先于启用，取消报文清除启用位，并拒绝异常控制报文。MG 仍是仅行车记录模式，这项修改不扩展其控制支持范围。

14. **修正大众／奥迪小转向扭矩的削减边界。** 在持续相同转向扭矩的保护调整中，将削减量限制在当前扭矩幅值以内，避免小扭矩被削减成相反方向的请求。

15. **移除 sunnylink 云服务。** 移除其注册、配对、云端配置／备份和界面入口。原有其他 API、Athena 连接及上传进程并未随之全部关闭。

16. **雅阁采用统一纵向控制。** 将巡航、跟车、模型停车意图和实际停车状态放到同一控制流程，协调跟踪、接近停车、保持和释放。加入有界的起步补偿与制动反馈；当前目标已经转弱、车辆仍过度减速时，加快释放残留的制动修正。生效范围见开头的雅阁平台条件。

17. **移除离线地图、Trips 和 Firehose 功能。** 删除 OSM 地图及地图弯道控速、Trips 页面和 Firehose 相关入口与后台实现。视觉弯道控速、车辆来源的限速信息和原始驾驶日志仍有各自的处理路径。

18. **改进视觉弯道控速。** 修正直道及弯道速度目标，根据当前弯道形状恢复速度；驾驶员在弯中踩油门后松开，会以松开时的速度表达本次弯道选择。转向饱和还需达到有效横向需求才参与限制，减少低速轻微转向误差带来的额外限速。需要启用对应视觉弯道控速功能。

19. **增加有条件的低速紧凑停车目标。** 持续低速跟随、前车身份稳定、确认正在停车且满足当前停车余量检查时，可采用更紧凑的目标间距；目标变化或观测异常后重新确认。舒适目标可以收紧到约 3 米，但不是对实际停车距离的保证。

20. **增加雅阁雷达目标可视化。** 显示目标位置、距离和横向边界，并略微加宽辅助生效时的路线显示；在已有目标标记上展示其是否影响当前控制以及作用强弱。尚未通过测量资格的雷达点为橙色，已通过资格但未被控制采用的点为绿色，参与控制的目标用由浅到深的红色表示。颜色不是检测置信度、碰撞概率或实测制动力。雷达叠加显示默认关闭。

21. **按纵向控制状态保存高清道路录像。** 高清前向／广角录像在纵向控制期间记录，退出后再保留 10 秒；对应开关当前默认开启，可停车设置、下次行程生效。低清录像与驾驶日志继续记录，不随高清录像窗口一起停止。

22. **将雅阁当前雷达目标用于纵向控制。** 通过新增的雷达处理通路，把符合条件的雷达目标与视觉目标结合，用于跟车、减速、停车和前车起步判断。正式规划使用两个前方目标槽位，并按当前路径与车道位置筛选；不是额外多目标预测避障功能。

23. **区分纵向规划故障和通信故障。** 纵向规划不可用时给出对应提示，保留独立的通信与输入健康检查，便于判断需要接管的功能。

24. **修正特定广角相机的道路叠图。** 根据当前设备实际消费的身份字段，对 `tici`、道路／广角传感器均为 `ox03c10`、1928×1208 内参坐标基底的广角画面应用投影校正。修复了仅按日志头中的 `tizi` 判断而漏用校正的问题；窄角画面和其他未验证身份不应用这组专用校正。这是界面叠图修正，不改变模型输入或镜头物理标定。

25. **增加短暂数值求解失败的有界衔接。** 仅在当前目标、车辆状态、限制和控制权限仍合格时，允许已确认执行的有效轨迹短暂接续。当前这类故障的接续最多 100 毫秒，还受原参考源、模型和计划的更早截止时间限制；持续异常或资格撤销仍需接管。

26. **完善极低速末段停车规划。** 规划器可以在首个预测间隔内表达“刹停后保持静止”，改善极短距离停车在离散预测中的表达，保留对实际制动能力的约束。

27. **分别处理转向与车速控制的故障和接管。** 在统一纵向模式且已开启 MADS 时，针对独立 EPS 故障或纵向规划故障，只有另一方向的独立输入及执行权限仍健康，才保留其已启用的辅助。停车接管保持单独判断：符合条件的纵向规划故障中，车辆已实际刹停且制动链路健康时，可在原退出期限外获得最多 3 秒的保持和倒计时，不会因规划恢复自动松刹起步。纵向故障退出后需确认恢复，再由驾驶员通过 SET 重新启用；踩刹车接管后及时清理对应提示，失效规划遗留的 FCW 状态也会清除。

28. **增加基于实测车身侵入的切入判断。** 要求雷达目标有连续的进入趋势，且当前测得的车身已侵入行驶区域；近距离采用更严格的侵入条件。缺乏同目标视觉确认的近距离侧向窄反射点，不能直接取得普通跟车／停车资格；视觉确认和实际车身切入分别判断。该功能不以物体类别识别来保证排除护栏或其他反射物。

29. **改进前车起步后的跟随释放。** 当前雷达或视觉持续确认前车起步、且没有当前停车或制动否决时，可以由跟车规划接续起步，减少仅等待模型自车速度明显变正造成的延迟。接续仍要求当前观测，最长 5 秒、最多前进 5 米，或车速达到 2 m/s 时退出该阶段。

30. **缩短统一纵向标准模式的部分跟车时距。** 标准模式在不超过 80 km/h 时采用 1.25 秒目标时距，80–100 km/h 平滑恢复至 1.45 秒。该变化使对应速度范围的跟车目标更紧凑；目标时距不是固定距离，也不代表保证的实际车距。

31. **按当前有效目标重新规划。** 目标被新观测撤销、过期或失去资格后，撤销相关控制约束，并重建依赖该目标的历史规划代价，避免旧目标继续保留停车或减速要求。正常异步扫描仍有短暂复用：测距和测速按各自最后真实测量计龄，消费上限约 200 毫秒；测距超过约 100 毫秒进入预测状态，或收到明确撤销时，会更早失去雷达控制资格。非活动目标身份最多保留 500 毫秒用于关联，不保留旧目标的停车／制动要求。普通有效计划交接不超过原计划／模型／参考的最早 250 毫秒截止时间，不能靠重复使用续期。

32. **加强雷达原始测距、测速的一致性检查。** 正式控制要求连续的原生距离与速度配对，并检查其与距离变化是否一致；改进同一扫描分批到达、目标槽位迁移及重新获取的处理。首次准入后仍持续核对原生测距与 AUX 速度，考虑量化及离散观测区间；矛盾的新观测会撤销原有资格。正式雷达纵向速度不再由位置差分或视觉速度代替。

33. **改进模型短暂延迟后的恢复和提示。** 连续健康的新模型输入至少持续 1 秒，并通过规划和实际消费检查后，允许尚未退出的控制恢复正常状态，减少历史丢帧统计持续触发退出。合并相关模型与定位延迟提示，并区分输入已恢复和仍需接管转向。已经退出的控制不会因此自动重新接管车辆。

34. **新增 Radar Signal 开关。** 入口位于 Cruise／巡航，默认开启；停车时可选择下一次行程是否采用雷达目标，整次行程保持该选择。雷达目标显示开关位于 Visuals／外观。关闭后，辅助驾驶及相关目标显示只采用视觉目标；车辆原始 CAN、安全状态和其他输入健康检查仍独立保留。

35. **新增 Sound 蜂鸣模式。** 普通／Sunnypilot 和 mici 设置页均可选择 None、Warning、All，默认 All。None 关闭全部已映射蜂鸣，包括紧急警告，并停止当前鸣响；Warning 保留警告、提醒和拒绝启用提示，屏蔽启用／退出提示；All 允许全部现有提示。选择会在蜂鸣器主循环中生效并持久保存，参与参数备份；不改变上游告警生成和车辆控制逻辑。参数库同时修复按标志枚举时名称缓冲区被复用的问题，新参数需要匹配的设备 ARM 库。

36. **改善无前车停车后的正常起步识别。** 仅在雅阁停车状态、原始／估计车速及四轮当前测量都支持停稳、没有当前前车、输入健康且模型明确支持前行时，允许有界的近零预测残差，避免极小的预测负速度一直挡住起步。实际轮速必须有新测量，缺失或过期不会当成零速；停车、制动及故障否决仍保留。

37. **完善 Blendful 的本地接入和异常输入处理。** 拒绝重复或倒序的摄像头观测重新建立修正，隔离名义模型动作历史，并完善异常输入处理、渐退和界面缓存。此项仅列本地增加的处理；上游算法本身及其车道历史行为见开头的适用条件。

38. **保留启动时已收到的雅阁固件信息并补充识别。** 启动查询回退时，保留同次启动已经实际收到的 EPS 回复，避免后续漏答抹掉已观察到的转向固件信息；补充本地记录的雅阁摄像头和雷达固件回复。该修改不代表新增所有雅阁车型支持。

<!--
维护依据：以下信息供核对本地来源、顺序与当前代码使用，不是安装验证报告。
repository: openpilot-3xl-0817
branch: c3xl-dev
baseline: b6c8ff1c6472dbcfec84a7f7d61015f2ad3314d0
source: 796e5921e382fae827430b5a9faa26bb7ac631ce
scope: local-only; upstream imports excluded; mixed entries describe only the local delta
order: first local-change commit in git log --first-parent --reverse baseline..source
来源判定使用提交正文、原始提交和已获取的远端历史，不仅凭作者或本地提交时间。
同主题后续本地修订按正文合并；同一提交引入的不同主题可有相同排序依据。
模型加载及车型接口修复起于 754ab98f9；车型控制边界包含 bb66cdf80，模型和转向边界包含 fe12cd4e2。
intro | item | current evidence (paths relative to repository)
c657fa948 | 1 | openpilot/selfdrive/controls/lib/motion_supervisor.py; openpilot/selfdrive/controls/lib/longcontrol.py
89b57eb61 | 2 | opendbc_repo/opendbc/car/honda/interface.py
683398a73 | 3 | openpilot/selfdrive/car/car_events.py
e729509e7 | 4 | openpilot/selfdrive/ui/layouts/settings/toggles.py; openpilot/selfdrive/ui/sunnypilot/layouts/settings/settings.py
cea261879 | 5 | openpilot/selfdrive/controls/lib/longitudinal_planner.py
51b20425a | 6 | launch_env.sh; openpilot/system/manager/process_config.py; openpilot/selfdrive/selfdrived/selfdrived.py
99859fd8f | 7 | openpilot/system/manager/process_config.py; removed:dashcam.py; removed:openpilot/selfdrive/c3_client.py
5d7286bb1 | 8 | launch_env.sh; openpilot/selfdrive/ui/layouts/sidebar.py; openpilot/selfdrive/ui/onroad/model_renderer.py; openpilot/selfdrive/car/honda_radar_native.py
06907bd0c | 9 | openpilot/selfdrive/selfdrived/beep.py
1115e7a94 | 10 | openpilot/selfdrive/controls/lib/longitudinal_planner.py; openpilot/selfdrive/controls/lib/cruise_response.py; openpilot/selfdrive/controls/lib/longcontrol.py
2ccb0539f | 11 | openpilot/selfdrive/controls/lib/longitudinal_planner.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py
754ab98f9 | 12 | openpilot/sunnypilot/modeld_v2/modeld.py; openpilot/sunnypilot/models/tinygrad_ref.py; openpilot/sunnypilot/models/tinygrad_vendor.json
754ab98f9 | 13 | opendbc_repo/opendbc/car/gwm/interface.py; opendbc_repo/opendbc/car/gwm/carcontroller.py; opendbc_repo/opendbc/car/gwm/carstate.py; opendbc_repo/opendbc/car/gwm/gwmcan.py; opendbc_repo/opendbc/car/mg/carcontroller.py; opendbc_repo/opendbc/safety/modes/gwm.h
fe12cd4e2 | 14 | opendbc_repo/opendbc/car/volkswagen/carcontroller.py
e18b1de11 | 15 | openpilot/system/manager/process_config.py; removed:openpilot/sunnypilot/sunnylink
58e5fc71b | 16 | openpilot/selfdrive/controls/lib/longitudinal_mode.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py; openpilot/selfdrive/controls/lib/launch_response.py
f84214c21 | 17 | openpilot/selfdrive/ui/sunnypilot/layouts/settings/settings.py; removed:openpilot/sunnypilot/mapd; removed:openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py
ad0a04ba1 | 18 | openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py
eff7e389f | 19 | openpilot/selfdrive/controls/lib/low_speed_stop_distance.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py
35087062a | 20 | openpilot/selfdrive/ui/sunnypilot/onroad/honda_radar_overlay.py; openpilot/selfdrive/ui/sunnypilot/onroad/lead_control_influence.py; openpilot/common/params_keys.h; openpilot/selfdrive/ui/sunnypilot/onroad/model_renderer.py
10e1a7b94 | 21 | openpilot/system/loggerd/road_recording.h; openpilot/selfdrive/ui/layouts/settings/toggles.py; openpilot/common/params_keys.h
ff9acb052 | 22 | openpilot/selfdrive/car/honda_radar_observer_config.py; openpilot/selfdrive/car/honda_radar_control.py; openpilot/selfdrive/controls/radard.py
3b5bbee60 | 23 | openpilot/selfdrive/selfdrived/selfdrived.py; openpilot/selfdrive/selfdrived/events.py
41b309524 | 24 | openpilot/selfdrive/ui/onroad/camera_projection.py
08cf26f4f | 25 | openpilot/selfdrive/controls/lib/longitudinal_continuation.py; openpilot/selfdrive/controls/lib/planner_continuation.py
a4a818335 | 26 | openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/interval_rest.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py
34ab08bd8 | 27 | openpilot/selfdrive/selfdrived/engagement_policy.py; openpilot/selfdrive/selfdrived/selfdrived.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py
4454b22ce | 28 | openpilot/selfdrive/car/honda_radar_control.py
3255b7ab4 | 29 | openpilot/selfdrive/controls/lib/lead_launch.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py
d522a554d | 30 | openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py
62845e719 | 31 | openpilot/selfdrive/car/honda_radar_control.py; openpilot/selfdrive/car/honda_radar_fusion.py; openpilot/selfdrive/controls/lib/lead_qualification.py; openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py; openpilot/selfdrive/controls/lib/reference_timing.py
cc0c60269 | 32 | openpilot/selfdrive/car/honda_radar_native.py; openpilot/selfdrive/car/honda_radar_control.py
4afacfc1d | 33 | openpilot/selfdrive/selfdrived/model_lag_recovery.py; openpilot/selfdrive/selfdrived/perception_alerts.py; openpilot/selfdrive/selfdrived/engagement_policy.py
0aff8a3b2 | 34 | openpilot/common/radar_signal.py; openpilot/selfdrive/ui/sunnypilot/layouts/settings/cruise.py; openpilot/selfdrive/ui/sunnypilot/layouts/settings/visuals.py
d5f78ebae | 35 | openpilot/common/sound_mode.py; openpilot/common/params_keys.h; openpilot/common/params_c.cc; openpilot/selfdrive/selfdrived/beep.py; openpilot/selfdrive/ui/layouts/settings/toggles.py; openpilot/selfdrive/ui/mici/layouts/settings/toggles.py
后续修订：ad81a3991 补充条目 8 的六项性能优化；5addda768 将条目 9 改为稳定 GPIO 输出；4d51d5731／68c86683a 修正条目 24 的生产身份条件；e8e3f5311 修复条目 35 的参数名称枚举。
13005eb52 | 36 | openpilot/selfdrive/controls/lib/model_reference.py; openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py; opendbc_repo/opendbc/car/honda/carstate.py
14afa9d50 | 37 | openpilot/sunnypilot/modeld_v2/modeld.py; openpilot/sunnypilot/modeld_v2/blendful.py
2b4873f0b | 38 | opendbc_repo/opendbc/car/car_helpers.py; opendbc_repo/opendbc/sunnypilot/car/honda/fingerprints_ext.py
-->
