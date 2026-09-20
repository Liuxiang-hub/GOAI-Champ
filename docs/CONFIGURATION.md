# 配置与执行模式

## 配置文件的职责

| 文件 | 用途 |
|---|---|
| robot_client/Pi05_PiperX/deploy.yml | 模型适配器、端点、相机映射和RTC初始化 |
| motion_gate.json（同目录） | 执行循环的模式、频率及动作安全限制；公开默认关闭硬件输出 |
| motion_gate.2223-runtime-snapshot.json | 2223历史现场参数，包含启用硬件输出；仅作版本对照 |
| motion_gate.robot1-v1.json | Robot 1同步前缀配置，存档中关闭硬件输出 |
| piper6_norm_stats.json | 收录的统计文件；模型实际加载统计来自服务端检查点assets |

执行循环由motion_gate.json读取模式和频率，RTC控制器由deploy.yml初始化。两侧涉及同一概念的参数必须核对一致，不能只改一侧并假定另一侧自动跟随。同步配置存档不会自动生效。

## 当前公开参数

| 参数 | 值 | 含义 |
|---|---|---|
| prediction horizon | 50 | 每块动作步数 |
| control_hz | 25 | 名义执行频率 |
| execute_steps | 15 | 每次消费预测动作块的前15步，再更新观测 |
| execution_mode | synchronous_prefix | 当前现场方案及公开默认模式 |

现场核查及配置来源见[现场运行说明](FIELD_STATUS_20260920.md)。25Hz描述动作块内的名义频率，完整循环还包含通信和推理等待。

## 实验功能参数

仓库保留rtc_trigger_step=20、rtc_initial_delay_steps=11、rtc_prewarm_guided=true等实验参数；仅在显式选择对应实验执行模式时使用。历史快照中的第5步触发和22步延迟估计属于旧配置，不代表当前默认值或实测延迟。

## 动作与观测

动作顺序为 `[left J1..J6, left gripper, right J1..J6, right gripper]`；关节按弧度，夹爪范围[0,1]。模型输出为绝对目标，关节增量转换在检查点输入变换链中完成。模型空间32维动作不能直接发送给机械臂。

顶部图像支持cam_head/cam_high别名；双腕使用cam_left_wrist和cam_right_wrist。适配器要求有效RGB uint8图像、有限状态值和非空任务指令。

## 映射与运行版本

服务端读取PI05_PIPER_SWAP_ARMS、PI05_PIPER_SWAP_J4_J5、PI05_PIPER_SWAP_LEFT_J4_J5、PI05_PIPER_SWAP_RIGHT_J4_J5和PI05_PIPER_RIGHT_WRIST_PERM环境变量。

原README记录右腕546、右臂交换开启；历史快照记录右腕654、右臂交换关闭。这些是不同文档记录，尚不能确定哪个等于最终现场值。不能将实验性重排作为检查点标准动作顺序。

## 硬件输出

motion_gate.json默认关闭输出；执行代码在非debug模式检查该门。debug模式会绕过该检查，所以不能把设置debug视作硬件隔离；离线验证必须使用不实例化硬件后端的测试或回放工具。

deploy.py保留由 /tmp/pi05_right_arm_probe_once.json 触发的右臂辨识路径，文档整理不改变此代码。真机操作前须确认该路径不会意外触发。
