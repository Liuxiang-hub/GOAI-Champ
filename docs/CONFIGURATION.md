# 配置与执行模式

## 配置文件的职责

| 文件 | 用途 |
|---|---|
| robot_client/Pi05_PiperX/deploy.yml | 模型适配器、端点和相机映射 |
| robot_client/HRT1/deploy.yml | 决赛初版 HRT1 现场策略配置 |
| robot_client/HRT1/motion_gate.json | HRT1 安全模板；默认关闭硬件输出 |
| robot_client/HRT1/motion_gate.robot6-finals-v1.json | 2026-09-21 Robot 6 运行快照，仅供审计 |
| motion_gate.json（同目录） | 执行循环的模式、频率及动作安全限制；公开默认关闭硬件输出 |
| motion_gate.2223-runtime-snapshot.json | 2223历史现场参数，包含启用硬件输出；仅作版本对照 |
| motion_gate.robot1-v1.json | Robot 1同步前缀配置，存档中关闭硬件输出 |
| piper6_norm_stats.json | 收录的统计文件；模型实际加载统计来自服务端检查点assets |

执行循环由motion_gate.json读取模式、频率、执行前缀和安全门控；机器人端
deploy.yml同时保存适配器参数。2026-09-21 决赛初版在两处均记录
`execute_steps=20`，切换配置时必须保持一致。旧 Pi05_PiperX 目录保留此前
15 步基线，不代表当前 HRT1 运行值。

## 当前公开参数

| 参数 | 值 | 含义 |
|---|---|---|
| prediction horizon | 50 | 每块动作步数 |
| control_hz | 25 | 名义执行频率 |
| execute_steps | 20 | 每次消费预测动作块的前20步，再更新观测 |
| execution_mode | synchronous_prefix | 当前现场方案及公开默认模式 |

## 检查点切换

L20运行配置是模型加载的唯一入口。同一训练配方下保持
`ckpt_name=real-piper6-lora`，只修改 `checkpoint_num`。机器人端配置中的
checkpoint文本不参与模型解析。

`checkpoint_num`、sampler默认`num_steps`和机器人端`execute_steps`分别表示
训练保存步、推理去噪次数和动作执行前缀，不能互相替代。

现场核查及配置来源见[现场运行说明](FIELD_STATUS_20260920.md)。25Hz描述动作块内的名义频率，完整循环还包含通信和推理等待。

部署文件中保留的历史兼容字段不参与 `synchronous_prefix` 执行循环，当前参数以 `motion_gate.json` 中的同步前缀配置为准。

## 动作与观测

动作顺序为 `[left J1..J6, left gripper, right J1..J6, right gripper]`；关节按弧度，夹爪范围[0,1]。模型输出为绝对目标，关节增量转换在检查点输入变换链中完成。模型空间32维动作不能直接发送给机械臂。

顶部图像支持cam_head/cam_high别名；双腕使用cam_left_wrist和cam_right_wrist。适配器要求有效RGB uint8图像、有限状态值和非空任务指令。

## 映射与运行版本

服务端读取PI05_PIPER_SWAP_ARMS、PI05_PIPER_SWAP_J4_J5、PI05_PIPER_SWAP_LEFT_J4_J5、PI05_PIPER_SWAP_RIGHT_J4_J5和PI05_PIPER_RIGHT_WRIST_PERM环境变量。

当前发布接口采用标准14维动作顺序。历史快照中的腕部排列和手臂交换记录仅用于版本追溯，不能作为当前检查点的标准动作顺序。

## 硬件输出

motion_gate.json默认关闭输出；执行代码在非debug模式检查该门。debug模式会绕过该检查，所以不能把设置debug视作硬件隔离；离线验证必须使用不实例化硬件后端的测试或回放工具。

deploy.py保留由 /tmp/pi05_right_arm_probe_once.json 触发的右臂辨识路径，文档整理不改变此代码。真机操作前须确认该路径不会意外触发。
