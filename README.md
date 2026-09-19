# GOAI-Champ

HUST HRT GOAI 双 PIPER-X 真机测评部署代码快照。

本仓库当前归档的是 2026-09-20 在 **2223 机械臂电脑**实际运行的
`Pi05_PiperX` 客户端适配器，以及用于本地模拟官方任务派发的工具。

## 当前链路

```text
2223 eval-runner
  -> ws://127.0.0.1:6007
  -> 2223 Pi05_PiperX adapter
  -> SSH tunnel 127.0.0.1:6198
  -> L20 Pi_05 inference server
  -> real-piper6-lora/7594
```

当前执行方式是模型内 RTC 异步双缓冲模式，执行频率为 25 Hz。旧动作块先按
最新真机状态重对齐，再经 checkpoint 的 DeltaActions、quantile Normalize 和
PadStatesAndActions 变换为 `(50, 32)` 模型空间约束。Pi0.5 在每个 JAX 去噪步
应用 VJP/PiGDM 引导，前段强约束、后段逐渐衰减。同步前缀模式仍作为配置回退保留。

每次 RTC 请求绑定 `generation/request_id`，迟到结果不得替换当前计划。首次
`rtc_start` 会在允许取动作前执行一次零权重 guided 预热，避免首次 JAX 编译使
50 步队列在 25 Hz 下耗尽。

## 目录

- `robot_client/Pi05_PiperX/`: 2223 当前策略适配和执行代码。
- `l20_server/Pi_05/`: L20 Pi0.5 模型适配、策略变换与 JAX sampler 覆盖文件。
- `tools/local_pi05_eval.py`: 本地模拟官方任务派发工具。
- `tools/recorded_pi05_rtc_*.py`: 只读记录回放、A/B 和延迟测试。
- `docs/runtime-snapshot-2223.md`: 运行环境、文件来源和已知差异。

## 安全说明

仓库默认的 `motion_gate.json` 已将硬件输出关闭。现场实际运行参数原样保存在
`motion_gate.2223-runtime-snapshot.json`，复制或启用该配置可能导致机械臂运动。
任何真机运行必须有现场急停监护并单独确认。

`deploy.py` 是现场源码快照，其中仍包含通过
`/tmp/pi05_right_arm_probe_once.json` 触发右臂辨识动作的兼容路径。生产部署前应
删除该路径或改为显式、受审计的运维命令。

本仓库不包含 checkpoint、相机数据、日志、令牌、密码、私钥或 `.env`。
相关 XPolicyLab 代码按仓库中的 Apache-2.0 `LICENSE` 分发。

## 当前 L20 映射

L20 运行文件已逐文件读取、备份、部署并记录哈希。启动脚本当前映射是：

- `swap_arms=false`
- `swap_left_q4_q5=true`
- `swap_right_q4_q5=true`
- `right_wrist_perm=546`

这些现场实验性重排不是 checkpoint 发布格式本身的要求，仍是尚未消除的真机风险。
本次 RTC 改动没有改变这些映射。
