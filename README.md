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

当前执行方式是同步前缀模式：每次用最新观测请求一个 `(50, 14)` 动作块，
执行前 15 步，再采集新观测重新推理。执行频率为 25 Hz，RTC 代码保留但当前
主执行循环未启用 RTC。

## 目录

- `robot_client/Pi05_PiperX/`: 2223 当前策略适配和执行代码。
- `tools/local_pi05_eval.py`: 本地模拟官方任务派发工具。
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

## 已知缺口

L20 当前运行的 `Pi_05/model.py` 启用了实验性关节重排，但新的 SSH 登录目前不可用，
因此本次提交没有把无法逐文件校验的 L20 源码冒充为最新快照。已观测到的运行参数是：

- `swap_arms=false`
- `swap_left_q4_q5=true`
- `swap_right_q4_q5=false`
- `right_wrist_perm=654`

这些重排不是 checkpoint 发布格式要求，详见运行快照文档。恢复 L20 文件访问后，
应将服务端代码和启动配置单独归档并记录 SHA256。
