# GOAI-Champ

HUST HRT GOAI 双 PIPER-X 真机测评部署代码快照。

## 决赛核查材料

本仓库是依赖XPolicyLab/OpenPI的部署覆盖代码，不是独立安装包。
材料入口：[核查说明](docs/FINAL_SUBMISSION.md)、[运行指南](docs/RUNNING.md)、
[配置说明](docs/CONFIGURATION.md)、[修订记录](docs/DOCUMENTATION_REVIEW.md)。
权重：[ModelScope · LiuXiangg/HRT_GOAI](https://www.modelscope.cn/models/LiuXiangg/HRT_GOAI)。
检查点对应关系及文件核验状态见[模型说明](docs/MODEL_WEIGHTS.md)。

项目以 Pi0.5 为核心，结合三路视觉、双臂关节与夹爪状态以及任务指令预测动作块，
通过机器人端适配器执行同步前缀控制。
2026-09-20 现场核查见[现场运行说明](docs/FIELD_STATUS_20260920.md)。
仓库同时保留历史部署快照与实验代码；最终一致性仍需核对实际加载文件。

## 系统结构

```text
测评环境（三路图像、双臂状态、任务指令）
  -> Pi05_PiperX / 现场 HRT 策略适配器
  -> L20 Pi0.5 模型服务
  -> 50 步双臂动作预测
  -> 同步前缀执行与下一轮观测更新
```

当前现场执行方案及仓库默认模式为 `synchronous_prefix`：模型返回50步动作，
客户端执行前15步后更新观测并请求下一块，名义控制频率为25 Hz。
该频率描述动作块内的节奏，完整循环还包含观测、通信和推理等待。
关节顺序、方向及夹爪尺度由适配层与检查点变换共同处理。

RTC异步调度与模型内VJP/PiGDM引导保留为实验功能，供回放与后续研究使用。
现场反馈尚未达到预期，当前正式执行方案采用同步前缀控制；实验代码不代表
已验证的真机性能收益。

## 目录

- `robot_client/Pi05_PiperX/`: 策略适配、执行代码和公开默认配置。
- `l20_server/Pi_05/`: L20 Pi0.5 模型适配、策略变换与 JAX sampler 覆盖文件。
- `tools/local_pi05_eval.py`: 本地模拟官方任务派发工具。
- `tools/recorded_pi05_rtc_*.py`: 只读记录回放、A/B 和延迟测试。
- `docs/FIELD_STATUS_20260920.md`: 最近现场配置与真机采集记录。
- `docs/runtime-snapshot-2223.md`: 历史运行环境、文件来源和已知差异。

## 安全说明

仓库默认的 `motion_gate.json` 已将硬件输出关闭。历史运行参数保存在
`motion_gate.2223-runtime-snapshot.json`，不代表最新现场配置，复制或启用可能导致机械臂运动。
任何真机运行必须有现场急停监护并单独确认。

`deploy.py` 是现场源码快照，其中仍包含通过
`/tmp/pi05_right_arm_probe_once.json` 触发右臂辨识动作的兼容路径。生产部署前应
删除该路径或改为显式、受审计的运维命令。

本仓库不包含 checkpoint、相机数据、日志、令牌、密码、私钥或 `.env`。
相关 XPolicyLab 代码按仓库中的 Apache-2.0 `LICENSE` 分发。

## L20映射记录（待现场核定）

本README原始部署记录给出的映射是：

- `swap_arms=false`
- `swap_left_q4_q5=true`
- `swap_right_q4_q5=true`
- `right_wrist_perm=546`

这些现场实验性重排不是 checkpoint 发布格式本身的要求，仍是尚未消除的真机风险。
历史快照另记右腕654、右臂交换关闭。最终值须以实际进程的配置核定，源码默认值不能证明现场值。
