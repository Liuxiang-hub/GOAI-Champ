# 🤖 GOAI-Champ：Pi0.5 双臂具身智能系统

![GOAI-Champ 双臂任务展示](assets/dual-arm-demo.png)

HUST HRT 面向 GOAI 2026 双臂赛道构建的 Pi0.5 真机部署与复现仓库。系统联合三路视觉、双臂关节与夹爪状态以及任务指令预测连续动作块，并通过同步前缀执行完成闭环控制。`HRT1` 是当前 Pi0.5 链路的比赛策略名与部署适配层，不代表另一套基础模型。

[运行指南](docs/RUNNING.md) · [配置说明](docs/CONFIGURATION.md) · [模型权重](docs/MODEL_WEIGHTS.md) · [决赛核查材料](docs/FINAL_SUBMISSION.md) · [现场记录](docs/FIELD_STATUS_20260920.md)

## 项目概览

| 项目 | 当前实现 |
|---|---|
| 核心策略 | Pi0.5，六项 GOAI 任务联合微调 |
| 机器人平台 | 双 PIPER-X，12 维机械臂关节 + 2 维夹爪 |
| 感知输入 | 顶部相机、左右腕部相机、双臂状态、任务指令 |
| 策略输出 | 50 步、14 维绝对关节与夹爪目标 |
| 执行方式 | 每轮执行动作块前 20 步，再更新观测并重新推理 |
| 名义控制频率 | 25 Hz；完整周期还包含观测、通信与模型推理时间 |
| 决赛初版检查点 | `pi05-goai6-piper/10000`；由 L20 运行配置选择 |
| 权重下载 | [ModelScope · LiuXiangg/HRT_GOAI](https://www.modelscope.cn/models/LiuXiangg/HRT_GOAI) |

> 本仓库保存决赛部署覆盖代码、配置、测试和复现文档，不是 XPolicyLab/OpenPI 的独立发行包。模型权重、原始数据、现场日志和访问凭据不提交到 Git。

## 核心技术路线

```text
三路相机 + 双臂状态 + 任务指令
                │
                ▼
       Pi0.5 多模态策略服务（L20）
                │
                ▼
        50 × 14 绝对目标动作块
                │
                ▼
   现场执行端坐标变换、关节映射与动作门控
                │
                ▼
      执行前 20 步 → 更新观测 → 下一轮
```

项目的重点不只是调用模型，而是完成模型动作空间与真实双 PIPER-X 控制链路之间的工程闭环：

- **多模态动作预测**：接入顶部与双腕图像、机器人本体状态和语言指令，输出双臂协同动作块。
- **检查点一致性**：服务端绑定训练配置、归一化统计与资产标识，避免只更换参数却遗漏配套统计。
- **双臂坐标适配**：处理关节顺序、方向和夹爪尺度，使模型输出与不同真机配置对齐。
- **同步前缀控制**：保留动作块的短时连贯性，并在每轮执行后重新获取观测，降低开环累积误差。
- **默认安全门控**：公开配置默认关闭硬件输出，真机启用需要现场急停监护和单独确认。

## 六项任务

| 任务 | 任务 | 任务 |
|---|---|---|
| Fill the Pen Holder | Insert the Charger | Put Objects into the Basket |
| Stack and Cover the Blocks | Stack the Bowls | Stand Up the Bottles |

封面图用于展示双臂操作场景，不作为单次模型成功率或比赛成绩证明。可复核的真机日志、模型版本及配置边界以[现场运行记录](docs/FIELD_STATUS_20260920.md)为准。

## 系统实现

| 项目 | 实现情况 | 说明 |
|---|---|---|
| Pi0.5 部署接口 | 已实现 | 50 步、14 维绝对动作；适配器和配置已开源 |
| 同步前缀执行 | 已实现 | 25 Hz 名义控制、每轮执行前 20 步 |
| 可选 RTC | 已实现、默认关闭 | 现场本地消费动作，后台异步请求 L20；单字段回退同步 20 步 |
| L20 策略适配器 | 已实现 | 公网入口转入 L20 本机适配器，再调用同机 10k 推理服务 |
| Robot 6 配置 | 已实现 | 记录端口、动作前缀和双侧 J5 坐标方向 |
| 三路视觉接入 | 已实现 | 顶部相机与左右腕部相机统一进入策略观测 |
| 数据产物保存 | 已实现 | 现场流程可保存任务 HDF5 与视频记录 |

现场配置、模型版本和运行记录见[现场运行说明](docs/FIELD_STATUS_20260920.md)。

## 目录结构

```text
GOAI-Champ/
├── assets/                         # README 展示素材
├── archive/                        # 与当前 Pi0.5 主线隔离的 LingBot 等历史技术栈
├── docs/                           # 核查、运行、配置、权重与现场记录
├── l20_server/                     # L20 模型服务适配与历史覆盖代码
├── l20_server/HRT1/                # 当前 Pi0.5 链路的 L20 HRT1 适配与路由单元
├── robot_client/HRT1/              # 当前 Pi0.5 链路的现场执行循环与运行快照
├── robot_client/Pi05_PiperX/       # 2026-09-20 的 Pi0.5 部署基线
├── tests/                          # 部署模式与变换回归测试
├── tools/                          # 本地任务派发、回放与链路验证
├── SOURCE_SHA256.md                # 历史源码文件校验记录
└── requirements-test.txt           # 公开测试依赖
```

## 快速核查

### 1. 获取代码

```bash
git clone https://github.com/Liuxiang-hub/GOAI-Champ.git
cd GOAI-Champ
```

### 2. 准备上游环境与权重

本仓库需要放入对应的 XPolicyLab/OpenPI 环境中使用。权重从 [ModelScope](https://www.modelscope.cn/models/LiuXiangg/HRT_GOAI) 获取，并保留发布目录中的参数、`assets` 与归一化统计。完整要求见[模型说明](docs/MODEL_WEIGHTS.md)。

### 3. 运行公开测试

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

### 4. 核对部署配置

正式运行前至少确认模型路径与 revision、训练配置、归一化资产、动作排列、左右臂映射、关节方向、服务端口及硬件输出门控。详见[运行指南](docs/RUNNING.md)与[配置说明](docs/CONFIGURATION.md)。

## 安全边界

- 仓库默认 `motion_gate.json` 关闭硬件输出。
- 历史运行配置不是通用真机参数，不应直接复制到另一台机械臂。
- `deploy.py` 中保留一次性右臂辨识兼容入口，生产部署前应删除或改成显式、可审计的运维命令。
- 真机运行必须由现场人员监护，并确认急停、关节限位、速度限制和坐标映射。

## 开源内容及价值

仓库公开 Pi0.5 双 PIPER-X 部署适配、双臂动作与状态变换、同步动作块执行、配置样例、回归测试以及决赛核查文档。它将模型权重、训练资产、服务端配置、机器人映射和现场证据组织成可追溯链路，帮助复现者判断“代码可运行”“推理输出有效”和“真机任务完成”分别由哪些材料支持。

上游框架、数据集、基础模型及机器人 SDK 受各自许可证约束。仓库内可分发代码以 [Apache-2.0](LICENSE) 发布。

## 文档索引

- [决赛技术真实性及一致性核查](docs/FINAL_SUBMISSION.md)
- [运行与部署步骤](docs/RUNNING.md)
- [配置字段与安全门控](docs/CONFIGURATION.md)
- [模型权重与检查点说明](docs/MODEL_WEIGHTS.md)
- [2026-09-20 现场运行核查](docs/FIELD_STATUS_20260920.md)
- [2026-09-21 决赛初版部署快照](docs/FINALS_V1_20260921.md)
- [文档修订记录](docs/DOCUMENTATION_REVIEW.md)
