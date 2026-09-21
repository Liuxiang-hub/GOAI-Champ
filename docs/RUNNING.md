# 安装、运行与验证

## 1. 软件边界与依赖

这是部署覆盖仓库，运行于匹配的XPolicyLab/OpenPI源码、模型配置和检查点环境中。

| 用途 | 依赖来源 |
|---|---|
| 本仓库CPU单元测试 | Python、NumPy；见 requirements-test.txt |
| 录像回放 | 另需h5py、XPolicyLab图像解码及WebSocket客户端 |
| Pi0.5模型端 | 匹配的OpenPI及其JAX/Flax、模型加载和分词等完整依赖 |
| 机器人端服务与执行 | XPolicyLab及现场机器人、相机、通信依赖 |

测试依赖清单用于仓库CPU测试；模型端和机器人端依赖按对应XPolicyLab/OpenPI环境安装，不在现场环境中直接升级JAX或其他核心依赖。

在两端各自的运行环境中，可用以下只读命令收集版本，审阅输出、移除凭据及敏感路径后再保存提交：

```bash
python --version
python -m pip list --format=freeze
git rev-parse HEAD
git status --short
```

对XPolicyLab和OpenPI分别记录提交号；若源码有未提交覆盖改动，另记文件哈希。GPU端还需记录驱动及JAX实际可见设备。

## 2. 放置代码和检查点

决赛初版现场端：将 `robot_client/HRT1` 对应到 XPolicyLab 的 `policy/HRT1`，
并先把仓库安全模板复制为现场 `motion_gate.json`。L20 端按
[`l20_server/HRT1`](../l20_server/HRT1/README.md)安装适配器和端口路由；模型
服务使用 `l20_server/Pi_05/deploy.pi05-goai10000.yml`。
覆盖前保存原文件，并核对上游版本；本仓库未完整收录上游工程。

权重获取与配置见[模型说明](MODEL_WEIGHTS.md)。模型服务使用上游XPolicyLab/OpenPI的模型端入口；机器人端deploy.yml仅用于策略适配器，不能用于启动模型服务。

同一训练配方下切换权重时，保持 `ckpt_name: pi05-goai6-piper` 不变，只修改
L20运行配置中的一行：

```yaml
checkpoint_num: <新的训练保存步>
```

重启L20模型服务后，从启动日志确认实际checkpoint路径、训练配置和assets加载成功，
再通过关闭硬件输出的观测回放确认返回形状仍为 `(50, 14)`。机器人端运行逻辑
和执行前缀不随训练保存步修改。

## 3. 先验证CPU逻辑

在本仓库根目录、隔离的测试环境中执行：

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

这些测试使用模拟对象，不调用机械臂，也不证明完整模型性能。

## 4. 适配器入口

在已准备好的XPolicyLab根目录、对应运行环境中执行：

```bash
python setup_policy_server.py --config_path policy/HRT1/deploy.yml
```

入口脚本来自外部XPolicyLab，不包含在本仓库中。

当前决赛初版为同步前缀执行：两份配置均设置
`execution_mode=synchronous_prefix`，每块 50 步预测执行前 20 步，名义控制
频率 25 Hz。执行循环以 `motion_gate.json` 为准。
公开配置保持hardware_output_enabled=false；历史2223快照不应用作当前配置替代品。
当前现场记录及验证范围见[决赛初版部署快照](FINALS_V1_20260921.md)。
`FIELD_STATUS_20260920.md` 记录的是上一版 Pi05_PiperX 链路，不作为当前 HRT1
端口、检查点或动作前缀的配置来源。

## 5. 辅助工具

| 工具 | 用途 |
|---|---|
| tools/local_pi05_eval.py | 本地模拟任务派发与评测流程 |

归档工具保留用于版本追溯，不属于当前同步前缀部署流程。相关脚本包含特定环境路径和端点，录像也不随仓库公开。

`tools/local_pi05_eval.py`会派发和启动评测任务，使用前需确认目标环境及硬件门控。

## 6. 保存运行记录

每次运行记录源码提交、模型标识、依赖版本、实际配置、端点、任务和耗时。同步前缀方案同时记录请求等待、动作执行节奏、任务次数及人工结果。
