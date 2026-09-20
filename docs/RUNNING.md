# 安装、运行与验证

## 1. 软件边界与依赖

这是部署覆盖仓库，需要匹配的XPolicyLab/OpenPI源码、模型配置和检查点。仓库未提供现场Python版本、上游提交号或完整锁文件；不能宣称任意新环境安装后即可复现现场。

| 用途 | 依赖来源 |
|---|---|
| 本仓库CPU单元测试 | Python、NumPy；见 requirements-test.txt |
| 录像回放 | 另需h5py、XPolicyLab图像解码及WebSocket客户端 |
| Pi0.5模型端 | 匹配的OpenPI及其JAX/Flax、模型加载和分词等完整依赖 |
| 机器人端服务与执行 | XPolicyLab及现场机器人、相机、通信依赖 |

测试依赖清单不是部署锁文件。不在现场环境中直接升级JAX或其他依赖；按现有上游锁文件安装，最终补充实际环境版本。

在两端各自的运行环境中，可用以下只读命令收集版本，审阅输出、移除凭据及敏感路径后再保存提交：

```bash
python --version
python -m pip list --format=freeze
git rev-parse HEAD
git status --short
```

对XPolicyLab和OpenPI分别记录提交号；若源码有未提交覆盖改动，另记文件哈希。GPU端还需记录驱动及JAX实际可见设备。

## 2. 放置代码和检查点

机器人端：将robot_client/Pi05_PiperX对应到XPolicyLab的policy/Pi05_PiperX。
模型端：按[l20_server说明](../l20_server/README.md)的三个文件映射部署到匹配工程。
覆盖前保存原文件，并核对上游版本；本仓库未完整收录上游工程。

权重获取与配置见[模型说明](MODEL_WEIGHTS.md)。模型端部署YAML及实际完整启动命令尚未随仓库收录，须从现场实际使用版本补充；不要使用机器人端deploy.yml启动模型端。

## 3. 先验证CPU逻辑

在本仓库根目录、隔离的测试环境中执行：

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

这些测试使用模拟对象，不调用机械臂，也不证明完整模型性能。

## 4. 模型端变换与引导测试

tools/test_pi05_model_rtc.py检查14到32维动作变换以及简化速度场中的引导效果。它依赖匹配的OpenPI、训练配置和真实assets统计，不是端到端GPU时延评测。

文件末尾检查点路径写死为现场路径；换机器应先将路径改为实际检查点，再在能导入覆盖后OpenPI的环境中运行：

```bash
python tools/test_pi05_model_rtc.py
```

## 5. 适配器入口

在已准备好的XPolicyLab根目录、对应运行环境中执行：

```bash
python setup_policy_server.py --config_path policy/Pi05_PiperX/deploy.yml
```

入口脚本来自外部XPolicyLab，不包含在本仓库中。该命令不是本次文档修订已执行的命令。

## 6. 无硬件回放与比较

| 工具 | 检查内容 |
|---|---|
| tools/recorded_pi05_rtc_ab.py | 固定录像观测上的普通/引导动作比较 |
| tools/recorded_pi05_rtc_multichunk.py | 多动作块RTC API运行与切换 |
| tools/benchmark_pi05_rtc_recorded.py | 普通与引导推理延迟 |

这些脚本当前硬编码了XPolicyLab路径、HDF5路径和端点，没有通用的路径命令行参数。先核对源文件中的常量及HDF5字段，再执行相应Python文件；录像未随仓库公开。回放会调用并重置模型服务，应使用独立测试会话，避免影响正在执行的任务。

tools/local_pi05_eval.py可以派发和启动评测任务，不能把它归类为纯只读回放工具。

## 7. 保存核查结果

每次验证记录源码提交/文件哈希、模型revision、依赖版本、实际配置、测试命令、结果及耗时。RTC重点记录预热后延迟分布、跨块切换偏差、过期响应数与队列耗尽情况。开环回放和简化引导测试均不能替代真机闭环成功率。
