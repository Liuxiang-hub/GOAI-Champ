# 模型权重与检查点

- ModelScope：[LiuXiangg/HRT_GOAI](https://www.modelscope.cn/models/LiuXiangg/HRT_GOAI)
- 配套代码：[GOAI-Champ](https://github.com/Liuxiang-hub/GOAI-Champ)
- 模型卡描述：HUST_HRT 基于 Pi0.5，通过 LoRA 对六项GOAI任务联合微调，面向双臂PIPER-X。
- 执行接口：50步、14维绝对目标，排列为左臂6关节、左夹爪、右臂6关节、右夹爪。

## 获取与配置

权重不包含在GitHub仓库中，请从上述ModelScope页面获取。保留发布目录内部结构，尤其是参数与assets；不要只复制参数而遗漏训练统计。

L20实际运行配置将模型族与训练保存步分开：

```yaml
ckpt_name: real-piper6-lora
checkpoint_num: 7594
```

对于相同训练配方的新权重，只修改 `checkpoint_num`。机器人端
`deploy.yml`中的 `ckpt_name` 只是适配器标签，不参与L20模型加载。

服务端使用训练配置 `pi05_base_piper6_lora_real`，从检查点的
`assets/yangchenjie/robodojo_piper6_v3` 读取统计。参数文件、训练配置和
assets必须属于同一训练配方。

以下三个“step”彼此独立：

- `checkpoint_num`：训练保存步，仅用于选择权重。
- sampler `num_steps`：单次推理的去噪迭代次数，当前代码默认10。
- `execute_steps`：机器人每轮执行的动作前缀，默认15。

更换同配方checkpoint时只改第一项；后两项是推理和控制参数，不随训练步数变化。

## 版本一致性

部署时以ModelScope发布目录、L20启动配置和启动日志共同确认模型版本。机器人端
维持固定的50步、14维物理动作与32维模型动作校验；不兼容的horizon或动作维度
会在动作下发前报错。

权重许可须以模型发布页的明确说明为准。
