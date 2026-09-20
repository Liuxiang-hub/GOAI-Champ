# 模型权重与检查点

- ModelScope：[LiuXiangg/HRT_GOAI](https://www.modelscope.cn/models/LiuXiangg/HRT_GOAI)
- 配套代码：[GOAI-Champ](https://github.com/Liuxiang-hub/GOAI-Champ)
- 模型卡描述：HUST_HRT 基于 Pi0.5，通过 LoRA 对六项GOAI任务联合微调，面向双臂PIPER-X。
- 执行接口：50步、14维绝对目标，排列为左臂6关节、左夹爪、右臂6关节、右夹爪。

## 获取与配置

权重不包含在GitHub仓库中，请从上述ModelScope页面获取。保留发布目录内部结构，尤其是参数与assets；不要只复制参数而遗漏训练统计。

检查点只在 `l20_server/Pi_05/deploy.yml` 中选择。对于相同
`real-piper6-lora` 训练配方的新权重，只修改 `ckpt_name` 路径末尾的训练步；机器人端
`deploy.yml`不保存检查点编号，而是从L20服务状态读取实际模型元数据。

服务端使用训练配置 `pi05_base_piper6_lora_real`，从检查点的
`assets/yangchenjie/robodojo_piper6_v3` 读取统计。参数文件、训练配置和
assets必须属于同一训练配方。

以下三个“step”彼此独立：

- `ckpt_name` 路径末尾：训练保存步，仅用于选择权重。
- `num_denoising_steps`：单次推理的去噪迭代次数，默认10。
- `execute_steps`：机器人每轮执行的动作前缀，默认15。

更换同配方checkpoint时只改第一项；后两项是推理和控制参数，不随训练步数变化。

## 版本一致性

部署时以ModelScope发布目录、L20配置和服务端 `status` 元数据共同确认模型版本。客户端同时校验策略类型、动作horizon和动作维度，避免错误技术栈或不兼容权重进入机械臂链路。

权重许可须以模型发布页的明确说明为准。
