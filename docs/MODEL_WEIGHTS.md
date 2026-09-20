# 模型权重与检查点

- ModelScope：[LiuXiangg/HRT_GOAI](https://www.modelscope.cn/models/LiuXiangg/HRT_GOAI)
- 配套代码：[GOAI-Champ](https://github.com/Liuxiang-hub/GOAI-Champ)
- 检查点标识：`real-piper6-lora/7594`。
- 模型卡描述：HUST_HRT 基于 Pi0.5，通过 LoRA 对六项GOAI任务联合微调，面向双臂PIPER-X。
- 执行接口：50步、14维绝对目标，排列为左臂6关节、左夹爪、右臂6关节、右夹爪。

## 获取与配置

权重不包含在GitHub仓库中，请从上述ModelScope页面获取。保留发布目录内部结构，尤其是参数与assets；不要只复制参数而遗漏训练统计。

服务端支持 `model_path` / `checkpoint_path` 显式路径，亦支持 `ckpt_name` 路径解析。实际加载路径以服务端配置及日志为准。机器人端deploy.yml中的ckpt_name标签不证明远端加载了同一份权重。

仓库模型变换测试使用训练配置 `pi05_base_piper6_lora_real`，从检查点的 `assets/yangchenjie/robodojo_piper6_v3` 读取统计。服务端必须配置与检查点一致的训练配置和assets标识；model.py中的默认 `pi05_aloha`、`1118` 不能当作本模型的已验证配置。

## 当前核验状态

2026-09-20最近一次只读检查中，模型卡可访问，描述与7594检查点匹配，但目录接口未成功返回文件，未下载或校验权重。该结果仅描述当次检查，不代表平台此后仍未上传完成。

上传完成后，需记录模型仓库revision、文件清单、大小、SHA256及加载结果。当前尚未确认发布物是完整推理参数还是仅LoRA增量，因此不提供未经验证的合并或下载命令。若是增量，发布方还需注明基础模型版本及加载步骤。

权重许可须以模型发布页的明确说明为准。
