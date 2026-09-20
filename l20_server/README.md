# L20 Pi0.5 deployment overlay

These files record the model-service overrides relative to the L20 XPolicyLab
checkout rooted at `/opt/goai/src/XPolicyLab-pi05`. The current deployment
uses synchronous prefix execution:

```text
l20_server/Pi_05/model.py
  -> policy/Pi_05/model.py
l20_server/Pi_05/openpi/src/openpi/policies/policy.py
  -> policy/Pi_05/openpi/src/openpi/policies/policy.py
l20_server/Pi_05/openpi/src/openpi/models/pi0.py
  -> policy/Pi_05/openpi/src/openpi/models/pi0.py
```

The active inference path applies the checkpoint input transform, normalization,
14-to-32D padding and the standard ten-step sampler before returning the
physical 14D action block. Archived asynchronous guidance branches are retained
only for source compatibility and are not part of the current deployment path.

Do not deploy these files over a different OpenPI revision without reviewing
the diff. `SOURCE_SHA256.md` records historical file hashes. The overlay must
be used with its matching XPolicyLab/OpenPI environment.

## 配置与依赖补充

本目录不含完整上游源码、依赖锁文件或模型端部署YAML。见
[运行指南](../docs/RUNNING.md)和[权重说明](../docs/MODEL_WEIGHTS.md)。
model.py默认pi05_aloha及repo_id=1118，不能当作7594已验证配置。
仓库变换测试使用pi05_base_piper6_lora_real及assets/yangchenjie/robodojo_piper6_v3，
实际配置须与检查点及现场记录核对。
