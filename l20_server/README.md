# L20 Pi0.5 model-internal RTC overlay

These files mirror the deployed RTC changes relative to the L20 XPolicyLab
checkout rooted at `/opt/goai/src/XPolicyLab-pi05`:

```text
l20_server/Pi_05/model.py
  -> policy/Pi_05/model.py
l20_server/Pi_05/openpi/src/openpi/policies/policy.py
  -> policy/Pi_05/openpi/src/openpi/policies/policy.py
l20_server/Pi_05/openpi/src/openpi/models/pi0.py
  -> policy/Pi_05/openpi/src/openpi/models/pi0.py
```

The policy layer carries physical RTC actions through the exact training input
transform chain. The sampler applies PiGDM/VJP guidance at each of the ten
Euler denoising steps. Ordinary inference follows the original branch when RTC
arguments are absent.

Do not deploy these files over a different OpenPI revision without reviewing
the diff. SOURCE_SHA256.md records historical file hashes, not the upstream
base revision. Exact XPolicyLab/OpenPI commits still require field capture.

## 配置与依赖补充

本目录不含完整上游源码、依赖锁文件或模型端部署YAML。见
[运行指南](../docs/RUNNING.md)和[权重说明](../docs/MODEL_WEIGHTS.md)。
model.py默认pi05_aloha及repo_id=1118，不能当作7594已验证配置。
仓库变换测试使用pi05_base_piper6_lora_real及assets/yangchenjie/robodojo_piper6_v3，
实际配置须与检查点及现场记录核对。
