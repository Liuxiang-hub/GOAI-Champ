# L20 Pi0.5 model-internal RTC overlay

These files retain the experimental RTC changes relative to the L20 XPolicyLab
checkout rooted at `/opt/goai/src/XPolicyLab-pi05`. The current field approach
uses synchronous prefix execution; these experimental branches are not evidence
of verified hardware performance improvements:

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
the diff. `SOURCE_SHA256.md` records historical file hashes, not the upstream
base revision. Exact XPolicyLab/OpenPI commits still require field capture.

## LingBot global_step_8884

`lingbot8884/goai-lingbot-10-denoising.conf` is the systemd drop-in used by
Robot 6 HRT. It serves `global_step_8884` on port 8008 with 10 denoising
steps. The recorded-observation smoke test on 2026-09-20 confirmed metadata
`num_denoising_steps=10`, finite `(15, 14)` output after HRT prefix slicing,
and approximately 1.77 seconds per inference. Hardware output was disabled
and all smoke-test actions were discarded.

## 配置与依赖补充

本目录不含完整上游源码、依赖锁文件或模型端部署YAML。见
[运行指南](../docs/RUNNING.md)和[权重说明](../docs/MODEL_WEIGHTS.md)。
model.py默认pi05_aloha及repo_id=1118，不能当作7594已验证配置。
仓库变换测试使用pi05_base_piper6_lora_real及assets/yangchenjie/robodojo_piper6_v3，
实际配置须与检查点及现场记录核对。
