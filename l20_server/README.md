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

本目录不含完整上游源码或依赖锁文件。决赛初版的 10k 运行配置和 HRT1
路由单元已保存于 `Pi_05/` 与 `HRT1/`；见
[运行指南](../docs/RUNNING.md)和[权重说明](../docs/MODEL_WEIGHTS.md)。
当前决赛初版实际配置使用 `ckpt_name=pi05-goai6-piper`、
`train_config_name=pi05_goai6_piper` 和
`repo_id=local/goai2026_real_piper_joint_rgb`；同配方新权重只修改
`checkpoint_num`。2026-09-21 决赛初版固定为 `checkpoint_num=10000`。
