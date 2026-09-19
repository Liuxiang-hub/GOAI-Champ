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
the diff. The deployed base revision is recorded in `SOURCE_SHA256.md`.
