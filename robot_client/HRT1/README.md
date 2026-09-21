# HRT1 field execution package

This directory records the field side of the 2026-09-21 finals-v1 deployment.
Copy it to `XPolicyLab/policy/HRT1` on the robot computer. The official
eval-runner imports `deploy.py` locally so hardware commands remain on the
field computer; `model.py` is also retained here for the rollback topology.

## Current contract

- Official policy name: `HRT1`
- Public WebSocket port: `6000`
- Model output: 50 x 14 absolute targets
- Execution: first 20 steps, then acquire a new observation and infer again
- Nominal control rate: 25 Hz
- Mode: `synchronous_prefix`
- Action order: left J1-J6, left gripper, right J1-J6, right gripper
- Robot 6 J5 mapping: left sign `-1`, right sign `+1`

`motion_gate.json` is the repository-safe template and keeps hardware output
disabled. It also sets `rtc_enabled: false`; changing that single field to
`true` selects field-local RTC, while `false` forces the synchronous 20-step
rollback path. `motion_gate.robot6-finals-v1.json` is the exact enabled field
snapshot captured for audit; do not copy it blindly to another robot.

The legacy `real-piper6-lora/7594` wording in the exact deployed `model.py`
docstring and one validation message is not the loaded checkpoint. The Pi0.5
service configuration, startup record and adapter metadata identify this
snapshot as `pi05-goai6-piper/10000`.

`HRT1` is the competition policy name and deployment adapter for this Pi0.5
route. It is independent from the archived LingBot/8884 materials.

The deployed adapter logs `HRT1_CLIENT_TIMING` per request. A completed official
record only proves that the software flow ended; it does not prove physical
task success.

RTC parameters remain present while the switch is off:

```json
{
  "rtc_enabled": false,
  "control_hz": 25.0,
  "rtc_trigger_step": 20,
  "rtc_initial_delay_steps": 5,
  "rtc_safety_margin_steps": 2,
  "rtc_blend_steps": 5,
  "rtc_action_ema_alpha": 0.4
}
```
