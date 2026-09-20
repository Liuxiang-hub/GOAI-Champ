# 2223 Pi05 PiperX runtime snapshot

Snapshot date: 2026-09-20 (Asia/Shanghai)

> 最新现场方案与真机记录见[现场运行说明](FIELD_STATUS_20260920.md)。下文保留历史参数和实验记录，不作为当前默认配置。

> Historical record, not a live status report. Current repository code includes
> model-internal guidance. Replay results below are not tied to that version.
> Mapping values also differ from the root README and need field reconciliation.
> See [修订记录](DOCUMENTATION_REVIEW.md).

## Provenance

The files under `robot_client/Pi05_PiperX/` were copied from the working tree
used by the process listening on port 6007:

```text
/home/user/goai_pi05/XPolicyLab-pi05/policy/Pi05_PiperX/
```

The running command was:

```text
/home/user/goai_pi05/venv/bin/python setup_policy_server.py \
  --config_path policy/Pi05_PiperX/deploy.yml
```

Port 6007 is managed by the enabled and active
`pi05-piperx-adapter.service`. The eval-runner listens on port 19200.

## Robot-side mapping

The 2223 robot configuration used:

```yaml
left_arm: can_left
right_arm: can_right
```

The adapter packs actions in checkpoint order:

```text
[left J1..J6, left gripper, right J1..J6, right gripper]
```

The model response is treated as absolute joint targets. Grippers are clipped
independently to `[0, 1]`.

## Active execution parameters

```yaml
execution_mode: rtc
rtc_start_steps: 5
rtc_initial_delay_steps: 22
control_hz: 25
upstream: ws://127.0.0.1:6198
checkpoint: real-piper6-lora/7594
```

The RTC controller uses an asynchronous double buffer, commits the latest
observation after every executed action, and rebases each returned absolute
action chunk. The build described by this historical snapshot was recorded as
not implementing guidance/inpainting. Current repository sampler code includes
rtc_guided_euler_step and jax.vjp; the historical statement does not describe it.

The released Pi05 checkpoint documentation describes 30 Hz training data, so
25 Hz execution makes a 15-step prefix last 0.6 s instead of 0.5 s.

## L20 mapping warning

The upstream L20 process was observed with these model adapter settings:

```text
swap_arms=false
swap_left_q4_q5=true
swap_right_q4_q5=false
right_wrist_perm=654
```

The checkpoint action layout does not require those joint permutations. A
saved static comparison on 2223 measured the maximum first-action wrist jump
as 0.01596 rad for permutation `564` and 0.04760 rad for the active `654`.
That metric alone cannot validate any non-standard permutation, but it does
show that `654` has not been justified by this test.

## Duplicate deployment tree

2223 contains two policy trees:

```text
/home/user/.xrobot/XPolicyLab/policy/Pi05_PiperX/
```

The RTC deployment synchronized `deploy.py`, `deploy.yml`, `motion_gate.json`,
and `rtc_preflight.py` between both trees. Their deployed hashes matched after
installation. Port 6007 loads the `goai_pi05` tree; eval-runner imports the
`.xrobot` tree.

## Read-only RTC verification

Recorded three-camera observations were replayed through both the local 6007
adapter and the official `47.97.37.69:6007` endpoint. Actions were discarded by
a fake environment; no hardware backend was instantiated.

- Local endpoint: two 120-step runs, 22 generation switches each.
- Official endpoint: two 60-step runs, 11 generation switches each.
- Both paths observed concurrent inference and completed without queue
  exhaustion.
- Official metadata reported `real-piper6-lora/7594` via upstream
  `ws://127.0.0.1:6198`.
