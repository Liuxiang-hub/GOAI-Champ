# 2223 Pi05 PiperX runtime snapshot

Snapshot date: 2026-09-20 (Asia/Shanghai)

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

The process was started manually. At snapshot time,
`pi05-piperx-adapter.service` was inactive while port 6007 was listening.

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
execution_mode: synchronous_prefix
execute_steps: 15
control_hz: 25
upstream: ws://127.0.0.1:6198
checkpoint: real-piper6-lora/7594
```

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

2223 also contains:

```text
/home/user/.xrobot/XPolicyLab/policy/Pi05_PiperX/
```

The Python files matched the working tree at snapshot time, but its
`deploy.yml` specified 10 Hz and loopback-only binding. The process listening
on 6007 loaded the `goai_pi05` tree included in this repository, not that
second configuration.
