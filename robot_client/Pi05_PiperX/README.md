# Pi05_PiperX adapter

XPolicyLab WebSocket adapter for a remote Pi0.5 dual-PIPER-X inference server.

## Runtime contract

- Downstream endpoint: `ws://0.0.0.0:6007`
- Upstream endpoint: `ws://127.0.0.1:6198`
- Checkpoint: `real-piper6-lora/7594`
- Action shape: `(50, 14)`
- Action order: `[left arm 6, left gripper, right arm 6, right gripper]`
- Action semantics: absolute joint targets
- Current deployment approach: synchronous prefix execution at nominal 25 Hz

The repository defaults to `synchronous_prefix`: execute the first 15 actions
of each 50-action prediction, update observations, and request the next chunk.
`deploy.py` reads the execution mode from `motion_gate.json` at episode start.
See the [field verification note](../../docs/FIELD_STATUS_20260920.md).

## Experimental RTC implementation

RTC is retained for experimental evaluation. Field feedback has not established
the expected benefit; it is not presented as the verified deployment approach.

RTC guidance remains in physical absolute-action space until the L20 policy
applies the checkpoint's DeltaActions, quantile normalization, and 14-to-32D
padding transforms. The JAX sampler then applies VJP guidance at every
denoising step. The adapter never synthesizes model normalization statistics.

`rtc_start` performs a zero-weight guided warmup before exposing the first
action. Every inference is tagged with `generation/request_id`; stale results
are counted and discarded.

## Configuration

`deploy.yml` configures endpoints, camera aliases, synchronous prefix and RTC
initialization: rtc_trigger_step=20, rtc_initial_delay_steps=11,
rtc_prewarm_guided=true, control_hz=25. Experimental parameters do not select
the execution mode; the synchronous prefix length is execute_steps=15.
The execution loop separately reads motion_gate.json. See
[运行指南](../../docs/RUNNING.md) and [配置说明](../../docs/CONFIGURATION.md).
`motion_gate.json` is the safe repository default and has hardware output
disabled. `motion_gate.2223-runtime-snapshot.json` is a historical profile,
not the latest field configuration, and must not be enabled without on-site supervision.

`motion_gate.robot1-v1.json` records the synchronous first-version parameters
validated on Robot 1: 25 Hz, the first 15 actions from each 50-action chunk,
and no initial-pose envelope. Hardware output remains disabled in the stored
profile. Joint limits, velocity limits, tracking checks, and gripper limits
remain active.

## Dependencies

The adapter is loaded inside an XPolicyLab checkout and imports its
`model_template`, WebSocket client, and robot-state helpers. It is not a
standalone Python package.
