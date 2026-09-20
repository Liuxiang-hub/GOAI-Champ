# Pi05_PiperX adapter

XPolicyLab WebSocket adapter for a remote Pi0.5 dual-PIPER-X inference server.

## Runtime contract

- Downstream endpoint: `ws://0.0.0.0:6007`
- Upstream endpoint: `ws://127.0.0.1:6198`
- Checkpoint: `real-piper6-lora/7594`
- Action shape: `(50, 14)`
- Action order: `[left arm 6, left gripper, right arm 6, right gripper]`
- Action semantics: absolute joint targets
- Current loop: model-guided RTC asynchronous double buffer at 25 Hz

`deploy.py` defaults to `rtc` mode. `synchronous_prefix` remains available as
a fallback by changing `execution_mode` in `motion_gate.json`.

RTC guidance remains in physical absolute-action space until the L20 policy
applies the checkpoint's DeltaActions, quantile normalization, and 14-to-32D
padding transforms. The JAX sampler then applies VJP guidance at every
denoising step. The adapter never synthesizes model normalization statistics.

`rtc_start` performs a zero-weight guided warmup before exposing the first
action. Every inference is tagged with `generation/request_id`; stale results
are counted and discarded.

## Configuration

`deploy.yml` configures the WebSocket endpoints and synchronous prefix.
`motion_gate.json` is the safe repository default and has hardware output
disabled. `motion_gate.2223-runtime-snapshot.json` records the parameters from
the running field machine and must not be enabled without on-site supervision.

`motion_gate.robot1-v1.json` records the synchronous first-version parameters
validated on Robot 1: 25 Hz, the first 15 actions from each 50-action chunk,
and no initial-pose envelope. Hardware output remains disabled in the stored
profile. Joint limits, velocity limits, tracking checks, and gripper limits
remain active.

`motion_gate.robot6-ok.json` records the field-validated Robot 6 profile from
2026-09-20: HRT on port 6009, Pi0.5 `real-piper6-lora/7594` upstream on port
6198, synchronous prefix execution at 25 Hz, the first 15 actions from each
50-action chunk, software safety disabled, and both left/right J5 coordinate
signs set to `-1`. Hardware output is deliberately disabled in the stored
profile. This label means the execution flow completed on Robot 6; it is not
evidence of task success or a profile for another robot.

## Robot 6 global_step_8884 switch

`model_8884_bridge.py` and `deploy.8884.yml` switch HRT port 6009 to the
verified local FINAL adapter on port 6008, which forwards to the L20
`global_step_8884` service. The Robot 6 execution profile remains synchronous
at 25 Hz and executes the first 15 actions from each 50-action chunk. Both J5
coordinate signs remain `-1` on observation input and hardware output.

The 2026-09-20 recorded-observation smoke test confirmed checkpoint
`global_step_8884`, normalization SHA256
`7a0bbbbdc9d83e3457fd47e178defb67739d3ae5d4fe0e259a4fa1d30c69f91a`,
finite `(15, 14)` output, and discarded all actions with hardware output
disabled. This validates the inference path only; it does not mark the 8884
variant as physically successful on Robot 6.

## Dependencies

The adapter is loaded inside an XPolicyLab checkout and imports its
`model_template`, WebSocket client, and robot-state helpers. It is not a
standalone Python package.
