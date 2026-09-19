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

## Dependencies

The adapter is loaded inside an XPolicyLab checkout and imports its
`model_template`, WebSocket client, and robot-state helpers. It is not a
standalone Python package.
