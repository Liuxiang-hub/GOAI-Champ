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

## Configuration

`deploy.yml` configures endpoints, camera aliases and the 25 Hz nominal
control rate. The synchronous prefix length is `execute_steps=15`.
The execution loop separately reads motion_gate.json. See
[运行指南](../../docs/RUNNING.md) and [配置说明](../../docs/CONFIGURATION.md).
`motion_gate.json` is the safe repository default and has hardware output
disabled. `motion_gate.2223-runtime-snapshot.json` is a historical profile,
not the latest field configuration, and must not be enabled without on-site supervision.

Legacy compatibility fields and archived asynchronous code remain in the
source history but do not participate in the `synchronous_prefix` execution
path.

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

## Archived Robot 6 compatibility record

`model_8884_bridge.py` and `deploy.8884.yml` switch HRT port 6009 to the
verified local FINAL adapter on port 6008, which forwards to the L20
`global_step_8884` service. The Robot 6 execution profile remains synchronous
at 25 Hz and executes the first 15 actions from each 50-action chunk. Both J5
coordinate signs remain `-1` on observation input and hardware output.

This compatibility record is separate from the current Pi0.5
`real-piper6-lora/7594` runtime contract above.

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
