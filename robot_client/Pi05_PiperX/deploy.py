"""Synchronous real-robot episode loop for the Pi0.5 Piper X adapter.

The upstream Pi0.5 server doesn't implement RTC guidance.  Therefore this
runner deliberately avoids the adapter-side pseudo-RTC path: every inference
starts from the latest observation and only the first configured action prefix
is executed before replanning.
"""

import json
import os
from pathlib import Path
import time

import numpy as np

from .safety import CommandLimiter


_RIGHT_ARM_PROBE_FLAG = Path("/tmp/pi05_right_arm_probe_once.json")


def _run_right_arm_probe(task_env):
    """Run one bounded q1-q6 plus gripper identification sweep."""
    request = json.loads(_RIGHT_ARM_PROBE_FLAG.read_text())
    _RIGHT_ARM_PROBE_FLAG.unlink()

    arm_delta = float(request.get("arm_delta_rad", 0.02))
    gripper_delta = float(request.get("gripper_delta", 0.05))
    ramp_steps = int(request.get("ramp_steps", 8))
    control_hz = float(request.get("control_hz", 25.0))
    if not 0 < arm_delta <= 0.03:
        raise ValueError("right-arm probe delta must be in (0, 0.03] rad")
    if not 0 < gripper_delta <= 0.1:
        raise ValueError("right gripper probe delta must be in (0, 0.1]")
    if not 4 <= ramp_steps <= 20 or not 5 <= control_hz <= 50:
        raise ValueError("invalid right-arm probe ramp settings")

    initial = task_env.get_obs()["state"]
    left = np.asarray(initial["left_arm_joint_state"], dtype=float).copy()
    right = np.asarray(initial["right_arm_joint_state"], dtype=float).copy()
    left_grip = np.asarray(initial["left_ee_joint_state"], dtype=float).copy()
    right_grip = np.asarray(initial["right_ee_joint_state"], dtype=float).copy()
    if left.shape != (6,) or right.shape != (6,):
        raise ValueError("right-arm probe requires two 6-DoF Piper arms")

    lo = np.deg2rad([-179, -45, -185, -120, -120, -180])
    hi = np.deg2rad([179, 220, 45, 120, 120, 180])
    pause = 1.0 / control_hz

    def command(right_target, grip_target):
        action = {
            "left_arm_joint_state": left.copy(),
            "left_ee_joint_state": left_grip.copy(),
            "right_arm_joint_state": np.asarray(right_target, dtype=float).copy(),
            "right_ee_joint_state": np.asarray([grip_target], dtype=float),
        }
        task_env.take_action(action)
        time.sleep(pause)

    print(
        "RIGHT_ARM_PROBE_START "
        + json.dumps({"right_q": right.tolist(), "right_gripper": right_grip.tolist()}),
        flush=True,
    )
    for joint_index in range(6):
        direction = 1.0 if right[joint_index] + arm_delta <= hi[joint_index] else -1.0
        if right[joint_index] + direction * arm_delta < lo[joint_index]:
            raise ValueError(f"right q{joint_index + 1} has no safe probe direction")
        for step in range(1, ramp_steps + 1):
            target = right.copy()
            target[joint_index] += direction * arm_delta * step / ramp_steps
            command(target, float(right_grip[0]))
        for step in range(ramp_steps - 1, -1, -1):
            target = right.copy()
            target[joint_index] += direction * arm_delta * step / ramp_steps
            command(target, float(right_grip[0]))
        measured = np.asarray(
            task_env.get_obs()["state"]["right_arm_joint_state"], dtype=float
        )
        print(
            "RIGHT_ARM_PROBE_JOINT "
            + json.dumps(
                {
                    "joint": joint_index + 1,
                    "command_delta_rad": direction * arm_delta,
                    "measured_after_return": measured.tolist(),
                }
            ),
            flush=True,
        )

    grip_direction = 1.0 if right_grip[0] + gripper_delta <= 1.0 else -1.0
    for step in range(1, ramp_steps + 1):
        command(right, float(right_grip[0] + grip_direction * gripper_delta * step / ramp_steps))
    for step in range(ramp_steps - 1, -1, -1):
        command(right, float(right_grip[0] + grip_direction * gripper_delta * step / ramp_steps))
    command(right, float(right_grip[0]))
    print("RIGHT_ARM_PROBE_COMPLETE", flush=True)


def _gate():
    cfg = json.loads(Path(__file__).with_name("motion_gate.json").read_text())
    if os.environ.get("EVAL_ENV_TYPE") != "debug" and not cfg["hardware_output_enabled"]:
        raise RuntimeError(
            "Pi05_PiperX motion gate is disabled; finish live preflight before enabling motion_gate.json"
        )
    return cfg


def validate_action(action, obs, limit):
    for side in ("left", "right"):
        joint = np.asarray(action[f"{side}_arm_joint_state"])
        state = np.asarray(obs["state"][f"{side}_arm_joint_state"])
        gripper = np.asarray(action[f"{side}_ee_joint_state"])
        if not np.isfinite(joint).all() or not np.isfinite(gripper).all():
            raise ValueError("Non-finite model action")
        if np.max(np.abs(joint - state)) > limit:
            raise ValueError(
                f"{side} arm target exceeds measured-state step limit {limit} rad"
            )
        if np.any(gripper < 0) or np.any(gripper > 1):
            raise ValueError(f"{side} gripper target is outside [0,1]")


def eval_one_episode(TASK_ENV, model_client):
    cfg = _gate()
    if _RIGHT_ARM_PROBE_FLAG.exists():
        return _run_right_arm_probe(TASK_ENV)
    debug = os.environ.get("EVAL_ENV_TYPE") == "debug"
    control_hz = float(cfg["control_hz"])
    prefix_steps = int(cfg.get("execute_steps", 15))
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if not 1 <= prefix_steps <= 50:
        raise ValueError("execute_steps must be between 1 and 50")

    model_client.call(func_name="reset")
    limiter = CommandLimiter(cfg, TASK_ENV.get_obs())

    while not TASK_ENV.is_episode_end():
        observation = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        chunk = model_client.call(func_name="get_action")
        if not isinstance(chunk, (list, tuple)) or not chunk:
            raise ValueError("Pi05_PiperX returned an empty or invalid action chunk")
        if len(chunk) > prefix_steps:
            chunk = chunk[:prefix_steps]

        for chunk_index, raw in enumerate(chunk):
            started = time.monotonic()
            observation = TASK_ENV.get_obs()
            if not debug:
                _gate()  # A local disarm takes effect before the next command.
                action = limiter.command(raw, observation)
            else:
                action = raw

            TASK_ENV.take_action(action)
            if not debug:
                print(
                    "SYNC15_COMMAND",
                    json.dumps(
                        {
                            "chunk_index": chunk_index,
                            "raw": {k: np.asarray(v).tolist() for k, v in raw.items()},
                            "command": {k: np.asarray(v).tolist() for k, v in action.items()},
                        }
                    ),
                    flush=True,
                )
            limiter.commit(action)
            time.sleep(max(0.0, 1.0 / control_hz - (time.monotonic() - started)))
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError(
        "Pi05_PiperX synchronous-prefix mode supports one real or debug environment per session"
    )
