"""Real-robot episode loops for the Pi0.5 Piper X adapter.

RTC uses an asynchronous double buffer, rebases guidance to the latest
observation, and delegates per-denoising-step guidance to the Pi0.5 server.
``execution_mode: synchronous_prefix`` remains available as a fallback.
"""

import json
import os
from pathlib import Path
import time

import numpy as np

from .safety import CommandLimiter


_RIGHT_ARM_PROBE_FLAG = Path("/tmp/pi05_right_arm_probe_once.json")


def _finite_action(raw):
    result = {}
    expected = {
        "left_arm_joint_state": (6,),
        "left_ee_joint_state": (1,),
        "right_arm_joint_state": (6,),
        "right_ee_joint_state": (1,),
    }
    for key, shape in expected.items():
        value = np.asarray(raw[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid non-finite or malformed action: {key}")
        result[key] = value.copy()
    return result


class _ActionEMA:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("rtc_action_ema_alpha must be in [0, 1]")
        self.previous = None

    def apply(self, raw):
        current = _finite_action(raw)
        if self.previous is None:
            result = current
        else:
            result = {
                key: self.alpha * value + (1.0 - self.alpha) * self.previous[key]
                for key, value in current.items()
            }
        self.previous = {key: value.copy() for key, value in result.items()}
        return result


def _command(raw, observation, cfg, limiter):
    hardware = _finite_action(raw)
    hardware["left_arm_joint_state"][4] *= float(cfg.get("left_j5_sign", 1.0))
    if bool(cfg.get("software_safety_enabled", True)):
        return limiter.command(hardware, observation)
    return hardware


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


def _eval_synchronous_prefix(task_env, model_client, cfg, debug):
    control_hz = float(cfg["control_hz"])
    prefix_steps = int(cfg.get("execute_steps", 15))
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if not 1 <= prefix_steps <= 50:
        raise ValueError("execute_steps must be between 1 and 50")

    model_client.call(func_name="reset")
    limiter = CommandLimiter(cfg, task_env.get_obs())

    while not task_env.is_episode_end():
        observation = task_env.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        chunk = model_client.call(func_name="get_action")
        if not isinstance(chunk, (list, tuple)) or not chunk:
            raise ValueError("Pi05_PiperX returned an empty or invalid action chunk")
        if len(chunk) > prefix_steps:
            chunk = chunk[:prefix_steps]

        for chunk_index, raw in enumerate(chunk):
            started = time.monotonic()
            observation = task_env.get_obs()
            if not debug:
                _gate()  # A local disarm takes effect before the next command.
                action = _command(raw, observation, cfg, limiter)
            else:
                action = raw

            task_env.take_action(action)
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
            if task_env.is_episode_end():
                break


def _eval_rtc(task_env, model_client, cfg, debug):
    control_hz = float(cfg["control_hz"])
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")

    model_client.call(func_name="reset")
    observation = task_env.get_obs()
    limiter = CommandLimiter(cfg, observation)
    ema = _ActionEMA(cfg.get("rtc_action_ema_alpha", 0.4))
    started_rtc = False
    try:
        status = model_client.call(func_name="rtc_start", obs=observation)
        started_rtc = True
        print("RTC_START " + json.dumps(status), flush=True)

        while not task_env.is_episode_end():
            started = time.monotonic()
            observation = task_env.get_obs()
            raw = model_client.call(func_name="rtc_next_action")
            smoothed = ema.apply(raw)
            if not debug:
                _gate()  # A local disarm takes effect before the next command.
                action = _command(smoothed, observation, cfg, limiter)
            else:
                action = smoothed

            task_env.take_action(action)
            if not debug:
                print(
                    "RTC_COMMAND "
                    + json.dumps(
                        {
                            "raw": {k: np.asarray(v).tolist() for k, v in raw.items()},
                            "command": {
                                k: np.asarray(v).tolist() for k, v in action.items()
                            },
                        }
                    ),
                    flush=True,
                )
            limiter.commit(action)
            time.sleep(max(0.0, 1.0 / control_hz - (time.monotonic() - started)))
            if task_env.is_episode_end():
                break

            status = model_client.call(func_name="rtc_commit", obs=task_env.get_obs())
            print("RTC_STATUS " + json.dumps(status), flush=True)
    finally:
        if started_rtc:
            model_client.call(func_name="rtc_stop")


def eval_one_episode(TASK_ENV, model_client):
    cfg = _gate()
    if _RIGHT_ARM_PROBE_FLAG.exists():
        return _run_right_arm_probe(TASK_ENV)
    debug = os.environ.get("EVAL_ENV_TYPE") == "debug"
    mode = cfg.get("execution_mode", "synchronous_prefix")
    if mode == "rtc":
        return _eval_rtc(TASK_ENV, model_client, cfg, debug)
    if mode == "synchronous_prefix":
        return _eval_synchronous_prefix(TASK_ENV, model_client, cfg, debug)
    raise ValueError(f"unsupported execution_mode: {mode}")


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError(
        "Pi05_PiperX supports one real or debug environment per session"
    )
