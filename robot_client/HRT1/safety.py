"""Commissioning command limits for the real Piper X deployment."""

import numpy as np


class CommandLimiter:
    def __init__(self, cfg, obs):
        self.cfg = cfg
        self.previous = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in obs["state"].items()
            if "joint_state" in key
        }
        self.initial_arm = {
            side: np.asarray(obs["state"][side + "_arm_joint_state"], dtype=float).copy()
            for side in ("left", "right")
        }

    def command(self, raw, obs):
        result = {}
        for side in ("left", "right"):
            key = side + "_arm_joint_state"
            target = np.asarray(raw[key], dtype=float)
            measured = np.asarray(obs["state"][key], dtype=float)
            if (
                target.shape != (6,)
                or measured.shape != (6,)
                or not np.isfinite(target).all()
                or not np.isfinite(measured).all()
            ):
                raise ValueError("Invalid arm action or feedback")

            lo = np.deg2rad([-179, -45, -185, -120, -120, -180])
            hi = np.deg2rad([179, 220, 45, 120, 120, 180])
            if np.any(target < lo) or np.any(target > hi):
                raise ValueError("Raw arm target exceeds joint range")

            envelope = self.cfg.get(side + "_arm_initial_envelope_rad")
            if envelope is not None:
                envelope = float(envelope)
                if envelope <= 0:
                    raise ValueError(side + " arm initial envelope must be positive")
                target = np.clip(
                    target,
                    self.initial_arm[side] - envelope,
                    self.initial_arm[side] + envelope,
                )

            if np.max(np.abs(target - measured)) > self.cfg["max_raw_target_error_rad"]:
                raise ValueError("Guarded arm target differs excessively from measured state")
            step = self.cfg["max_command_velocity_rad_s"] / self.cfg["control_hz"]
            result[key] = self.previous[key] + np.clip(
                target - self.previous[key], -step, step
            )
            if np.any(result[key] < lo) or np.any(result[key] > hi):
                raise ValueError(
                    "Command starts outside joint range; verify calibration and initial pose"
                )
            if (
                np.max(np.abs(result[key] - measured))
                > self.cfg["max_command_tracking_error_rad"]
            ):
                raise ValueError("Robot is not following commanded trajectory")

            key = side + "_ee_joint_state"
            grip = np.asarray(raw[key], dtype=float)
            if (
                grip.shape != (1,)
                or not np.isfinite(grip).all()
                or np.any(grip < -0.005)
                or np.any(grip > 1.005)
            ):
                raise ValueError("Invalid gripper target")
            step = self.cfg["max_gripper_rate_s"] / self.cfg["control_hz"]
            result[key] = self.previous[key] + np.clip(
                np.clip(grip, 0, 1) - self.previous[key], -step, step
            )
        return result

    def commit(self, command):
        self.previous = {key: np.array(value, copy=True) for key, value in command.items()}
