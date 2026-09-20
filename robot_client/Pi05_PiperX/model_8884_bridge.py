"""HRT bridge to the verified FINAL/global_step_8884 policy adapter."""

import threading

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from client_server.ws import WsModelClient


NORM_SHA256 = "7a0bbbbdc9d83e3457fd47e178defb67739d3ae5d4fe0e259a4fa1d30c69f91a"
_ACTION_SHAPES = {
    "left_arm_joint_state": (6,),
    "left_ee_joint_state": (1,),
    "right_arm_joint_state": (6,),
    "right_ee_joint_state": (1,),
}


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = dict(model_cfg)
        if self.cfg["action_type"] != "joint":
            raise ValueError("global_step_8884 supports joint actions only")
        for key in ("left_j5_sign", "right_j5_sign"):
            if float(self.cfg.get(key, 1.0)) not in (-1.0, 1.0):
                raise ValueError(f"{key} must be -1 or 1")
        self.execute_steps = int(self.cfg.get("execute_steps", 15))
        if not 1 <= self.execute_steps <= 50:
            raise ValueError("execute_steps must be between 1 and 50")
        self.timeout = float(self.cfg.get("request_timeout_s", 60))
        self.url = self.cfg.get("final_url", "ws://127.0.0.1:6008")
        self.lock = threading.RLock()
        self.client = WsModelClient(
            url=self.url,
            evaluation_id="hrt-8884-bridge",
            trial_id="adapter-session",
            request_timeout_s=self.timeout,
            connect_timeout_s=60.0,
        )
        status = self.client.call(func_name="status")
        metadata = status.get("metadata", {}) if isinstance(status, dict) else {}
        expected = {
            "checkpoint": "global_step_8884",
            "norm_sha256": NORM_SHA256,
            "robot_name": "goai_piper_x",
            "action_semantics": "absolute_joint_positions",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Upstream FINAL metadata mismatch: {metadata}")
        self.metadata = metadata
        self.observation = None
        self.last_diagnostics = {}
        self.reset()

    def _adapt_observation(self, obs):
        adapted = dict(obs)
        prompt = adapted.get("instruction", adapted.get("instructions"))
        if not prompt:
            prompt = self.cfg.get("default_instruction", "")
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HRT observation and config are missing an instruction")
        adapted["instruction"] = prompt
        state = dict(obs["state"])
        for side in ("left", "right"):
            key = f"{side}_arm_joint_state"
            arm = np.asarray(state[key], dtype=np.float32).copy()
            if arm.shape != (6,) or not np.isfinite(arm).all():
                raise ValueError(f"Invalid {key}")
            arm[4] *= float(self.cfg.get(f"{side}_j5_sign", 1.0))
            state[key] = arm
        adapted["state"] = state
        return adapted

    @staticmethod
    def _validate_step(step):
        result = {}
        for key, shape in _ACTION_SHAPES.items():
            value = np.asarray(step[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid upstream action: {key} {value.shape}")
            result[key] = value.copy()
        return result

    def update_obs(self, obs):
        self.observation = self._adapt_observation(obs)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise NotImplementedError("HRT 8884 bridge supports one environment")
        self.update_obs(obs_list[0])

    def get_action(self):
        if self.observation is None:
            raise RuntimeError("update_obs must precede get_action")
        with self.lock:
            self.client.call(func_name="update_obs", obs=self.observation)
            response = self.client.call(func_name="get_action")
        if not isinstance(response, (list, tuple)) or len(response) != 50:
            raise ValueError("FINAL must return exactly 50 actions")
        actions = [self._validate_step(step) for step in response]
        self.last_diagnostics = {
            "upstream": self.url,
            "checkpoint": self.metadata["checkpoint"],
            "execute_steps": self.execute_steps,
            "default_instruction": self.cfg.get("default_instruction", ""),
        }
        return actions[: self.execute_steps]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list not in (None, [0]):
            raise NotImplementedError("HRT 8884 bridge supports one environment")
        return [self.get_action()]

    def reset(self):
        self.observation = None
        self.last_diagnostics = {}
        with self.lock:
            self.client.call(func_name="reset")

    def rtc_start(self, _obs):
        raise RuntimeError("HRT Robot 6 profile uses synchronous_prefix; RTC is disabled")

    def rtc_next_action(self):
        raise RuntimeError("RTC is disabled")

    def rtc_commit(self, _obs):
        raise RuntimeError("RTC is disabled")

    def rtc_stop(self):
        return None

    def status(self):
        return {
            "metadata": self.metadata,
            "upstream": self.url,
            "rtc_enabled": False,
            "execute_steps": self.execute_steps,
            "default_instruction": self.cfg.get("default_instruction", ""),
            "j5_signs": {
                "left": float(self.cfg.get("left_j5_sign", 1.0)),
                "right": float(self.cfg.get("right_j5_sign", 1.0)),
            },
            "diagnostics": self.last_diagnostics,
        }
