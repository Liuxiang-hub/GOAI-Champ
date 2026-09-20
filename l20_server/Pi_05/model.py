#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_PIPER_SWAP_ARMS = _env_flag("PI05_PIPER_SWAP_ARMS")
_PIPER_SWAP_J4_J5 = _env_flag("PI05_PIPER_SWAP_J4_J5")
_PIPER_SWAP_LEFT_J4_J5 = _env_flag("PI05_PIPER_SWAP_LEFT_J4_J5", _PIPER_SWAP_J4_J5)
_PIPER_SWAP_RIGHT_J4_J5 = _env_flag("PI05_PIPER_SWAP_RIGHT_J4_J5", _PIPER_SWAP_J4_J5)


def _wrist_permutation(name, default):
    value = os.environ.get(name, default).strip()
    if len(value) != 3 or set(value) != {"4", "5", "6"}:
        raise ValueError(f"{name} must be a permutation of 456, got {value!r}")
    return tuple(int(joint) - 4 for joint in value), value


_RIGHT_WRIST_PERM, _RIGHT_WRIST_PERM_LABEL = _wrist_permutation(
    "PI05_PIPER_RIGHT_WRIST_PERM",
    "546" if _PIPER_SWAP_RIGHT_J4_J5 else "456",
)


def _swap_piper_arms(values):
    """Convert between physical and checkpoint left/right arm ordering."""
    result = np.asarray(values).copy()
    if result.shape[-1] != 14:
        raise ValueError(f"Piper arm swap requires 14-D state/action, got {result.shape}")
    physical_left = result[..., :7].copy()
    result[..., :7] = result[..., 7:14]
    result[..., 7:14] = physical_left
    return result


def _swap_piper_j4_j5(values):
    """Convert between standard Piper X and checkpoint q4/q5 ordering."""
    result = np.asarray(values).copy()
    if result.shape[-1] < 12:
        raise ValueError(f"Piper q4/q5 swap requires 14-D state/action, got {result.shape}")
    pairs = []
    if _PIPER_SWAP_LEFT_J4_J5:
        pairs.append((3, 4))
    if _PIPER_SWAP_RIGHT_J4_J5:
        pairs.append((10, 11))
    for q4, q5 in pairs:
        original_q4 = result[..., q4].copy()
        result[..., q4] = result[..., q5]
        result[..., q5] = original_q4
    return result


def _right_wrist_to_checkpoint(values):
    """Map physical right q4/q5/q6 into the checkpoint's wrist ordering."""
    result = np.asarray(values).copy()
    if result.shape[-1] != 14:
        raise ValueError(f"Piper wrist permutation requires 14-D state/action, got {result.shape}")
    wrist = result[..., 10:13].copy()
    result[..., 10:13] = wrist[..., list(_RIGHT_WRIST_PERM)]
    return result


def _right_wrist_to_piper(values):
    """Invert the checkpoint wrist ordering for physical right-arm commands."""
    result = np.asarray(values).copy()
    if result.shape[-1] != 14:
        raise ValueError(f"Piper wrist permutation requires 14-D state/action, got {result.shape}")
    checkpoint_wrist = result[..., 10:13].copy()
    physical_wrist = np.empty_like(checkpoint_wrist)
    physical_wrist[..., list(_RIGHT_WRIST_PERM)] = checkpoint_wrist
    result[..., 10:13] = physical_wrist
    return result


def _to_checkpoint_state(state):
    result = np.asarray(state, dtype=np.float32).copy()
    if _PIPER_SWAP_ARMS:
        result = _swap_piper_arms(result)
    if _PIPER_SWAP_LEFT_J4_J5:
        original_q4 = result[..., 3].copy()
        result[..., 3] = result[..., 4]
        result[..., 4] = original_q4
    if _RIGHT_WRIST_PERM_LABEL != "456":
        result = _right_wrist_to_checkpoint(result)
    return result


def _to_piper_actions(actions):
    result = np.asarray(actions)
    if _RIGHT_WRIST_PERM_LABEL != "456":
        result = _right_wrist_to_piper(result)
    if _PIPER_SWAP_LEFT_J4_J5:
        result = result.copy()
        original_q4 = result[..., 3].copy()
        result[..., 3] = result[..., 4]
        result[..., 4] = original_q4
    return _swap_piper_arms(result) if _PIPER_SWAP_ARMS else result


def _to_checkpoint_images(images):
    if not _PIPER_SWAP_ARMS:
        return images
    return {
        "cam_high": images["cam_high"],
        "cam_left_wrist": images["cam_right_wrist"],
        "cam_right_wrist": images["cam_left_wrist"],
    }


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self._last_rtc_model_actions: list[np.ndarray] = []
        self._last_rtc_context: list[dict[str, int]] = []

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy
        fixed_noise_seed = os.environ.get("PI05_FIXED_NOISE_SEED", "").strip()
        self._fixed_noise_seed = int(fixed_noise_seed) if fixed_noise_seed else None
        self._fixed_noise_counters: dict[int, int] = {}
        print(
            "[PIPER] checkpoint mappings: "
            f"swap_arms={_PIPER_SWAP_ARMS}, "
            f"swap_q4_q5={_PIPER_SWAP_J4_J5}, "
            f"swap_left_q4_q5={_PIPER_SWAP_LEFT_J4_J5}, "
            f"swap_right_q4_q5={_PIPER_SWAP_RIGHT_J4_J5}, "
            f"right_wrist_perm={_RIGHT_WRIST_PERM_LABEL}",
            flush=True,
        )

    def get_model(self, model_cfg: dict[str, Any]):
        self.train_config_name = model_cfg.get(
            "train_config_name", "pi05_base_piper6_lora_real"
        )
        self.repo_id = model_cfg.get(
            "repo_id", "yangchenjie/robodojo_piper6_v3"
        )
        self.model_root = _resolve_pi05_model_root(model_cfg)
        self.checkpoint_family = str(model_cfg.get("ckpt_name", self.model_root.parent.name))
        self.checkpoint_step = str(
            model_cfg.get("checkpoint_num", self.model_root.name)
        )
        self.num_denoising_steps = int(model_cfg.get("num_denoising_steps", 10))
        if self.num_denoising_steps <= 0:
            raise ValueError("num_denoising_steps must be positive")

        config = _config.get_config(self.train_config_name)
        norm_stats = None
        if self.repo_id is not None:
            norm_stats = _normalize.load(
                self.model_root / "assets" / str(self.repo_id)
            )

        policy = _policy_config.create_trained_policy(
            config, str(self.model_root), norm_stats=norm_stats
        )
        policy._sample_kwargs = dict(policy._sample_kwargs)
        policy._sample_kwargs["num_steps"] = self.num_denoising_steps
        return policy

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_with_metadata(self, **kwargs):
        """Return physical actions plus exact model-space RTC metadata."""
        actions = self.get_action(**kwargs)
        return {
            "actions": actions,
            "_rtc_model_actions": self._last_rtc_model_actions[0],
            "_rtc_context": self._last_rtc_context[0],
        }

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        rtc_model_actions = []
        rtc_contexts = []

        for batch_index, env_idx in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            infer_kwargs = dict(kwargs)
            if self._fixed_noise_seed is not None:
                counter = self._fixed_noise_counters.get(int(env_idx), 0)
                self._fixed_noise_counters[int(env_idx)] = counter + 1
                seed = np.random.SeedSequence([self._fixed_noise_seed, int(env_idx), counter])
                rng = np.random.default_rng(seed)
                action_horizon = int(self.policy._model.action_horizon)
                action_dim = int(self.policy._model.action_dim)
                infer_kwargs["noise"] = rng.standard_normal(
                    (action_horizon, action_dim), dtype=np.float32
                )
            result = self.policy.infer(single_observation, **infer_kwargs)
            actions = _to_piper_actions(result["actions"])
            rtc_model_actions.append(
                np.asarray(result["_rtc_model_actions"], dtype=np.float32)
            )
            rtc_contexts.append(dict(result["_rtc_context"]))
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        self._last_rtc_model_actions = rtc_model_actions
        self._last_rtc_context = rtc_contexts
        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._last_rtc_model_actions = []
        self._last_rtc_context = []
        self._fixed_noise_counters.clear()

    def reset_obsrvationwindows(self):
        self.reset()

    def status(self):
        return {
            "metadata": {
                "policy_family": "pi05",
                "checkpoint_family": self.checkpoint_family,
                "checkpoint_step": self.checkpoint_step,
                "train_config_name": self.train_config_name,
                "repo_id": self.repo_id,
                "action_horizon": int(self.policy._model.action_horizon),
                "model_action_dim": int(self.policy._model.action_dim),
                "physical_action_dim": 14,
                "num_denoising_steps": self.num_denoising_steps,
            }
        }


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = _to_checkpoint_state(observation["state"])
        images = _to_checkpoint_images({
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        })
        prompt = observation.get("instruction")
        encoded = {"state": state, "images": images, "prompt": prompt}
        return add_rtc_payload(encoded, observation)

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = _to_checkpoint_images({
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    })
    state = _to_checkpoint_state(
        pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")
    )
    prompt = observation.get("instruction")
    encoded = {"state": state, "images": images, "prompt": prompt}
    return add_rtc_payload(encoded, observation)


def add_rtc_payload(encoded, observation):
    """Map physical RTC targets into checkpoint ordering without normalizing."""
    context = observation.get("_rtc_context")
    if context is None:
        context = {"generation": 0, "request_id": 0}
    if not isinstance(context, dict):
        raise ValueError("_rtc_context must be a dictionary")
    try:
        generation = int(context["generation"])
        request_id = int(context["request_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("_rtc_context requires integer generation/request_id") from exc
    encoded["_rtc_context"] = {
        "generation": generation,
        "request_id": request_id,
    }

    rtc = observation.get("_rtc")
    if rtc is None:
        return encoded
    if not isinstance(rtc, dict):
        raise ValueError("_rtc must be a dictionary")
    actions = np.asarray(rtc.get("actions"), dtype=np.float32)
    weights = np.asarray(rtc.get("weights"), dtype=np.float32)
    if actions.shape != (50, 14):
        raise ValueError(f"_rtc.actions must have shape (50, 14), got {actions.shape}")
    if weights.shape != (50,):
        raise ValueError(f"_rtc.weights must have shape (50,), got {weights.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(weights).all():
        raise ValueError("_rtc contains NaN or Inf")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("_rtc.weights must be in [0, 1]")
    beta = float(rtc.get("beta", 5.0))
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("_rtc.beta must be finite and positive")
    if int(rtc.get("generation", generation)) != generation:
        raise ValueError("_rtc generation does not match _rtc_context")
    if int(rtc.get("request_id", request_id)) != request_id:
        raise ValueError("_rtc request_id does not match _rtc_context")
    encoded["_rtc"] = {
        "actions": _to_checkpoint_state(actions),
        "weights": weights,
        "beta": beta,
    }
    return encoded


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    stacked = {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }
    if any("_rtc" in obs for obs in obs_list):
        if not all("_rtc" in obs for obs in obs_list):
            raise ValueError("batched observations must either all use RTC or none use it")
        stacked["_rtc"] = {
            "actions": np.stack([obs["_rtc"]["actions"] for obs in obs_list]),
            "weights": np.stack([obs["_rtc"]["weights"] for obs in obs_list]),
            "beta": np.asarray([obs["_rtc"]["beta"] for obs in obs_list]),
        }
    stacked["_rtc_context"] = [obs["_rtc_context"] for obs in obs_list]
    return stacked


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    sliced = {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }
    if "_rtc" in obs:
        sliced["_rtc"] = {
            "actions": obs["_rtc"]["actions"][batch_index],
            "weights": obs["_rtc"]["weights"][batch_index],
            "beta": obs["_rtc"]["beta"][batch_index],
        }
    sliced["_rtc_context"] = obs["_rtc_context"][batch_index]
    return sliced


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
