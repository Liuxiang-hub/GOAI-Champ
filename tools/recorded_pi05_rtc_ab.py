#!/usr/bin/env python3
"""Recorded-observation Pi0.5 RTC A/B. This script never calls robot actions."""

import json
import sys

import h5py
import numpy as np

sys.path.insert(0, "/home/user/.xrobot/XPolicyLab")
from XPolicyLab.utils.process_data import decode_image_bit
from client_server.ws.model_client import WsModelClient


HDF5 = "/home/user/eval-runner-slim-0.1.1/data/stack_bowls/x-one-piperX-orbbec/42.hdf5"
URL = "ws://127.0.0.1:6198"
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def observation(handle, frame, context, rtc=None):
    state = np.concatenate([
        np.asarray(handle["left_arm/joint"][frame], dtype=np.float32),
        np.asarray([handle["left_arm/gripper"][frame]], dtype=np.float32),
        np.asarray(handle["right_arm/joint"][frame], dtype=np.float32),
        np.asarray([handle["right_arm/gripper"][frame]], dtype=np.float32),
    ])
    images = {
        "cam_high": decode_image_bit(handle["cam_head/color"][frame]),
        "cam_left_wrist": decode_image_bit(handle["cam_left_wrist/color"][frame]),
        "cam_right_wrist": decode_image_bit(handle["cam_right_wrist/color"][frame]),
    }
    obs = {
        "images": {key: np.transpose(value, (2, 0, 1)) for key, value in images.items()},
        "state": state,
        "instruction": "Stack the bowls on the table.",
        "_rtc_context": context,
    }
    if rtc is not None:
        obs["_rtc"] = rtc
    return obs, state


def pack(steps):
    keys = (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )
    return np.stack([
        np.concatenate([np.asarray(step[key]).ravel() for key in keys])
        for step in steps
    ]).astype(np.float32)


def infer(client, obs):
    client.call(func_name="reset")
    client.call(func_name="update_obs", obs=obs)
    result = client.call(func_name="get_action_with_metadata")
    assert result["_rtc_context"] == obs["_rtc_context"]
    actions = pack(result["actions"])
    model_actions = np.asarray(result["_rtc_model_actions"], dtype=np.float32)
    assert actions.shape == (50, 14)
    assert model_actions.shape == (50, 32)
    assert np.isfinite(actions).all() and np.isfinite(model_actions).all()
    return actions, model_actions


def weights(horizon, start, delay):
    result = np.zeros(horizon, dtype=np.float32)
    result[:delay] = 1.0
    overlap = horizon - start
    denominator = overlap - delay + 1
    for index in range(delay, overlap):
        c_i = (overlap - index) / denominator
        result[index] = c_i * np.expm1(c_i) / np.expm1(1.0)
    return result


def main():
    client = WsModelClient(
        url=URL,
        evaluation_id="pi05-model-rtc-recorded-ab",
        trial_id="hardware-output-disabled",
        request_timeout_s=120,
        max_connect_attempts=3,
    )
    try:
        with h5py.File(HDF5, "r") as handle:
            old_obs, old_state = observation(
                handle, 0, {"generation": 0, "request_id": 0}
            )
            old_chunk, _ = infer(client, old_obs)

            latest_obs, latest_state = observation(
                handle, 5, {"generation": 0, "request_id": 1}
            )
            rebased = old_chunk.copy()
            rebased[:, JOINTS] += latest_state[JOINTS] - old_state[JOINTS]
            target = np.zeros_like(rebased)
            target[:45] = rebased[5:]
            mask = weights(50, start=5, delay=3)
            guided_obs = dict(latest_obs)
            guided_obs["_rtc"] = {
                "actions": target,
                "weights": mask,
                "beta": 5.0,
                "generation": 0,
                "request_id": 1,
            }
            guided_physical, guided_model = infer(client, guided_obs)

            plain_obs = dict(latest_obs)
            plain_obs["_rtc_context"] = {"generation": 0, "request_id": 2}
            plain_physical, plain_model = infer(client, plain_obs)

        overlap = mask > 0
        tail = mask == 0
        weighted_plain = float(np.mean(np.square((plain_model - guided_model)[overlap])))
        # Physical-space continuity remains interpretable after output transforms;
        # model-space differences show how far guidance propagates into the tail.
        plain_overlap_error = float(np.mean(np.square(plain_physical[overlap] - target[overlap])))
        guided_overlap_error = float(np.mean(np.square(guided_physical[overlap] - target[overlap])))
        switch_index = 3
        metrics = {
            "hardware_output": False,
            "actions_discarded": True,
            "recording": HDF5,
            "recorded_frame": 5,
            "fixed_noise_required": True,
            "plain_first_joint_jump_rad": float(np.max(np.abs(plain_physical[0, JOINTS] - latest_state[JOINTS]))),
            "guided_first_joint_jump_rad": float(np.max(np.abs(guided_physical[0, JOINTS] - latest_state[JOINTS]))),
            "plain_switch_error_rad": float(np.max(np.abs(plain_physical[switch_index, JOINTS] - target[switch_index, JOINTS]))),
            "guided_switch_error_rad": float(np.max(np.abs(guided_physical[switch_index, JOINTS] - target[switch_index, JOINTS]))),
            "plain_overlap_mse": plain_overlap_error,
            "guided_overlap_mse": guided_overlap_error,
            "plain_vs_guided_model_overlap_mse": weighted_plain,
            "plain_vs_guided_model_tail_mse": float(np.mean(np.square(plain_model[tail] - guided_model[tail]))),
            "guided_tail_std": float(np.std(guided_physical[tail])),
            "guided_gripper_range": [
                float(guided_physical[:, [6, 13]].min()),
                float(guided_physical[:, [6, 13]].max()),
            ],
        }
        assert guided_overlap_error < plain_overlap_error
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
