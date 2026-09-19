#!/usr/bin/env python3
"""Measure recorded-observation Pi0.5 plain/guided latency; no actuation."""

import json
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, "/home/user/.xrobot/XPolicyLab")
from XPolicyLab.utils.process_data import decode_image_bit
from client_server.ws.model_client import WsModelClient


HDF5 = "/home/user/eval-runner-slim-0.1.1/data/stack_bowls/x-one-piperX-orbbec/42.hdf5"


def load_obs():
    with h5py.File(HDF5, "r") as handle:
        state = np.concatenate([
            handle["left_arm/joint"][5],
            [handle["left_arm/gripper"][5]],
            handle["right_arm/joint"][5],
            [handle["right_arm/gripper"][5]],
        ]).astype(np.float32)
        images = {
            "cam_high": decode_image_bit(handle["cam_head/color"][5]),
            "cam_left_wrist": decode_image_bit(handle["cam_left_wrist/color"][5]),
            "cam_right_wrist": decode_image_bit(handle["cam_right_wrist/color"][5]),
        }
    return {
        "images": {key: np.transpose(value, (2, 0, 1)) for key, value in images.items()},
        "state": state,
        "instruction": "Stack the bowls on the table.",
    }


def timed(client, obs):
    client.call(func_name="reset")
    started = time.monotonic()
    client.call(func_name="update_obs", obs=obs)
    result = client.call(func_name="get_action_with_metadata")
    return (time.monotonic() - started) * 1000.0, result


def main():
    client = WsModelClient(
        url="ws://127.0.0.1:6198",
        evaluation_id="pi05-model-rtc-latency",
        trial_id="hardware-output-disabled",
        request_timeout_s=180,
        max_connect_attempts=3,
    )
    try:
        base = load_obs()
        plain = dict(base, _rtc_context={"generation": 0, "request_id": 0})
        plain_ms, plain_result = timed(client, plain)
        physical = []
        for step in plain_result["actions"]:
            physical.append(np.concatenate([
                step["left_arm_joint_state"], step["left_ee_joint_state"],
                step["right_arm_joint_state"], step["right_ee_joint_state"],
            ]))
        target = np.asarray(physical, dtype=np.float32)
        mask = np.linspace(1.0, 0.0, 50, dtype=np.float32)
        guided_times = []
        for request_id in range(1, 4):
            context = {"generation": request_id - 1, "request_id": request_id}
            guided = dict(base, _rtc_context=context)
            guided["_rtc"] = {
                "actions": target,
                "weights": mask,
                "beta": 5.0,
                **context,
            }
            elapsed, _ = timed(client, guided)
            guided_times.append(elapsed)
        print(json.dumps({
            "hardware_output": False,
            "actions_discarded": True,
            "plain_ms": plain_ms,
            "guided_ms": guided_times,
        }, indent=2), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
