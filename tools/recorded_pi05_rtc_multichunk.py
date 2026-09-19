#!/usr/bin/env python3
"""Drive the RTC policy API from a recording; never call the robot environment."""

import json
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, "/home/user/.xrobot/XPolicyLab")
from XPolicyLab.utils.process_data import decode_image_bit
from client_server.ws.model_client import WsModelClient


HDF5 = "/home/user/eval-runner-slim-0.1.1/data/stack_bowls/x-one-piperX-orbbec/42.hdf5"


def observation(handle, frame):
    return {
        "vision": {
            "cam_head": {"color": decode_image_bit(handle["cam_head/color"][frame])},
            "cam_left_wrist": {"color": decode_image_bit(handle["cam_left_wrist/color"][frame])},
            "cam_right_wrist": {"color": decode_image_bit(handle["cam_right_wrist/color"][frame])},
        },
        "state": {
            "left_arm_joint_state": np.asarray(handle["left_arm/joint"][frame], dtype=np.float32),
            "left_ee_joint_state": np.asarray([handle["left_arm/gripper"][frame]], dtype=np.float32),
            "right_arm_joint_state": np.asarray(handle["right_arm/joint"][frame], dtype=np.float32),
            "right_ee_joint_state": np.asarray([handle["right_arm/gripper"][frame]], dtype=np.float32),
        },
        "instruction": "Stack the bowls on the table.",
    }


def main():
    client = WsModelClient(
        url="ws://127.0.0.1:6007",
        evaluation_id="pi05-model-rtc-recorded-multichunk",
        trial_id="hardware-output-disabled",
        request_timeout_s=120,
        max_connect_attempts=3,
    )
    generations = []
    cursors = []
    started = False
    try:
        client.call(func_name="reset")
        with h5py.File(HDF5, "r") as handle:
            status = client.call(func_name="rtc_start", obs=observation(handle, 0))
            started = True
            generations.append(status["generation"])
            for step in range(1, 151):
                action = client.call(func_name="rtc_next_action")
                assert set(action) == {
                    "left_arm_joint_state",
                    "left_ee_joint_state",
                    "right_arm_joint_state",
                    "right_ee_joint_state",
                }
                assert all(np.isfinite(np.asarray(value)).all() for value in action.values())
                time.sleep(1.0 / 25.0)
                status = client.call(
                    func_name="rtc_commit", obs=observation(handle, step)
                )
                generations.append(status["generation"])
                cursors.append(status["cursor"])
    finally:
        if started:
            client.call(func_name="rtc_stop")
        client.close()

    summary = {
        "hardware_output": False,
        "actions_discarded": True,
        "steps_replayed": 150,
        "max_generation": max(generations),
        "generation_transitions": sum(
            current != previous
            for previous, current in zip(generations, generations[1:])
        ),
        "max_cursor": max(cursors),
    }
    assert summary["max_generation"] >= 2
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
