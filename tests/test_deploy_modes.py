import os
import unittest
from unittest import mock

import numpy as np

from robot_client.Pi05_PiperX import deploy


def action(value=0.0):
    return {
        "left_arm_joint_state": np.full(6, value, dtype=np.float32),
        "left_ee_joint_state": np.array([0.5], dtype=np.float32),
        "right_arm_joint_state": np.full(6, value, dtype=np.float32),
        "right_ee_joint_state": np.array([0.5], dtype=np.float32),
    }


def observation():
    return {"state": action(), "vision": {}, "instruction": "test"}


class FakeEnv:
    def __init__(self, steps):
        self.steps = steps
        self.count = 0

    def get_obs(self):
        return observation()

    def take_action(self, _):
        self.count += 1

    def is_episode_end(self):
        return self.count >= self.steps


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, func_name, **_):
        self.calls.append(func_name)
        if func_name == "rtc_start":
            return {"generation": 0, "cursor": 0, "inflight": False}
        if func_name == "rtc_next_action":
            return action()
        if func_name == "rtc_commit":
            return {"generation": 1, "cursor": 1, "inflight": True}
        if func_name == "get_action":
            return [action() for _ in range(50)]
        return None


class DeployModeTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {"EVAL_ENV_TYPE": "debug"})
        self.environment.start()
        self.sleep = mock.patch.object(deploy.time, "sleep")
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()
        self.environment.stop()

    @staticmethod
    def config(mode):
        return {
            "hardware_output_enabled": False,
            "control_hz": 25.0,
            "execute_steps": 15,
            "execution_mode": mode,
            "right_arm_initial_envelope_rad": 0.35,
            "max_command_velocity_rad_s": 2.0,
            "max_raw_target_error_rad": 1.0,
            "max_command_tracking_error_rad": 0.5,
            "max_gripper_rate_s": 5.0,
        }

    def test_rtc_commits_latest_observation_and_stops(self):
        client = FakeClient()
        env = FakeEnv(steps=4)
        with mock.patch.object(deploy, "_gate", return_value=self.config("rtc")):
            deploy.eval_one_episode(env, client)

        self.assertEqual(env.count, 4)
        self.assertEqual(client.calls.count("rtc_next_action"), 4)
        self.assertEqual(client.calls.count("rtc_commit"), 3)
        self.assertEqual(client.calls[-1], "rtc_stop")

    def test_synchronous_prefix_remains_available(self):
        client = FakeClient()
        env = FakeEnv(steps=5)
        with mock.patch.object(
            deploy, "_gate", return_value=self.config("synchronous_prefix")
        ):
            deploy.eval_one_episode(env, client)

        self.assertEqual(env.count, 5)
        self.assertEqual(client.calls.count("get_action"), 1)
        self.assertNotIn("rtc_start", client.calls)

    def test_action_ema_and_reset(self):
        first = action(0.0)
        second = action(1.0)
        ema = deploy._ActionEMA(0.4)
        np.testing.assert_allclose(ema.apply(first)["left_arm_joint_state"], 0.0)
        np.testing.assert_allclose(ema.apply(second)["left_arm_joint_state"], 0.4)
        reset = deploy._ActionEMA(0.4)
        np.testing.assert_allclose(reset.apply(second)["left_arm_joint_state"], 1.0)

    def test_safety_bypass_only_rejects_non_finite_values(self):
        cfg = self.config("rtc")
        cfg["software_safety_enabled"] = False
        raw = action(10.0)
        result = deploy._command(raw, observation(), cfg, mock.Mock())
        np.testing.assert_array_equal(result["left_arm_joint_state"], 10.0)
        raw["left_arm_joint_state"][0] = np.nan
        with self.assertRaises(ValueError):
            deploy._command(raw, observation(), cfg, mock.Mock())

    def test_left_j5_hardware_direction_is_adapted_once(self):
        cfg = self.config("synchronous_prefix")
        cfg["software_safety_enabled"] = False
        cfg["left_j5_sign"] = -1.0
        raw = action(0.0)
        raw["left_arm_joint_state"][4] = 0.25
        result = deploy._command(raw, observation(), cfg, mock.Mock())
        self.assertEqual(result["left_arm_joint_state"][4], -0.25)
        self.assertEqual(raw["left_arm_joint_state"][4], 0.25)

    def test_right_j5_hardware_direction_is_adapted_once(self):
        cfg = self.config("synchronous_prefix")
        cfg["software_safety_enabled"] = False
        cfg["right_j5_sign"] = -1.0
        raw = action(0.0)
        raw["right_arm_joint_state"][4] = 0.25
        result = deploy._command(raw, observation(), cfg, mock.Mock())
        self.assertEqual(result["right_arm_joint_state"][4], -0.25)
        self.assertEqual(raw["right_arm_joint_state"][4], 0.25)


if __name__ == "__main__":
    unittest.main()
