from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CheckpointSwitchContractTest(unittest.TestCase):
    def test_checkpoint_is_selected_only_on_l20(self):
        server_config = (ROOT / "l20_server/Pi_05/deploy.yml").read_text()
        robot_config = (
            ROOT / "robot_client/Pi05_PiperX/deploy.yml"
        ).read_text()

        self.assertIn("ckpt_name: real-piper6-lora/", server_config)
        self.assertIn("train_config_name: pi05_base_piper6_lora_real", server_config)
        self.assertIn("repo_id: yangchenjie/robodojo_piper6_v3", server_config)
        self.assertNotIn("ckpt_name:", robot_config)
        self.assertNotIn("checkpoint_num:", robot_config)
        self.assertNotIn("execute_steps:", robot_config)
        self.assertIn(
            '"execute_steps":',
            (ROOT / "robot_client/Pi05_PiperX/motion_gate.json").read_text(),
        )

    def test_robot_adapter_uses_upstream_shape_metadata(self):
        source = (
            ROOT / "robot_client/Pi05_PiperX/model.py"
        ).read_text()

        self.assertIn("self.upstream_metadata['action_horizon']", source)
        self.assertIn("self.upstream_metadata['model_action_dim']", source)
        self.assertNotIn("real-piper6-lora/" + "7594", source)


if __name__ == "__main__":
    unittest.main()
