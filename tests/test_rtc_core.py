import threading
import time
import unittest

import numpy as np

from robot_client.Pi05_PiperX.rtc_core import (
    RTCConfig,
    RTCStaleResponse,
    RealTimeChunkingController,
    build_rtc_guidance,
    soft_mask_weights,
)


def response_for(request, value=0.0):
    return {
        "action": np.full((50, 14), value, dtype=np.float32),
        "_rtc_model_actions": np.full((50, 32), value, dtype=np.float32),
        "_rtc_context": dict(request["_rtc_context"]),
    }


class RTCContractTest(unittest.TestCase):
    def test_guidance_keeps_physical_actions_and_decays(self):
        chunk = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
        guidance = build_rtc_guidance(
            chunk,
            start_steps=5,
            delay_steps=3,
            beta=5.0,
            generation=7,
            request_id=11,
        )

        np.testing.assert_array_equal(guidance["actions"][:45], chunk[5:])
        np.testing.assert_array_equal(guidance["actions"][45:], 0.0)
        self.assertEqual(guidance["generation"], 7)
        self.assertEqual(guidance["request_id"], 11)
        weights = guidance["weights"]
        np.testing.assert_array_equal(weights[:3], 1.0)
        self.assertTrue(np.all(np.diff(weights[3:45]) <= 0.0))
        np.testing.assert_array_equal(weights[45:], 0.0)

    def test_soft_mask_rejects_delay_without_overlap(self):
        with self.assertRaises(ValueError):
            soft_mask_weights(horizon=50, start_steps=48, delay_steps=2)

    def test_blend_window_limits_model_guidance_transition(self):
        weights = soft_mask_weights(
            horizon=50, start_steps=30, delay_steps=13, blend_steps=5
        )
        np.testing.assert_array_equal(weights[:13], 1.0)
        self.assertTrue(np.all(np.diff(weights[13:18]) < 0.0))
        np.testing.assert_array_equal(weights[18:], 0.0)

    def test_response_identity_is_mandatory(self):
        controller = RealTimeChunkingController(
            lambda request: response_for(request),
            RTCConfig(
                minimum_execution_steps=2,
                initial_delay_steps=1,
                prewarm_guided=False,
            ),
        )
        stale = response_for({"_rtc_context": {"generation": 1, "request_id": 9}})
        with self.assertRaises(RTCStaleResponse):
            controller._validate_response(
                stale, expected_generation=2, expected_request_id=9
            )

    def test_late_response_after_stop_never_replaces_plan(self):
        inference_started = threading.Event()
        release_inference = threading.Event()
        calls = []

        def infer(request):
            calls.append(request)
            if len(calls) > 1:
                inference_started.set()
                release_inference.wait(timeout=2.0)
            return response_for(request, value=float(len(calls)))

        controller = RealTimeChunkingController(
            infer,
            RTCConfig(
                minimum_execution_steps=1,
                initial_delay_steps=1,
                prewarm_guided=False,
            ),
        )
        controller.start({"state": np.zeros(14, dtype=np.float32)})
        controller.next_action()
        controller.commit({"state": np.zeros(14, dtype=np.float32)})
        self.assertTrue(inference_started.wait(timeout=1.0))

        stopper = threading.Thread(target=controller.stop, kwargs={"timeout": 2.0})
        stopper.start()
        time.sleep(0.02)
        release_inference.set()
        stopper.join(timeout=2.0)

        self.assertFalse(stopper.is_alive())
        self.assertFalse(controller.status()["running"])
        self.assertGreaterEqual(controller.status()["generation"], 1)
        self.assertEqual(controller.status()["stale_responses"], 1)
        self.assertIn("stopped", controller.status()["last_discard_reason"])

    def test_start_prewarms_guided_branch_before_running(self):
        requests = []

        def infer(request):
            requests.append(request)
            return response_for(request)

        controller = RealTimeChunkingController(
            infer,
            RTCConfig(minimum_execution_steps=2, initial_delay_steps=1),
        )
        controller.start({"state": np.zeros(14, dtype=np.float32)})
        try:
            self.assertEqual(len(requests), 2)
            self.assertNotIn("_rtc", requests[0])
            self.assertIn("_rtc", requests[1])
            np.testing.assert_array_equal(requests[1]["_rtc"]["weights"], 0.0)
            self.assertEqual(controller.status()["request_id"], 1)
        finally:
            controller.stop()


if __name__ == "__main__":
    unittest.main()
