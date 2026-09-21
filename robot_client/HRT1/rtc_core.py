"""Real-Time Chunking controller for asynchronous Pi0.5 deployment.

This module implements the controller side of RTC (arXiv:2506.07339): the
robot consumes the current action chunk at a fixed rate while a background
thread asks the flow policy to inpaint the next chunk. Model-side guidance is
applied by the patched OpenPI sampler on every denoising step.

The API deliberately separates issuing and committing an action.  A caller
must execute the action returned by :meth:`next_action`, then call
:meth:`commit` with the observation captured after that execution.  This
avoids treating an action as executed before the robot actually consumed it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
import math
import threading
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class RTCError(RuntimeError):
    """Base error for the RTC runtime."""


class RTCPlanExhausted(RTCError):
    """No safe action remains while the next inference is unavailable."""


class RTCStaleResponse(RTCError):
    """An inference result belongs to an obsolete request or generation."""


@dataclass(frozen=True)
class RTCConfig:
    prediction_horizon: int = 50
    minimum_execution_steps: int = 15
    control_hz: float = 25.0
    initial_delay_steps: int = 10
    delay_buffer_size: int = 10
    safety_margin_steps: int = 1
    blend_steps: int | None = None
    guidance_beta: float = 5.0
    prewarm_guided: bool = True

    def __post_init__(self) -> None:
        if self.prediction_horizon < 2:
            raise ValueError("prediction_horizon must be >= 2")
        if not 1 <= self.minimum_execution_steps < self.prediction_horizon:
            raise ValueError("minimum_execution_steps must be in [1, horizon)")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not 0 <= self.initial_delay_steps + self.safety_margin_steps < (
            self.prediction_horizon - self.minimum_execution_steps
        ):
            raise ValueError("initial delay plus safety margin leaves no RTC overlap")
        if self.delay_buffer_size < 1:
            raise ValueError("delay_buffer_size must be >= 1")
        if self.safety_margin_steps < 0:
            raise ValueError("safety_margin_steps must be >= 0")
        if self.blend_steps is not None and self.blend_steps < 1:
            raise ValueError("blend_steps must be >= 1 when configured")
        if self.guidance_beta <= 0:
            raise ValueError("guidance_beta must be positive")


def latency_to_steps(latency_ms: float, control_hz: float, margin_steps: int = 1) -> int:
    """Convert end-to-end latency to a conservative controller-step count."""
    if latency_ms < 0 or control_hz <= 0 or margin_steps < 0:
        raise ValueError("invalid latency, frequency, or margin")
    return int(math.ceil(latency_ms * control_hz / 1000.0)) + margin_steps


def soft_mask_weights(
    horizon: int,
    start_steps: int,
    delay_steps: int,
    blend_steps: int | None = None,
) -> np.ndarray:
    """Return the exponential RTC soft mask from Eq. 4 of the paper."""
    if not 0 <= start_steps < horizon:
        raise ValueError("start_steps must be in [0, horizon)")
    overlap = horizon - start_steps
    if not 0 <= delay_steps < overlap:
        raise ValueError(
            f"delay_steps={delay_steps} must be smaller than overlap={overlap}"
        )

    weights = np.zeros(horizon, dtype=np.float32)
    weights[:delay_steps] = 1.0
    transition_steps = overlap - delay_steps
    if blend_steps is not None:
        transition_steps = min(transition_steps, blend_steps)
    denominator = transition_steps + 1
    for index in range(delay_steps, delay_steps + transition_steps):
        c_i = (delay_steps + transition_steps - index) / denominator
        weights[index] = c_i * np.expm1(c_i) / np.expm1(1.0)
    return weights


def build_rtc_guidance(
    action_chunk: np.ndarray,
    start_steps: int,
    delay_steps: int,
    beta: float,
    generation: int,
    request_id: int,
    blend_steps: int | None = None,
) -> dict[str, Any]:
    """Build a right-padded physical-action guidance payload."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"normalized chunk must be rank 2, got {chunk.shape}")
    horizon = chunk.shape[0]
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains NaN or Inf")

    remaining = chunk[start_steps:]
    target = np.zeros_like(chunk)
    target[: remaining.shape[0]] = remaining
    return {
        "actions": target,
        "weights": soft_mask_weights(
            horizon, start_steps, delay_steps, blend_steps=blend_steps
        ),
        "beta": float(beta),
        "start_steps": int(start_steps),
        "delay_steps": int(delay_steps),
        "generation": int(generation),
        "request_id": int(request_id),
    }


class RealTimeChunkingController:
    """Threaded RTC double buffer around a synchronous policy ``infer`` call.

    ``infer`` must accept one observation dictionary and return full physical
    action arrays, exact model-space actions, and the echoed RTC context.
    """

    def __init__(self, infer: Callable[[dict[str, Any]], Mapping[str, Any]], config: RTCConfig, rebase_guidance=None):
        self._infer = infer
        self._rebase_guidance = rebase_guidance
        self._plan_origin = None
        self.config = config
        self._condition = threading.Condition()
        self._delay_steps = deque(
            [config.initial_delay_steps], maxlen=config.delay_buffer_size
        )
        self._thread: threading.Thread | None = None
        self._running = False
        self._inflight = False
        self._issued = False
        self._cursor = 0
        self._generation = 0
        self._request_id = 0
        self._stale_responses = 0
        self._last_discard_reason: str | None = None
        self._latest_observation: dict[str, Any] | None = None
        self._physical_chunk: dict[str, np.ndarray] | None = None
        self._model_chunk: np.ndarray | None = None
        self._background_error: BaseException | None = None

    @property
    def delay_history(self) -> tuple[int, ...]:
        with self._condition:
            return tuple(self._delay_steps)

    @property
    def cursor(self) -> int:
        with self._condition:
            return self._cursor

    def status(self):
        with self._condition:
            return dict(cursor=self._cursor, generation=self._generation,
                        inflight=self._inflight, running=self._running,
                        request_id=self._request_id,
                        stale_responses=self._stale_responses,
                        last_discard_reason=self._last_discard_reason,
                        delay_history=list(self._delay_steps))

    def _validate_response(
        self,
        response: Mapping[str, Any],
        *,
        expected_generation: int,
        expected_request_id: int,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        context = response.get("_rtc_context")
        if not isinstance(context, Mapping):
            raise RTCError("server response is missing _rtc_context")
        actual = (context.get("generation"), context.get("request_id"))
        expected = (expected_generation, expected_request_id)
        if actual != expected:
            raise RTCStaleResponse(
                f"discarding stale RTC response {actual}, expected {expected}"
            )

        if "_rtc_model_actions" not in response:
            raise RTCError("server response is missing _rtc_model_actions")
        model_actions = np.asarray(response["_rtc_model_actions"], dtype=np.float32)
        if model_actions.ndim == 3 and model_actions.shape[0] == 1:
            model_actions = model_actions[0]
        expected_horizon = self.config.prediction_horizon
        if model_actions.ndim != 2 or model_actions.shape[0] != expected_horizon:
            raise RTCError(
                f"model action shape {model_actions.shape} does not start with "
                f"horizon {expected_horizon}"
            )
        if not np.isfinite(model_actions).all():
            raise RTCError("model-space response contains NaN or Inf")

        physical: dict[str, np.ndarray] = {}
        for key, value in response.items():
            if key.startswith("_") or key == "server_timing":
                continue
            array = np.asarray(value)
            if array.ndim >= 1 and array.shape[0] == expected_horizon:
                if not np.isfinite(array).all():
                    raise RTCError(f"physical action {key} contains NaN or Inf")
                physical[key] = array.copy()
        if not physical:
            raise RTCError("server response contains no full physical action chunk")
        return physical, model_actions.copy()

    def start(self, initial_observation: Mapping[str, Any]) -> None:
        """Synchronously obtain the initial chunk, then start background RTC."""
        with self._condition:
            if self._running:
                raise RTCError("RTC controller is already running")

        request = dict(initial_observation)
        request["_rtc_context"] = {"generation": 0, "request_id": 0}
        physical, model_actions = self._validate_response(
            self._infer(request), expected_generation=0, expected_request_id=0
        )
        initial_request_id = 0
        if self.config.prewarm_guided:
            if "action" not in physical:
                raise RTCError("RTC response has no canonical physical action chunk")
            initial_request_id = 1
            warmup = dict(initial_observation)
            warmup["_rtc"] = {
                "actions": physical["action"].copy(),
                "weights": np.zeros(self.config.prediction_horizon, dtype=np.float32),
                "beta": self.config.guidance_beta,
                "start_steps": 0,
                "delay_steps": 0,
                "generation": 0,
                "request_id": initial_request_id,
            }
            warmup["_rtc_context"] = {
                "generation": 0,
                "request_id": initial_request_id,
            }
            # Compile and exercise the guided JAX branch before any action can
            # be issued. The warmup result is deliberately discarded.
            self._validate_response(
                self._infer(warmup),
                expected_generation=0,
                expected_request_id=initial_request_id,
            )
        with self._condition:
            self._physical_chunk = physical
            self._model_chunk = model_actions
            self._latest_observation = dict(initial_observation)
            self._plan_origin = dict(initial_observation)
            self._cursor = 0
            self._generation = 0
            self._request_id = initial_request_id
            self._stale_responses = 0
            self._last_discard_reason = None
            self._issued = False
            self._background_error = None
            self._running = True
            self._thread = threading.Thread(
                target=self._inference_loop,
                name="lingbot-rtc-inference",
                daemon=True,
            )
            self._thread.start()

    def next_action(self) -> dict[str, np.ndarray]:
        """Issue exactly one action; call ``commit`` only after it executes."""
        with self._condition:
            self._raise_background_error()
            if not self._running or self._physical_chunk is None:
                raise RTCError("RTC controller is not running")
            if self._issued:
                raise RTCError("previous action has not been committed")
            if self._cursor >= self.config.prediction_horizon:
                raise RTCPlanExhausted(
                    "50-step plan exhausted before inference completed; stop the robot"
                )
            action = {
                key: values[self._cursor].copy()
                for key, values in self._physical_chunk.items()
            }
            self._issued = True
            return action

    def commit(self, observation_after_action: Mapping[str, Any]) -> None:
        """Confirm execution and publish the newest observation to inference."""
        with self._condition:
            self._raise_background_error()
            if not self._issued:
                raise RTCError("commit called without an issued action")
            self._cursor += 1
            self._issued = False
            self._latest_observation = dict(observation_after_action)
            self._condition.notify_all()

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._running = False
            self._generation += 1
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RTCError("RTC inference thread did not stop within timeout")

    def _raise_background_error(self) -> None:
        if self._background_error is not None:
            raise RTCError("RTC background inference failed") from self._background_error

    def _inference_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._running
                    or (
                        not self._inflight
                        and self._cursor >= self.config.minimum_execution_steps
                        and self._latest_observation is not None
                    )
                )
                if not self._running:
                    return
                assert self._physical_chunk is not None
                start_steps = self._cursor
                overlap = self.config.prediction_horizon - start_steps
                predicted_delay = max(self._delay_steps) + self.config.safety_margin_steps
                if predicted_delay >= overlap:
                    self._background_error = RTCPlanExhausted(
                        f"predicted delay {predicted_delay} leaves no overlap {overlap}"
                    )
                    self._running = False
                    self._condition.notify_all()
                    return
                request = dict(self._latest_observation)
                if "action" not in self._physical_chunk:
                    self._background_error = RTCError(
                        "RTC response has no canonical physical action chunk"
                    )
                    self._running = False
                    self._condition.notify_all()
                    return
                guidance_chunk = self._physical_chunk["action"]
                if self._rebase_guidance is not None:
                    try:
                        guidance_chunk = self._rebase_guidance(
                            guidance_chunk, self._plan_origin, request)
                    except Exception as exc:
                        self._background_error = exc
                        self._running = False
                        self._condition.notify_all()
                        return
                generation = self._generation
                self._request_id += 1
                request_id = self._request_id
                request["_rtc"] = build_rtc_guidance(
                    guidance_chunk,
                    start_steps=start_steps,
                    delay_steps=predicted_delay,
                    beta=self.config.guidance_beta,
                    generation=generation,
                    request_id=request_id,
                    blend_steps=self.config.blend_steps,
                )
                request["_rtc_context"] = {
                    "generation": generation,
                    "request_id": request_id,
                }
                self._inflight = True

            try:
                physical, model_actions = self._validate_response(
                    self._infer(request),
                    expected_generation=generation,
                    expected_request_id=request_id,
                )
            except RTCStaleResponse as exc:
                with self._condition:
                    self._stale_responses += 1
                    self._last_discard_reason = str(exc)
                    self._inflight = False
                    self._condition.notify_all()
                logger.warning("%s", exc)
                continue
            except BaseException as exc:
                with self._condition:
                    self._background_error = exc
                    self._inflight = False
                    self._running = False
                    self._condition.notify_all()
                return

            with self._condition:
                if not self._running:
                    self._stale_responses += 1
                    self._last_discard_reason = (
                        f"discarding request {request_id}: RTC stopped while inference was in flight"
                    )
                    logger.warning("%s", self._last_discard_reason)
                    return
                if generation != self._generation:
                    self._stale_responses += 1
                    self._last_discard_reason = (
                        f"discarding request {request_id}: generation changed from "
                        f"{generation} to {self._generation}"
                    )
                    logger.warning("%s", self._last_discard_reason)
                    self._inflight = False
                    self._condition.notify_all()
                    continue
                observed_delay = self._cursor - start_steps
                self._physical_chunk = physical
                self._model_chunk = model_actions
                self._plan_origin = {k: v for k, v in request.items() if not k.startswith('_rtc')}
                self._cursor = max(0, observed_delay)
                self._generation += 1
                self._delay_steps.append(max(0, observed_delay))
                self._inflight = False
                self._condition.notify_all()
