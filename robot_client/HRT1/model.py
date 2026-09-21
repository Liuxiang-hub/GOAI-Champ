"""XPolicyLab protocol adapter for the remote Pi05 (real-piper6-lora/7594) service.

Modelled on policy/FinalGOAI: this adapter is itself an XPolicyLab ws server
(the official Collector/eval-runner connect to it) whose Model forwards
inference to the Pi05 policy server on L20 via the XPolicyLab ws protocol
(default ws://127.0.0.1:6198, SSH-forwarded from the field machine).

The adapter keeps RTC targets in physical absolute-action space.  The Pi0.5
server maps them through the checkpoint's real DeltaActions, Normalize and
PadStatesAndActions transforms before applying per-denoising-step guidance.
"""
import json
import math
import threading
import time

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)
from client_server.ws import WsModelClient

from .rtc_core import RTCConfig, RealTimeChunkingController


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = dict(model_cfg)
        if self.cfg['action_type'] != 'joint':
            raise ValueError('Pi05 real-piper6-lora/7594 supports joint actions only')
        self.dims = get_robot_action_dim_info(self.cfg['env_cfg_type'])
        if self.dims != {'arm_dim': [6, 6], 'ee_dim': [1, 1]}:
            raise ValueError(f'Checkpoint requires two 6-joint arms and two grippers, got {self.dims}')
        self.action_dim = sum(self.dims['arm_dim']) + sum(self.dims['ee_dim'])
        self.execute_steps = int(self.cfg.get('execute_steps', 15))
        if not 1 <= self.execute_steps <= 50:
            raise ValueError('execute_steps must be between 1 and 50')
        self.timeout = float(self.cfg.get('request_timeout_s', 60))

        self.url = self.cfg.get('pi05_url', 'ws://127.0.0.1:6198')
        self.lock = threading.RLock()
        self.client = WsModelClient(
            url=self.url,
            evaluation_id='pi05-piperx-adapter',
            trial_id='adapter-session',
            request_timeout_s=self.timeout,
            connect_timeout_s=60.0,
        )
        self.rtc = None
        self.observation = None
        self.last_diagnostics = {}
        self.request_seq = 0
        self.last_translate_ms = None
        self.reset()

    # ---------- observation/action translation ----------

    def _translate(self, obs):
        state = pack_robot_state(obs, 'joint', self.dims).astype(np.float32)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError('Invalid robot state')
        prompt = obs.get('instruction', obs.get('instructions'))
        if not prompt:
            prompt = self.cfg.get('default_instruction', '')
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ''
        if isinstance(prompt, bytes):
            prompt = prompt.decode('utf-8')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Official observation must supply a nonempty instruction')
        images = {}
        for target, aliases in self.cfg['camera_mapping'].items():
            name = next((key for key in aliases if key in obs['vision']), None)
            if name is None:
                raise KeyError(f'Missing camera: {aliases}')
            value = obs['vision'][name]
            if isinstance(value, dict):
                value = value.get('color', value.get('rgb'))
            if isinstance(value, (bytes, str)):
                value = decode_image_bit(value)
            img = np.asarray(value)
            if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
                raise ValueError(f'{name}: expected decoded RGB uint8 HWC, got {img.shape}/{img.dtype}')
            images[target] = np.ascontiguousarray(img).copy()
        translated = {'images': images, 'state': state, 'instruction': prompt}
        for key in ('_rtc', '_rtc_context'):
            if key in obs:
                translated[key] = obs[key]
        return translated

    @staticmethod
    def _pack_steps(steps):
        rows = []
        for step in steps:
            row = np.concatenate([
                np.asarray(step['left_arm_joint_state'], dtype=np.float64).ravel(),
                np.asarray(step['left_ee_joint_state'], dtype=np.float64).ravel(),
                np.asarray(step['right_arm_joint_state'], dtype=np.float64).ravel(),
                np.asarray(step['right_ee_joint_state'], dtype=np.float64).ravel(),
            ])
            rows.append(row)
        arr = np.stack(rows).astype(np.float32)
        if arr.shape != (50, 14) or not np.isfinite(arr).all():
            raise ValueError(f'Invalid remote action chunk: {arr.shape}')
        return arr

    def _call(self, translated):
        started = time.perf_counter()
        with self.lock:
            self.client.call(func_name='update_obs', obs=translated)
            updated = time.perf_counter()
            response = self.client.call(func_name='get_action_with_metadata')
        inferred = time.perf_counter()
        if not isinstance(response, dict):
            raise ValueError('Pi0.5 server did not return inference metadata')
        model_actions = np.asarray(response.get('_rtc_model_actions'), dtype=np.float32)
        if model_actions.shape != (50, 32) or not np.isfinite(model_actions).all():
            raise ValueError(f'Invalid Pi0.5 model-space action chunk: {model_actions.shape}')
        context = response.get('_rtc_context')
        if not isinstance(context, dict):
            raise ValueError('Pi0.5 server response is missing _rtc_context')
        actions = self._pack_steps(response.get('actions'))
        packed = time.perf_counter()
        self.request_seq += 1
        timing = {
            'request_seq': self.request_seq,
            'translate_ms': self.last_translate_ms,
            'upstream_update_obs_ms': round((updated - started) * 1000, 3),
            'upstream_get_action_ms': round((inferred - updated) * 1000, 3),
            'action_validate_pack_ms': round((packed - inferred) * 1000, 3),
            'upstream_total_ms': round((packed - started) * 1000, 3),
        }
        print('HRT1_UPSTREAM_TIMING ' + json.dumps(timing), flush=True)
        return {
            'action': actions,
            '_rtc_model_actions': model_actions,
            '_rtc_context': context,
        }

    # ---------- ModelTemplate interface ----------

    def update_obs(self, obs):
        started = time.perf_counter()
        self.observation = self._translate(obs)
        self.last_translate_ms = round((time.perf_counter() - started) * 1000, 3)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise NotImplementedError('Pi05_PiperX supports one real environment per session')
        self.update_obs(obs_list[0])

    def get_action(self):
        if self.observation is None:
            raise RuntimeError('update_obs must precede get_action')
        response = self._call(self.observation)
        actions = response['action']
        self.last_diagnostics = {'upstream': self.url, 'execute_steps': self.execute_steps}
        # Server returns absolute joint targets in [left arm, left gripper,
        # right arm, right gripper] order. No second state addition or scaling.
        return unpack_robot_state(actions[:self.execute_steps].copy(), 'joint', self.dims)

    def get_action_with_metadata(self):
        """Return the full chunk for a field-local RTC controller."""
        if self.observation is None:
            raise RuntimeError('update_obs must precede get_action_with_metadata')
        response = self._call(self.observation)
        self.last_diagnostics = {
            'upstream': self.url,
            'execute_steps': self.execute_steps,
            'metadata_response': True,
        }
        return response

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list not in (None, [0]):
            raise NotImplementedError('Pi05_PiperX supports one real environment per session')
        return [self.get_action()]

    def reset(self):
        self.rtc_stop()
        self.observation, self.last_diagnostics = None, {}
        with self.lock:
            self.client.call(func_name='reset')

    # ---------- RTC ----------

    def _rebase(self, chunk, old, new):
        """Express the old joint trajectory relative to the latest state."""
        result = np.array(chunk, copy=True)
        cols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        delta = np.asarray(new['state'])[cols] - np.asarray(old['state'])[cols]
        result[:, cols] += delta
        return result

    def _rtc_infer(self, request):
        return self._call(request)

    def rtc_start(self, obs):
        self.rtc_stop()
        self.update_obs(obs)
        horizon = 50
        lookahead_ratio = float(self.cfg.get('rtc_lookahead_ratio', 0.4))
        if not 0.0 < lookahead_ratio < 1.0:
            raise ValueError('rtc_lookahead_ratio must be between 0 and 1')
        trigger_cursor = int(self.cfg.get(
            'rtc_trigger_step', math.ceil(horizon * (1.0 - lookahead_ratio))))
        if not 1 <= trigger_cursor < horizon:
            raise ValueError('rtc_trigger_step must be in [1, horizon)')
        self.rtc = RealTimeChunkingController(self._rtc_infer, RTCConfig(
            prediction_horizon=horizon,
            minimum_execution_steps=trigger_cursor,
            control_hz=float(self.cfg.get('control_hz', 10)),
            initial_delay_steps=int(self.cfg.get('rtc_initial_delay_steps', 11)),
            delay_buffer_size=int(self.cfg.get('rtc_latency_window', 5)),
            safety_margin_steps=int(self.cfg.get('rtc_safety_margin_steps', 2)),
            blend_steps=int(self.cfg.get('rtc_blend_steps', 5)),
            prewarm_guided=bool(self.cfg.get('rtc_prewarm_guided', True)),
        ), rebase_guidance=self._rebase)
        self.rtc.start(self.observation)
        return self.rtc.status()

    def rtc_next_action(self):
        action = np.asarray(self.rtc.next_action()['action'], dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Invalid RTC action')
        return unpack_robot_state(action, 'joint', self.dims)

    def rtc_commit(self, obs):
        self.update_obs(obs)
        self.rtc.commit(self.observation)
        return self.rtc.status()

    def rtc_stop(self):
        if getattr(self, 'rtc', None) is not None:
            self.rtc.stop(timeout=self.timeout + 2)
            self.rtc = None

    def status(self):
        return {'metadata': {'upstream': self.url, 'ckpt': self.cfg.get('ckpt_name')},
                'rtc': self.rtc.status() if self.rtc else None,
                'diagnostics': self.last_diagnostics}
