"""XPolicyLab protocol adapter for a compatible remote Pi0.5 service.

Modelled on policy/FinalGOAI: this adapter is itself an XPolicyLab ws server
(the official Collector/eval-runner connect to it) whose Model forwards
inference to the Pi05 policy server on L20 via the XPolicyLab ws protocol
(default ws://127.0.0.1:6198, SSH-forwarded from the field machine).

The adapter keeps RTC targets in physical absolute-action space.  The Pi0.5
server maps them through the checkpoint's real DeltaActions, Normalize and
PadStatesAndActions transforms before applying per-denoising-step guidance.
"""
import math
import threading

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
        if float(self.cfg.get('left_j5_sign', 1.0)) not in (-1.0, 1.0):
            raise ValueError('left_j5_sign must be -1 or 1')
        if float(self.cfg.get('right_j5_sign', 1.0)) not in (-1.0, 1.0):
            raise ValueError('right_j5_sign must be -1 or 1')
        if self.cfg['action_type'] != 'joint':
            raise ValueError('Pi0.5 Piper-X adapter supports joint actions only')
        self.dims = get_robot_action_dim_info(self.cfg['env_cfg_type'])
        if self.dims != {'arm_dim': [6, 6], 'ee_dim': [1, 1]}:
            raise ValueError(f'Checkpoint requires two 6-joint arms and two grippers, got {self.dims}')
        self.action_dim = sum(self.dims['arm_dim']) + sum(self.dims['ee_dim'])
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
        upstream_status = self.client.call(func_name='status')
        self.upstream_metadata = (
            upstream_status.get('metadata', {})
            if isinstance(upstream_status, dict) else {}
        )
        if self.upstream_metadata.get('policy_family') != 'pi05':
            raise ValueError(
                f"Expected Pi0.5 upstream, got {self.upstream_metadata!r}"
            )
        self.horizon = int(self.upstream_metadata['action_horizon'])
        self.model_action_dim = int(self.upstream_metadata['model_action_dim'])
        if self.horizon <= 0 or self.model_action_dim <= 0:
            raise ValueError('Upstream action dimensions must be positive')
        if int(self.upstream_metadata.get('physical_action_dim', -1)) != self.action_dim:
            raise ValueError(
                f"Upstream physical action dim does not match Piper-X: "
                f"{self.upstream_metadata!r}"
            )
        self.rtc = None
        self.observation = None
        self.last_diagnostics = {}
        self.reset()

    # ---------- observation/action translation ----------

    def _translate(self, obs):
        state = pack_robot_state(obs, 'joint', self.dims).astype(np.float32)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError('Invalid robot state')
        state[4] *= float(self.cfg.get('left_j5_sign', 1.0))
        state[11] *= float(self.cfg.get('right_j5_sign', 1.0))
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

    def _pack_steps(self, steps):
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
        if arr.shape != (self.horizon, self.action_dim) or not np.isfinite(arr).all():
            raise ValueError(f'Invalid remote action chunk: {arr.shape}')
        return arr

    def _call(self, translated):
        with self.lock:
            self.client.call(func_name='update_obs', obs=translated)
            response = self.client.call(func_name='get_action_with_metadata')
        if not isinstance(response, dict):
            raise ValueError('Pi0.5 server did not return inference metadata')
        model_actions = np.asarray(response.get('_rtc_model_actions'), dtype=np.float32)
        expected_model_shape = (self.horizon, self.model_action_dim)
        if model_actions.shape != expected_model_shape or not np.isfinite(model_actions).all():
            raise ValueError(f'Invalid Pi0.5 model-space action chunk: {model_actions.shape}')
        context = response.get('_rtc_context')
        if not isinstance(context, dict):
            raise ValueError('Pi0.5 server response is missing _rtc_context')
        return {
            'action': self._pack_steps(response.get('actions')),
            '_rtc_model_actions': model_actions,
            '_rtc_context': context,
        }

    # ---------- ModelTemplate interface ----------

    def update_obs(self, obs):
        self.observation = self._translate(obs)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise NotImplementedError('Pi05_PiperX supports one real environment per session')
        self.update_obs(obs_list[0])

    def get_action(self):
        if self.observation is None:
            raise RuntimeError('update_obs must precede get_action')
        response = self._call(self.observation)
        actions = response['action']
        self.last_diagnostics = {'upstream': self.url, 'action_horizon': self.horizon}
        # Server returns absolute joint targets in [left arm, left gripper,
        # right arm, right gripper] order. No second state addition or scaling.
        return unpack_robot_state(actions.copy(), 'joint', self.dims)

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
        horizon = self.horizon
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
        return {'metadata': {'upstream': self.url, **self.upstream_metadata},
                'rtc': self.rtc.status() if self.rtc else None,
                'diagnostics': self.last_diagnostics}
