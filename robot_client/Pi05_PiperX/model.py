"""XPolicyLab protocol adapter for the remote Pi05 (real-piper6-lora/7594) service.

Modelled on policy/FinalGOAI: this adapter is itself an XPolicyLab ws server
(the official Collector/eval-runner connect to it) whose Model forwards
inference to the Pi05 policy server on L20 via the XPolicyLab ws protocol
(default ws://127.0.0.1:6198, SSH-forwarded from the field machine).

Differences from FinalGOAI:
- Upstream is XPolicyLab ws (WsModelClient), not the LingBot msgpack wire.
- The vendored openpi build has no RTC guidance/inpainting support, so `_rtc`
  payloads are constructed by rtc_core but stripped before going upstream;
  chunk replacement relies on the rebase hook only.
- `_normalized_actions` are synthesized adapter-side from the physical chunk
  using the piper6 norm stats (openpi q01/q99 normalization), because the
  server does not return normalized actions.
"""
import json
import os
from pathlib import Path
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

        stats = json.loads(
            Path(__file__).with_name('piper6_norm_stats.json').read_text()
        )['norm_stats']['actions']
        self.q01 = np.asarray(stats['q01'], dtype=np.float64)
        self.q99 = np.asarray(stats['q99'], dtype=np.float64)
        if self.q01.shape != (self.action_dim,) or self.q99.shape != (self.action_dim,):
            raise ValueError('piper6_norm_stats.json action stats must be 14-dim')

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
        self.reset()

    # ---------- observation/action translation ----------

    def _translate(self, obs):
        state = pack_robot_state(obs, 'joint', self.dims).astype(np.float32)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError('Invalid robot state')
        prompt = obs.get('instruction', obs.get('instructions'))
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
        return {'images': images, 'state': state, 'instruction': prompt}

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
        # Gripper undershoot below 0 is a normalization artifact (observed down
        # to -0.011, i.e. ~1 mm past closed); the physical stroke is [0, 1].
        arr[:, 6] = np.clip(arr[:, 6], 0.0, 1.0)
        arr[:, 13] = np.clip(arr[:, 13], 0.0, 1.0)
        return arr

    def _normalize(self, chunk):
        return np.clip(
            (np.asarray(chunk, dtype=np.float64) - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0,
            -1.0, 1.0,
        ).astype(np.float32)

    def _call(self, translated):
        with self.lock:
            self.client.call(func_name='update_obs', obs=translated)
            steps = self.client.call(func_name='get_action')
        return self._pack_steps(steps)

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
        actions = self._call(self.observation)
        self.last_diagnostics = {'upstream': self.url, 'execute_steps': self.execute_steps}
        # Server returns absolute joint targets in [left arm, left gripper,
        # right arm, right gripper] order. No second state addition or scaling.
        return unpack_robot_state(actions[:self.execute_steps].copy(), 'joint', self.dims)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list not in (None, [0]):
            raise NotImplementedError('Pi05_PiperX supports one real environment per session')
        return [self.get_action()]

    def reset(self):
        self.rtc_stop()
        self.observation, self.last_diagnostics = None, {}
        with self.lock:
            self.client.call(func_name='reset')

    # ---------- RTC (no server-side guidance; rebase only) ----------

    def _rebase(self, chunk, old, new):
        result = np.array(chunk, copy=True)
        cols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        scale = 2.0 / (self.q99[cols] - self.q01[cols] + 1e-6)
        delta = (np.asarray(old['state'])[cols] - np.asarray(new['state'])[cols]) * scale
        result[:, cols] += delta
        return result

    def _rtc_infer(self, request):
        # The vendored openpi build has no RTC guidance support; strip the
        # adapter-internal keys so the server never sees them.
        clean = {k: v for k, v in request.items() if not k.startswith('_rtc')}
        actions = self._call(clean)
        return {'action': actions, '_normalized_actions': self._normalize(actions)}

    def rtc_start(self, obs):
        self.rtc_stop()
        self.update_obs(obs)
        self.rtc = RealTimeChunkingController(self._rtc_infer, RTCConfig(
            minimum_execution_steps=int(self.cfg.get('rtc_start_steps', 5)),
            control_hz=float(self.cfg.get('control_hz', 10)),
            initial_delay_steps=int(self.cfg.get('rtc_initial_delay_steps', 22)),
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
