"""Exercise the official episode loop and real GPU with recorded cameras only."""
import argparse
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
from client_server.ws import WsModelClient
from XPolicyLab.utils.process_data import decode_image_bit, unpack_robot_state, get_robot_action_dim_info
from XPolicyLab.policy.Pi05_PiperX.deploy import eval_one_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=120)
    args = p.parse_args()
    os.environ['EVAL_ENV_TYPE'] = 'debug'
    rows = []
    for encoded in (False, True):
        with h5py.File(args.trajectory) as f:
            state = np.concatenate([np.r_[f[arm+'/joint'][0], f[arm+'/gripper'][0]] for arm in ('left_arm','right_arm')]).astype(np.float32)
            observation = dict(instruction='stack the bowls', state=unpack_robot_state(state,'joint',get_robot_action_dim_info('piper_x')), vision={})
            for cam in ('cam_head','cam_left_wrist','cam_right_wrist'):
                value = f[cam+'/color'][0]
                observation['vision'][cam] = {'color': value if encoded else decode_image_bit(value)}
        client = WsModelClient(url='ws://127.0.0.1:6007', evaluation_id='pi05-rtc-readonly-audit', trial_id='discard', request_timeout_s=60, max_connect_attempts=2)
        trace=[]
        class RecordingClient:
            def call(self, **kwargs):
                result = client.call(**kwargs)
                if kwargs['func_name'] == 'rtc_commit': trace.append(result)
                return result
        class RecordedEnv:
            count = 0
            def get_obs(self): return observation
            def is_episode_end(self): return self.count >= args.steps
            def take_action(self, action):
                assert all(np.isfinite(v).all() for v in action.values())
                self.count += 1  # Discard, never instantiate a hardware backend.
        started=time.monotonic()
        try:
            eval_one_episode(RecordedEnv(), RecordingClient())
            assert any(x['inflight'] for x in trace), 'No concurrent inference observed'
            assert max(x['generation'] for x in trace) >= 2, 'Fewer than two completed switches'
            row=dict(encoded=encoded, steps=args.steps, elapsed_s=time.monotonic()-started, trace=trace, metadata=client.call(func_name='status')['metadata'])
            rows.append(row)
            print('PASS', encoded, args.steps, 'generations', max(x['generation'] for x in trace), flush=True)
        finally:
            client.close()
    Path(args.output).write_text(json.dumps(dict(hardware_output=False, actions_discarded=True, rows=rows), indent=2))


if __name__ == '__main__': main()
