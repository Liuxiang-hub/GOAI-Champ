"""Replay recorded observations through official RPC; discard every action."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from client_server.ws import WsModelClient
from XPolicyLab.utils.process_data import decode_image_bit, unpack_robot_state, pack_robot_state, get_robot_action_dim_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='ws://127.0.0.1:6007')
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dims = get_robot_action_dim_info('piper_x')
    joints = [0,1,2,3,4,5,7,8,9,10,11,12]
    client = WsModelClient(url=args.url, evaluation_id='pi05-local-readonly-audit', trial_id='discard-all-actions', request_timeout_s=60, max_connect_attempts=2)
    rows = []
    try:
        with h5py.File(args.trajectory) as f:
            def state_at(frame):
                return np.concatenate([np.r_[f[arm+'/joint'][frame], f[arm+'/gripper'][frame]] for arm in ('left_arm','right_arm')]).astype(np.float32)
            for encoded in [False, True]:
                for frame in [0,30,60,94]:
                    state = state_at(frame)
                    target = state_at(min(frame+1,len(f['left_arm/joint'])-1))
                    obs = dict(instruction='stack the bowls', state=unpack_robot_state(state,'joint',dims), vision={}, env_idx=0)
                    for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
                        data = f[cam+'/color'][frame]
                        obs['vision'][cam] = {'color': data if encoded else decode_image_bit(data)}
                    client.call(func_name='reset')
                    started=time.perf_counter()
                    client.call(func_name='update_obs',obs=obs)
                    actions=client.call(func_name='get_action')
                    assert len(actions)==15
                    packed=np.stack([pack_robot_state({'state':x},'joint',dims) for x in actions])
                    assert packed.shape==(15,14) and np.isfinite(packed).all()
                    row=dict(frame=frame,encoded=encoded,shape=list(packed.shape),latency_ms=(time.perf_counter()-started)*1000,max_first_jump_rad=float(np.abs(packed[0,joints]-state[joints]).max()),max_nextstate_error_rad=float(np.abs(packed[0,joints]-target[joints]).max()),gripper_min=float(packed[:,[6,13]].min()),gripper_max=float(packed[:,[6,13]].max()))
                    rows.append(row)
                    print(json.dumps(row),flush=True)
    finally:
        client.close()
    args.output.write_text(json.dumps(dict(hardware_output=False,actions_discarded=True,target='next_state',rows=rows),indent=2))


if __name__=='__main__':
    main()
