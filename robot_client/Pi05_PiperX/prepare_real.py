"""Read-only live preflight; --arm enables the gate but sends no robot commands."""
import argparse
import json
from pathlib import Path
import sys
from client_server.ws import WsModelClient
from XPolicyLab.utils.process_data import unpack_robot_state, get_robot_action_dim_info
from XPolicyLab.policy.Pi05_PiperX.safety import CommandLimiter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm', action='store_true')
    p.add_argument('--disarm', action='store_true')
    args=p.parse_args()
    path=Path(__file__).with_name('motion_gate.json')
    cfg=json.loads(path.read_text())
    cfg['hardware_output_enabled']=False
    path.write_text(json.dumps(cfg,indent=2)+'\n')
    if args.disarm:
        print('DISARMED. No new episode-loop commands permitted. This is not a hardware emergency stop.')
        return
    sys.path.insert(0,'/home/user/GOAI/goai_client_console')
    from pi05_live_observation_preflight import real_state_14, read_camera
    state, arms=real_state_14()
    obs=dict(instruction='stack the bowls',state=unpack_robot_state(state,'joint',get_robot_action_dim_info('piper_x')),vision={})
    for name,role in [('cam_head','head'),('cam_left_wrist','left_wrist'),('cam_right_wrist','right_wrist')]:
        obs['vision'][name]={'color':read_camera(role)}
    client=WsModelClient(url='ws://127.0.0.1:6007',evaluation_id='pi05-live-readonly-preflight',trial_id='discard',request_timeout_s=60,max_connect_attempts=2)
    try:
        client.call(func_name='reset')
        client.call(func_name='rtc_start',obs=obs)
        raw=client.call(func_name='rtc_next_action')
        print(json.dumps(dict(state=state.tolist(),arm_warnings={k:v['warnings'] for k,v in arms.items()},raw_action={k:list(v) for k,v in raw.items()},actions_discarded=True),default=float), flush=True)
        command=CommandLimiter(cfg,obs).command(raw,obs)
        print(json.dumps(dict(metadata=client.call(func_name='status')['metadata'],arm_warnings={k:v['warnings'] for k,v in arms.items()},raw_action={k:list(v) for k,v in raw.items()},limited_command={k:list(v) for k,v in command.items()},actions_discarded=True),default=float))
    finally:
        client.call(func_name='rtc_stop')
        client.close()
    if args.arm:
        cfg['hardware_output_enabled']=True
        path.write_text(json.dumps(cfg,indent=2)+'\n')
        print('ARMED. Start a valid official trial in Collector to execute. This command did not move the robot.')
    else:
        print('LIVE_FIRST_ACTION_OK. Gate remains disabled; full task motion has not been tested.')


if __name__=='__main__': main()
