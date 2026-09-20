# LingBot global_step_8884 archive

This directory preserves the Robot 6 compatibility bridge and service settings
for the LingBot-VLA 2.0 `global_step_8884` deployment.

It is a separate technology stack from the repository's current Pi0.5
`real-piper6-lora/7594` deployment. Nothing in this directory is loaded by
the Pi0.5 default configuration.

## Archived files

- `model_8884_bridge.py`: HRT adapter for the FINAL/LingBot service.
- `deploy.8884.yml`: Robot 6 adapter configuration for `global_step_8884`.
- `goai-lingbot-10-denoising.conf`: systemd override that starts
  `start_lingbot_vla_v2_rtc_server.sh` from Final_GOAI.

These files are retained only for historical reproducibility and must not be
copied into `robot_client/Pi05_PiperX/` or `l20_server/Pi_05/`.
