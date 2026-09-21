# HRT1 adapter on L20

HRT1 is the deployment adapter and competition policy name for the current
Pi0.5 route; it is not a separate foundation model and does not use the
archived LingBot/8884 stack.

This directory records the optimized finals-v1 policy route:

```text
field eval-runner
  -> public port 6000
  -> hrt1-l20-public socket proxy
  -> 127.0.0.1:16010 (HRT1 adapter)
  -> 127.0.0.1:6199 (Pi0.5 10k inference server)
```

Install the Python package under `XPolicyLab/policy/HRT1`, place
`deploy.hrt1-l20.yml` in the L20 configuration directory, and install the
systemd units from `systemd/`. Paths in the supplied units match the recorded
L20 layout and must be reviewed on another host.

The `hrt1-field-public` units preserve the rollback route from public port 6000
to the field reverse tunnel on localhost port 16000. Only one public socket may
own port 6000. Switch routes only after confirming that no official trial or
HDF5/MP4 recording is active.

No credentials, model weights, recordings, or private network keys are stored
in this directory.
