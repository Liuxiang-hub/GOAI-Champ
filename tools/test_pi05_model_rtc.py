import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models.pi0 import rtc_guided_euler_step
from openpi.policies import aloha_policy
from openpi.shared import normalize
from openpi.training import config


def test_exact_transform(checkpoint, train_config_name, repo_id, horizon):
    train_config = config.get_config(train_config_name)
    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )
    stats = normalize.load(Path(checkpoint) / "assets" / repo_id)
    transform = transforms.compose([
        aloha_policy.AlohaInputs(adapt_to_pi=False),
        transforms.DeltaActions(transforms.make_bool_mask(6, -1, 6, -1)),
        transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    state = np.linspace(-0.3, 0.3, 14, dtype=np.float32)
    physical = np.tile(state, (horizon, 1))
    physical[:, :6] += 0.04
    physical[:, 7:13] -= 0.03
    physical[:, [6, 13]] = [0.25, 0.75]
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    result = transform({
        "state": state.copy(),
        "actions": physical.copy(),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": "test",
    })
    assert result["state"].shape == (32,)
    assert result["actions"].shape == (horizon, 32)
    assert np.isfinite(result["actions"]).all()
    assert np.allclose(result["actions"][:, 14:], 0.0)

    delta = physical.copy()
    mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
    delta[:, mask] -= state[mask]
    q01 = stats["actions"].q01
    q99 = stats["actions"].q99
    expected = (delta - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    assert np.allclose(result["actions"][:, :14], expected, atol=1e-6)
    print("exact_transform: PASS", result["actions"].shape)


def test_guided_step(horizon, denoising_steps):
    x = jnp.ones((1, horizon, 32), dtype=jnp.float32)
    target = jnp.zeros_like(x)
    guided_prefix = min(8, horizon)
    weights = jnp.concatenate([
        jnp.ones((1, guided_prefix), dtype=jnp.float32),
        jnp.zeros((1, horizon - guided_prefix), dtype=jnp.float32),
    ], axis=1)

    def velocity(value):
        return value * 0.2

    plain = x
    guided = x
    time = 1.0
    dt = -1.0 / denoising_steps
    for _ in range(denoising_steps):
        plain = plain + dt * velocity(plain)
        guided = rtc_guided_euler_step(
            velocity, guided, time, dt, target, weights, jnp.asarray(5.0)
        )
        time += dt

    plain_prefix_error = float(jnp.mean(jnp.square(plain[:, :guided_prefix] - target[:, :guided_prefix])))
    guided_prefix_error = float(jnp.mean(jnp.square(guided[:, :guided_prefix] - target[:, :guided_prefix])))
    assert guided_prefix_error < plain_prefix_error
    assert np.isfinite(np.asarray(guided)).all()
    print("guided_step: PASS", plain_prefix_error, guided_prefix_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-config", default="pi05_base_piper6_lora_real"
    )
    parser.add_argument(
        "--repo-id", default="yangchenjie/robodojo_piper6_v3"
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--denoising-steps", type=int, default=10)
    args = parser.parse_args()
    if args.horizon <= 0 or args.denoising_steps <= 0:
        parser.error("--horizon and --denoising-steps must be positive")
    test_exact_transform(
        args.checkpoint, args.train_config, args.repo_id, args.horizon
    )
    test_guided_step(args.horizon, args.denoising_steps)
