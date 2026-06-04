# fmt: off
"""Shared MAM config builder for MAGComp scenarios.

MAM follows MAT's joint sequence policy structure, but replaces attention with
Mamba variants: BiMamba in the encoder, causal Mamba in the decoder, and
CrossMamba for observation-action conditioning. These scenario configs keep the
environment and reward shaping identical to the corresponding MAPPO baselines.
"""
from __future__ import annotations

import copy

import jax.numpy as jnp


def make_mam_config(
    base_config: dict,
    *,
    name: str,
    tags: list[str],
    d_state: int,
    delta_rank: int,
    learning_rate: float = 1.5e-4,
    lr_decay_start_fraction: float = 0.05,
    entropy_start: float = 0.05,
    entropy_end: float = 0.01,
    kl_threshold: float = 0.03,
    kl_warmup_fraction: float = 0.0,
    ratio_max_threshold: float = 3.0,
    n_embd: int = 128,
    n_block: int = 1,
    learning_epochs: int | None = None,
    mini_batches: int | None = None,
    rollouts: int | None = None,
    value_hidden_sizes: list[int] | None = None,
    reward_clip: tuple[float, float] | None = None,
    sort_agents_by_type: bool = False,
    type_cycle_len: int = 3,
) -> dict:
    """Convert a MAGComp MAPPO config into a MAM config."""
    cfg = copy.deepcopy(base_config)

    experiment = cfg["experiment"]
    experiment["name"] = name
    experiment["agent_type"] = "mam"
    experiment["wandb_kwargs"] = {
        **experiment.get("wandb_kwargs", {}),
        "tags": tags,
    }

    mam = cfg.pop("mappo")
    mam.update(
        {
            # Conservative MAM PPO settings used by the repo's warehouse and
            # continuous-coord configs after the NaN/ratio-instability fixes.
            "learning_rate": learning_rate,
            "lr_decay_start_fraction": lr_decay_start_fraction,
            "update_state_preprocessor_in_update": False,
            "update_shared_state_preprocessor_in_update": False,
            "entropy_loss_scale": 0.01,
            "entropy_loss_scale_start": entropy_start,
            "entropy_loss_scale_end": entropy_end,
            "value_loss_scale": 1.0,
            "kl_threshold": kl_threshold,
            "kl_warmup_fraction": kl_warmup_fraction,
            "ratio_max_threshold": ratio_max_threshold,
            # MAM architecture: one Mamba block and d_conv=4 match Daniel et
            # al. (2024) defaults; d_state/delta_rank scale with agent count.
            "n_embd": n_embd,
            "n_block": n_block,
            "d_state": d_state,
            "d_conv": 4,
            "delta_rank": delta_rank,
            "sort_agents_by_type": sort_agents_by_type,
            "type_cycle_len": type_cycle_len,
        }
    )
    if learning_epochs is not None:
        mam["learning_epochs"] = learning_epochs
    if mini_batches is not None:
        mam["mini_batches"] = mini_batches
    if rollouts is not None:
        mam["rollouts"] = rollouts
        cfg["memory"]["size"] = rollouts
    if reward_clip is not None:
        lo, hi = reward_clip
        mam["rewards_shaper"] = lambda rewards, *_: jnp.clip(rewards, lo, hi)
    cfg["mam"] = mam

    cfg["policy"] = {"unnormalized_log_prob": True}
    if value_hidden_sizes is not None:
        cfg["value"]["hidden_sizes"] = value_hidden_sizes
    return cfg
