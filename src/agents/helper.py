from typing import Any


def build_mappo_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate the JSON config into the dict expected by skrl's MAPPO."""
    m = cfg["mappo"]
    exp = cfg["experiment"]
    return {
        "rollouts": m["rollouts"],
        "learning_epochs": m["learning_epochs"],
        "mini_batches": m["mini_batches"],
        "discount_factor": m["discount_factor"],
        "lambda": m["lambda"],
        "learning_rate": m["learning_rate"],
        "learning_rate_scheduler": None,
        "learning_rate_scheduler_kwargs": m.get("learning_rate_scheduler_kwargs", {}),
        "state_preprocessor": None,
        "state_preprocessor_kwargs": m.get("state_preprocessor_kwargs", {}),
        "shared_state_preprocessor": None,
        "shared_state_preprocessor_kwargs": m.get(
            "shared_state_preprocessor_kwargs", {}
        ),
        "value_preprocessor": None,
        "value_preprocessor_kwargs": m.get("value_preprocessor_kwargs", {}),
        "random_timesteps": m["random_timesteps"],
        "learning_starts": m["learning_starts"],
        "grad_norm_clip": m["grad_norm_clip"],
        "ratio_clip": m["ratio_clip"],
        "value_clip": m["value_clip"],
        "clip_predicted_values": m["clip_predicted_values"],
        "entropy_loss_scale": m["entropy_loss_scale"],
        "value_loss_scale": m["value_loss_scale"],
        "kl_threshold": m["kl_threshold"],
        "rewards_shaper": None,
        "time_limit_bootstrap": m["time_limit_bootstrap"],
        "experiment": {
            "directory": exp["directory"],
            "experiment_name": exp["name"],
            "write_interval": exp["write_interval"],
            "checkpoint_interval": exp["checkpoint_interval"],
            "store_separately": exp["store_separately"],
            "wandb": exp["wandb"],
            "wandb_kwargs": exp.get("wandb_kwargs", {}),
        },
    }
