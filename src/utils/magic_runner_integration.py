"""
magic_runner_integration.py
============================
Drop-in additions to BaseRunner for the MAGIC analysis mode.

Usage
-----
Add the following method to your MagicRunner class:

    def analyze_magic(
        self,
        checkpoint_path: str | None = None,
        n_episodes: int = 30,
        output_dir: str = "eval_plots/magic",
    ) -> None:
        from utils.magic_runner_integration import run_magic_analysis
        run_magic_analysis(self, checkpoint_path, n_episodes, output_dir)

Then call it from your script / CLI:

    runner.analyze_magic(
        checkpoint_path="runs/magic_blindspot_v2/checkpoints/agent_100000.pt",
        n_episodes=30,
    )

Alternatively, route through your existing BaseRunner.analyze() by adding an
"agent_type == magic" branch there that delegates to this module.

──────────────────────────────────────────────────────────────────────────────
What the MAGIC policy net MUST expose
──────────────────────────────────────────────────────────────────────────────
The collector reads communication internals from the `outputs` dict that
MAGICMAPPO.act() receives back from policy.act().  Add the lines below
to the RETURN statement of your MAGICPolicyNet.__call__ (or wherever
your network assembles the outputs dict):

    outputs["adj_matrices"]  = adj_soft      # (R, B, N, N) Gumbel-Softmax weights
    outputs["hard_adj"]      = hard_adj      # (B, N, N) binary mask, optional
    outputs["messages"]      = raw_msg       # (B, N, message_dim) pre-aggregation
    outputs["agg_messages"]  = agg_msg       # (B, N, message_dim) post-GAT

    # "net_output" (the policy logits) is already expected by CategoricalMAPPO
    # — just make sure it stays in the dict.
    outputs["net_output"]    = logits        # (B*N, num_actions)

If your implementation uses different tensor shapes, the collector's
`_extract_comm_outputs` helper will attempt to reshape them automatically
(it handles (R,B,N,N), (B,R,N,N), (R,N,N), and (N,N) for adjacency tensors).

Quick-add snippet for a typical MAGIC Flax/JAX implementation
-------------------------------------------------------------

    # Inside MAGICPolicyNet.__call__:

    # 1. Scheduler: GAT encoder + Gumbel-Softmax → soft directed graph
    scheduler_logits = self.scheduler(obs_enc)          # (B, N, N)  per round
    adj_soft = jax.nn.sigmoid(scheduler_logits / self.temperature)
    # Stack rounds: adj_stack shape = (R, B, N, N)
    adj_stack = jnp.stack([adj_r for adj_r in adj_rounds], axis=0)

    # 2. Message Processor: GAT over dynamic graph
    raw_msg  = self.msg_encoder(obs_enc)                # (B, N, D)
    agg_msg  = self.gat(raw_msg, adj_stack[-1])         # (B, N, D)

    # 3. Policy head
    policy_in = jnp.concatenate([obs_enc, agg_msg], axis=-1)
    logits     = self.policy_head(policy_in)             # (B*N, A)

    outputs = {
        "net_output":   logits,
        "adj_matrices": adj_stack,
        "hard_adj":     (adj_stack[-1] > 0.5).astype(jnp.float32),
        "messages":     raw_msg,
        "agg_messages": agg_msg,
    }
    return actions, log_prob, outputs

──────────────────────────────────────────────────────────────────────────────
Integrating with BaseRunner.analyze()
──────────────────────────────────────────────────────────────────────────────
In runners/base_runner.py, add a branch inside analyze():

    def analyze(self, checkpoint_path=None, n_episodes=30, output_dir="eval_plots"):
        ...
        agent_type = self._cfg.get("experiment", {}).get("agent_type", "mappo")
        if agent_type == "magic":
            from utils.magic_runner_integration import run_magic_analysis
            run_magic_analysis(self, checkpoint_path, n_episodes,
                               output_dir=f"{output_dir}/magic")
            return

        # existing non-MAGIC analysis path
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runners.base_runner import BaseRunner


def run_magic_analysis(
    runner: "BaseRunner",
    checkpoint_path: str | None = None,
    n_episodes: int = 30,
    output_dir: str = "eval_plots/magic",
) -> None:
    """
    Full MAGIC communication analysis pipeline.

    Parameters
    ----------
    runner          : a BaseRunner (or subclass) instance already holding
                      a constructed MAGIC agent and environment.
    checkpoint_path : path to a saved checkpoint; if None, uses the path from
                      runner._cfg["eval"]["checkpoint_path"].
    n_episodes      : number of evaluation episodes to collect.
    output_dir      : directory where figures (PDF + PNG) are written.
    """
    from utils.magic_comm_analysis import MAGICCommCollector
    from utils.magic_comm_visualizer import save_all_magic_figures

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    path = checkpoint_path or runner._cfg.get("eval", {}).get("checkpoint_path")
    if path:
        runner._load_checkpoint(path)

    # ── Pull configuration ───────────────────────────────────────────────────
    cfg = runner._cfg
    env_cfg = cfg.get("env", {})
    magic_cfg = cfg.get("magic", {})
    exp_name = cfg.get("experiment", {}).get("name", "magic_experiment")
    env_id = env_cfg.get("id", "coingame")

    if env_id == "simple_adversary":
        from utils.magic_sa_comm_analysis import MAGICSASCommCollector
        from utils.magic_sa_comm_visualizer import save_all_magic_sa_figures

        collector_sa = MAGICSASCommCollector(
            num_comm_rounds=magic_cfg.get("num_comm_rounds", 2),
            message_dim=magic_cfg.get("message_dim", 128),
        )

        print(f"\n[MAGIC SA Analysis] Collecting {n_episodes} episodes …")
        data_sa = collector_sa.collect(
            env=runner._env,
            agent=runner._agent,
            n_episodes=n_episodes,
        )

        save_all_magic_sa_figures(
            data_sa,
            output_dir=output_dir,
            prefix=exp_name,
        )
        return

    collector = MAGICCommCollector(
        grid_size=env_cfg.get("grid_size", 9),
        num_traps=env_cfg.get("num_traps", 5),
        max_cycles=env_cfg.get("max_cycles", 100),
        num_comm_rounds=magic_cfg.get("num_comm_rounds", 2),
        message_dim=magic_cfg.get("message_dim", 128),
    )

    print(f"\n[MAGIC Analysis] Collecting {n_episodes} episodes …")
    data = collector.collect(
        env=runner._env,
        agent=runner._agent,
        n_episodes=n_episodes,
    )

    save_all_magic_figures(
        data,
        output_dir=output_dir,
        prefix=exp_name,
    )
