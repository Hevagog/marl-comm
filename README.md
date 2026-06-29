# marl-comm

A research framework for **communication in multi-agent reinforcement learning** (MARL),
built on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax) and
[skrl](https://skrl.readthedocs.io/). It implements and compares several agent families on
cooperative, coordination-heavy environments to study **when explicit inter-agent
communication helps, and how it should be structured**.

## Agents

All agents share a common PPO/MAPPO training core (CTDE) and a unified
train / record / analyze pipeline. They are selected per-config via `experiment.agent_type`.

| `agent_type`  | Architecture | Communication |
|---------------|--------------|---------------|
| `mappo`       | Multi-Agent PPO — strong non-communicating baseline (Yu et al., 2022) | none |
| `magic`       | Graph-attention message routing with a learned communication graph (Niu et al., 2021) | learned GAT |
| `commformer`  | Transformer with a learnable communication graph (Hu et al., ICLR 2024) | learned graph |
| `mam`         | Multi-Agent Mamba — bidirectional state-space encoders with selective-SSM communication (Daniel et al., 2024) | SSM |

MAGIC additionally supports a recurrent encoder slot, set via `magic.recurrent_type`:

- `lstm` / `gru` — classic recurrent memory
- `hopfield` (**EH**, episodic Hopfield over a buffer of recent observations)
- `hopfield_state` (**PH**, prototype Hopfield state cell with learned attractors)

## Environments

Selected per-config via `env.id`:

`warehouse` (multi-stage pick→treat→deliver grid with heterogeneous agents, batteries,
rendezvous, task deadlines and configurable fault/comm-loss profiles — the main testbed),
`coingame` / `coingame-partialobs`, `blindspot`, `continuous_coord`, `overcooked`,
`intersection` (highway), `flatland`, and `simple_adversary` (PettingZoo MPE).

## Installation

Requires Python ≥ 3.12 and a CUDA-capable GPU (JAX CUDA build). Uses
[uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

The entry point is `src/cli.py`, driven by a Python config file and a task:

```bash
# Train
PYTHONPATH=src python src/cli.py --task train  --config configs/magcomp/wh_s2_rendezvous_mappo.py

# Resume training from a checkpoint (timestep inferred from filename)
PYTHONPATH=src python src/cli.py --task train  --config <cfg> --resume runs/<exp>/checkpoints/agent_1000000.pickle

# Record a rollout video (add --record-comm for a split-screen comm-graph panel; MAGIC/CommFormer)
PYTHONPATH=src python src/cli.py --task record --config <cfg> --checkpoint <ckpt>

# Analyze behavior / communication; writes plots to --aout (default: eval_plots)
PYTHONPATH=src python src/cli.py --task analyze --config <cfg> --checkpoint <ckpt> --aout eval_plots
```


## Configuration

Configs are plain Python modules exposing a `CONFIG` dict. Key sections:

- `experiment` — `agent_type`, run name/`directory`, Weights & Biases settings, checkpoint intervals
- `env` — `id` plus environment-specific parameters (grid size, agents, rewards, faults, …)
- `training` — `timesteps`, `seed`
- `eval` / `record` — eval/recording timesteps and checkpoint paths
- `mappo` (and per-agent blocks like `magic`, `mam`) — optimizer, PPO clipping, schedulers, preprocessors
- `policy` / `value` — network hidden sizes
- `memory` — rollout buffer size

Defaults live in `configs/*_default.py`; experiment configs override them. The largest
suite is `configs/magcomp/` (warehouse scenarios S1–S7 across all agents).

## Project layout

```
src/
  cli.py             # entry point: env factory + task dispatch
  agents/            # mappo, magic, commformer, mam (each: model, train runner)
  environments/      # warehouse_grid, gridworld (coingame/blindspot), continuous_coord, ...
  utils/             # per-agent analysis/plotting, comm-graph rendering, stats
configs/             # experiment configs (Python modules with a CONFIG dict)
tests/               # pytest suite (architecture, training, env, paper-fidelity checks)
runs/                # training outputs + checkpoints
```

