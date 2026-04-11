"""In-place curriculum scheduler for Flatland training.

The curriculum varies only environment knobs whose dimensionality does not
depend on ``num_agents`` or ``tree_depth``: grid size, city count, rail
density, and malfunction toggles.  Keeping ``num_agents`` fixed lets us
rebuild the vectorised Flatland env between stages *without* having to
rebuild or reinitialise the shared MAPPO critic (whose input dim depends on
``num_agents * (obs_dim + status_dim)``).

Changing ``num_agents`` mid-run would invalidate the value network's
first-layer shape, the running state/value scalers, and the autoregressive
decoder's expected sequence length.  If a team-size curriculum is needed,
run it as a sequence of separate training phases with explicit checkpoint
hand-offs — it cannot be done in a single ``trainer.train()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import FlatlandConfig


@dataclass(frozen=True)
class CurriculumStage:
    """Knobs for one stage of the in-place Flatland curriculum.

    Attributes
    ----------
    width, height:
        Rail grid dimensions passed to ``sparse_rail_generator``.
    max_num_cities:
        Upper bound on cities placed by the generator.  More cities mean
        longer, more branching paths and more conflict points.
    max_rails_between_cities, max_rail_pairs_in_city:
        Density of rails between / inside cities.  Lower = simpler layout.
    use_malfunctions:
        Whether to enable the stochastic malfunction generator.  Disable
        in early stages to remove a source of untraceable reward variance.
    completion_threshold:
        Exponential-moving-average completion ratio the policy must hit
        before advancing.  Set to > 1.0 to disable advancement (terminal
        stage).
    min_steps:
        Minimum rollout steps spent in this stage before advancement is
        even considered.  Protects against noisy early-epoch spikes.
    """

    width: int
    height: int
    max_num_cities: int
    max_rails_between_cities: int
    max_rail_pairs_in_city: int
    use_malfunctions: bool
    completion_threshold: float
    min_steps: int


# Default in-place stages — all keep ``num_agents`` fixed.  Difficulty grows
# along three axes: grid size, city count/rail density, and malfunctions.
FLATLAND_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        width=20,
        height=20,
        max_num_cities=2,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=1,
        use_malfunctions=False,
        completion_threshold=0.50,
        min_steps=2_000,
    ),
    CurriculumStage(
        width=24,
        height=22,
        max_num_cities=3,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=2,
        use_malfunctions=False,
        completion_threshold=0.45,
        min_steps=3_000,
    ),
    CurriculumStage(
        width=28,
        height=24,
        max_num_cities=3,
        max_rails_between_cities=3,
        max_rail_pairs_in_city=2,
        use_malfunctions=True,
        completion_threshold=0.35,
        min_steps=4_000,
    ),
    # Terminal stage: full "hard" config.  ``completion_threshold > 1.0``
    # means ``maybe_advance`` never fires here.
    CurriculumStage(
        width=32,
        height=24,
        max_num_cities=4,
        max_rails_between_cities=3,
        max_rail_pairs_in_city=2,
        use_malfunctions=True,
        completion_threshold=1.01,
        min_steps=0,
    ),
)


class CurriculumScheduler:
    """Tracks which curriculum stage is active and decides when to advance.

    The runner polls :meth:`maybe_advance` every training segment with the
    most recent completion-ratio signal and the number of env steps the
    segment consumed.  When the EMA crosses the current stage's threshold
    *and* the minimum dwell time has elapsed, the scheduler advances to
    the next stage and returns ``True`` so the runner can rebuild the env.

    The rolling statistic is an EMA with a fixed momentum (0.9) — this
    smooths the per-segment noise without introducing a configurable knob
    that would only ever be tuned empirically anyway.
    """

    _EMA_MOMENTUM: float = 0.9

    def __init__(
        self,
        base_config: FlatlandConfig,
        stages: tuple[CurriculumStage, ...] = FLATLAND_CURRICULUM,
    ) -> None:
        if not stages:
            raise ValueError("CurriculumScheduler requires at least one stage")
        self._base = base_config
        self._stages = stages
        self._idx = 0
        self._steps_in_stage = 0
        self._ema_completion = 0.0

    @property
    def current(self) -> CurriculumStage:
        return self._stages[self._idx]

    @property
    def stage_index(self) -> int:
        return self._idx

    @property
    def num_stages(self) -> int:
        return len(self._stages)

    @property
    def ema_completion(self) -> float:
        return self._ema_completion

    def config_for(self, stage: CurriculumStage) -> FlatlandConfig:
        """Return a new ``FlatlandConfig`` with this stage's knobs applied.

        The base config's ``num_agents``, ``tree_depth``, and reward-shaping
        parameters are preserved — only the knobs explicitly listed on
        :class:`CurriculumStage` are overridden.
        """
        return replace(
            self._base,
            width=stage.width,
            height=stage.height,
            max_num_cities=stage.max_num_cities,
            max_rails_between_cities=stage.max_rails_between_cities,
            max_rail_pairs_in_city=stage.max_rail_pairs_in_city,
            use_malfunctions=stage.use_malfunctions,
        )

    def maybe_advance(self, completion_ratio: float, rollout_steps: int) -> bool:
        """Feed a new observation and possibly advance the active stage.

        Parameters
        ----------
        completion_ratio:
            Mean completion ratio measured over the latest training segment.
        rollout_steps:
            Number of environment steps in that segment.  Used for the
            ``min_steps`` dwell check.

        Returns
        -------
        bool
            ``True`` iff the active stage has changed and the caller should
            rebuild the env with :meth:`config_for` applied to
            :attr:`current`.
        """
        self._steps_in_stage += int(rollout_steps)
        self._ema_completion = self._EMA_MOMENTUM * self._ema_completion + (
            1.0 - self._EMA_MOMENTUM
        ) * float(completion_ratio)
        stage = self.current
        if (
            self._ema_completion >= stage.completion_threshold
            and self._steps_in_stage >= stage.min_steps
            and self._idx + 1 < len(self._stages)
        ):
            self._idx += 1
            self._steps_in_stage = 0
            # Carry a small seed of the previous EMA into the new stage so
            # we don't restart from a cold zero (which would delay the next
            # advancement check by ~10 segments at the EMA momentum of 0.9).
            self._ema_completion *= 0.5
            return True
        return False
