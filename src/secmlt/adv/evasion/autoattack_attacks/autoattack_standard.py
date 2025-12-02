"""Wrapper exposing the full AutoAttack standard suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autoattack import AutoAttack

from .autoattack_base import BaseAutoAttack

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from secmlt.models.base_model import BaseModel
    from secmlt.trackers.trackers import Tracker


class AutoAttackStandard(BaseAutoAttack):
    """Run the complete AutoAttack pipeline (standard version)."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        *,
        version: str = "standard",
        attacks_to_run: Sequence[str] | None = None,
        seed: int = 0,
        verbose: bool = False,
        log_path: str | None = None,
        trackers: list[Tracker] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Configure the AutoAttack standard pipeline wrapper.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model requested for the attack.
        epsilon : float
            Radius of the perturbation constraint.
        version : str, optional
            AutoAttack version to run. Defaults to "standard".
        attacks_to_run : Sequence[str] | None, optional
            Optional subset of attacks to run. If None, runs the full suite.
        seed : int, optional
            Random seed for AutoAttack.
        verbose : bool, optional
            Whether to enable verbose AutoAttack output.
        log_path : str | None, optional
            Optional path for AutoAttack logging.
        trackers : list[Tracker] | None, optional
            Trackers (not supported by AutoAttack, for API compatibility).
        device : torch.device | None, optional
            Device that the attack should run on. Must match the wrapped
            model device if provided.
        """
        super().__init__(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            trackers=trackers,
            device=device,
        )
        self.version = version
        self.attacks_to_run = attacks_to_run
        self.seed = seed
        self.verbose = verbose
        self.log_path = log_path

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._validate_model(model)

        attack = AutoAttack(
            model=model,
            norm=self._autoattack_norm(),
            eps=self.epsilon,
            version=self.version,
            device=str(self.device),
            seed=self.seed,
            verbose=self.verbose,
            log_path=self.log_path,
        )

        if self.attacks_to_run is not None:
            attack.attacks_to_run = list(self.attacks_to_run)

        advx = attack.run_standard_evaluation(
            samples,
            labels,
            bs=samples.shape[0],
        ).detach()
        delta = advx - samples
        return advx, delta
