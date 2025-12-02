"""AutoPGD attack implemented via the AutoAttack backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from autoattack.autopgd_base import APGDAttack

from .autoattack_base import BaseAutoAttack

if TYPE_CHECKING:
    from secmlt.models.base_model import BaseModel
    from secmlt.trackers.trackers import Tracker


class APGDAutoAttack(BaseAutoAttack):
    """AutoPGD wrapper that delegates to the AutoAttack implementation."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        n_restarts: int,
        loss: str,
        seed: int,
        eot_iter: int,
        rho: float,
        topk: float | None,
        verbose: bool,
        use_largereps: bool,
        y_target: int | None = None,
        trackers: list[Tracker] | None = None,
        device: torch.device | None = None,
        **kwargs,
    ) -> None:
        """Initialize the AutoAttack AutoPGD wrapper."""
        super().__init__(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            trackers=trackers,
            device=device,
        )
        self.num_steps = num_steps
        self.n_restarts = n_restarts
        self.loss = loss
        self.y_target = y_target
        if self.y_target is not None:
            msg = "Targeted AutoPGD not supported via AutoAttack backend."
            raise ValueError(msg)
        self.seed = seed
        self.eot_iter = eot_iter
        self.rho = rho
        self.topk = topk
        self.verbose = verbose
        self.use_largereps = use_largereps

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._validate_model(model)

        attack_loss = self.loss
        attack_device = (
            str(self.device) if isinstance(self.device, torch.device) else self.device
        )
        attack = APGDAttack(
            predict=model,
            n_iter=self.num_steps,
            norm=self._autoattack_norm(),
            n_restarts=self.n_restarts,
            eps=self.epsilon,
            seed=self.seed,
            loss=attack_loss,
            eot_iter=self.eot_iter,
            rho=self.rho,
            topk=self.topk,
            verbose=self.verbose,
            device=attack_device,
            use_largereps=self.use_largereps,
        )

        advx = attack.perturb(samples, labels).detach()

        delta = advx - samples
        return advx, delta
