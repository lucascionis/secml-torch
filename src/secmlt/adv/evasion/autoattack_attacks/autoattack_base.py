"""Shared helpers for AutoAttack-based wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from secmlt.models.base_model import BaseModel
    from secmlt.trackers.trackers import Tracker


class BaseAutoAttack(BaseEvasionAttack):
    """Base class implementing common AutoAttack wrapper logic."""

    _NORM_MAPPING: ClassVar[dict[str, str]] = {
        LpPerturbationModels.LINF: "Linf",
        LpPerturbationModels.L2: "L2",
        LpPerturbationModels.L1: "L1",
    }

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        *,
        trackers: list[Tracker] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the shared AutoAttack wrapper state.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model requested for the attack.
        epsilon : float
            Radius of the perturbation constraint.
        trackers : list[Tracker] | None, optional
            Trackers (not supported by AutoAttack, for API compatibility).
        device : torch.device | str | None, optional
            Device hint for the attack. Samples are already moved by the
            base class; this is passed downstream when provided.
        """
        self._trackers = None
        if trackers is not None:
            self.trackers = trackers
        self.check_perturbation_model_available(perturbation_model)
        self.perturbation_model = perturbation_model
        self.epsilon = epsilon
        super().__init__(device=device)

    @classmethod
    def _trackers_allowed(cls) -> bool:
        return False

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        """Return the perturbation models supported by AutoAttack."""
        return set(cls._NORM_MAPPING)

    def _autoattack_norm(self) -> str:
        """Return the AutoAttack norm identifier for the configured model."""
        return self._NORM_MAPPING[self.perturbation_model]

    def _validate_model(self, model: BaseModel) -> BasePytorchClassifier:
        """Ensure the wrapped model can be consumed by AutoAttack."""
        if not isinstance(model, BasePytorchClassifier):
            msg = "Model type not supported."
            raise NotImplementedError(msg)
        return model
