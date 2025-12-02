"""AutoPGD attack creator that switches between backends."""

from __future__ import annotations

import importlib.util
from functools import partial
from typing import TYPE_CHECKING

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.modular_attacks.modular_attack_fixed_eps import (
    ModularEvasionAttackFixedEps,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    ClipConstraint,
    L2Constraint,
    LInfConstraint,
)
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer, RandomLpInitializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class APGD(BaseEvasionAttackCreator):
    """Creator for the Auto-PGD attack."""

    def __new__(
        cls,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float | None = None,
        n_restarts: int = 1,
        loss: str = "ce",
        seed: int = 0,
        eot_iter: int = 1,
        rho: float = 0.75,
        topk: float | None = None,
        verbose: bool = False,
        use_largereps: bool = False,
        lb: float = 0.0,
        ub: float = 1.0,
        random_start: bool = False,
        backend: str = Backends.AUTOATTACK,
        y_target: int | None = None,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """Create an Auto-PGD attack with the requested backend."""
        if backend == Backends.NATIVE:
            msg = (
                "Native AutoPGD implementation is not available. "
                "Use the AutoAttack or adv_lib backends instead."
            )
            raise NotImplementedError(msg)
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        implementation.check_perturbation_model_available(perturbation_model)
        if backend == Backends.AUTOATTACK and y_target is not None:
            msg = "Targeted AutoPGD is not supported through AutoAttack."
            raise ValueError(msg)
        if backend == Backends.NATIVE and n_restarts != 1:
            msg = "Native AutoPGD does not support restarts yet."
            raise NotImplementedError(msg)

        if backend == Backends.NATIVE:
            return implementation(
                perturbation_model=perturbation_model,
                epsilon=epsilon,
                num_steps=num_steps,
                step_size=step_size,
                rho=rho,
                loss=loss,
                random_start=random_start,
                lb=lb,
                ub=ub,
                y_target=y_target,
                trackers=trackers,
                **kwargs,
            )

        extra_kwargs = dict(kwargs)
        if backend == Backends.ADVLIB:
            extra_kwargs.setdefault("lb", lb)
            extra_kwargs.setdefault("ub", ub)
        return implementation(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            n_restarts=n_restarts,
            loss=loss,
            seed=seed,
            eot_iter=eot_iter,
            rho=rho,
            topk=topk,
            verbose=verbose,
            use_largereps=use_largereps,
            y_target=y_target,
            trackers=trackers,
            **extra_kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get implementations available for AutoPGD."""
        return [Backends.AUTOATTACK, Backends.ADVLIB, Backends.NATIVE]

    @classmethod
    def get_implementation(cls, backend: str) -> BaseEvasionAttack:
        """
        Get the Auto-PGD implementation for the requested backend.
        """
        cls.check_backend_available(backend)
        implementations = {
            Backends.AUTOATTACK: cls._get_autoattack_implementation,
            Backends.ADVLIB: cls._get_advlib_implementation,
            Backends.NATIVE: cls._get_native_implementation,
        }
        return implementations[backend]()

    @staticmethod
    def _get_autoattack_implementation() -> type[BaseEvasionAttack]:
        if importlib.util.find_spec("autoattack.autopgd_base", None) is not None:
            from secmlt.adv.evasion.autoattack_attacks.autoattack_apgd import (
                APGDAutoAttack,
            )

            return APGDAutoAttack
        msg = "AutoAttack extra not installed."
        raise ImportError(msg)

    @staticmethod
    def _get_advlib_implementation() -> type[BaseEvasionAttack]:
        if importlib.util.find_spec("adv_lib", None) is not None:
            from secmlt.adv.evasion.advlib_attacks import APGDAdvLib

            return APGDAdvLib
        msg = "adv_lib extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_native_implementation() -> type[BaseEvasionAttack]:
        msg = "Native AutoPGD implementation is not available."
        raise NotImplementedError(msg)
