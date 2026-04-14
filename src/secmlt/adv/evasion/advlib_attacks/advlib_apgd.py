"""Wrapper of the Auto-PGD attack implemented in Adversarial Library."""

from __future__ import annotations  # noqa: I001

from functools import partial
from typing import TYPE_CHECKING

from adv_lib.attacks.auto_pgd import apgd
from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

if TYPE_CHECKING:
    from secmlt.trackers.trackers import Tracker


class APGDAdvLib(BaseAdvLibEvasionAttack):
    """Adversarial Library Auto-PGD wrapper."""

    EPSILON_KWARG = "eps"

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        n_restarts: int = 1,
        loss: str = "ce",
        seed: int = 0,
        eot_iter: int = 1,
        rho: float = 0.75,
        topk: float | None = None,
        verbose: bool = False,
        use_largereps: bool = False,
        y_target: int | None = None,
        trackers: list[Tracker] | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        best_loss: bool = False,
        use_rs: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize an Auto-PGD attack with the Adversarial Library backend.

        Parameters
        ----------
        perturbation_model : str
            Desired perturbation model.
        epsilon : float
            Attack radius.
        num_steps : int
            Number of attack iterations.
        n_restarts : int, optional
            Number of restarts. Defaults to 1.
        loss : str, optional
            Loss to optimize. Select from ``'ce'`` and ``'dlr'``.
        seed : int, optional
            Not used for the Adversarial Library backend.
        eot_iter : int, optional
            Expectation-over-transformation iterations.
        rho : float, optional
            Step-size reduction factor.
        topk : float | None, optional
            Not supported by the Adversarial Library backend.
        verbose : bool, optional
            Not supported by the Adversarial Library backend.
        use_largereps : bool, optional
            Whether to use the large-repetition schedule.
        y_target : int | None, optional
            Target label. If None, perform untargeted attack.
        trackers : list[Tracker] | None, optional
            Trackers are not supported by the Adversarial Library backend.
        lb : float, optional
            Lower bound for the input domain. Defaults to 0.0.
        ub : float, optional
            Upper bound for the input domain. Defaults to 1.0.
        best_loss : bool, optional
            Keep the perturbation with the best loss per restart.
        use_rs : bool, optional
            Random start flag for large repetitions.
        kwargs : dict
            Additional arguments forwarded to :func:`adv_lib.attacks.auto_pgd.apgd`.
        """
        if seed != 0:
            raise NotImplementedError(
                "Seeding is not implemented for Adversarial Library Auto-PGD."
            )
        if topk is not None:
            raise NotImplementedError(
                "The 'topk' argument is not supported by the AdvLib Auto-PGD backend."
            )
        if verbose:
            raise NotImplementedError(
                "Verbose mode is not available for the AdvLib Auto-PGD backend."
            )

        loss = loss.lower()
        supported_losses = {"ce", "dlr"}
        if loss not in supported_losses:
            msg = f"Unsupported loss '{loss}'. Available options: {sorted(supported_losses)}."
            raise ValueError(msg)

        use_large_reps = kwargs.pop("use_large_reps", use_largereps)
        best_loss = kwargs.pop("best_loss", best_loss)
        use_rs = kwargs.pop("use_rs", use_rs)

        norm_mapping = {
            LpPerturbationModels.L1: 1,
            LpPerturbationModels.L2: 2,
            LpPerturbationModels.LINF: float("inf"),
        }
        norm = norm_mapping.get(perturbation_model)
        if norm is None:
            msg = "Auto-PGD via Adversarial Library supports only L1, L2 and Linf."
            raise NotImplementedError(msg)

        advlib_attack = partial(
            apgd,
            n_iter=num_steps,
            n_restarts=n_restarts,
            loss_function=loss,
            eot_iter=eot_iter,
            rho=rho,
            use_large_reps=use_large_reps,
            use_rs=use_rs,
            best_loss=best_loss,
            norm=norm,
        )

        super().__init__(
            advlib_attack=advlib_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """Return perturbation models supported by the Adversarial Library Auto-PGD."""
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }
