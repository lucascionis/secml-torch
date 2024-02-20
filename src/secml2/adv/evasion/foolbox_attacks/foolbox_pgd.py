from typing import Optional
from secml2.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack
from secml2.adv.evasion.perturbation_models import PerturbationModels

from foolbox.attacks.projected_gradient_descent import (
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
)


class PGDFoolbox(BaseFoolboxEvasionAttack):
    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool,
        y_target: Optional[int] = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs
    ) -> None:
        perturbation_models = {
            PerturbationModels.L1: L1ProjectedGradientDescentAttack,
            PerturbationModels.L2: L2ProjectedGradientDescentAttack,
            PerturbationModels.LINF: LinfProjectedGradientDescentAttack,
        }
        foolbox_attack_cls = perturbation_models.get(perturbation_model, None)
        if foolbox_attack_cls is None:
            raise NotImplementedError(
                "This threat model is not implemented in foolbox."
            )

        foolbox_attack = foolbox_attack_cls(
            abs_stepsize=step_size, steps=num_steps, random_start=random_start
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
        )
