from typing import Optional, List

import torch

from src.adv.evasion.composite_attack import CompositeEvasionAttack, CE_LOSS, SGD, AdditiveManipulation, \
    GradientNormalizerProcessing, Initializer
from src.adv.evasion.foolbox import BaseFoolboxEvasionAttack

from src.adv.evasion.perturbation_models import PerturbationModels
from src.adv.backends import Backends
from src.adv.evasion.base_evasion_attack import (
    BaseEvasionAttackCreator,
)

from foolbox.attacks.projected_gradient_descent import (
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
)

from src.optimization.constraints import ClipConstraint, L1Constraint, L2Constraint, LInfConstraint, Constraint


class PGD(BaseEvasionAttackCreator):
    def __new__(
            cls,
            perturbation_model: str,
            epsilon: float,
            num_steps: int,
            step_size: float,
            random_start: bool,
            y_target: Optional[int] = None,
            lb: float = 0.0,
            ub: float = 1.0,
            backend: str = Backends.FOOLBOX,
            **kwargs
    ):
        cls.check_perturbation_model_available(perturbation_model)
        implementation = cls.get_implementation(backend)
        return implementation(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=random_start,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs
        )

    @staticmethod
    def get_foolbox_implementation():
        return PGDFoolbox

    @staticmethod
    def get_native_implementation():
        return PGDNative


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


class PGDNative(CompositeEvasionAttack):
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
            PerturbationModels.L1: L1Constraint,
            PerturbationModels.L2: L2Constraint,
            PerturbationModels.LINF: LInfConstraint
        }
        initializer = Initializer() if random_start else Initializer()
        # TODO add random init with different LP norms
        self.epsilon = epsilon
        super().__init__(y_target=y_target, num_steps=num_steps, step_size=step_size, loss_function=CE_LOSS,
                         optimizer_cls=SGD, manipulation_function=AdditiveManipulation(),
                         domain_constraints=[ClipConstraint(lb=lb, ub=ub)],
                         perturbation_constraints=[perturbation_models[perturbation_model]],
                         gradient_processing=GradientNormalizerProcessing(perturbation_model),
                         initializer=initializer)

    def init_perturbation_constraints(self, center: torch.Tensor) -> List[Constraint]:
        return [p(center, self.epsilon) for p in self.perturbation_constraints]
