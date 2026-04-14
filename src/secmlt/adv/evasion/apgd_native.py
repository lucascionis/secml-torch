"""Native Auto-PGD attack implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from secmlt.adv.evasion.losses import DLRLoss, TargetedDLRLoss
from secmlt.adv.evasion.modular_attacks.modular_attack_fixed_eps import (
    ModularEvasionAttackFixedEps,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    ClipConstraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer, RandomLpInitializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory
from secmlt.utils.tensor_utils import atleast_kd

if TYPE_CHECKING:
    from secmlt.models.base_model import BaseModel
    from secmlt.trackers.trackers import Tracker
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

CE_LOSS = "ce"
DLR_LOSS = "dlr"

_LOSS_FUNCTIONS = {
    CE_LOSS: lambda _targeted: torch.nn.CrossEntropyLoss(reduction="none"),
    DLR_LOSS: lambda targeted: TargetedDLRLoss() if targeted else DLRLoss(),
}


def _check_oscillation(
    loss_history: torch.Tensor,
    iteration: int,
    k: int,
    rho: float,
) -> torch.Tensor:
    """Check per-sample oscillation over the last k iterations.

    Returns a boolean mask (batch,) where True means oscillation is detected
    (loss improved fewer than rho fraction of the time).
    """
    if iteration < k:
        return torch.zeros(loss_history.shape[0], dtype=torch.bool)

    window = loss_history[:, iteration - k + 1 : iteration + 1]
    # Count how many times loss improved (decreased) step-to-step
    diffs = window[:, 1:] - window[:, :-1]
    # For a minimization attack, loss going down is good.
    # But APGD tracks the *attack* loss which we maximize (multiplier flips sign).
    # In the loop, losses are already multiplied. A "good" step means loss went down
    # (since multiplier makes it so lower = better for the attacker).
    # Oscillation = too many steps where loss didn't improve.
    n_improved = (diffs < 0).sum(dim=1).float()
    return n_improved <= k * rho


class APGDNative(ModularEvasionAttackFixedEps):
    """Native Auto-PGD attack with adaptive step-size scheduling."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        rho: float = 0.75,
        loss: str = CE_LOSS,
        random_start: bool = True,
        y_target: int | None = None,
        n_restarts: int = 1,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> None:
        """Create native Auto-PGD attack.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model (l1, l2, linf).
        epsilon : float
            Perturbation budget.
        num_steps : int
            Number of attack iterations.
        rho : float, optional
            Oscillation threshold for step-size adaptation. Default 0.75.
        loss : str, optional
            Loss function: 'ce' or 'dlr'. Default 'ce'.
        random_start : bool, optional
            Whether to use random initialization. Default True.
        y_target : int | None, optional
            Target label for targeted attack. None for untargeted.
        n_restarts : int, optional
            Number of random restarts. Default 1.
        lb : float, optional
            Lower bound of input domain. Default 0.0.
        ub : float, optional
            Upper bound of input domain. Default 1.0.
        trackers : list[Tracker] | None, optional
            Trackers for monitoring attack metrics.
        """
        self.rho = rho
        self.n_restarts = n_restarts
        self.epsilon = epsilon
        self.perturbation_model = perturbation_model

        perturbation_models = {
            LpPerturbationModels.L1: L1Constraint,
            LpPerturbationModels.L2: L2Constraint,
            LpPerturbationModels.LINF: LInfConstraint,
        }

        if random_start:
            initializer = RandomLpInitializer(
                perturbation_model=perturbation_model,
                radius=epsilon,
            )
        else:
            initializer = Initializer()

        targeted = y_target is not None
        if loss not in _LOSS_FUNCTIONS:
            msg = f"Loss '{loss}' not supported. Use 'ce' or 'dlr'."
            raise ValueError(msg)
        loss_fn = _LOSS_FUNCTIONS[loss](targeted)

        gradient_processing = LinearProjectionGradientProcessing(perturbation_model)
        perturbation_constraints = [
            perturbation_models[perturbation_model](radius=epsilon),
        ]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )

        # Initial step size: 2*epsilon for Linf/L2, epsilon for L1
        if perturbation_model == LpPerturbationModels.L1:
            step_size = epsilon
        else:
            step_size = 2.0 * epsilon

        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=loss_fn,
            optimizer_cls=OptimizerFactory.create_sgd(step_size),
            scheduler_cls=LRSchedulerFactory.create_no_scheduler(),
            manipulation_function=manipulation_function,
            gradient_processing=gradient_processing,
            initializer=initializer,
            trackers=trackers,
        )

    def _run_loop(
        self,
        model: BaseModel,
        delta: torch.Tensor,
        samples: torch.Tensor,
        target: torch.Tensor,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        multiplier: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = samples.shape[0]
        ndim = len(samples.shape)

        # Checkpoint schedule
        k = max(int(0.22 * self.num_steps), 1)
        k_min = max(int(0.06 * self.num_steps), 1)
        size_decr = max(int(0.03 * self.num_steps), 1)

        # Per-sample step sizes
        step_size = torch.full(
            (batch_size,), self.step_size, dtype=samples.dtype, device=samples.device
        )

        # State tracking
        best_losses = torch.full((batch_size,), torch.inf)
        best_delta = torch.zeros_like(samples)
        loss_history = torch.full((batch_size, self.num_steps), torch.inf)
        loss_best_at_checkpoint = torch.full((batch_size,), torch.inf)

        # Momentum state
        delta_prev = delta.detach().clone()
        grad_before_processing = torch.zeros_like(delta)

        # Initial projection
        x_adv, delta = self.manipulation_function(samples, delta)

        checkpoint_counter = 0

        for i in range(self.num_steps):
            # Project
            x_adv.data, delta.data = self.manipulation_function(
                samples.data, delta.data
            )
            delta_before_processing = delta.detach().clone()

            # Compute loss and gradient
            optimizer.zero_grad()
            scores, losses = self._loss_and_grad(
                model=model,
                samples=samples,
                delta=delta,
                target=target,
                multiplier=multiplier,
            )

            # Track best (losses are multiplied, so lower = better for attacker)
            improved = losses.detach().cpu() < best_losses
            best_delta.data = torch.where(
                atleast_kd(improved, ndim),
                delta_before_processing.detach().cpu().data,
                best_delta.data,
            )
            best_losses.data = torch.where(
                improved, losses.detach().cpu(), best_losses.data
            )
            loss_history[:, i] = losses.detach().cpu()

            # Fire trackers
            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(
                        i,
                        losses.detach().cpu().data,
                        scores.detach().cpu().data,
                        x_adv.detach().cpu().data,
                        delta.detach().cpu().data,
                        grad_before_processing.detach().cpu().data,
                    )

            # Save raw gradient before processing
            grad_before_processing = delta.grad.data.clone()

            # Process gradient (sign for Linf, normalize for L2, etc.)
            grad_processed = self.gradient_processing(delta.grad.data)

            # Momentum update: new_delta = delta + a*(delta - delta_prev) + step*grad
            # For APGD, we do the gradient step then apply momentum
            a = 0.75 if i > 0 else 1.0

            # Gradient step
            grad_step = step_size.view(-1, *([1] * (ndim - 1))) * grad_processed
            delta_after_step = delta.data - grad_step  # minus because we minimize

            # Project after gradient step
            x_adv_temp, delta_after_step = self.manipulation_function(
                samples.data, delta_after_step
            )

            # Apply momentum: blend current step with previous direction
            delta_new = (
                delta_after_step.data
                + a * (delta_after_step.data - delta.data)
                + (1.0 - a) * (delta.data - delta_prev.data)
            )

            # Save previous delta before updating
            delta_prev.data = delta.data.clone()

            # Update delta
            delta.data = delta_new.data

            # Project final result
            x_adv.data, delta.data = self.manipulation_function(
                samples.data, delta.data
            )

            checkpoint_counter += 1

            # Checkpoint: adaptive step-size reduction
            if checkpoint_counter == k and i > 0:
                oscillating = _check_oscillation(loss_history, i, k, self.rho)
                no_progress = loss_best_at_checkpoint <= best_losses
                should_reduce = (oscillating | no_progress).to(samples.device)

                # Halve step size for samples that need it
                step_size = torch.where(
                    should_reduce, step_size / 2.0, step_size
                )

                # Reset to best delta for samples that need reduction
                reduce_mask = atleast_kd(should_reduce, ndim)
                delta.data = torch.where(
                    reduce_mask, best_delta.data.to(delta.device), delta.data
                )
                delta_prev.data = delta.data.clone()

                loss_best_at_checkpoint = best_losses.clone()
                checkpoint_counter = 0
                k = max(k - size_decr, k_min)

        # Return best adversarial
        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run APGD with optional restarts."""
        if self.n_restarts <= 1:
            return super()._run(model, samples, labels, init_deltas)

        # Multiple restarts: track best across restarts
        best_x_adv = samples.clone()
        best_delta = torch.zeros_like(samples)
        best_losses = torch.full((samples.shape[0],), torch.inf)

        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)

        for _restart in range(self.n_restarts):
            delta = self.initializer(samples.data)
            delta.requires_grad = True

            optimizer = self._create_optimizer(delta, **self.optim_kwargs)
            scheduler = self._create_scheduler(optimizer, **self.scheduler_kwargs)

            x_adv, restart_delta = self._run_loop(
                model, delta, samples, target, optimizer, scheduler, multiplier
            )

            # Evaluate final losses for this restart
            with torch.no_grad():
                x_adv_eval, _ = self.manipulation_function(
                    samples.data, restart_delta.data
                )
                scores, losses = self.forward_loss(model, x_adv_eval, target)
                losses = losses * multiplier

            improved = losses.detach().cpu() < best_losses
            ndim = len(samples.shape)
            best_x_adv.data = torch.where(
                atleast_kd(improved, ndim),
                x_adv.detach().cpu().data,
                best_x_adv.data,
            )
            best_delta.data = torch.where(
                atleast_kd(improved, ndim),
                restart_delta.detach().cpu().data,
                best_delta.data,
            )
            best_losses.data = torch.where(
                improved, losses.detach().cpu(), best_losses.data
            )

        return best_x_adv, best_delta

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        """Return supported perturbation models."""
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }
