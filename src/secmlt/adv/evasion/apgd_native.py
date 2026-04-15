"""Native Auto-PGD attack implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

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
from secmlt.optimization.gradient_processing import (
    LinearProjectionGradientProcessing,
    SparseL1GradientProcessing,
)
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
_SPARSITY_IMPROVEMENT_THRESHOLD = 0.95

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

    Returns a boolean mask (batch,) where True means the loss improved
    fewer than ``rho`` fraction of the time (i.e. oscillation detected).
    """
    if iteration < k:
        return torch.zeros(loss_history.shape[0], dtype=torch.bool)

    window = loss_history[:, iteration - k + 1 : iteration + 1]
    diffs = window[:, 1:] - window[:, :-1]
    n_improved = (diffs < 0).sum(dim=1).float()
    return n_improved <= k * rho


class APGDNative(ModularEvasionAttackFixedEps):
    """Native Auto-PGD attack with adaptive step-size scheduling."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: Union[float, torch.Tensor],
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
        epsilon : float | torch.Tensor
            Perturbation budget. Scalar or per-sample tensor.
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
        self._is_l1 = perturbation_model == LpPerturbationModels.L1

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

        # Gradient processing: sparse topk for L1, standard for others
        # For L1, a placeholder is created here; batch-sized topk is set in _run_loop
        if self._is_l1:
            gradient_processing = SparseL1GradientProcessing(
                topk=torch.tensor([0.2]), n_fts=1
            )
        else:
            gradient_processing = LinearProjectionGradientProcessing(
                perturbation_model
            )

        perturbation_constraints = [
            perturbation_models[perturbation_model](radius=epsilon),
        ]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )

        # Initial step size: epsilon for L1, 2*epsilon for Linf/L2
        if self._is_l1:
            step_size = epsilon if isinstance(epsilon, float) else epsilon.max().item()
        else:
            step_size = (
                2.0 * epsilon
                if isinstance(epsilon, float)
                else 2.0 * epsilon.max().item()
            )

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

    def _init_loop_state(
        self, samples: torch.Tensor, delta: torch.Tensor
    ) -> dict:
        """Initialize all loop state variables."""
        batch_size = samples.shape[0]
        device = samples.device

        eps_t = (
            torch.as_tensor(self.epsilon, dtype=samples.dtype, device=device)
            if not isinstance(self.epsilon, torch.Tensor)
            else self.epsilon.to(device)
        )
        alpha = 1.0 if self._is_l1 else 2.0

        if self._is_l1:
            k = max(int(0.04 * self.num_steps), 1)
            k_min = max(int(0.02 * self.num_steps), 1)
            size_decr = max(int(0.01 * self.num_steps), 1)
            n_fts = samples[0].numel()
            topk = torch.full((batch_size,), 0.2)
            self.gradient_processing = SparseL1GradientProcessing(
                topk=topk, n_fts=n_fts
            )
            sp_old = torch.full((batch_size,), float(n_fts))
        else:
            k = max(int(0.22 * self.num_steps), 1)
            k_min = max(int(0.06 * self.num_steps), 1)
            size_decr = max(int(0.03 * self.num_steps), 1)
            n_fts = 0
            sp_old = None

        step_size = torch.full(
            (batch_size,), self.step_size, dtype=samples.dtype, device=device
        )
        x_adv, delta = self.manipulation_function(samples, delta)

        return {
            "eps_t": eps_t,
            "alpha": alpha,
            "k": k,
            "k_min": k_min,
            "size_decr": size_decr,
            "n_fts": n_fts,
            "sp_old": sp_old,
            "step_size": step_size,
            "best_losses": torch.full((batch_size,), torch.inf),
            "best_delta": torch.zeros_like(samples),
            "loss_history": torch.full((batch_size, self.num_steps), torch.inf),
            "loss_best_at_checkpoint": torch.full((batch_size,), torch.inf),
            "delta_prev": delta.detach().clone(),
            "grad_before_processing": torch.zeros_like(delta),
            "x_adv": x_adv,
            "delta": delta,
        }

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
        ndim = len(samples.shape)
        device = samples.device
        state = self._init_loop_state(samples, delta)

        step_size = state["step_size"]
        best_losses = state["best_losses"]
        best_delta = state["best_delta"]
        loss_history = state["loss_history"]
        loss_best_at_checkpoint = state["loss_best_at_checkpoint"]
        delta_prev = state["delta_prev"]
        grad_before_processing = state["grad_before_processing"]
        x_adv = state["x_adv"]
        delta = state["delta"]
        k = state["k"]
        k_min = state["k_min"]
        size_decr = state["size_decr"]

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

            # Process gradient
            grad_processed = self.gradient_processing(delta.grad.data)

            # Momentum parameter
            a = 0.75 if i > 0 else 1.0

            # Gradient step
            grad_step = step_size.view(-1, *([1] * (ndim - 1))) * grad_processed
            delta_after_step = delta.data - grad_step  # minus because we minimize

            # Project after gradient step
            _, delta_after_step = self.manipulation_function(
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
                if self._is_l1:
                    self._l1_checkpoint(
                        delta=delta,
                        delta_prev=delta_prev,
                        best_delta=best_delta,
                        samples=samples,
                        step_size=step_size,
                        sp_old=state["sp_old"],
                        alpha=state["alpha"],
                        eps_t=state["eps_t"],
                        n_fts=state["n_fts"],
                        ndim=ndim,
                    )
                else:
                    self._linf_l2_checkpoint(
                        delta=delta,
                        delta_prev=delta_prev,
                        best_delta=best_delta,
                        step_size=step_size,
                        loss_history=loss_history,
                        best_losses=best_losses,
                        loss_best_at_checkpoint=loss_best_at_checkpoint,
                        i=i,
                        k=k,
                        ndim=ndim,
                        device=device,
                    )

                loss_best_at_checkpoint = best_losses.clone()
                checkpoint_counter = 0
                k = max(k - size_decr, k_min)

        # Return best adversarial
        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta

    def _l1_checkpoint(
        self,
        delta: torch.Tensor,
        delta_prev: torch.Tensor,
        best_delta: torch.Tensor,
        samples: torch.Tensor,
        step_size: torch.Tensor,
        sp_old: torch.Tensor,
        alpha: float,
        eps_t: torch.Tensor,
        n_fts: int,
        ndim: int,
    ) -> None:
        """L1-specific checkpoint: adapt step size based on sparsity."""
        device = samples.device
        sp_curr = best_delta.to(device).flatten(1).norm(p=0, dim=1)
        fl_redtopk = (
            sp_curr / sp_old.to(device).clamp(min=1)
        ) < _SPARSITY_IMPROVEMENT_THRESHOLD

        # Update topk in gradient processing
        self.gradient_processing.topk = sp_curr.cpu() / n_fts / 1.5

        # Reset step size if sparsity improving, else reduce by 1.5x
        alpha_eps = alpha * eps_t
        step_size.copy_(
            torch.where(fl_redtopk, alpha_eps.expand_as(step_size), step_size / 1.5)
        )
        step_size.clamp_(
            (alpha_eps / 10.0).expand_as(step_size),
            alpha_eps.expand_as(step_size),
        )
        sp_old.copy_(sp_curr.cpu())

        # Reset to best delta for samples where sparsity improved
        reduce_mask = atleast_kd(fl_redtopk, ndim)
        delta.data = torch.where(
            reduce_mask, best_delta.data.to(device), delta.data
        )
        delta_prev.data = delta.data.clone()

    def _linf_l2_checkpoint(
        self,
        delta: torch.Tensor,
        delta_prev: torch.Tensor,
        best_delta: torch.Tensor,
        step_size: torch.Tensor,
        loss_history: torch.Tensor,
        best_losses: torch.Tensor,
        loss_best_at_checkpoint: torch.Tensor,
        i: int,
        k: int,
        ndim: int,
        device: torch.device,
    ) -> None:
        """Linf/L2 checkpoint: adapt step size based on oscillation."""
        oscillating = _check_oscillation(loss_history, i, k, self.rho)
        no_progress = loss_best_at_checkpoint <= best_losses
        should_reduce = (oscillating | no_progress).to(device)

        step_size.copy_(
            torch.where(should_reduce, step_size / 2.0, step_size)
        )

        reduce_mask = atleast_kd(should_reduce, ndim)
        delta.data = torch.where(
            reduce_mask, best_delta.data.to(device), delta.data
        )
        delta_prev.data = delta.data.clone()

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
