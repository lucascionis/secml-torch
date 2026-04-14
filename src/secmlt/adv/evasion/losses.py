"""Loss functions for adversarial attacks."""

from __future__ import annotations

import torch


class DLRLoss(torch.nn.Module):
    """Difference of Logits Ratio loss (untargeted).

    Computes -(z_y - z_max') / (z_1 - z_3) where z_y is the true-class logit,
    z_max' is the largest logit excluding the true class, and z_1, z_3 are the
    first and third largest logits.
    """

    def __init__(self, reduction: str = "none") -> None:
        """Initialize loss with reduction mode."""
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute DLR loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits of shape (batch, num_classes).
        labels : torch.Tensor
            True labels of shape (batch,).

        Returns
        -------
        torch.Tensor
            Per-sample DLR loss of shape (batch,).
        """
        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Best logit excluding the true class
        best_other = torch.where(
            sorted_logits[:, 0] == true_logits,
            sorted_logits[:, 1],
            sorted_logits[:, 0],
        )

        denom = sorted_logits[:, 0] - sorted_logits[:, 2] + 1e-12
        loss = -(true_logits - best_other) / denom

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TargetedDLRLoss(torch.nn.Module):
    """Difference of Logits Ratio loss (targeted).

    Computes -(z_target - z_max') / (z_1 - z_avg34) where z_target is the
    target-class logit, z_max' is the largest logit excluding the target,
    and z_avg34 is the average of the 3rd and 4th largest logits.
    """

    def __init__(self, reduction: str = "none") -> None:
        """Initialize loss with reduction mode."""
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute targeted DLR loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits of shape (batch, num_classes).
        labels : torch.Tensor
            Target labels of shape (batch,).

        Returns
        -------
        torch.Tensor
            Per-sample targeted DLR loss of shape (batch,).
        """
        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Best logit excluding the target class
        best_other = torch.where(
            sorted_logits[:, 0] == target_logits,
            sorted_logits[:, 1],
            sorted_logits[:, 0],
        )

        denom = sorted_logits[:, 0] - (
            sorted_logits[:, 2] + sorted_logits[:, 3]
        ) / 2.0 + 1e-12
        loss = -(target_logits - best_other) / denom

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
