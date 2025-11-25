"""Utility functions for computing and plotting robustness curves."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels

# Default rounding precision for different Lp norms
ROUNDING_DECIMALS = {
    LpPerturbationModels.L0: 0,
    LpPerturbationModels.L1: 1,
    LpPerturbationModels.L2: 3,
    LpPerturbationModels.LINF: -1,
}


def compute_security_evaluation_curve(
    distances: Union[torch.Tensor, np.ndarray],
    p: LpPerturbationModels,
    decimals: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the security evaluation curve from perturbation distances.

    Parameters
    ----------
    distances : (n_samples, n_steps) tensor/array
        Per-sample, per-iteration perturbation distances. Only the last column is used.
    p : LpPerturbationModels
        Perturbation norm type.

    Returns
    -------
    distances_unique : np.ndarray
        Sorted unique perturbation distances (including 0).
    robust_acc : np.ndarray
        Robust accuracy at each distance threshold.
    """
    # To numpy
    if isinstance(distances, torch.Tensor):
        distances = distances.detach().cpu().numpy()

    # Use final distance for each sample
    final_dists = distances[:, -1]

    # Rounding
    decimals = decimals if decimals is not None else ROUNDING_DECIMALS.get(p, 3)
    if decimals >= 0:
        final_dists = np.round(final_dists, decimals)

    # Robust accuracy curve from final distances
    distances_unique, counts = np.unique(final_dists, return_counts=True)
    robust_acc = 1.0 - counts.cumsum() / len(final_dists)

    return distances_unique, robust_acc
