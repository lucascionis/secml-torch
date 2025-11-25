import numpy as np
import pytest
import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.utils.security_evaluation_curve_utils import (
    compute_security_evaluation_curve,
)


@pytest.mark.parametrize(
    "distances, perturbation_model, expected_distances, expected_robust_acc",
    [
        (
            # shape (n_samples, n_steps) -> here n_steps=1
            torch.tensor([[0.1], [0.2], [0.1]]),
            LpPerturbationModels.L2,
            np.array([0.1, 0.2]),
            np.array([1 / 3, 0.0]),
        ),
        (
            torch.tensor([[0.0], [0.0], [1.0]]),
            LpPerturbationModels.L0,
            np.array([0.0, 1.0]),
            np.array([1 / 3, 0.0]),
        ),
    ],
)
def test_compute_robust_accuracy_curve(
    distances, perturbation_model, expected_distances, expected_robust_acc
):
    distances_unique, robust_acc = compute_security_evaluation_curve(
        distances, perturbation_model
    )
    np.testing.assert_array_almost_equal(distances_unique, expected_distances)
    np.testing.assert_array_almost_equal(robust_acc, expected_robust_acc)
