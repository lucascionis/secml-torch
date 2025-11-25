import matplotlib.pyplot as plt
import torchvision
from robustbench.utils import load_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.fmn import FMN
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.trackers.trackers import (
    BestPerturbationNormTracker,
    PerturbationNormTracker,
)
from secmlt.utils.security_evaluation_curve_utils import (
    compute_security_evaluation_curve,
)
from torch.utils.data import DataLoader, Subset

# Load pretrained robust model from RobustBench
net = load_model(
    model_name="Rony2019Decoupling", dataset="cifar10", threat_model="L2"
).to("cuda")
model = BasePytorchClassifier(net)

model.model.eval()

# Load CIFAR-10 test set (subset for demo)
test_dataset = torchvision.datasets.CIFAR10(
    root=".",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_dataset = Subset(test_dataset, list(range(50)))
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# Compute clean accuracy
clean_accuracy = Accuracy()(model, test_loader)
print(f"Clean Accuracy: {clean_accuracy.item():.2%}")

perturbation_norm = LpPerturbationModels.L2

# Configure attack
attack = FMN(
    perturbation_model=perturbation_norm,
    num_steps=100,
    step_size=0.3,
    y_target=None,
    backend=Backends.NATIVE,
    trackers=[
        PerturbationNormTracker(perturbation_norm),  # Final distances
        BestPerturbationNormTracker(perturbation_norm),  # Best distances
    ],
)

# Run attack
print("Running FMN attack...")
adv_loader = attack(model, test_loader)
print("Attack completed.")

# Compute robust accuracy
robust_accuracy = Accuracy()(model, adv_loader)
print(f"Robust Accuracy: {robust_accuracy.item():.2%}")

# Extract tracked distances
final_distances = attack.trackers[0].get()  # Final perturbation at last step
best_distances = attack.trackers[1].get()  # Best (smallest successful) perturbation

# Compute robustness curves
final_dists, final_acc = compute_security_evaluation_curve(
    final_distances, perturbation_norm
)
best_dists, best_acc = compute_security_evaluation_curve(
    best_distances, perturbation_norm
)

# Plot curves
plt.figure(figsize=(10, 6))
plt.plot(
    final_dists,
    final_acc,
    marker="o",
    markersize=3,
    label="Final Distances",
    linewidth=2,
    alpha=0.8,
)
plt.plot(
    best_dists,
    best_acc,
    marker="s",
    markersize=3,
    label="Best Distances",
    linewidth=2,
    linestyle="--",
    alpha=0.8,
)

plt.xlabel("Perturbation Distance", fontsize=12)
plt.ylabel("Robust Accuracy", fontsize=12)
plt.title("Security Evaluation Curve (FMN Attack)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(visible=True, alpha=0.3)
plt.ylim([0, 1])
plt.xlim(left=0)
plt.tight_layout()
plt.show()
