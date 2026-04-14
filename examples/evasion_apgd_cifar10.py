"""Test APGD wrappers on a RobustBench CIFAR-10 model (Addepalli2022Efficient_RN18)."""

import torch
import torchvision
from robustbench.utils import load_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.apgd import APGD
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from torch.utils.data import DataLoader, Subset

device = "cpu"

# Load RobustBench model (Addepalli et al. 2022, ResNet-18, Linf)
net = load_model(
    model_name="Addepalli2022Efficient_RN18",
    dataset="cifar10",
    threat_model="Linf",
).to(device)
net.eval()

# Load CIFAR-10 test set (small subset for demo)
test_dataset = torchvision.datasets.CIFAR10(
    root="example_data/datasets/",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_dataset = Subset(test_dataset, list(range(50)))
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

model = BasePyTorchClassifier(net)

# Clean accuracy
accuracy = Accuracy()(model, test_loader)
print(f"Clean accuracy: {accuracy.item():.2%}")

perturbation_model = LpPerturbationModels.LINF
epsilon = 8 / 255
num_steps = 100

# --- AutoAttack APGD ---
print("\nRunning AutoAttack APGD...")
aa_attack = APGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    backend=Backends.AUTOATTACK,
)
aa_adv_ds = aa_attack(model, test_loader)
aa_robust = Accuracy()(model, aa_adv_ds)
print(f"Robust accuracy (AutoAttack APGD): {aa_robust.item():.2%}")

# --- AdvLib APGD ---
print("\nRunning AdvLib APGD...")
advlib_attack = APGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    backend=Backends.ADVLIB,
)
advlib_adv_ds = advlib_attack(model, test_loader)
advlib_robust = Accuracy()(model, advlib_adv_ds)
print(f"Robust accuracy (AdvLib APGD): {advlib_robust.item():.2%}")

# --- Compare perturbation norms ---
original, _ = next(iter(test_loader))
aa_adv, _ = next(iter(aa_adv_ds))
advlib_adv, _ = next(iter(advlib_adv_ds))

p = float("inf")
aa_dists = (aa_adv - original).flatten(1).norm(p=p, dim=1)
advlib_dists = (advlib_adv - original).flatten(1).norm(p=p, dim=1)

print(f"\nAutoAttack  - max Linf dist: {aa_dists.max():.6f}, mean: {aa_dists.mean():.6f}")
print(f"AdvLib      - max Linf dist: {advlib_dists.max():.6f}, mean: {advlib_dists.mean():.6f}")
print(f"Epsilon budget: {epsilon:.6f}")
