import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from src.adv.backends import Backends
from src.adv.evasion.pgd import PGD
from src.adv.evasion.perturbation_models import PerturbationModels

from src.metrics.classification import Accuracy
from src.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from robustbench.utils import load_model

net = load_model(model_name="Rony2019Decoupling", dataset="cifar10", threat_model="L2")
device = "mps"
net.to(device)
test_dataset = torchvision.datasets.CIFAR10(
    transform=torchvision.transforms.ToTensor(), train=False, root=".", download=True
)
test_dataset = Subset(test_dataset, list(range(5)))
test_data_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_data_loader)
print(accuracy)

# Create and run attack
epsilon = 0.5
num_steps = 50
step_size = 0.05
perturbation_model = PerturbationModels.LINF
y_target = None
native_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=y_target,
    backend=Backends.NATIVE,
)
native_adv_ds = native_attack(model, test_data_loader)

# Test accuracy on adversarial examples
n_robust_accuracy = Accuracy()(model, native_adv_ds)
print(n_robust_accuracy)

# Create and run attack
foolbox_attack = PGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    step_size=step_size,
    random_start=False,
    y_target=None,
    backend=Backends.FOOLBOX,
)
f_adv_ds = foolbox_attack(model, test_data_loader)

# Test accuracy on adversarial examples
f_robust_accuracy = Accuracy()(model, f_adv_ds)
print(f_robust_accuracy)

native_data, native_labels = next(iter(native_adv_ds))
f_data, f_labels = next(iter(f_adv_ds))

distance = torch.linalg.norm(native_data.flatten(start_dim=1).to(device) - f_data.flatten(start_dim=1), ord=2, dim=1)
print(distance)
