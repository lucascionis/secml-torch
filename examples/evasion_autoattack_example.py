from loaders.get_loaders import get_mnist_loader
from models.mnist_net import get_mnist_model
from secmlt.adv.evasion.autoattack_attacks.autoattack_standard import (
    AutoAttackStandard,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.metrics.classification import Accuracy
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

device = "cpu"
model_path = "example_data/models/mnist"
dataset_path = "example_data/datasets/"
net = get_mnist_model(model_path).to(device)
test_loader = get_mnist_loader(dataset_path)

# Wrap model
model = BasePytorchClassifier(net)

# Test accuracy on original data
accuracy = Accuracy()(model, test_loader)
print(f"test accuracy: {accuracy.item():.2f}")

# Configure AutoAttack (standard suite)
perturbation_model = LpPerturbationModels.LINF
epsilon = 0.3

aa_attack = AutoAttackStandard(
    perturbation_model=perturbation_model, epsilon=epsilon, device=device, verbose=True
)

aa_adv_ds = aa_attack(model, test_loader)
aa_robust_accuracy = Accuracy()(model, aa_adv_ds)
print("robust accuracy AutoAttack (standard): ", aa_robust_accuracy)

# Inspect perturbation norms on the first batch
aa_data, _ = next(iter(aa_adv_ds))
real_data, _ = next(iter(test_loader))
p = LpPerturbationModels.get_p(perturbation_model)
aa_distances = (real_data - aa_data).flatten(start_dim=1).norm(p=p, dim=-1)
print("AutoAttack (standard) distances: ", aa_distances)
