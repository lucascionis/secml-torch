from loaders.get_loaders import get_mnist_loader
from models.mnist_net import get_mnist_model
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.apgd import APGD
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

num_steps = 100
perturbation_model = LpPerturbationModels.LINF
y_target = None
epsilon = 0.3

# AutoAttack backend (untargeted only)
aa_attack = APGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    backend=Backends.AUTOATTACK,
    device=device,
)
aa_adv_ds = aa_attack(model, test_loader)
aa_robust_accuracy = Accuracy()(model, aa_adv_ds)
print("robust accuracy AutoAttack APGD: ", aa_robust_accuracy)

# AdvLib backend
advlib_attack = APGD(
    perturbation_model=perturbation_model,
    epsilon=epsilon,
    num_steps=num_steps,
    backend=Backends.ADVLIB,
    device=device,
)
advlib_adv_ds = advlib_attack(model, test_loader)
advlib_robust_accuracy = Accuracy()(model, advlib_adv_ds)
print("robust accuracy AdvLib APGD: ", advlib_robust_accuracy)

# Compare perturbation magnitudes
aa_data, _ = next(iter(aa_adv_ds))
advlib_data, _ = next(iter(advlib_adv_ds))
original_data, _ = next(iter(test_loader))

p = LpPerturbationModels.get_p(perturbation_model)
aa_distances = (original_data - aa_data).flatten(start_dim=1).norm(p=p, dim=-1)
advlib_distances = (original_data - advlib_data).flatten(start_dim=1).norm(p=p, dim=-1)
print("AutoAttack distances: ", aa_distances)
print("AdvLib distances: ", advlib_distances)
