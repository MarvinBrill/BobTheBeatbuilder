import torch
from torch.nn import functional as F
from app.ml.phrase_losses import binary_loss


def test_directml_compatible_loss_matches_pytorch_values_and_gradients():
    logits = torch.tensor([-100.0, -3, 0, 2, 100], requires_grad=True)
    targets = torch.tensor([0.0, 1, 1, 0, 1])
    weight = torch.tensor(3.0)
    expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight, reduction="none")
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    actual = binary_loss(logits, targets, weight)
    actual_gradient = torch.autograd.grad(actual.sum(), logits)[0]
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6)
