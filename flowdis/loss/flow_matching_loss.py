import torch
from torch import Tensor
import torch.nn.functional as F

def flow_matching_loss(
    pred: Tensor,
    target: Tensor
):
    """
    Flow matching loss for the velocity field.
    Args:
        pred: The predicted velocity field
        target: The target velocity field

    Returns:
        The flow matching loss
    """
    return F.mse_loss(pred, target)