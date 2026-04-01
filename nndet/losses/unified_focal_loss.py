"""
Pytorch adaptation of original Unified Focal Loss (https://github.com/mlyg/unified-focal-loss/)
The implementation follows nnDetection's naming convention (PascalCase) and logic (loss and wrapper).

The original definition assumed the output of the model to have already non-linearities applied,
this implementation allows one to directly specify if output are logits and which non-linearity to apply.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Callable
from nndet.losses.base import reduction_helper
from nndet.utils import make_onehot_batch


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation
def identify_axis(shape):
    """
    Identifies the spatial axes for summation based on the input shape.
    Assumes shape is (N, C, Spatial...).
    For 4D (N, C, H, W), returns [2, 3].
    For 5D (N, C, D, H, W), returns [2, 3, 4].
    """
    return list(range(2, len(shape)))


def prepare_input(
    pred: torch.Tensor, target: torch.Tensor, nonlin: Optional[Callable] = None
):
    """
    Prepares input for loss calculation:
    1. Applies nonlinearity (sigmoid/softmax) if provided
    2. Converts target to one-hot if necessary
    """
    if nonlin is not None:
        pred = nonlin(pred)

    # Check if target needs one-hot encoding
    # Condition: target has fewer dimensions than pred OR target has 1 channel where pred has > 1
    if pred.ndim != target.ndim:
        # target is likely (N, Spatial...), pred is (N, C, Spatial...)
        target = make_onehot_batch(target, n_classes=pred.shape[1]).float()
    elif pred.shape[1] != target.shape[1] and target.shape[1] == 1:
        # target is (N, 1, Spatial...), pred is (N, C, Spatial...)
        target = target.squeeze(1)
        target = make_onehot_batch(target, n_classes=pred.shape[1]).float()

    return pred, target


################################
#           Dice loss          #
################################
def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.5,
    smooth: float = 1e-6,
    reduction: str = "mean",
):
    """
    Dice loss
    """
    axis = identify_axis(pred.shape)
    tp = torch.sum(target * pred, dim=axis)
    fn = torch.sum(target * (1 - pred), dim=axis)
    fp = torch.sum((1 - target) * pred, dim=axis)

    dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    loss = 1 - dice_class
    return reduction_helper(loss, reduction)


class DiceLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * dice_loss(
            pred, target, self.delta, self.smooth, self.reduction
        )


################################
#         Tversky loss         #
################################
def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.7,
    smooth: float = 1e-6,
    reduction: str = "mean",
):
    """
    Tversky loss
    """
    axis = identify_axis(pred.shape)
    tp = torch.sum(target * pred, dim=axis)
    fn = torch.sum(target * (1 - pred), dim=axis)
    fp = torch.sum((1 - target) * pred, dim=axis)

    tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    loss = 1 - tversky_class
    return reduction_helper(loss, reduction)


class TverskyLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        smooth: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * tversky_loss(
            pred, target, self.delta, self.smooth, self.reduction
        )


################################
#       Dice coefficient       #
################################
def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.5,
    smooth: float = 1e-6,
    reduction: str = "mean",
):
    """
    Dice coefficient
    """
    axis = identify_axis(pred.shape)
    tp = torch.sum(target * pred, dim=axis)
    fn = torch.sum(target * (1 - pred), dim=axis)
    fp = torch.sum((1 - target) * pred, dim=axis)

    dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    return reduction_helper(dice_class, reduction)


class DiceCoefficient(nn.Module):
    def __init__(
        self,
        delta: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * dice_coefficient(
            pred, target, self.delta, self.smooth, self.reduction
        )


################################
#          Combo loss          #
################################
def combo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
    reduction: str = "mean",
):
    """
    Combo Loss
    """
    # Dice component
    dice = dice_coefficient(pred, target, delta=0.5, smooth=smooth, reduction="mean")

    # Cross entropy component
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    cross_entropy = -target * torch.log(pred)

    if beta is not None:
        # beta_weight = np.array([beta, 1-beta])
        beta_weight = torch.tensor(
            [beta, 1 - beta], device=pred.device, dtype=pred.dtype
        )
        # Reshape for broadcasting: (1, 2, 1, 1...)
        shape = [1] * pred.ndim
        shape[1] = 2
        beta_weight = beta_weight.view(*shape)

        if pred.shape[1] == 2:
            cross_entropy = beta_weight * cross_entropy

    # sum over classes (axis=[-1] in Keras -> dim=1 in PyTorch)
    cross_entropy = torch.mean(torch.sum(cross_entropy, dim=1))

    if alpha is not None:
        loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
    else:
        loss = cross_entropy - dice

    return loss


class ComboLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * combo_loss(
            pred, target, self.alpha, self.beta, self.smooth, self.reduction
        )


################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1e-6,
    reduction: str = "mean",
):
    """
    Focal Tversky loss
    """
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

    axis = identify_axis(pred.shape)
    tp = torch.sum(target * pred, dim=axis)
    fn = torch.sum(target * (1 - pred), dim=axis)
    fp = torch.sum((1 - target) * pred, dim=axis)

    tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    loss = torch.pow((1 - tversky_class), gamma)

    return reduction_helper(loss, reduction)


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * focal_tversky_loss(
            pred, target, self.delta, self.gamma, self.smooth, self.reduction
        )


################################
#          Focal loss          #
################################
def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[Union[float, List[float]]] = None,
    gamma_f: float = 2.0,
    reduction: str = "mean",
):
    """
    Focal loss
    """
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    cross_entropy = -target * torch.log(pred)

    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_tensor = torch.tensor(alpha, device=pred.device, dtype=pred.dtype)
        else:
            alpha_tensor = torch.tensor(alpha, device=pred.device, dtype=pred.dtype)

        if alpha_tensor.ndim > 0:
            shape = [1] * pred.ndim
            shape[1] = alpha_tensor.shape[0]
            alpha_tensor = alpha_tensor.view(*shape)

        loss = alpha_tensor * torch.pow(1 - pred, gamma_f) * cross_entropy
    else:
        loss = torch.pow(1 - pred, gamma_f) * cross_entropy

    loss = torch.sum(loss, dim=1)
    return reduction_helper(loss, reduction)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Union[float, List[float]]] = None,
        gamma_f: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma_f = gamma_f
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * focal_loss(
            pred, target, self.alpha, self.gamma_f, self.reduction
        )


################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.7,
    gamma: float = 2.0,
    reduction: str = "mean",
):
    """
    Symmetric Focal loss (Binary)
    """
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    cross_entropy = -target * torch.log(pred)

    # calculate losses separately for each class
    # Assuming binary: channel 0 is background, channel 1 is foreground
    back_ce = torch.pow(1 - pred[:, 0, ...], gamma) * cross_entropy[:, 0, ...]
    back_ce = (1 - delta) * back_ce

    fore_ce = torch.pow(1 - pred[:, 1, ...], gamma) * cross_entropy[:, 1, ...]
    fore_ce = delta * fore_ce

    loss = back_ce + fore_ce

    return reduction_helper(loss, reduction)


class SymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * symmetric_focal_loss(
            pred, target, self.delta, self.gamma, self.reduction
        )


#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.7,
    gamma: float = 0.75,
    reduction: str = "mean",
):
    """
    Symmetric Focal Tversky loss (Binary)
    """
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

    axis = identify_axis(pred.shape)
    tp = torch.sum(target * pred, dim=axis)
    fn = torch.sum(target * (1 - pred), dim=axis)
    fp = torch.sum((1 - target) * pred, dim=axis)

    dice_class = (tp + epsilon) / (tp + delta * fn + (1 - delta) * fp + epsilon)

    back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -gamma)
    fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -gamma)

    loss = torch.stack([back_dice, fore_dice], dim=1)

    print(f"Symmetric Focal Tversky loss: {loss}")

    return reduction_helper(loss, reduction)


class SymmetricFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * symmetric_focal_tversky_loss(
            pred, target, self.delta, self.gamma, self.reduction
        )


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.7,
    gamma: float = 2.0,
    reduction: str = "mean",
):
    """
    Asymmetric Focal loss (Binary)
    """
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    cross_entropy = -target * torch.log(pred)

    back_ce = torch.pow(1 - pred[:, 0, ...], gamma) * cross_entropy[:, 0, ...]
    back_ce = (1 - delta) * back_ce

    fore_ce = cross_entropy[:, 1, ...]
    fore_ce = delta * fore_ce

    loss = back_ce + fore_ce

    return reduction_helper(loss, reduction)


class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * asymmetric_focal_loss(
            pred, target, self.delta, self.gamma, self.reduction
        )


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.7,
    gamma: float = 0.75,
    reduction: str = "mean",
):
    """
    Asymmetric Focal Tversky loss (Binary)
    """
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

    axis = identify_axis(pred.shape)
    tp = torch.sum(target * pred, dim=axis)
    fn = torch.sum(target * (1 - pred), dim=axis)
    fp = torch.sum((1 - target) * pred, dim=axis)

    dice_class = (tp + epsilon) / (tp + delta * fn + (1 - delta) * fp + epsilon)

    back_dice = 1 - dice_class[:, 0]
    fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -gamma)

    loss = torch.stack([back_dice, fore_dice], dim=1)

    return reduction_helper(loss, reduction)


class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * asymmetric_focal_tversky_loss(
            pred, target, self.delta, self.gamma, self.reduction
        )


###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 0.5,
    delta: float = 0.6,
    gamma: float = 0.5,
    reduction: str = "mean",
):
    """
    Symmetric Unified Focal loss
    """
    symmetric_ftl = symmetric_focal_tversky_loss(
        pred, target, delta=delta, gamma=gamma, reduction=reduction
    )
    symmetric_fl = symmetric_focal_loss(
        pred, target, delta=delta, gamma=gamma, reduction=reduction
    )

    if weight is not None:
        return (weight * symmetric_ftl) + ((1 - weight) * symmetric_fl)
    else:
        return symmetric_ftl + symmetric_fl


class SymUnifiedFocalLoss(nn.Module):
    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * sym_unified_focal_loss(
            pred, target, self.weight, self.delta, self.gamma, self.reduction
        )


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 0.5,
    delta: float = 0.6,
    gamma: float = 0.5,
    reduction: str = "mean",
):
    """
    Asymmetric Unified Focal loss
    """
    asymmetric_ftl = asymmetric_focal_tversky_loss(
        pred, target, delta=delta, gamma=gamma, reduction=reduction
    )
    asymmetric_fl = asymmetric_focal_loss(
        pred, target, delta=delta, gamma=gamma, reduction=reduction
    )

    if weight is not None:
        return (weight * asymmetric_ftl) + ((1 - weight) * asymmetric_fl)
    else:
        return asymmetric_ftl + asymmetric_fl


class AsymUnifiedFocalLoss(nn.Module):
    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        nonlin: Optional[Callable] = None,
    ):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.nonlin = nonlin

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = prepare_input(pred, target, self.nonlin)
        return self.loss_weight * asym_unified_focal_loss(
            pred, target, self.weight, self.delta, self.gamma, self.reduction
        )
