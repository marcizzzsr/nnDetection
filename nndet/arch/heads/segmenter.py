"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Union, Sequence, Optional, Tuple, TypeVar

from nndet.arch.conv import compute_padding_for_kernel, conv_kwargs_helper
from nndet.arch.heads.comb import AbstractHead
from nndet.arch.layers.interpolation import InterpolateToShapes
from nndet.losses.segmentation import SoftDiceLoss, TopKLoss
from nndet.losses.unified_focal_loss import (
    SymmetricFocalTverskyLoss,
    SymmetricFocalLoss,
    AsymmetricFocalTverskyLoss,
    AsymmetricFocalLoss
)


class Segmenter(AbstractHead):
    def __init__(
        self,
        seg_classes: int,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        **kwargs,
    ):
        """
        Abstract interface for segmentation head

        Args:
            seg_classes: number of foreground classes
                (!! internally +1 added for background)!!)
            in_channels: number of input channels at all decoder levels
            decoder_levels: decoder levels used for detection
        """
        super().__init__()
        self.seg_classes = seg_classes + 1
        self.in_channels = in_channels
        self.decoder_levels = decoder_levels


class DiCESegmenter(Segmenter):
    def __init__(
        self,
        conv,
        seg_classes: int,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        internal_channels: Optional[int] = None,
        num_internal: int = 0,
        add_norm: bool = True,
        add_act: bool = True,
        kernel_size: Union[int, Sequence[int]] = 3,
        alpha: float = 0.5,
        ce_kwargs: Optional[dict] = None,
        dice_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Basic Segmentation Head with dice and CE loss
        (num_internal x conv [kernel_size]) -> final conv [1x1]

        Args:
            conv: Convolution modules which handles a single layer
            seg_classes: number of foreground classes
                (!! internally +1 added for background)!!)
            in_channels: number of input channels at all decoder levels
            decoder_levels: decoder levels used for detection
            internal_channels: number of channels of internal convolutions
            num_internal: number of internal convolutions
            add_norm: add normalization layers to internal convolutions
            add_act: add activation layers to internal convolutions
            kernel_size: kernel size of conv
            alpha: weight dice and ce loss (alpha * ce + (1-alpha) * soft_dice)
            ce_kwargs: keyword arguments passed to CE loss
            dice_kwargs: keyword arguments passed to dice loss
        """
        super().__init__(
            seg_classes=seg_classes,
            in_channels=in_channels,
            decoder_levels=decoder_levels,
        )
        self.num_internal = num_internal

        if internal_channels is None:
            self.internal_channels = self.in_channels[0]
        else:
            self.internal_channels = internal_channels

        self.conv_out = self.build_conv_out(conv)
        self.conv_intermediate = self.build_conv_internal(
            conv,
            kernel_size=kernel_size,
            add_norm=add_norm,
            add_act=add_act,
            **kwargs,
        )

        if dice_kwargs is None:
            dice_kwargs = {}
        dice_kwargs.setdefault("smooth_nom", 1e-5)
        dice_kwargs.setdefault("smooth_denom", 1e-5)
        dice_kwargs.setdefault("do_bg", False)
        self.dice_loss = SoftDiceLoss(nonlin=torch.nn.Softmax(dim=1), **dice_kwargs)

        if ce_kwargs is None:
            ce_kwargs = {}
        self.ce_loss = torch.nn.CrossEntropyLoss(**ce_kwargs)

        self.logits_convert_fn = nn.Softmax(dim=1)
        self.alpha = alpha

    def build_conv_out(self, conv) -> nn.Module:
        """
        Build output convolution
        """
        _intermediate_channels = (
            self.internal_channels if self.num_internal > 0 else self.in_channels[0]
        )
        return conv(
            _intermediate_channels,
            self.seg_classes,
            kernel_size=1,
            padding=0,
            add_norm=None,
            add_act=None,
            bias=True,
        )

    def build_conv_internal(
        self,
        conv,
        kernel_size: Union[int, Tuple[int]],
        add_norm: bool,
        add_act: bool,
        **kwargs,
    ) -> Optional[nn.Module]:
        """
        Buld internal convolutions
        """
        padding = compute_padding_for_kernel(kernel_size)
        if self.num_internal > 0:
            _intermediate = torch.nn.Sequential()
            for i in range(self.num_internal):
                _intermediate.add_module(
                    f"c_intermediate{i}",
                    conv(
                        self.in_channels if i == 0 else self.internal_channels,
                        self.internal_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=1,
                        add_norm=add_norm,
                        add_act=add_act,
                        **kwargs,
                    ),
                )
        else:
            _intermediate = None
        return _intermediate

    def forward(
        self,
        x: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: all features produced by decoder. Largest to smallest.

        Returns:
            torch.Tensor: result
        """
        x = x[0]
        if self.conv_intermediate is not None:
            x = self.conv_intermediate(x)
        return {"seg_logits": self.conv_out(x)}

    def compute_loss(
        self,
        pred_seg: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted dice and cross entropy loss

        Args:
            pred_seg: segmentation predictions
                `seg_logits`: predicted logits
            target: ground truth segmentation of top layer

        Returns:
            Dict[str, torch.Tensor]: computed loss (contained in key seg)
        """
        seg_logits = pred_seg["seg_logits"]
        return {
            "seg_ce": self.alpha * self.ce_loss(seg_logits, target.long()),
            "seg_dice": (1 - self.alpha) * self.dice_loss(seg_logits, target),
        }

    def postprocess_for_inference(
        self,
        prediction: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. convert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
                `seg_logits`: predicted logits

        Returns:
            Dict[str, torch.Tensor]: postprocessed predictions
                `pred_seg`: predicted probabilities [N, C, dims]
        """
        return {"pred_seg": self.logits_convert_fn(prediction["seg_logits"])}


class DiCESegmenterFgBg(DiCESegmenter):
    def __init__(
        self,
        conv,
        seg_classes: int,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        internal_channels: Optional[int] = None,
        num_internal: int = 0,
        add_norm: bool = True,
        add_act: bool = True,
        kernel_size: Union[int, Sequence[int]] = 3,
        alpha: float = 0.5,
        **kwargs,
    ):
        """
        Basic Segmentation Head with dice and CE loss which only
        differentiates foreground and background
        (num_internal x conv [kernel_size]) -> final conv [1x1]

        Args:
            conv: Convolution modules which handles a single layer
            seg_classes: ignored!
            in_channels: number of input channels at all decoder levels
            decoder_levels: decoder levels used for detection
            internal_channels: number of channels of internal convolutions
            num_internal: number of internal convolutions
            add_norm: add normalization layers to internal convolutions
            add_act: add activation layers to internal convolutions
            kernel_size: kernel size of conv
            alpha: weight dice and ce loss (alpha * ce + (1-alpha) * soft_dice)
            ce_kwargs: keyword arguments passed to CE loss
            dice_kwargs: keyword arguments passed to dice loss

        Warnings:
            If this class is used, the reportet dice scores during training
            are wrong if multiple classes are present in the dataset.
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            seg_classes=1,
            decoder_levels=decoder_levels,
            internal_channels=internal_channels,
            num_internal=num_internal,
            add_norm=add_norm,
            add_act=add_act,
            kernel_size=kernel_size,
            alpha=alpha,
            **kwargs,
        )

    def compute_loss(
        self,
        pred_seg: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted dice and cross entropy loss

        Args:
            pred_seg: segmentation predictions
                `seg_logits`: predicted logits
            target: ground truth segmentation of top layer

        Returns:
            Dict[str, torch.Tensor]: computed loss (contained in key seg)
        """
        target[target > 0] = 1
        return super().compute_loss(pred_seg, target)


class DiceTopKSegmenter(DiCESegmenter):
    def __init__(
        self,
        conv,
        seg_classes: int,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        internal_channels: Optional[int] = None,
        num_internal: int = 0,
        add_norm: bool = True,
        add_act: bool = True,
        kernel_size: Union[int, Sequence[int]] = 3,
        alpha: float = 0.5,
        topk: float = 0.1,
        **kwargs,
    ):
        """
        Basic Segmentation Head with dice and TopK loss
        (num_internal x conv [kernel_size]) -> final conv [1x1]

        Args:
            conv: Convolution modules which handles a single layer
            seg_classes: number of foreground classes
                (!! internally +1 added for background)!!)
            in_channels: number of input channels at all decoder levels
            decoder_levels: decoder levels used for detection
            internal_channels: number of channels of internal convolutions
            num_internal: number of internal convolutions
            add_norm: add normalization layers to internal convolutions
            add_act: add activation layers to internal convolutions
            kernel_size: kernel size of conv
            alpha: weight dice and ce loss (alpha * ce + (1-alpha) * soft_dice)
            ce_kwargs: keyword arguments passed to CE loss
            topk: percentage of all entries to use for loss computation
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            seg_classes=seg_classes,
            decoder_levels=decoder_levels,
            internal_channels=internal_channels,
            num_internal=num_internal,
            add_norm=add_norm,
            add_act=add_act,
            kernel_size=kernel_size,
            alpha=alpha,
            ce_kwargs=None,
            **kwargs,
        )
        self.ce_loss = TopKLoss(topk=topk)


class DiceTopKSegmenterFgBg(DiCESegmenterFgBg):
    def __init__(
        self,
        conv,
        seg_classes: int,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        internal_channels: Optional[int] = None,
        num_internal: int = 0,
        add_norm: bool = True,
        add_act: bool = True,
        kernel_size: Union[int, Sequence[int]] = 3,
        alpha: float = 0.5,
        topk: float = 0.1,
        **kwargs,
    ):
        """
        Basic Segmentation Head with dice and CE loss which only
        differentiates foreground and background
        (num_internal x conv [kernel_size]) -> final conv [1x1]

        Args:
            conv: Convolution modules which handles a single layer
            seg_classes: ignored!
            in_channels: number of input channels at all decoder levels
            decoder_levels: decoder levels used for detection
            internal_channels: number of channels of internal convolutions
            num_internal: number of internal convolutions
            add_norm: add normalization layers to internal convolutions
            add_act: add activation layers to internal convolutions
            kernel_size: kernel size of conv
            alpha: weight dice and ce loss (alpha * ce + (1-alpha) * soft_dice)
            ce_kwargs: keyword arguments passed to CE loss
            topk: percentage of all entries to use for loss computation

        Warnings:
            If this class is used, the reportet dice scores during training
            are wrong if multiple classes are present in the dataset.
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            seg_classes=seg_classes,
            decoder_levels=decoder_levels,
            internal_channels=internal_channels,
            num_internal=num_internal,
            add_norm=add_norm,
            add_act=add_act,
            kernel_size=kernel_size,
            alpha=alpha,
            **kwargs,
        )
        self.ce_loss = TopKLoss(topk=topk)


class DeepSupervisionSegmenterFGBG(DiCESegmenterFgBg):
    def __init__(
        self,
        conv,
        seg_classes: int,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        internal_channels: Optional[int] = None,
        num_internal: int = 0,
        add_norm: bool = True,
        add_act: bool = True,
        kernel_size: Union[int, Sequence[int]] = 3,
        alpha: float = 0.5,
        dsv_weight: float = 1.0,
        **kwargs,
    ):
        """
        Deep supervision segmenation which trains with CE and Dice
        to differentitate foreground and background
        (num_internal x conv [kernel_size]) -> final conv [1x1]

        Args:
            conv: Convolution modules which handles a single layer
            seg_classes: ignored!
                (!! internally +1 added for background)!!)
            in_channels: number of input channels at all decoder levels
            decoder_levels: decoder levels used for detection
            internal_channels: number of channels of internal convolutions
            num_internal: number of internal convolutions
            add_norm: add normalization layers to internal convolutions
            add_act: add activation layers to internal convolutions
            kernel_size: kernel size of conv
            alpha: weight dice and ce loss (alpha * ce + (1-alpha) * soft_dice)
            ce_kwargs: keyword arguments passed to CE loss
            dice_kwargs: keyword arguments passed to dice loss
            dsv_weight: additional weight for dsv losses
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            seg_classes=1,
            decoder_levels=decoder_levels,
            internal_channels=internal_channels,
            num_internal=num_internal,
            add_norm=add_norm,
            add_act=add_act,
            kernel_size=kernel_size,
            alpha=alpha,
            **kwargs,
        )

        assert len(self.decoder_levels) > 0
        self.dsv_conv = conv(
            self.in_channels[-1],
            2,
            kernel_size=3,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )
        self.interpolator = InterpolateToShapes()
        self.dsv_weight = dsv_weight

    def forward(
        self,
        x: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: all features produced by decoder. Largest to smallest.

        Returns:
            torch.Tensor: result
        """
        predictions = {}
        if self.intermediate is not None:
            predictions["seg_logits"] = self.conv_out(self.conv_intermediate(x[0]))
        else:
            predictions["seg_logits"] = self.conv_out(x[0])

        for dl in self.decoder_levels:
            predictions[f"dsv_logits_{dl}"] = self.dsv_conv(x[dl])
        return predictions

    def compute_loss(
        self,
        pred_seg: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted dice and cross entropy loss

        Args:
            pred_seg: segmentation predictions
                `seg_logits`: predicted logits
            target: ground truth segmentation of top layer

        Returns:
            Dict[str, torch.Tensor]: computed loss (contained in key seg)
        """
        target[target > 0] = 1

        loss = self._compute_loss(pred_seg["seg_logits"], target)

        preds_decoder_level = [
            pred_seg[f"dsv_logits_{dl}"] for dl in self.decoder_levels
        ]
        targets_interpolated = self.interpolator(preds_decoder_level, target)

        for pred, target in zip(preds_decoder_level, targets_interpolated):
            loss = loss + self.dsv_weight * self._compute_loss(pred, target)

        return {"seg_loss": loss / (len(self.decoder_levels) + 1)}

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce_loss(pred, target.long()) + (
            1 - self.alpha
        ) * self.dice_loss(pred, target)


SegmenterType = TypeVar("SegmenterType", bound=Segmenter)


class SymmetricUFLSegmenter(DiCESegmenter):
    def __init__(
        self,
        conv,
        seg_classes,
        in_channels,
        decoder_levels,
        internal_channels=None,
        num_internal=0,
        add_norm=True,
        add_act=True,
        kernel_size=3,
        alpha=0.5,
        ce_kwargs=None,
        dice_kwargs=None,
        ce_weight=1.0,
        seg_weight=1.0,
        delta=0.6,
        gamma=0.5,
        reduction="mean",
        **kwargs,
    ):
        """
        SymmetricUFLSegmenter is a segmentation head that combines symmetric focal loss and symmetric focal Tversky loss for robust segmentation, particularly in imbalanced scenarios.
        Inherits from:
            DiCESegmenter
        Args:
            conv (nn.Module): Convolutional layer to use in the segmentation head.
            seg_classes (int): Number of segmentation classes.
            in_channels (int): Number of input channels.
            decoder_levels (int): Number of decoder levels.
            internal_channels (Optional[int]): Number of internal channels for intermediate layers.
            num_internal (int): Number of internal layers.
            add_norm (bool): Whether to add normalization layers.
            add_act (bool): Whether to add activation layers.
            kernel_size (int): Kernel size for convolutional layers.
            alpha (float): Weighting factor between CE and Dice losses (0.0 = only Dice, 1.0 = only CE).
            ce_kwargs (Optional[dict]): Additional keyword arguments for the CE loss.
            dice_kwargs (Optional[dict]): Additional keyword arguments for the Dice loss.
            ce_weight (float, optional): Weight for the 'ce' component of loss function. Defaults to 1.
            seg (float, optional): Weight for the 'seg' component of loss function. Defaults to 1.
            delta (float): Delta parameter for the symmetric focal losses.
            gamma (float): Gamma parameter for the symmetric focal losses.
            reduction (str): Reduction method for the losses ('mean', 'sum', etc.).
            **kwargs: Additional keyword arguments.
        Attributes:
            loss_weight (float): Weight applied to the total loss.
            ce_loss (SymmetricFocalLoss): Symmetric focal loss instance for cross-entropy.
            dice_loss (SymmetricFocalTverskyLoss): Symmetric focal Tversky loss instance.
        Methods:
            compute_loss(pred_seg: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
                Computes the segmentation loss as a weighted sum of symmetric focal loss and symmetric focal Tversky loss.
                Args:
                    pred_seg (Dict[str, torch.Tensor]): Dictionary containing segmentation logits.
                    target (torch.Tensor): Ground truth segmentation mask.
                Returns:
                    Dict[str, torch.Tensor]: Dictionary with computed loss components.
        """
        super().__init__(
            conv,
            seg_classes,
            in_channels,
            decoder_levels,
            internal_channels,
            num_internal,
            add_norm,
            add_act,
            kernel_size,
            alpha,
            ce_kwargs,
            dice_kwargs,
            **kwargs,
        )

        self.ce_weight = ce_weight
        self.seg_weight = seg_weight

        self.ce_loss = SymmetricFocalLoss(
            delta=delta,
            gamma=gamma,
            reduction=reduction,
            nonlin=torch.nn.Softmax(dim=1),
        )
        self.dice_loss = SymmetricFocalTverskyLoss(
            delta=delta,
            gamma=gamma,
            reduction=reduction,
            nonlin=torch.nn.Softmax(dim=1),
        )

    def compute_loss(
        self,
        pred_seg: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target[target > 0] = 1  # skip multiclass
        seg_logits = pred_seg["seg_logits"]
        return {
            "seg_ce": self.ce_weight * self.alpha * self.ce_loss(seg_logits, target.long()),
            "seg_dice": self.seg_weight * (1 - self.alpha) * self.dice_loss(seg_logits, target),
        }


class AsymmetricUFLSegmenter(SymmetricUFLSegmenter):
    def __init__(
        self,
        conv,
        seg_classes,
        in_channels,
        decoder_levels,
        internal_channels=None,
        num_internal=0,
        add_norm=True,
        add_act=True,
        kernel_size=3,
        alpha=0.5,
        ce_kwargs=None,
        dice_kwargs=None,
        ce_weight=1.0,
        seg_weight=1.0,
        delta=0.6,
        gamma=0.5,
        reduction="mean",
        **kwargs,
    ):
        """
        Initializes the segmenter head with specified configuration.

        Args:
            conv (nn.Module): Convolutional layer to use in the head.
            seg_classes (int): Number of segmentation classes.
            in_channels (int): Number of input channels.
            decoder_levels (int): Number of decoder levels.
            internal_channels (Optional[int], optional): Number of internal channels. Defaults to None.
            num_internal (int, optional): Number of internal layers. Defaults to 0.
            add_norm (bool, optional): Whether to add normalization layers. Defaults to True.
            add_act (bool, optional): Whether to add activation layers. Defaults to True.
            kernel_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
            alpha (float, optional): Weighting factor for loss calculation. Defaults to 0.5.
            ce_kwargs (Optional[dict], optional): Additional keyword arguments for cross-entropy loss. Defaults to None.
            dice_kwargs (Optional[dict], optional): Additional keyword arguments for dice loss. Defaults to None.
            ce_weight (float, optional): Weight for the 'ce' component of loss function. Defaults to 1.
            seg (float, optional): Weight for the 'seg' component of loss function. Defaults to 1.
            delta (float, optional): Delta parameter for asymmetric loss functions. Defaults to 0.6.
            gamma (float, optional): Gamma parameter for asymmetric loss functions. Defaults to 0.5.
            reduction (str, optional): Reduction method for loss ('mean', 'sum', etc.). Defaults to "mean".
            **kwargs: Additional keyword arguments.

        Attributes:
            ce_loss (AsymmetricFocalLoss): Asymmetric focal loss with softmax nonlinearity.
            dice_loss (AsymmetricFocalTverskyLoss): Asymmetric focal Tversky loss with softmax nonlinearity.
        
        Methods:
            compute_loss(pred_seg: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
                Computes the segmentation loss as a weighted sum of asymmetric focal loss and asymmetric focal Tversky loss.
                Args:
                    pred_seg (Dict[str, torch.Tensor]): Dictionary containing segmentation logits.
                    target (torch.Tensor): Ground truth segmentation mask.
                Returns:
                    Dict[str, torch.Tensor]: Dictionary with computed loss components.
        """
        super().__init__(
            conv,
            seg_classes,
            in_channels,
            decoder_levels,
            internal_channels,
            num_internal,
            add_norm,
            add_act,
            kernel_size,
            alpha,
            ce_kwargs,
            dice_kwargs,
            ce_weight,
            seg_weight,
            delta,
            gamma,
            reduction,
            **kwargs,
        )

        self.ce_loss = AsymmetricFocalLoss(delta=delta, gamma=gamma, reduction=reduction, nonlin=torch.nn.Softmax(dim=1))
        self.dice_loss = AsymmetricFocalTverskyLoss(delta=delta, gamma=gamma, reduction=reduction, nonlin=torch.nn.Softmax(dim=1))
