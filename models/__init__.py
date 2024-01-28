# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .unet_model import UNet
from .other_network import R2U_Net, AttU_Net, NestedUNet
from .IN_Unet import InstanceNormalization_UNet


def get_models(model_name:str, args):
    """option: in_unet, bn_unet"""
    if model_name == "in_unet":
        model = InstanceNormalization_UNet(n_channels=args.in_channel,n_classes=args.classes)
    elif model_name == "bn_unet":
        model = UNet(n_channels=args.in_channel,n_classes=args.classes)
    else:
        raise NotImplementedError(f"{model_name} has not implemented")

    return model
