from collections import OrderedDict
import torch.nn as nn


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (tuple): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels

        for layer_idx, layers_features in enumerate(layers, 1):
            d[f"mask_fcn{layer_idx}"] = nn.Conv2d(next_feature,
                                                  layers_features,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=dilation,
                                                  dilation=dilation)
            d[f"relu{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layers_features

        super().__init__(d)
        # initial params
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")