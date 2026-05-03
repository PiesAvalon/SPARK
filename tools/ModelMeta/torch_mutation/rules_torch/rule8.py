"""
变异Conv算子，bias+=delta
已验证，成功
"""

import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from torch_mutation.rules_torch.constant import *
import copy


class TransLayer_rule8(nn.Module):
    def __init__(self, layer_conv):
        super(TransLayer_rule8, self).__init__()
        if not isinstance(layer_conv, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")

        self.layer_conv = nn.Conv2d(
            in_channels=layer_conv.in_channels,
            out_channels=layer_conv.out_channels,
            kernel_size=layer_conv.kernel_size,
            stride=layer_conv.stride,
            padding=layer_conv.padding,
            dilation=layer_conv.dilation,
            bias=(layer_conv.bias is not None)
        )

        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0].astype("float32")).to(device)

        with (torch.no_grad()):
            self.layer_conv.weight.data = layer_conv.weight.data
            if layer_conv.bias is not None:
                self.layer_conv.bias.data = self.delta + layer_conv.bias.data

    def forward(self, x):
        return self.layer_conv(x) - self.delta


