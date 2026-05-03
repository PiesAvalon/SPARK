"""
变异Conv算子，input.transpose, weight.transpose
已验证，成功
"""

import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from torch_mutation.rules_torch.constant import *


class TransLayer_rule7(nn.Module):
    def __init__(self, layer_conv):
        super(TransLayer_rule7, self).__init__()
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

        self.delta = tensor(1 + np.random.uniform(-DELTA, DELTA, 1)[0].astype(DTYPE)).to(device)

        with torch.no_grad():
            self.layer_conv.weight.data = layer_conv.weight.transpose(2,3)
            if layer_conv.bias is not None:
                self.layer_conv.bias.copy_(layer_conv.bias)

    def forward(self, x):
        mut_x = x.transpose(2,3)
        return ((self.layer_conv(mut_x) - self.layer_conv.bias.reshape(-1, 1, 1))).transpose(2,3) + self.layer_conv.bias.reshape(-1, 1, 1)


