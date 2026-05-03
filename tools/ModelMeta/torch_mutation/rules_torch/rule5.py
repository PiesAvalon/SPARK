# conv
import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from torch_mutation.rules_torch.constant import *


class TransLayer_rule5(nn.Module):
    def __init__(self, layer_conv):
        super(TransLayer_rule5, self).__init__()
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
            self.layer_conv.weight.copy_(layer_conv.weight)
            if layer_conv.bias is not None:
                self.layer_conv.bias.copy_(layer_conv.bias)

    def forward(self, x):
        mut_x = x * self.delta
        return ((self.layer_conv(mut_x) - self.layer_conv.bias.reshape(-1, 1, 1)) / self.delta
                + self.layer_conv.bias.reshape(-1, 1, 1))

