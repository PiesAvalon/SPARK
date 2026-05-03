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


class TransLayer_rule6(nn.Module):
    def __init__(self, layer_conv):
        super(TransLayer_rule6, self).__init__()
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
            self.layer_conv.weight.data = layer_conv.weight.data * self.delta
            if layer_conv.bias is not None:
                self.layer_conv.bias.data = layer_conv.bias.data

    def forward(self, x):
        return ((self.layer_conv(x) - self.layer_conv.bias.reshape(-1, 1, 1)) / self.delta
                + self.layer_conv.bias.reshape(-1, 1, 1))


if __name__ == "__main__" and False: 
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    print(conv_layer)
    #trans_layer, d = FollowModel_4(conv_layer, device)
    trans_layer = TransLayer_rule6(conv_layer)
    print(trans_layer)
    x = torch.randn(1, 1, 5, 5).to(device)  # Batch size 1, 1 channel, 5x5 image

    conv_layer_output = conv_layer(x)
    trans_layer_output = trans_layer(x)
    
    print("delta:")
    print(trans_layer.delta)
    print("Original Conv Layer:")
    print(conv_layer.weight)
    print("TransLayer:")
    print(trans_layer.layer_conv.weight)


    print("Original Conv Layer Output:")
    print(conv_layer_output)
    print("TransLayer Output:")
    print(trans_layer_output)
    print("delta:")
    print(trans_layer.delta)

    dis = torch.sum(torch.abs_(conv_layer_output - trans_layer_output))
    print("Difference between outputs:")
    print(trans_layer_output - conv_layer_output)
    print(dis)
