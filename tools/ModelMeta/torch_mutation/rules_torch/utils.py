"""
一些仅用于改变输入张量的工具层，可能有用
"""

import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from constant import *

# 适用于Conv层，data*delta
class ScaleInput(nn.Module):
    def __init__(self, layer):
        super(ScaleInput, self).__init__()
        self.layer_name = type(layer).__name__
        if type(self.layer_name) == 'Conv2d':
            self.delta = tensor(1 + np.random.uniform(-DELTA, DELTA, 1)[0].astype(DTYPE)).to(device)
        elif type(self.layer_name) == 'MaxPool2d':
            self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)
        elif type(self.layer_name) == 'AvgPool2d':
            self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)

    def forward(self, x):
        return x * self.delta


# 适用于pool层、softmax层，data+delta
class ShiftInput(nn.Module):
    def __init__(self, layer):
        super(ShiftInput, self).__init__()
        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)

    def forward(self, x):
        return x + self.delta


# 适用于pool层，张量转置
class TransposeInputPool(nn.Module):
    def __init__(self):
        super(TransposeInputPool, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 2, 3)


# 适用于sigmoid层，张量转置
class TransposeInputSigmoid(nn.Module):
    def __init__(self):
        super(TransposeInputSigmoid, self).__init__()

    def forward(self, x):
        if FORMAT == "NHWC":
            return torch.from_numpy(x.cpu().numpy().transpose(0, 2, 1, 3)).to(device)

        elif FORMAT == "NCHW":
            return torch.from_numpy(x.cpu().numpy().transpose(0, 1, 3, 2)).to(device)
