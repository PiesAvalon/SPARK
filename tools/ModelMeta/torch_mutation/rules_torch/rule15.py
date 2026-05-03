"""
变异relu算子，input*delta【relu leakyrelu】
已验证，成功
"""

import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *


class TransLayer_rule15_ReLU(nn.Module):
    def __init__(self, layer_relu):
        super(TransLayer_rule15_ReLU, self).__init__()

        self.layer_relu = layer_relu
        self.delta = tensor(np.random.uniform(0, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x * self.delta
        return self.layer_relu(mut_x) / self.delta

class TransLayer_rule15_LeakyReLU(nn.Module):
    def __init__(self, layer_relu):
        super(TransLayer_rule15_LeakyReLU, self).__init__()

        self.layer_relu = layer_relu
        self.delta = tensor(np.random.uniform(0, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x * self.delta
        return self.layer_relu(mut_x) / self.delta

if __name__ == "__main__" and False:
    relu = nn.LeakyReLU(negative_slope=0.01).to(device)

    # 创建变异层实例
    trans_layer = TransLayer_rule15_ReLU(relu)

    # 生成随机数据
    x = torch.randn(5, 10).to(device)  # 5个样本，每个样本10维

    # 计算正常 ReLU 的输出
    with torch.no_grad():
        original_output = relu(x)
        print("Original ReLU Output:")
        print(original_output)

    # 计算变异 ReLU 的输出
    with torch.no_grad():
        mutated_output = trans_layer(x)
        print("\nMutated ReLU Output:")
        print(mutated_output)

    # 打印变异系数
    print("\nDelta used for mutation:")
    print(trans_layer.delta)

    print(original_output - mutated_output)
    diff = torch.abs(original_output - mutated_output).max().item()
    print("\n最大输出差异:", diff)
