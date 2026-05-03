"""
变异tanh算子，input*(-1)
已验证，成功
"""

import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from torch_mutation.rules_torch.constant import *


class TransLayer_rule18(nn.Module):
    def __init__(self, layer_tanh):
        super(TransLayer_rule18, self).__init__()
        self.layer_tanh = layer_tanh

    def forward(self, x):
        return -self.layer_tanh(-x)


"""
# 创建 Tanh 层
tanh = nn.Tanh().to(device)

# 创建变异层实例
trans_layer = TransLayer(tanh)

# 生成随机数据
x = torch.randn(5, 10).to(device)  # 5个样本，每个样本10维

# 计算正常 tanh 的输出
with torch.no_grad():
    original_output = tanh(x)
    print("Original Tanh Output:")
    print(original_output)

# 计算变异 tanh 的输出
with torch.no_grad():
    mutated_output = trans_layer(x)
    print("\nMutated Tanh Output:")
    print(mutated_output)

print(original_output - mutated_output)
"""