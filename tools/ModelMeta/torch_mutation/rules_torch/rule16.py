"""
变异sigmoid算子，input.transpose
已验证，成功
"""

import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *


class TransLayer_rule16(nn.Module):
    def __init__(self, layer_sigmoid):
        super(TransLayer_rule16, self).__init__()

        self.layer_sigmoid = layer_sigmoid
        self.delta = tensor(np.random.uniform(0, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x * self.delta
        if FORMAT == "NHWC":
            # mut_x = torch.from_numpy(x.cpu().numpy().transpose(0, 2, 1, 3)).to(device) 9.28 zgb
            mut_x = torch.from_numpy(x.detach().cpu().numpy().transpose(0, 2, 1, 3)).to(device)
            return torch.from_numpy(self.layer_sigmoid(mut_x).cpu().numpy().transpose(0, 2, 1, 3)).to(device)
        elif FORMAT == "NCHW":
            mut_x = torch.from_numpy(x.cpu().numpy().transpose(0, 1, 3, 2)).to(device)
            return torch.from_numpy(self.layer_sigmoid(mut_x).cpu().numpy().transpose(0, 1, 3, 2)).to(device)


"""
sigmoid = nn.Sigmoid().to(device)

# 创建变异层实例
trans_layer = TransLayer(sigmoid)

# 生成随机数据
x = torch.randn(5, 10, 20, 20).to(device)  # 示例数据，假设为 (batch_size, channels, height, width)

# 计算正常 Sigmoid 的输出
with torch.no_grad():
    original_output = sigmoid(x)
    print("Original Sigmoid Output:")
    print(original_output)

# 计算变异 Sigmoid 的输出
with torch.no_grad():
    mutated_output = trans_layer(x)
    print("\nMutated Sigmoid Output:")
    print(mutated_output)

# 打印变异系数
print("\nDelta used for mutation:")
print(trans_layer.delta)

print(original_output - mutated_output)
"""