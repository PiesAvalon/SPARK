"""
变异softmax算子，input+=delta
已验证，成功
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *

class TransLayer_rule17(nn.Module):
    def __init__(self, layer_softmax):
        super(TransLayer_rule17, self).__init__()

        self.layer_softmax = layer_softmax
        self.delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)

    def forward(self, x):
        mut_x = x + self.delta
        return self.layer_softmax(mut_x)


"""
# 创建 Softmax 层
softmax = nn.Softmax(dim=1).to(device)

# 创建变异层实例
trans_layer = TransLayer(softmax)

# 生成随机数据
x = torch.randn(5, 10).to(device)  # 5个样本，每个样本10维

# 计算正常 softmax 的输出
with torch.no_grad():
    original_output = softmax(x)
    print("Original Softmax Output:")
    print(original_output)

# 计算变异 softmax 的输出
with torch.no_grad():
    mutated_output = trans_layer(x)
    print("\nMutated Softmax Output:")
    print(mutated_output)

print(original_output - mutated_output)

# 打印变异系数
print("\nDelta used for mutation:")
print(trans_layer.delta)
"""