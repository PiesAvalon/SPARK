"""
变异BatchNorm算子，bias+=delta
已验证，成功
"""

import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch_mutation.rules_torch.constant import *
import copy


class TransLayer_rule9(nn.Module):
    def __init__(self, layer_bn):
        super(TransLayer_rule9, self).__init__()

        self.layer_bn = layer_bn

        if self.layer_bn.affine:
            self.delta = torch.tensor(np.random.uniform(-DELTA, DELTA, 1)[0].astype(DTYPE), device=device)
            self.layer_bn.bias.data += self.delta

        if self.layer_bn.track_running_stats:
            self.layer_bn.register_buffer('running_mean', self.layer_bn.running_mean.clone())
            self.layer_bn.register_buffer('running_var', self.layer_bn.running_var.clone())

    def forward(self, x):
        return self.layer_bn(x) - self.delta


if __name__ == "__main__" and False:
    # 创建一个标准的 BatchNorm 层
    batch_norm = nn.BatchNorm2d(10, affine=True).to(device)
    batch_norm.train()

    # 测试数据
    x = torch.randn(5, 10, 32, 32).to(device)

    # 通过 BatchNorm 层
    bn_output = batch_norm(x)
    print(batch_norm.bias)

    # 创建 TransLayer 层，初始化为与 batch_norm 层相同的参数
    trans_layer = TransLayer_rule9(batch_norm).to(device)



    # 通过 TransLayer 层
    trans_output = trans_layer(x)


    print(trans_layer.layer_bn.bias)
    print(trans_layer.delta)

    # 打印输出结果
    print("\nBatchNorm output:")
    print(bn_output)

    print("\nTransLayer output:")
    print(trans_output)

    # 计算和打印输出差异
    dis = torch.sum(torch.abs_(bn_output - trans_output))
    # 计算和打印输出差异
    print("\nMaximum difference between BatchNorm and TransLayer outputs:")
    print(dis)
