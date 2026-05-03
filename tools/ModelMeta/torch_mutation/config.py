import device_op

if device_op.device_option=='cpu':
    device='cpu'
elif device_op.device_option=='gpu':
    device='cuda:6'


# device='cpu'
# device='cuda'

import torch
from torch_mutation.rules_torch import rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,\
                                        rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18
rules_dict = {
            # torch.nn.Conv2d: [rule1, rule2, rule3, rule5, rule6, rule7, rule8],# 图像分类
            torch.nn.Conv2d: [rule1, rule3, rule5, rule6, rule7, rule8],# 非图像分类
            torch.nn.AvgPool2d: [rule1, rule3, rule12, rule13, rule14],
            torch.nn.MaxPool2d: [rule1, rule3, rule12, rule13, rule14],
            torch.nn.ReLU: [rule1, rule15],
            torch.nn.ReLU6: [rule1],
            torch.nn.BatchNorm2d: [rule1, rule4, rule9, rule10, rule11],
            torch.nn.Linear: [rule1],
            torch.nn.Flatten: [rule1],
            torch.nn.Hardsigmoid: [rule1],
            torch.nn.Sigmoid: [rule16, rule1],
            torch.nn.Softmax: [rule17, rule1],
            torch.nn.Tanh: [rule18, rule1],

            torch.nn.ConvTranspose2d: [rule1],
            torch.nn.LeakyReLU: [rule1,rule15],
            torch.nn.AdaptiveAvgPool2d: [rule1,rule12,rule13,rule14],
            torch.nn.Dropout: [rule1],
            torch.nn.Embedding: [rule1],
            torch.nn.LSTM: [rule1]
        }