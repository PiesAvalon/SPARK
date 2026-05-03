import torch
import mindspore
import mindspore.nn as nn_ms
from mindspore.nn.optim import AdamWeightDecay


def get_optimizer(optimize_name):
    optimizer = {}
    optimizer['SGD'] = [nn_ms.SGD, torch.optim.SGD]
    optimizer['adam'] = [mindspore.nn.Adam, torch.optim.Adam]
    optimizer['adamweightdecay'] = [AdamWeightDecay, torch.optim.AdamW]
    return optimizer[optimize_name]
