import torch
import torch.nn as nn
from torch import tensor
import numpy as np


DTYPE = "float32"  # 数据类型
DELTA = 10         # 随机生成张量的范围，可自定义
device = "cuda:6"     # 设备
FORMAT = "NHWC"    # 格式（？） 仅用于sigmoid层

# scale
# delta = tensor(1 + np.random.uniform(-DELTA, DELTA, 1)[0].astype(DTYPE)).to(device)  # conv层
# delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device) pool层
# delta = tensor(np.random.uniform(0, DELTA, 1)[0]).to(device) relu层

# shift
# delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)
