# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train scripts"""

import os
import argparse
import time
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import mindspore
from mindspore import Tensor
mindspore.set_context(device_target="CPU")
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
import mindspore.ops as ops
from mindspore.common import set_seed
from src.model.generator import get_generator
from src.model.discriminator import get_discriminator
from src.dataset.traindataset import create_traindataset
from src.dataset.testdataset import create_testdataset
from src.loss.psnr_loss import PSNRLoss
from src.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from src.trainonestep.train_psnr import TrainOnestepPSNR
from src.trainonestep.train_gan import TrainOneStepD
from src.trainonestep.train_gan import TrainOnestepG
def get_model_width(model):
    """统计模型最大宽度（输出通道数/特征数）"""
    import mindspore
    max_width = 0
    for _, cell in model.cells_and_names():
        if not cell.cells():
            if isinstance(cell, mindspore.nn.Conv2d):
                width = cell.out_channels
            elif isinstance(cell, mindspore.nn.Dense):
                width = cell.out_channels
            elif isinstance(cell, mindspore.nn.BatchNorm2d):
                width = cell.num_features
            else:
                continue  # 跳过无明确宽度的层（如激活函数）
            if width > max_width:
                max_width = width
    return max_width

def count_layers(model):
    """统计模型深度（有效层数）"""
    depth = 0
    for _, cell in model.cells_and_names():
        # 过滤容器类层（如 Sequential、CellList），只统计叶子层
        if not cell.cells():
            # 只统计具有参数的层
            if cell.get_parameters():
                depth += 1
    return depth

if __name__ == '__main__':

    generator = get_generator(4, 0.02)

    batch, channels, h, w = 1, 3, 24, 24
    upscale = 4
    init_gain = 0.02
    net = get_generator(upscale_factor=upscale, init_gain=init_gain)
    net.set_train(False)

    # 随机输入
    batch, channels, h, w = 1, 3, 24, 24
    upscale = 4
    # 低分辨率 LR: 24×24
    lr_np = np.random.randn(batch, channels, h, w).astype(np.float32)
    # 高分辨率 HR: (h*upscale)×(w*upscale) = 96×96
    hr_np = np.random.randn(batch, channels, h*upscale, w*upscale).astype(np.float32)

    lr = Tensor(lr_np)
    hr = Tensor(hr_np)
    inp = Tensor(lr)
    print(inp.shape)
    loss = PSNRLoss(net)
    loss_loss = loss(hr,lr)
    print(loss_loss)
    # # 前向
    # out = net(inp)
    # discriminator = get_discriminator(96, 0.02)
    # w1 = get_model_width(generator)
    # w2 = get_model_width(discriminator)
    # d1 = count_layers(generator)
    # d2 = count_layers(discriminator)
    # print(d1+d2)
    # print(max(w1,w2))
