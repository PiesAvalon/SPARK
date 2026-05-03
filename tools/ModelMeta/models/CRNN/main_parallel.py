# Copyright 2020-21 Huawei Technologies Co., Ltd
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
"""crnn training"""
import os

import mindspore
import mindspore.nn as nn
import numpy as np
import torch.nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.serialization import load_checkpoint
from torch.nn import functional as F
from src.crnn import CRNNV2
from src.dataset import create_dataset
from src.logger import get_logger
from src.loss import CTCLoss
from src.metric import CRNNAccuracy
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id
from src.model_utils.lr_scheduler import cosine_decay_lr_with_start_step

set_seed(1)


def apply_eval(eval_param):
    evaluation_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = evaluation_model.eval(eval_ds)
    return res[metrics_name]


def set_default():
    config.rank_id = int(os.getenv('RANK_ID', '0'))
    config.log_dir = os.path.join(config.output_dir, 'log', 'rank_%s' % config.rank_id)
    config.save_ckpt_dir = os.path.join(config.output_dir, 'ckpt')


def train():
    set_default()
    config.logger = get_logger(config.log_dir, config.rank_id)
    # config.logger.info("config : %s", config)
     #config.logger.info("Please check the above information for the configurations")

    if config.device_target == 'Ascend':
        device_id = get_device_id()
        context.set_context(device_id=device_id)

    if config.model_version == 'V1' and config.device_target != 'Ascend':
        raise ValueError("model version V1 is only supported on Ascend, pls check the config.")

    # lr_scale = 1
    if config.run_distribute:
        if config.device_target == 'Ascend':
            init()
            # lr_scale = 1
            device_num = get_device_num()
            rank = get_rank_id()
        else:
            init()
            # lr_scale = 1
            device_num = get_group_size()
            rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        device_num = 1
        rank = 0

    if config.resume_ckpt:
        resume_param = load_checkpoint(config.resume_ckpt,
                                       choice_func=lambda x: not x.startswith(('learning_rate', 'global_step')))
        config.train_start_epoch = int(resume_param.get('epoch_num', 0).asnumpy().item())
        config.logger.info("train_start_epoch: %d", config.train_start_epoch)
    max_text_length = config.max_text_length
    # create dataset
    dataset = create_dataset(name=config.train_dataset, dataset_path=config.train_dataset_path,
                             batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank, config=config)
    config.steps_per_epoch = dataset.get_dataset_size()
    config.logger.info("per_epoch_step_size: %d", config.steps_per_epoch)
    # define lr
    lr_init = config.learning_rate
    lr = cosine_decay_lr_with_start_step(0.0, lr_init, config.epoch_size * config.steps_per_epoch,
                                         config.steps_per_epoch, config.epoch_size,
                                         config.train_start_epoch * config.steps_per_epoch)
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    net = CRNNV2(config)
    for data in dataset:
        print(data[0].shape)
        print(net(data[0]).shape)
        break
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum, nesterov=config.nesterov)
    net_with_loss = WithLossCell(net, loss)


class CTCLoss_torch(torch.nn.Module):
    """
     CTCLoss definition

     Args:
        max_sequence_length(int): max number of sequence length. For text images, the value is equal to image
        width
        max_label_length(int): max number of label length for each input.
        batch_size(int): batch size of input logits
     """

    def __init__(self, max_sequence_length, max_label_length, batch_size):
        super(CTCLoss_torch, self).__init__()
        self.sequence_length = torch.tensor(np.array([max_sequence_length] * batch_size), dtype=torch.int32)
        labels_indices = []
        for i in range(batch_size):
            for j in range(max_label_length):
                labels_indices.append([i, j])
        self.labels_indices = torch.tensor(np.array(labels_indices), dtype=torch.int64)
        self.reshape = torch.reshape
        # self.ctc_loss = torch.nn.CTCLoss()

    def forward(self, logit, label):
        batch_size = logit.size(1)
        targets = torch.randint(1, 20, (batch_size, 2), dtype=torch.long)
        input_lengths = torch.ones(logit.size(1)).long() * logit.size(0)
        target_lengths = torch.ones(logit.size(1)).long() * 2
        loss = F.ctc_loss(logit, targets, input_lengths, target_lengths)
        return loss


if __name__ == '__main__':
    device = 'cuda:7'
    mindspore.set_context(device_target='GPU', device_id=6, pynative_synchronize=True)
    set_default()
    config.logger = get_logger(config.log_dir, config.rank_id)
    # config.logger.info("config : %s", config)
    # config.logger.info("Please check the above information for the configurations")
    device_num = 1
    rank = 0
    max_text_length = config.max_text_length
    dataset = create_dataset(name=config.train_dataset, dataset_path=config.train_dataset_path,
                             batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank, config=config)
    config.steps_per_epoch = dataset.get_dataset_size()
    print("dataset size: ", config.steps_per_epoch)
    dataset = dataset.create_tuple_iterator(output_numpy=True)
    config.logger.info("per_epoch_step_size: %d", config.steps_per_epoch)
    lr_init = config.learning_rate
    lr = cosine_decay_lr_with_start_step(0.0, lr_init, config.epoch_size * config.steps_per_epoch,
                                         config.steps_per_epoch, config.epoch_size,
                                         config.train_start_epoch * config.steps_per_epoch)
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    losser_torch = CTCLoss_torch(max_sequence_length=config.num_step,
                                 max_label_length=max_text_length,
                                 batch_size=config.batch_size)
    net = CRNNV2(config)
    from src.crnn_torch import CRNNV2 as CRNNV2_torch

    net_torch = CRNNV2_torch(config).to(device)
    # for data in dataset:
    #     print(data[0].shape)
    #     print(net(data[0]).shape)
    #     break
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum, nesterov=config.nesterov)
    opt_torch = torch.optim.SGD(net_torch.parameters(), lr=lr_init, momentum=config.momentum, nesterov=config.nesterov)
    losser = WithLossCell(net, loss)
    epoch_num = 6
    valid_dataset = create_dataset(name=config.eval_dataset,
                                   dataset_path=config.eval_dataset_path,
                                   batch_size=config.batch_size,
                                   is_training=False,
                                   config=config)
    metric = CRNNAccuracy(config, False)
    metric_torch = CRNNAccuracy(config, False)

    def forward_fn(data, label):
        loss = losser(data, label)
        return loss


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)


    def train_step(data, label):
        (loss), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, opt(grads))
        return loss


    for epoch in range(epoch_num):
        for data in dataset:
            loss = train_step(mindspore.Tensor(data[0]), mindspore.Tensor(data[1]))
            opt_torch.zero_grad()
            output = net_torch(torch.tensor(data[0], dtype=torch.float32).to(device))
            loss_torch = losser_torch(output, torch.tensor(data[1], dtype=torch.float32).to(device))
            loss_torch.backward()
            opt_torch.step()
            print("loss: ", loss)
            print("loss_torch: ", loss_torch)
            break
        metric.clear()
        metric_torch.clear()
        for tdata in valid_dataset:
            # print("y_pre", testnet(tdata[0]).shape)
            # indexes is [0, 2], using x as logits, y2 as label.
            metric.update(net(tdata[0]), tdata[1])
            metric_torch.update(mindspore.Tensor(net_torch(torch.tensor(tdata[0].asnumpy(),
                                                                        dtype=torch.float32).to(device)).detach().cpu().numpy()),
                                mindspore.Tensor(torch.tensor(tdata[1].asnumpy(),
                                                              dtype=torch.int64).to(device).detach().cpu().numpy()))
            break
        accuracy = metric.eval()
        accuracy_torch = metric_torch.eval()
        print("accuracy: ", accuracy)
        print("accuracy_torch: ", accuracy_torch)
