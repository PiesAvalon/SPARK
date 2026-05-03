# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""
#################train textcnn example on movie review########################
python issue6.py
"""
import os
import math
from mindspore import Tensor
import mindspore
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.common import set_seed

from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id
from model_utils.config import config
from src.textcnn import TextCNN
from src.textcnn import SoftmaxCrossEntropyExpand, EvalCallback
from src.dataset import MovieReview, SST2, Subjectivity

set_seed(1)


config.checkpoint_path = os.path.join(config.output_path, str(get_rank_id()), config.checkpoint_path)
config.epoch_num=2

def loss_com(logit, label):
    """
    construct
    """
    print("logit", logit.shape, "label", label.shape)
    exp = ops.Exp()
    reduce_sum = ops.ReduceSum(keep_dims=True)
    onehot = ops.OneHot()
    on_value = Tensor(1.0, mindspore.float32)
    off_value = Tensor(0.0, mindspore.float32)
    div = ops.Div()
    log = ops.Log()
    sum_cross_entropy = ops.ReduceSum(keep_dims=False)
    mul = ops.Mul()
    reduce_mean = ops.ReduceMean(keep_dims=False)
    reduce_max = ops.ReduceMax(keep_dims=True)
    sub = ops.Sub()

    logit_max = reduce_max(logit, -1)
    print(logit_max)
    exp0 = exp(sub(logit, logit_max))
    print(exp0)
    exp_sum = reduce_sum(exp0, -1)
    print(exp_sum)
    softmax_result = div(exp0, exp_sum)
    print(softmax_result)
    label = onehot(Tensor(label, dtype=mindspore.int32), ops.shape(logit)[1], on_value, off_value)
    print(label)
    softmax_result_log = log(softmax_result)
    print("softmax_result_log.shape1", softmax_result_log.shape)
    loss = sum_cross_entropy((mul(softmax_result_log, label)), -1)
    print(loss)
    loss = mul(ops.scalar_to_tensor(-1.0), loss)
    loss = reduce_mean(loss, -1)
    return loss



if __name__ == '__main__':
    """train net"""
    # set context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
    ms.set_context(device_id=get_device_id())
    if config.dataset == 'MR':
        instance = MovieReview(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SUBJ':
        instance = Subjectivity(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
        if config.device_target == "GPU":
            ms.set_context(enable_graph_kernel=True)
    elif config.dataset == 'SST2':
        instance = SST2(root_dir=config.data_path, maxlen=config.word_len, split=0.9)

    train_dataset = instance.create_train_dataset(batch_size=config.batch_size, epoch_size=config.epoch_num)
    test_dataset = instance.create_train_dataset(batch_size=config.batch_size, epoch_size=config.epoch_num)
    train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=config.epoch_num)
    test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=config.epoch_num)

    batch_num = train_dataset.get_dataset_size()
    if config.sink_size == -1:
        config.sink_size = batch_num

    base_lr = float(config.base_lr)
    learning_rate = []
    warm_up = [base_lr / math.floor(config.epoch_size / 5) * (i + 1) for _ in range(batch_num) for i in
               range(math.floor(config.epoch_size / 5))]
    shrink = [base_lr / (16 * (i + 1)) for _ in range(batch_num) for i in range(math.floor(config.epoch_size * 3 / 5))]
    normal_run = [base_lr for _ in range(batch_num) for i in
                  range(config.epoch_size - math.floor(config.epoch_size / 5) - math.floor(config.epoch_size * 2 / 5))]
    learning_rate = learning_rate + warm_up + normal_run + shrink

    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=config.word_len,
                  num_classes=config.num_classes, vec_length=config.vec_length)

    loss_fun = SoftmaxCrossEntropyExpand(sparse=True)


    for item in train_iter:
        text_array, targets_array = item['data'], item['label']
        # print(net(imgs_array).shape)
        text_tensor, targets_tensor = Tensor(text_array, dtype=mindspore.int32), Tensor(targets_array,
                                                                                        dtype=mindspore.int32)
        output_ms = net(text_tensor)
        loss1 = loss_com(output_ms, targets_tensor)
        print("========================================================")
        loss2 = loss_fun(output_ms, targets_tensor)
        print("loss1:", loss1)
        print("loss2:", loss2)

    # # Continue training if set pre_trained to be True
    # if config.pre_trained:
    #     param_dict = ms.load_checkpoint(config.checkpoint_path)
    #     ms.load_param_into_net(net, param_dict)

    # opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()),
    #               learning_rate=learning_rate, weight_decay=float(config.weight_decay))
    #
    # model = Model(net, loss_fn=loss_fun, optimizer=opt, metrics={'acc': Accuracy()})
    #
    # config_ck = CheckpointConfig(save_checkpoint_steps=config.sink_size,
    #                              keep_checkpoint_max=config.keep_checkpoint_max)
    # time_cb = TimeMonitor(data_size=batch_num)
    # ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    # ckpoint_cb = ModelCheckpoint(prefix="train_textcnn", directory=ckpt_save_dir, config=config_ck)
    # loss_cb = LossMonitor()
    # eval_callback = EvalCallback(model, test_dataset, save_path=ckpt_save_dir)
    # if config.device_target == "CPU":
    #     model.train(config.epoch_size, train_dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    # else:
    #     epoch_count = config.epoch_size * batch_num // config.sink_size
    #     model.train(epoch_count, train_dataset, callbacks=[time_cb, ckpoint_cb, loss_cb, eval_callback],
    #                 sink_size=config.sink_size, dataset_sink_mode=True)
    print("train success")


