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
"""eval resnet."""
import os
import mindspore as ms
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from models.resnet50.resnet50 import create_cifar10_dataset

ms.set_seed(1)


# if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152"):
#     if config.net_name == "resnet18":
#         from src.resnet import resnet18 as resnet
#     elif config.net_name == "resnet34":
#         from src.resnet import resnet34 as resnet
#     elif config.net_name == "resnet50":
#         from src.resnet import resnet50 as resnet
#     else:
#         from src.resnet import resnet152 as resnet
#     # if config.dataset == "cifar10":
#     #     from src.dataset import create_dataset1 as create_dataset
#     # else:
#     #     from src.dataset import create_dataset2 as create_dataset
#
# # elif config.net_name == "resnet101":
# #     from src.resnet import resnet101 as resnet
# #     from src.dataset import create_dataset3 as create_dataset
# # else:
# #     from src.resnet import se_resnet50 as resnet
# #     from src.dataset import create_dataset4 as create_dataset
# from src.dataset import create_dataset1 as create_dataset


def eval_net(net):
    """eval net"""
    target = "CPU"
    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)

    # create dataset
    dataset = create_cifar10_dataset(data_home="../../datasets/cifar10", image_size=224, batch_size=32,
                                     training=False)

    # define net

    # load checkpoint
    param_dict = ms.load_checkpoint("/data1/CKPTS/resnet50/resnet50_ascend_v190_cifar10_official_cv_top1acc91.00.ckpt")
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    # if config.dataset == "imagenet2012":
    #     if not config.use_label_smooth:
    #         config.label_smooth_factor = 0.0
    #     loss = CrossEntropySmooth(sparse=True, reduction='mean',
    #                               smooth_factor=config.label_smooth_factor,
    #                               num_classes=config.class_num)
    # else:
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res)


# if __name__ == '__main__':
#     eval_net()
