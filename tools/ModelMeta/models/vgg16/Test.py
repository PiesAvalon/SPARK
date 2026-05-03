# Copyright 2020 Huawei Technologies Co., Ltd
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
#################train vgg16 example on cifar10########################
"""
import datetime
import os
import time

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
import numpy as np
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset

from src.crossentropy import CrossEntropy
from src.warmup_step_lr import warmup_step_lr
from src.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
from src.warmup_step_lr import lr_steps
from src.utils.logging import get_logger
from src.utils.util import get_param_groups
from src.vgg import vgg16

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

import random
set_seed(1)


def update_params(old_op,new_op):
    attrs_list=list(old_op.__dict__.items())
    edit_flag=False
    for i in range(len(attrs_list)):
        if "grad_ops_label" in attrs_list[i][0]:
            edit_flag=True
            continue
        if edit_flag and "grad_ops_label" not in attrs_list[i][0]:
            if "Prim" in str(attrs_list[i][1]) and "<" in str(attrs_list[i][1]):
                edit_flag=False
                continue
            if hasattr(new_op,attrs_list[i][0]):
                #print("old op:" + str(new_op))
                #print("Update param:"+attrs_list[i][0])
                setattr(new_op,str(attrs_list[i][0]),attrs_list[i][1])
                #print("new op:" +str(new_op))
    return new_op


class ops2Cell(nn.Cell):
    def __init__(self, op2convert):
        super(ops2Cell, self).__init__()
        self.opcell = op2convert

    def construct(self, x):
        return self.opcell()(x)


class EmptyCell(nn.Cell):
    def __init__(self):
        super(EmptyCell, self).__init__()

    def construct(self, x):
        return x

class ops_reshape(nn.Cell):
    def __init__(self,out_shape):
        super(ops_reshape, self).__init__()
        self.out_shape=out_shape

    def construct(self, x):
        return ops.reshape(x,input_shape=self.out_shape)


class Repacle(nn.Cell):
    def product(self,shape_list):
        pro=1
        for val in shape_list:
            pro*=val
        return pro

    def __init__(self,in_shape,out_shape):
        super(Repacle, self).__init__()
        self.in_shape=in_shape
        self.out_shape=out_shape
        # self.random_seed = r_seed
        self.in_feature=self.product(self.in_shape)
        self.out_feature=self.product(self.out_shape)
        self.in_channel = in_shape[1]
        self.out_channel = out_shape[1]
        self.in_h = in_shape[2]
        self.out_h = out_shape[2]
        self.in_w = in_shape[3]
        self.out_w = out_shape[3]
        self.det=abs(self.out_feature-self.in_feature)
        self.operate_col_nums = int(self.det / in_shape[0])
        self.batch_size=in_shape[0]

        self.flatten_del=nn.Flatten()
        self.reshape_del=ops_reshape(self.out_shape)
        self.concat_del=ops_concat(1)
        self.cast_del = ops.Cast()

    def construct(self, x):
        x=self.cast_del(x,mindspore.float32)

        if self.out_feature>self.in_feature:
            x=self.flatten_del(x)
            concat_data=mindspore.Tensor(np.ones((self.batch_size,self.operate_col_nums)),mindspore.float32)
            x=self.concat_del(x,concat_data)

        elif self.out_feature<self.in_feature:
            if self.in_channel>self.out_channel:
                del_cols = np.random.randint(low=0, high=self.in_channel, size=(self.out_channel), dtype='int')
                x=x[:,list(del_cols),:,:]
                x = x.resize((self.batch_size,self.out_channel,self.out_h,self.out_w))
            elif self.in_channel==self.out_channel:
                x = x.resize((self.batch_size,self.out_channel,self.out_h,self.out_w))
            elif self.in_channel<self.out_channel:
                x = x.resize((self.batch_size,self.in_channel,self.out_h,self.out_w))
                add_channels=self.out_channel-self.in_channel
                concat_data = mindspore.Tensor(np.ones((self.batch_size,add_channels,self.out_h, self.out_w)), mindspore.float32)
                x = self.concat_del(x, concat_data)
            return x

        x = self.reshape_del(x)
        return x

class ops_concat(nn.Cell):
    def __init__(self,axis):
        super(ops_concat, self).__init__()
        self.axis=axis

    def construct(self, x1,x2):
        return ops.concat((x1,x2),axis=self.axis)

class BasicOPUtils:
    def __init__(self):
        self.available_activations = BasicOPUtils.available_activations()

    @staticmethod
    def available_activations():
        activations = {}
        activations['relu'] = nn.ReLU()
        activations['relu6'] = nn.ReLU6()
        activations['tanh'] = nn.Tanh()
        activations['sigmoid'] = nn.Sigmoid()
        activations['leakyrelu'] = nn.LeakyReLU()
        activations['elu'] = nn.ELU()
        activations['gelu'] =nn.GELU()
        activations['mish'] = nn.Mish()
        activations['softmax']=nn.Softmax()


        return activations


    @staticmethod
    def available_convs(in_channel, out_channel,kernel_size,stride,name):
        name=name.lower()
        if "1d" in name:
            convs=[nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,stride=stride),nn.Conv1dTranspose(in_channel, out_channel, kernel_size=kernel_size,stride=stride)]
            convs=np.random.permutation(convs)
            return convs[0]

        if "2d" in name:
            convs=[nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride),nn.Conv2dTranspose(in_channel, out_channel, kernel_size=kernel_size,stride=stride)]
            convs = np.random.permutation(convs)
            return convs[0]

        if "3d" in name:
            convs=[nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size,stride=stride),nn.Conv3dTranspose(in_channel, out_channel, kernel_size=kernel_size,stride=stride)]
            convs = np.random.permutation(convs)
            return convs[0]

        convs = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride),
                 nn.Conv2dTranspose(in_channel, out_channel, kernel_size=kernel_size, stride=stride)]
        convs = np.random.permutation(convs)
        return convs[0]


    @staticmethod
    def available_embedding(vocab_size,embedding_size,name):
       if "embedding"==name:
           return nn.Embedding(vocab_size,embedding_size)
       elif "embeddinglookup" == name:
            return nn.EmbeddingLookup(vocab_size, embedding_size)
    @staticmethod
    def available_Dense(in_feature, out_feature):
        return nn.Dense(in_feature, out_feature)

    @staticmethod
    def available_Dropout(p):
        return nn.Dropout(p)


    @staticmethod
    def available_BN(num_features,name):
        if "1d" in name:
            return nn.BatchNorm1d(num_features)
        if "2d" in name:
            return nn.BatchNorm2d(num_features)
        if "3d" in name:
            return nn.BatchNorm3d(num_features)

    @staticmethod
    def available_LN(shape_list):
        shape_list_nobatch=shape_list[1:]
        return nn.LayerNorm(shape_list_nobatch,begin_norm_axis=1, begin_params_axis=1)

    @staticmethod
    def available_pool(output_size,stride,name):
        if "1d" in name:
            pools=[nn.AvgPool1d(output_size,stride),nn.MaxPool1d(output_size,stride)]#nn.AdaptiveAvgPool1d(output_size),nn.AdaptiveMaxPool1d(output_size),
            pools=np.random.permutation(pools)
            return pools[0]
        if "2d" in name:
            pools=[ nn.AvgPool2d(output_size,stride),nn.MaxPool2d(output_size,stride)]#nn.AdaptiveAvgPool2d(output_size),nn.AdaptiveMaxPool2d(output_size),
            pools = np.random.permutation(pools)
            return pools[0]
        # if "3d" in name:
        #     return nn.AdaptiveAvgPool3d(output_size)

        pools = [nn.AvgPool2d(output_size, stride), nn.MaxPool2d(output_size,  stride)]  # nn.AdaptiveAvgPool2d(output_size),nn.AdaptiveMaxPool2d(output_size),
        pools = np.random.permutation(pools)
        return pools[0]





    @staticmethod
    def available_flatten():
       return nn.Flatten()



    @staticmethod
    def available_ops(size):
        """
        just for ResizeBilinear
        :param size: tuple
        :return:
        """
        ops_list=[ops2Cell(ops.ResizeBilinearV2(size)),ops2Cell(ops.Mul()),ops2Cell(ops.ReduceAll()),ops2Cell(ops.ReduceMean()),ops2Cell(ops.ReduceAny())
        ,ops2Cell(ops.ReduceMax()),ops2Cell(ops.ReduceMin()),ops2Cell(ops.MatMul()),ops2Cell(ops.Cast()),ops2Cell(ops.Concat()),ops2Cell(ops.Reshape())
        ,ops2Cell(ops.Tile())
        ,ops2Cell(ops.Squeeze())]
        ops_list=np.random.permutation(ops_list)
        return ops_list[0]


mutate_OPparam_names = {"kernel_size":"tuple_int",
                        # "padding":"tuple_int",
                        # "pad_mode":["enum",["same","valid","pad"]],
                        "keep_dims":"Bool",
                        "in_channels":"int",
                        "out_channels":"int",
                        "stride":"tuple_int",
                        }

mutate_OPparam_valranges = {"kernel_size":"(1,10)",
                        # "padding":"tuple_int",
                        # "pad_mode":["enum",["same","valid","pad"]],
                        "in_channels":"4",
                        "out_channels":"4",
                        "stride":"(1,20)",
                        }

if __name__ == '__main__':
    #run_train()
    config.device_target="GPU"
    config.data_dir="/root/MSTest/data"
    config.image_size=(224,224)
    config.rank = get_rank_id()
    config.group_size = get_device_num()
    import mindspore.ops as ops
    stdnormal = ops.StandardNormal(seed=2)
    input_size=(1, 3, 224, 224)
    input_data= stdnormal(input_size)
    model= vgg16(config.num_classes, config)

    layers = model.cells_and_names()
    layer_names = []
    for layer in layers:
        if "" == layer[0] or "del" in layer[0] or "empty" in layer[0]:
            continue
        print(layer[0])
        layer_names.append(layer[0])

    replace_layer = Repacle((1,64,222,222), (1,64,220,220))

    params_ori = list(mutate_OPparam_names.keys())
    params_ran = np.random.permutation(params_ori)
    mutate_param_selname = params_ran[0]
    mutate_param_selname = "in_channels"

    suit_for_mutate_layer_idxs = []
    for layer_name in layer_names:
        if hasattr(model.get_layers(layer_name), mutate_param_selname):
            suit_for_mutate_layer_idxs.append(layer_name)

    suit_for_mutate_layers_type = [model.get_layers(val).__class__.__name__ for val in suit_for_mutate_layer_idxs]
    print("suit_for_mutate_layer_idx:" + str(suit_for_mutate_layers_type))
    suit_for_mutate_layer_idxs = np.random.permutation(suit_for_mutate_layer_idxs)
    suit_for_mutate_layer_idx = suit_for_mutate_layer_idxs[0]
    suit_for_mutate_layer_idx = "layers.3"
    mutate_layer = model.get_layers(suit_for_mutate_layer_idx)
    mutate_layer_order = model.get_order(suit_for_mutate_layer_idx)
    print("order:" + str(mutate_layer_order))
    mutate_layer_output_shape = model.get_outshape(suit_for_mutate_layer_idx)
    mutate_layer_type = mutate_layer.__class__.__name__

    # find the input
    mutate_layer_input_shape = 0
    for layer_name in layer_names:
        if model.get_order(layer_name) == mutate_layer_order - 1:
            mutate_layer_input_shape = model.get_outshape(layer_name)
            break
    mutate_layer_input_shape[0] = input_size[0]
    mutate_layer_output_shape[0] = input_size[0]

    print("select op: " + str(
        suit_for_mutate_layer_idx) + " selected param:" + mutate_param_selname + " input_shape:" + str(
        mutate_layer_input_shape) + " output_shape:" + str(mutate_layer_output_shape))

    value_range_type = mutate_OPparam_names[mutate_param_selname]

    if "int" == value_range_type:
        Magnification = int(mutate_OPparam_valranges[mutate_param_selname])
        index = [i for i in range(-1 * Magnification, Magnification + 1)]
        change_rates = [-1 / val for val in index if val < 0] + [val for val in index if val > 0]

        print("change_rates: " + str(change_rates))

        change_rate = 1.0
        while change_rate == 1.0:
            change_rate = np.random.permutation(change_rates)[0]

        new_value = int(change_rate * getattr(mutate_layer, mutate_param_selname))

    new_value=128
    print("new value: " + str(new_value))
    print("------------")
    print(mutate_layer)
    print("------------")
    from copy import deepcopy
    test_input_data_shape = deepcopy(mutate_layer_input_shape)
    if "conv" in mutate_layer_type.lower():
        if "in_channels" == mutate_param_selname:
            mutate_replace_layer = BasicOPUtils.available_convs(new_value, mutate_layer.out_channels,
                                                                mutate_layer.kernel_size, mutate_layer.stride,
                                                                mutate_layer_type)
            test_input_data_shape[1] = new_value

    print(mutate_replace_layer)
    stdnormal = ops.StandardNormal(seed=2)
    test_input_data = stdnormal(tuple(test_input_data_shape))
    print("newop_in_shape:" + str(test_input_data.shape))
    new_op_outshape = mutate_replace_layer(test_input_data).shape
    print("newop_out_shape:" + str(new_op_outshape))

    replace_cell_outshape = [mutate_layer_input_shape[0], new_value, mutate_layer_input_shape[2],
                             mutate_layer_input_shape[3]]
    replace_cell1 = Repacle(tuple(mutate_layer_input_shape), tuple(test_input_data_shape))
    replace_cell2 = Repacle(tuple(new_op_outshape), tuple(mutate_layer_output_shape))
    replace_layer = nn.SequentialCell([replace_cell1, mutate_replace_layer, replace_cell2])

    print("***********************")
    print(tuple(mutate_layer_input_shape))
    print(tuple(mutate_layer_output_shape))
    print("***********************")

    # replace_layer = Repacle(tuple(mutate_layer_input_shape), tuple(mutate_layer_output_shape))
    #
    # print("********** " + str(replace_layer(test_input_data).shape))
    model.set_layers(suit_for_mutate_layer_idx, replace_layer)
    #
    # layers = model.cells_and_names()
    # layer_names = []
    # for layer in layers:
    #     print(layer[0])
    #     if "" == layer[0] or "del" in layer[0] or "empty" in layer[0]:
    #         continue
    #
    model(input_data)








