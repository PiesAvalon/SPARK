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
"""TextCNN"""

import os

import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.cell import Cell
import mindspore
from mindspore.train.callback import Callback


class EvalCallback(Callback):
    """
    Evaluation per epoch, and save the best accuracy checkpoint.
    """

    def __init__(self, model, eval_ds, begin_eval_epoch=1, save_path="./"):
        self.model = model
        self.eval_ds = eval_ds
        self.begin_eval_epoch = begin_eval_epoch
        self.best_acc = 0
        self.save_path = save_path

    def epoch_end(self, run_context):
        """
        evaluate at epoch end.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.begin_eval_epoch:
            res = self.model.eval(self.eval_ds)
            acc = res["acc"]
            if acc > self.best_acc:
                self.best_acc = acc
                mindspore.save_checkpoint(cb_params.train_network, os.path.join(self.save_path, "best_acc.ckpt"))
                print("the best epoch is", cur_epoch, "best acc is", self.best_acc)


class SoftmaxCrossEntropyExpand(Cell):
    r"""
    Computes softmax cross entropy between logits and labels. Implemented by expanded formula.

    This is a wrapper of several functions.

    .. math::
        \ell(x_i, t_i) = -log\left(\frac{\exp(x_{t_i})}{\sum_j \exp(x_j)}\right),
    where :math:`x_i` is a 1D score Tensor, :math:`t_i` is the target class.

    Note:
        When argument sparse is set to True, the format of label is the index
        range from :math:`0` to :math:`C - 1` instead of one-hot vectors.

    Args:
        sparse(bool): Specifies whether labels use sparse format or not. Default: False.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **label** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, a scalar tensor including the mean loss.

    Examples:
        >>> loss = nn.SoftmaxCrossEntropyExpand(sparse=True)
        >>> input_data = Tensor(np.ones([64, 512]), dtype=mindspore.float32)
        >>> label = Tensor(np.ones([64]), dtype=mindspore.int32)
        >>> loss(input_data, label)
    """

    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.div = ops.Div()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.cast = ops.Cast()
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.reduce_max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, logit, label):
        """
        construct
        """
        print("logit", logit.shape, "label", label.shape)
        logit_max = self.reduce_max(logit, -1)
        print(logit_max)
        exp = self.exp(self.sub(logit, logit_max))
        print(exp)
        exp_sum = self.reduce_sum(exp, -1)
        print(exp_sum)
        softmax_result = self.div(exp, exp_sum)
        print(softmax_result)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
            print(label)
        softmax_result_log = self.log(softmax_result)
        print("softmax_result_log.shape2", softmax_result_log.shape)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        print(loss)
        loss = self.mul2(ops.scalar_to_tensor(-1.0), loss)
        loss = self.reduce_mean(loss, -1)

        return loss


def make_conv_layer(kernel_size):
    return nn.Conv2d(in_channels=1, out_channels=96, kernel_size=kernel_size, padding=1,
                     pad_mode="pad", has_bias=True)


class TextCNN(nn.Cell):
    """
    TextCNN architecture
    """

    def __init__(self, vocab_len, word_len, num_classes, vec_length, embedding_table='uniform'):
        super(TextCNN, self).__init__()
        self.vec_length = vec_length
        self.word_len = word_len
        self.num_classes = num_classes

        self.unsqueeze = ops.ExpandDims()
        self.embedding = nn.Embedding(vocab_len, self.vec_length, embedding_table=embedding_table)

        self.slice = ops.Slice()
        self.layer1 = self.make_layer(kernel_height=3)
        self.layer2 = self.make_layer(kernel_height=4)
        self.layer3 = self.make_layer(kernel_height=5)

        self.concat = ops.Concat(1)

        self.fc = nn.Dense(96 * 3, self.num_classes)
        self.drop = nn.Dropout(keep_prob=0.5)
        self.reducemax = ops.ReduceMax(keep_dims=False)

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

        self.out_shapes = {
            "INPUT": [-1, 51],
            "embedding": [-1, 1, 51, 40],
            "layer1.0": [-1, 96, 51, 3],
            "layer1.1": [-1, 96, 51, 3],
            "layer1.2": [-1, 96, 3, 3],
            "layer2.0": [-1, 96, 50, 3],
            "layer2.1": [-1, 96, 50, 3],
            "layer2.2": [-1, 96, 3, 3],
            "layer3.0": [-1, 96, 49, 3],
            "layer3.1": [-1, 96, 49, 3],
            "layer3.2": [-1, 96, 3, 3],
            "drop": [-1, 288],
            "fc": [-1, 2],
            "OUTPUT": [-1, 2]
        }
        self.in_shapes = {
            "INPUT": [-1, 51],
            "embedding": [-1, 1, 51],
            "layer1.0": [-1, 1, 51, 40],
            "layer1.1": [-1, 96, 51, 3],
            "layer1.2": [-1, 96, 51, 3],
            "layer2.0": [-1, 1, 51, 40],
            "layer2.1": [-1, 96, 50, 3],
            "layer2.2": [-1, 96, 50, 3],
            "layer3.0": [-1, 1, 51, 40],
            "layer3.1": [-1, 96, 49, 3],
            "layer3.2": [-1, 96, 49, 3],
            "drop": [-1, 288],
            "fc": [-1, 288],
            "OUTPUT": [-1, 2]
        }

        self.orders = {
            "embedding": ["INPUT", ["layer1.0", "layer2.0", "layer3.0"]],
            "layer1.0": ["embedding", "layer1.1"],
            "layer1.1": ["layer1.0", "layer1.2"],
            "layer1.2": ["layer1.1", "drop"],
            "layer2.0": ["embedding", "layer2.1"],
            "layer2.1": ["layer2.0", "layer2.2"],
            "layer2.2": ["layer2.1", "drop"],
            "layer3.0": ["embedding", "layer3.1"],
            "layer3.1": ["layer3.0", "layer3.2"],
            "layer3.2": ["layer3.1", "drop"],
            "drop": [["layer1.2", "layer2.2", "layer3.2"], "fc"],
            "fc": ["drop", "OUTPUT"]
        }
        self.layer_names = {
            "embedding": self.embedding,
            "layer1": self.layer1,
            "layer1.0": self.layer1[0],
            "layer1.1": self.layer1[1],
            "layer1.2": self.layer1[2],
            "layer2": self.layer2,
            "layer2.0": self.layer2[0],
            "layer2.1": self.layer2[1],
            "layer2.2": self.layer2[2],
            "layer3": self.layer3,
            "layer3.0": self.layer3[0],
            "layer3.1": self.layer3[1],
            "layer3.2": self.layer3[2],
            "drop": self.drop,
            "fc": self.fc,
        }






    def make_layer(self, kernel_height):
        return nn.SequentialCell(
            [
                make_conv_layer((kernel_height, self.vec_length)), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.word_len - kernel_height + 1, 1)),
            ]
        )

    def construct(self, x):
        """
        construct
        """
        x = self.unsqueeze(x, 1)
        #print("unsqueeze.dtype: ", x.dtype)

        x=ops.Cast()(x,mindspore.int32)

        x = self.embedding(x)
        #print("embedding.dtype: ", x.dtype)

        x = ops.Cast()(x, mindspore.float32)
        x1 = self.layer1(x)
        #print("layer1.dtype: ", x.dtype)

        x2 = self.layer2(x)
        #print("layer2.dtype: ", x.dtype)

        x3 = self.layer3(x)
        #print("layer3.dtype: ", x.dtype)

        x1 = ops.Cast()(x1, mindspore.float32)
        x2 = ops.Cast()(x2, mindspore.float32)
        x3 = ops.Cast()(x3, mindspore.float32)


        x1 = self.reducemax(x1, (2, 3))
        x2 = self.reducemax(x2, (2, 3))
        x3 = self.reducemax(x3, (2, 3))

        #print("x1.dtype", x1.dtype)
        #print("x2.dtype", x2.dtype)
        #print("x3.dtype", x3.dtype)


        x = self.concat((x1, x2, x3))

        x = self.drop(x)
        #x = ops.Cast()(x, mindspore.float32)
        #print("drop.dtype", x.dtype)

        x = self.fc(x)
        #print("fc.dtype", x.dtype)

        return x

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


    def set_layers(self,layer_name,new_layer):
        if 'embedding' == layer_name:
            self.embedding= new_layer
            self.layer_names["embedding"]=new_layer

        elif 'layer1' == layer_name:
            self.layer1= new_layer
            self.layer_names["layer1"]=new_layer

        elif 'layer1.0' == layer_name:
            self.layer1[0]= new_layer
            self.layer_names["layer1.0"]=new_layer

        elif 'layer1.1' == layer_name:
            self.layer1[1]= new_layer
            self.layer_names["layer1.1"]=new_layer
            self.origin_layer_names["layer1.1"]=new_layer
        elif 'layer1.2' == layer_name:
            self.layer1[2]= new_layer
            self.layer_names["layer1.2"]=new_layer

        elif 'layer2' == layer_name:
            self.layer2= new_layer
            self.layer_names["layer2"]=new_layer

        elif 'layer2.0' == layer_name:
            self.layer2[0]= new_layer
            self.layer_names["layer2.0"]=new_layer

        elif 'layer2.1' == layer_name:
            self.layer2[1]= new_layer
            self.layer_names["layer2.1"]=new_layer

        elif 'layer2.2' == layer_name:
            self.layer2[2]= new_layer
            self.layer_names["layer2.2"]=new_layer

        elif 'layer3' == layer_name:
            self.layer3= new_layer
            self.layer_names["layer3"]=new_layer

        elif 'layer3.0' == layer_name:
            self.layer3[0]= new_layer
            self.layer_names["layer3.0"]=new_layer

        elif 'layer3.1' == layer_name:
            self.layer3[1]= new_layer
            self.layer_names["layer3.1"]=new_layer

        elif 'layer3.2' == layer_name:
            self.layer3[2]= new_layer
            self.layer_names["layer3.2"]=new_layer

        elif 'drop' == layer_name:
            self.drop= new_layer
            self.layer_names["drop"]=new_layer

        elif 'fc' == layer_name:
            self.fc= new_layer
            self.layer_names["fc"]=new_layer




    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name, order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name] = order

    def get_outshape(self, layer_name):
        if layer_name not in self.out_shapes.keys():
            return False
        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name, out):
        if layer_name not in self.out_shapes.keys():
            return False
        self.out_shapes[layer_name] = out

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False
        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False
        self.in_shapes[layer_name] = out



    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out