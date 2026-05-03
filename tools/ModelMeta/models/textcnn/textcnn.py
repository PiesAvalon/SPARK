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
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


def make_conv_layer(kernel_size):
    return nn.Conv2d(in_channels=1, out_channels=96, kernel_size=kernel_size, padding=1,
                     pad_mode="pad", has_bias=True)


class TextCNN(nn.Cell):
    def __init__(self, vocab_len, word_len, num_classes, vec_length, embedding_table='uniform'):
        super(TextCNN, self).__init__()
        self.vec_length = vec_length
        self.word_len = word_len
        self.num_classes = num_classes

        self.unsqueeze = mindspore.ops.unsqueeze
        self.embedding = nn.Embedding(vocab_len, self.vec_length, embedding_table=embedding_table)

        self.layer1 = self.make_layer(kernel_height=3)
        self.layer2 = self.make_layer(kernel_height=4)
        self.layer3 = self.make_layer(kernel_height=5)

        self.concat = ops.concat

        self.fc = nn.Dense(96 * 3, self.num_classes)
        self.drop = nn.Dropout(keep_prob=0.5)
        self.reducemax = ops.ReduceMax(keep_dims=False)

    def make_layer(self, kernel_height):
        return nn.SequentialCell(
            [
                make_conv_layer((kernel_height, self.vec_length)), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.word_len - kernel_height + 1, 1)),
            ]
        )

    def construct(self, x):
        # print("x", x.shape)
        x = self.unsqueeze(x, 1)

        x = ops.Cast()(x, mindspore.int32)

        x = self.embedding(x)

        x = ops.Cast()(x, mindspore.float32)
        x1 = self.layer1(x)

        x2 = self.layer2(x)

        x3 = self.layer3(x)

        x1 = ops.Cast()(x1, mindspore.float32)
        x2 = ops.Cast()(x2, mindspore.float32)
        x3 = ops.Cast()(x3, mindspore.float32)
        x1 = self.reducemax(x1, (2, 3))
        x2 = self.reducemax(x2, (2, 3))
        x3 = self.reducemax(x3, (2, 3))

        x = self.concat((x1, x2, x3), axis=1)

        x = self.drop(x)
        x = ops.Cast()(x, mindspore.float32)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    mindspore.set_context(device_target="GPU")
    model = TextCNN(vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
    # model2 = copy.deepcopy(model)
    a = mindspore.Tensor(np.random.randn(1, 51), dtype=mindspore.int32)
    # train_eval_TextCNN(model, model2, "../../datasets/rt-polaritydata", batch_size=4096)
    # print("type", a.dtype)
    print(model(a).shape)
    # mindspore.export(model, mindspore.Tensor(np.random.randn(1, 51), dtype=mindspore.int32), file_name="textcnn.onnx", file_format="ONNX")
