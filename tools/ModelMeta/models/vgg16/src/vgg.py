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
Image classifiation.
"""
import math
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
from .utils.var_init import default_recurisive_init, KaimingNormal
import mindspore
import mindspore.ops as ops


def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    # base=[64]
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               # padding=args.padding,
                               # pad_mode=args.pad_mode,
                               # has_bias=args.has_bias,
                               pad_mode="valid",
                               has_bias=False
                               )  # weight_init=weight
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train",
                 include_top=True):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.include_top = include_top
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))  # nn.Dense(512*1*1,512*7*7)
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        has_dropout = False
        if not has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])

        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None
        self.in_shapes = {'INPUT': [-1, 3, 224, 224],
                          'layers.0': [-1, 3, 224, 224],
                          'layers.1': [-1, 64, 222, 222],
                          'layers.2': [-1, 64, 222, 222],
                          'layers.3': [-1, 64, 222, 222],
                          'layers.4': [-1, 64, 220, 220],
                          'layers.5': [-1, 64, 220, 220],
                          'layers.6': [-1, 64, 220, 220],
                          'layers.7': [-1, 64, 110, 110],
                          'layers.8': [-1, 128, 108, 108],
                          'layers.9': [-1, 128, 108, 108],
                          'layers.10': [-1, 128, 108, 108],
                          'layers.11': [-1, 128, 106, 106],
                          'layers.12': [-1, 128, 106, 106],
                          'layers.13': [-1, 128, 106, 106],
                          'layers.14': [-1, 128, 53, 53],
                          'layers.15': [-1, 256, 51, 51],
                          'layers.16': [-1, 256, 51, 51],
                          'layers.17': [-1, 256, 51, 51],
                          'layers.18': [-1, 256, 49, 49],
                          'layers.19': [-1, 256, 49, 49],
                          'layers.20': [-1, 256, 49, 49],
                          'layers.21': [-1, 256, 47, 47],
                          'layers.22': [-1, 256, 47, 47],
                          'layers.23': [-1, 256, 47, 47],
                          'layers.24': [-1, 256, 23, 23],
                          'layers.25': [-1, 512, 21, 21],
                          'layers.26': [-1, 512, 21, 21],
                          'layers.27': [-1, 512, 21, 21],
                          'layers.28': [-1, 512, 19, 19],
                          'layers.29': [-1, 512, 19, 19],
                          'layers.30': [-1, 512, 19, 19],
                          'layers.31': [-1, 512, 17, 17],
                          'layers.32': [-1, 512, 17, 17],
                          'layers.33': [-1, 512, 17, 17],
                          'layers.34': [-1, 512, 8, 8],
                          'layers.35': [-1, 512, 6, 6],
                          'layers.36': [-1, 512, 6, 6],
                          'layers.37': [-1, 512, 6, 6],
                          'layers.38': [-1, 512, 4, 4],
                          'layers.39': [-1, 512, 4, 4],
                          'layers.40': [-1, 512, 4, 4],
                          'layers.41': [-1, 512, 2, 2],
                          'layers.42': [-1, 512, 2, 2],
                          'layers.43': [-1, 512, 2, 2],
                          'avgpool': [-1, 512, 1, 1],
                          'flatten': [-1, 512, 7, 7],
                          'classifier.0': [-1, 25088],
                          'classifier.1': [-1, 4096],
                          'classifier.2': [-1, 4096],
                          'classifier.3': [-1, 4096],
                          'classifier.4': [-1, 4096],
                          'classifier.5': [-1, 4096],
                          'classifier.6': [-1, 4096],
                          'OUTPUT': [-1, 10]}

        self.origin_layer_names = {
            'layers': self.layers,
            'layers.0': self.layers[0],
            'layers.1': self.layers[1],
            'layers.2': self.layers[2],
            'layers.3': self.layers[3],
            'layers.4': self.layers[4],
            'layers.5': self.layers[5],
            'layers.6': self.layers[6],
            'layers.7': self.layers[7],
            'layers.8': self.layers[8],
            'layers.9': self.layers[9],
            'layers.10': self.layers[10],
            'layers.11': self.layers[11],
            'layers.12': self.layers[12],
            'layers.13': self.layers[13],
            'layers.14': self.layers[14],
            'layers.15': self.layers[15],
            'layers.16': self.layers[16],
            'layers.17': self.layers[17],
            'layers.18': self.layers[18],
            'layers.19': self.layers[19],
            'layers.20': self.layers[20],
            'layers.21': self.layers[21],
            'layers.22': self.layers[22],
            'layers.23': self.layers[23],
            'layers.24': self.layers[24],
            'layers.25': self.layers[25],
            'layers.26': self.layers[26],
            'layers.27': self.layers[27],
            'layers.28': self.layers[28],
            'layers.29': self.layers[29],
            'layers.30': self.layers[30],
            'layers.31': self.layers[31],
            'layers.32': self.layers[32],
            'layers.33': self.layers[33],
            'layers.34': self.layers[34],
            'layers.35': self.layers[35],
            'layers.36': self.layers[36],
            'layers.37': self.layers[37],
            'layers.38': self.layers[38],
            'layers.39': self.layers[39],
            'layers.40': self.layers[40],
            'layers.41': self.layers[41],
            'layers.42': self.layers[42],
            'layers.43': self.layers[43],
            'avgpool': self.avgpool,
            'flatten': self.flatten,
            'classifier': self.classifier,
            'classifier.0': self.classifier[0],
            'classifier.1': self.classifier[1],
            'classifier.2': self.classifier[2],
            'classifier.3': self.classifier[3],
            'classifier.4': self.classifier[4],
            'classifier.5': self.classifier[5],
            'classifier.6': self.classifier[6],
        }

        self.orders = {
            'layers.0': ["INPUT", "layers.1"],
            'layers.1': ["layers.0", "layers.2"],
            'layers.2': ["layers.1", "layers.3"],
            'layers.3': ["layers.2", "layers.4"],
            'layers.4': ["layers.3", "layers.5"],
            'layers.5': ["layers.4", "layers.6"],
            'layers.6': ["layers.5", "layers.7"],
            'layers.7': ["layers.6", "layers.8"],
            'layers.8': ["layers.7", "layers.9"],
            'layers.9': ["layers.8", "layers.10"],
            'layers.10': ["layers.9", "layers.11"],
            'layers.11': ["layers.10", "layers.12"],
            'layers.12': ["layers.11", "layers.13"],
            'layers.13': ["layers.12", "layers.14"],
            'layers.14': ["layers.13", "layers.15"],
            'layers.15': ["layers.14", "layers.16"],
            'layers.16': ["layers.15", "layers.17"],
            'layers.17': ["layers.16", "layers.18"],
            'layers.18': ["layers.17", "layers.19"],
            'layers.19': ["layers.18", "layers.20"],
            'layers.20': ["layers.19", "layers.21"],
            'layers.21': ["layers.20", "layers.22"],
            'layers.22': ["layers.21", "layers.23"],
            'layers.23': ["layers.22", "layers.24"],
            'layers.24': ["layers.23", "layers.25"],
            'layers.25': ["layers.24", "layers.26"],
            'layers.26': ["layers.25", "layers.27"],
            'layers.27': ["layers.26", "layers.28"],
            'layers.28': ["layers.27", "layers.29"],
            'layers.29': ["layers.28", "layers.30"],
            'layers.30': ["layers.29", "layers.31"],
            'layers.31': ["layers.30", "layers.32"],
            'layers.32': ["layers.31", "layers.33"],
            'layers.33': ["layers.32", "layers.34"],
            'layers.34': ["layers.33", "layers.35"],
            'layers.35': ["layers.34", "layers.36"],
            'layers.36': ["layers.35", "layers.37"],
            'layers.37': ["layers.36", "layers.38"],
            'layers.38': ["layers.37", "layers.39"],
            'layers.39': ["layers.38", "layers.40"],
            'layers.40': ["layers.39", "layers.41"],
            'layers.41': ["layers.40", "layers.42"],
            'layers.42': ["layers.41", "layers.43"],
            'layers.43': ["layers.42", "avgpool"],
            'avgpool': ["layers.43", "flatten"],
            'flatten': ["avgpool", "classifier.0"],
            'classifier.0': ["flatten", "classifier.1"],
            'classifier.1': ["classifier.0", "classifier.2"],
            'classifier.2': ["classifier.1", "classifier.3"],
            'classifier.3': ["classifier.2", "classifier.4"],
            'classifier.4': ["classifier.3", "classifier.5"],
            'classifier.5': ["classifier.4", "classifier.6"],
            'classifier.6': ["classifier.5", "OUTPUT"]
        }
        self.layer_names = {
            'layers': self.layers,
            'layers.0': self.layers[0],
            'layers.1': self.layers[1],
            'layers.2': self.layers[2],
            'layers.3': self.layers[3],
            'layers.4': self.layers[4],
            'layers.5': self.layers[5],
            'layers.6': self.layers[6],
            'layers.7': self.layers[7],
            'layers.8': self.layers[8],
            'layers.9': self.layers[9],
            'layers.10': self.layers[10],
            'layers.11': self.layers[11],
            'layers.12': self.layers[12],
            'layers.13': self.layers[13],
            'layers.14': self.layers[14],
            'layers.15': self.layers[15],
            'layers.16': self.layers[16],
            'layers.17': self.layers[17],
            'layers.18': self.layers[18],
            'layers.19': self.layers[19],
            'layers.20': self.layers[20],
            'layers.21': self.layers[21],
            'layers.22': self.layers[22],
            'layers.23': self.layers[23],
            'layers.24': self.layers[24],
            'layers.25': self.layers[25],
            'layers.26': self.layers[26],
            'layers.27': self.layers[27],
            'layers.28': self.layers[28],
            'layers.29': self.layers[29],
            'layers.30': self.layers[30],
            'layers.31': self.layers[31],
            'layers.32': self.layers[32],
            'layers.33': self.layers[33],
            'layers.34': self.layers[34],
            'layers.35': self.layers[35],
            'layers.36': self.layers[36],
            'layers.37': self.layers[37],
            'layers.38': self.layers[38],
            'layers.39': self.layers[39],
            'layers.40': self.layers[40],
            'layers.41': self.layers[41],
            'layers.42': self.layers[42],
            'layers.43': self.layers[43],
            'avgpool': self.avgpool,
            'flatten': self.flatten,
            'classifier': self.classifier,
            'classifier.0': self.classifier[0],
            'classifier.1': self.classifier[1],
            'classifier.2': self.classifier[2],
            'classifier.3': self.classifier[3],
            'classifier.4': self.classifier[4],
            'classifier.5': self.classifier[5],
            'classifier.6': self.classifier[6],
        }
        self.out_shapes = {
            "INPUT": [1, 3, 224, 224],
            'layers.0': [1, 64, 222, 222],
            'layers.1': [1, 64, 222, 222],
            'layers.2': [1, 64, 222, 222],
            'layers.3': [1, 64, 220, 220],
            'layers.4': [1, 64, 220, 220],
            'layers.5': [1, 64, 220, 220],
            'layers.6': [1, 64, 110, 110],
            'layers.7': [1, 128, 108, 108],
            'layers.8': [1, 128, 108, 108],
            'layers.9': [1, 128, 108, 108],
            'layers.10': [1, 128, 106, 106],
            'layers.11': [1, 128, 106, 106],
            'layers.12': [1, 128, 106, 106],
            'layers.13': [1, 128, 53, 53],
            'layers.14': [1, 256, 51, 51],
            'layers.15': [1, 256, 51, 51],
            'layers.16': [1, 256, 51, 51],
            'layers.17': [1, 256, 49, 49],
            'layers.18': [1, 256, 49, 49],
            'layers.19': [1, 256, 49, 49],
            'layers.20': [1, 256, 47, 47],
            'layers.21': [1, 256, 47, 47],
            'layers.22': [1, 256, 47, 47],
            'layers.23': [1, 256, 23, 23],
            'layers.24': [1, 512, 21, 21],
            'layers.25': [1, 512, 21, 21],
            'layers.26': [1, 512, 21, 21],
            'layers.27': [1, 512, 19, 19],
            'layers.28': [1, 512, 19, 19],
            'layers.29': [1, 512, 19, 19],
            'layers.30': [1, 512, 17, 17],
            'layers.31': [1, 512, 17, 17],
            'layers.32': [1, 512, 17, 17],
            'layers.33': [1, 512, 8, 8],
            'layers.34': [1, 512, 6, 6],
            'layers.35': [1, 512, 6, 6],
            'layers.36': [1, 512, 6, 6],
            'layers.37': [1, 512, 4, 4],
            'layers.38': [1, 512, 4, 4],
            'layers.39': [1, 512, 4, 4],
            'layers.40': [1, 512, 2, 2],
            'layers.41': [1, 512, 2, 2],
            'layers.42': [1, 512, 2, 2],
            'layers.43': [1, 512, 1, 1],
            'avgpool': [1, 512, 7, 7],
            'flatten': [1, 25088],
            'classifier.0': [1, 4096],
            'classifier.1': [1, 4096],
            'classifier.2': [1, 4096],
            'classifier.3': [1, 4096],
            'classifier.4': [1, 4096],
            'classifier.5': [1, 4096],
            'classifier.6': [1, 10],
            "OUTPUT": [1, 10]
        }

    def construct(self, x):
        # N x 3 x 224 x 224
        x = self.layers(x)
        x = self.avgpool(x)
        if self.include_top:
            # N x 512 x 7 x 7
            x = self.flatten(x)
            # N x 512*7*7
            x = self.classifier(x)
        return x

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):

        if 'layers' == layer_name:
            self.layers = new_layer
            self.layer_names['layers'] = new_layer
            self.origin_layer_names['layers'] = new_layer

        elif 'layers.0' == layer_name:
            self.layers[0] = new_layer
            self.layer_names['layers.0'] = new_layer
            self.origin_layer_names['layers.0'] = new_layer

        elif 'layers.1' == layer_name:
            self.layers[1] = new_layer
            self.layer_names['layers.1'] = new_layer
            self.origin_layer_names['layers.1'] = new_layer

        elif 'layers.2' == layer_name:
            self.layers[2] = new_layer
            self.layer_names['layers.2'] = new_layer
            self.origin_layer_names['layers.2'] = new_layer

        elif 'layers.3' == layer_name:
            self.layers[3] = new_layer
            self.layer_names['layers.3'] = new_layer
            self.origin_layer_names['layers.3'] = new_layer

        elif 'layers.4' == layer_name:
            self.layers[4] = new_layer
            self.layer_names['layers.4'] = new_layer
            self.origin_layer_names['layers.4'] = new_layer

        elif 'layers.5' == layer_name:
            self.layers[5] = new_layer
            self.layer_names['layers.5'] = new_layer
            self.origin_layer_names['layers.5'] = new_layer

        elif 'layers.6' == layer_name:
            self.layers[6] = new_layer
            self.layer_names['layers.6'] = new_layer
            self.origin_layer_names['layers.6'] = new_layer

        elif 'layers.7' == layer_name:
            self.layers[7] = new_layer
            self.layer_names['layers.7'] = new_layer
            self.origin_layer_names['layers.7'] = new_layer

        elif 'layers.8' == layer_name:
            self.layers[8] = new_layer
            self.layer_names['layers.8'] = new_layer
            self.origin_layer_names['layers.8'] = new_layer

        elif 'layers.9' == layer_name:
            self.layers[9] = new_layer
            self.layer_names['layers.9'] = new_layer
            self.origin_layer_names['layers.9'] = new_layer

        elif 'layers.10' == layer_name:
            self.layers[10] = new_layer
            self.layer_names['layers.10'] = new_layer
            self.origin_layer_names['layers.10'] = new_layer

        elif 'layers.11' == layer_name:
            self.layers[11] = new_layer
            self.layer_names['layers.11'] = new_layer
            self.origin_layer_names['layers.11'] = new_layer

        elif 'layers.12' == layer_name:
            self.layers[12] = new_layer
            self.layer_names['layers.12'] = new_layer
            self.origin_layer_names['layers.12'] = new_layer

        elif 'layers.13' == layer_name:
            self.layers[13] = new_layer
            self.layer_names['layers.13'] = new_layer
            self.origin_layer_names['layers.13'] = new_layer

        elif 'layers.14' == layer_name:
            self.layers[14] = new_layer
            self.layer_names['layers.14'] = new_layer
            self.origin_layer_names['layers.14'] = new_layer

        elif 'layers.15' == layer_name:
            self.layers[15] = new_layer
            self.layer_names['layers.15'] = new_layer
            self.origin_layer_names['layers.15'] = new_layer

        elif 'layers.16' == layer_name:
            self.layers[16] = new_layer
            self.layer_names['layers.16'] = new_layer
            self.origin_layer_names['layers.16'] = new_layer

        elif 'layers.17' == layer_name:
            self.layers[17] = new_layer
            self.layer_names['layers.17'] = new_layer
            self.origin_layer_names['layers.17'] = new_layer

        elif 'layers.18' == layer_name:
            self.layers[18] = new_layer
            self.layer_names['layers.18'] = new_layer
            self.origin_layer_names['layers.18'] = new_layer

        elif 'layers.19' == layer_name:
            self.layers[19] = new_layer
            self.layer_names['layers.19'] = new_layer
            self.origin_layer_names['layers.19'] = new_layer

        elif 'layers.20' == layer_name:
            self.layers[20] = new_layer
            self.layer_names['layers.20'] = new_layer
            self.origin_layer_names['layers.20'] = new_layer

        elif 'layers.21' == layer_name:
            self.layers[21] = new_layer
            self.layer_names['layers.21'] = new_layer
            self.origin_layer_names['layers.21'] = new_layer

        elif 'layers.22' == layer_name:
            self.layers[22] = new_layer
            self.layer_names['layers.22'] = new_layer
            self.origin_layer_names['layers.22'] = new_layer

        elif 'layers.23' == layer_name:
            self.layers[23] = new_layer
            self.layer_names['layers.23'] = new_layer
            self.origin_layer_names['layers.23'] = new_layer

        elif 'layers.24' == layer_name:
            self.layers[24] = new_layer
            self.layer_names['layers.24'] = new_layer
            self.origin_layer_names['layers.24'] = new_layer

        elif 'layers.25' == layer_name:
            self.layers[25] = new_layer
            self.layer_names['layers.25'] = new_layer
            self.origin_layer_names['layers.25'] = new_layer

        elif 'layers.26' == layer_name:
            self.layers[26] = new_layer
            self.layer_names['layers.26'] = new_layer
            self.origin_layer_names['layers.26'] = new_layer

        elif 'layers.27' == layer_name:
            self.layers[27] = new_layer
            self.layer_names['layers.27'] = new_layer
            self.origin_layer_names['layers.27'] = new_layer

        elif 'layers.28' == layer_name:
            self.layers[28] = new_layer
            self.layer_names['layers.28'] = new_layer
            self.origin_layer_names['layers.28'] = new_layer

        elif 'layers.29' == layer_name:
            self.layers[29] = new_layer
            self.layer_names['layers.29'] = new_layer
            self.origin_layer_names['layers.29'] = new_layer

        elif 'layers.30' == layer_name:
            self.layers[30] = new_layer
            self.layer_names['layers.30'] = new_layer
            self.origin_layer_names['layers.30'] = new_layer

        elif 'layers.31' == layer_name:
            self.layers[31] = new_layer
            self.layer_names['layers.31'] = new_layer
            self.origin_layer_names['layers.31'] = new_layer

        elif 'layers.32' == layer_name:
            self.layers[32] = new_layer
            self.layer_names['layers.32'] = new_layer
            self.origin_layer_names['layers.32'] = new_layer

        elif 'layers.33' == layer_name:
            self.layers[33] = new_layer
            self.layer_names['layers.33'] = new_layer
            self.origin_layer_names['layers.33'] = new_layer

        elif 'layers.34' == layer_name:
            self.layers[34] = new_layer
            self.layer_names['layers.34'] = new_layer
            self.origin_layer_names['layers.34'] = new_layer

        elif 'layers.35' == layer_name:
            self.layers[35] = new_layer
            self.layer_names['layers.35'] = new_layer
            self.origin_layer_names['layers.35'] = new_layer

        elif 'layers.36' == layer_name:
            self.layers[36] = new_layer
            self.layer_names['layers.36'] = new_layer
            self.origin_layer_names['layers.36'] = new_layer

        elif 'layers.37' == layer_name:
            self.layers[37] = new_layer
            self.layer_names['layers.37'] = new_layer
            self.origin_layer_names['layers.37'] = new_layer

        elif 'layers.38' == layer_name:
            self.layers[38] = new_layer
            self.layer_names['layers.38'] = new_layer
            self.origin_layer_names['layers.38'] = new_layer

        elif 'layers.39' == layer_name:
            self.layers[39] = new_layer
            self.layer_names['layers.39'] = new_layer
            self.origin_layer_names['layers.39'] = new_layer

        elif 'layers.40' == layer_name:
            self.layers[40] = new_layer
            self.layer_names['layers.40'] = new_layer
            self.origin_layer_names['layers.40'] = new_layer

        elif 'layers.41' == layer_name:
            self.layers[41] = new_layer
            self.layer_names['layers.41'] = new_layer
            self.origin_layer_names['layers.41'] = new_layer

        elif 'layers.42' == layer_name:
            self.layers[42] = new_layer
            self.layer_names['layers.42'] = new_layer
            self.origin_layer_names['layers.42'] = new_layer

        elif 'layers.43' == layer_name:
            self.layers[43] = new_layer
            self.layer_names['layers.43'] = new_layer
            self.origin_layer_names['layers.43'] = new_layer

        elif 'avgpool' == layer_name:
            self.avgpool = new_layer
            self.layer_names['avgpool'] = new_layer
            self.origin_layer_names['avgpool'] = new_layer

        elif 'flatten' == layer_name:
            self.flatten = new_layer
            self.layer_names['flatten'] = new_layer
            self.origin_layer_names['flatten'] = new_layer

        elif 'classifier' == layer_name:
            self.classifier = new_layer
            self.layer_names['classifier'] = new_layer
            self.origin_layer_names['classifier'] = new_layer

        elif 'classifier.0' == layer_name:
            self.classifier[0] = new_layer
            self.layer_names['classifier.0'] = new_layer
            self.origin_layer_names['classifier.0'] = new_layer

        elif 'classifier.1' == layer_name:
            self.classifier[1] = new_layer
            self.layer_names['classifier.1'] = new_layer
            self.origin_layer_names['classifier.1'] = new_layer

        elif 'classifier.2' == layer_name:
            self.classifier[2] = new_layer
            self.layer_names['classifier.2'] = new_layer
            self.origin_layer_names['classifier.2'] = new_layer

        elif 'classifier.3' == layer_name:
            self.classifier[3] = new_layer
            self.layer_names['classifier.3'] = new_layer
            self.origin_layer_names['classifier.3'] = new_layer

        elif 'classifier.4' == layer_name:
            self.classifier[4] = new_layer
            self.layer_names['classifier.4'] = new_layer
            self.origin_layer_names['classifier.4'] = new_layer

        elif 'classifier.5' == layer_name:
            self.classifier[5] = new_layer
            self.layer_names['classifier.5'] = new_layer
            self.origin_layer_names['classifier.5'] = new_layer

        elif 'classifier.6' == layer_name:
            self.classifier[6] = new_layer
            self.layer_names['classifier.6'] = new_layer
            self.origin_layer_names['classifier.6'] = new_layer

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

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes=1000, args=None, phase="train", **kwargs):
    """
    Get Vgg16 neural network with Batch Normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.
        args(namespace): param for net init.
        phase(str): train or test mode.

    Returns:
        Cell, cell instance of Vgg16 neural network with Batch Normalization.

    Examples:
        >>> vgg16(num_classes=1000, args=args, **kwargs)
    """
    num_classes = 10
    args = {'T_max': 90,
            'batch_norm': True,
            'batch_size': 64,
            'buffer_size': 10,
            'checkpoint_url': '',
            'ckpt_file': '',
            'ckpt_interval': 5,
            'ckpt_path': 'outputs/',
            'config_path': '/data/MSTest/models/vgg16/model_utils/../cifar10_config.yaml',
            'data_dir': '/data/MSTest/data',
            'data_path': '/cache/data',
            'data_url': '',
            'dataset': 'cifar10',
            'dataset_name': 'cifar10',
            'device_target': 'GPU',
            'enable_modelarts': False,
            'eta_min': 0.0,
            'file_format': 'MINDIR',
            'file_name': 'vgg16',
            'graph_ckpt': 1,
            'group_size': 1,
            'has_bias': False,
            'has_dropout': False,
            'image_size': (224, 224),
            'initialize_mode': 'XavierUniform',
            'is_distributed': 0,
            'is_save_on_master': 1,
            'keep_checkpoint_max': 10,
            'label_dir': '',
            'label_smooth': 0,
            'label_smooth_factor': 0,
            'load_path': '/cache/checkpoint_path',
            'log_interval': 100,
            'log_path': 'outputs/',
            'loss_scale': 1.0,
            'lr': 0.01,
            'lr_epochs': '30,60,90,120',
            'lr_gamma': 0.1,
            'lr_init': 0.01,
            'lr_max': 0.1,
            'lr_scheduler': 'step',
            'max_epoch': 70,
            'modelarts_dataset_unzip_name': 'cifar10',
            'momentum': 0.9,
            'need_modelarts_dataset_unzip': True,
            'num_classes': 10,
            'output_path': '/cache/train',
            'pad_mode': 'same',
            'padding': 0,
            'per_batch_size': 32,
            'pre_trained': '',
            'rank': 0,
            'result_dir': '',
            'result_path': './preprocess_Result/',
            'train_url': '',
            'warmup_epochs': 5,
            'weight_decay': 0.0005}

    net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=1, phase=phase, **kwargs)
    return net
