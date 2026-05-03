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

from network.cv.unet.src.unet_medical.unet_parts import DoubleConv, Down, Up1, Up2, Up3, Up4, OutConv
import mindspore.nn as nn
import mindspore
import numpy as np
import mindspore.ops.operations as F
from mindspore.ops import functional as F2

class UNetMedical(nn.Cell):
    def __init__(self, n_channels, n_classes):
        super(UNetMedical, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up1(1024, 512)
        self.up2 = Up2(512, 256)
        self.up3 = Up3(256, 128)
        self.up4 = Up4(128, 64)
        self.outc = OutConv(64, n_classes)


        self.in_shapes = {
             'INPUT': [-1, 1, 572, 572],
             'inc.double_conv.0': [-1, 1, 572, 572],
             'inc.double_conv.1': [-1, 64, 570, 570],
             'inc.double_conv.2': [-1, 64, 570, 570],
             'inc.double_conv.3': [-1, 64, 568, 568],
             'down1.maxpool_conv.0': [-1, 64, 568, 568],
             'down1.maxpool_conv.1.double_conv.0': [-1, 64, 284, 284],
             'down1.maxpool_conv.1.double_conv.1': [-1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.2': [-1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.3': [-1, 128, 280, 280],
             'down2.maxpool_conv.0': [-1, 128, 280, 280],
             'down2.maxpool_conv.1.double_conv.0': [-1, 128, 140, 140],
             'down2.maxpool_conv.1.double_conv.1': [-1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.2': [-1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.3': [-1, 256, 136, 136],
             'down3.maxpool_conv.0': [-1, 256, 136, 136],
             'down3.maxpool_conv.1.double_conv.0': [-1, 256, 68, 68],
             'down3.maxpool_conv.1.double_conv.1': [-1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.2': [-1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.3': [-1, 512, 64, 64],
             'down4.maxpool_conv.0': [-1, 512, 64, 64],
             'down4.maxpool_conv.1.double_conv.0': [-1, 512, 32, 32],
             'down4.maxpool_conv.1.double_conv.1': [-1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.2': [-1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.3': [-1, 1024, 28, 28],
             'up1.up': [-1, 1024, 28, 28],
             'up1.relu': [-1, 512, 56, 56],
             'up1.center_crop': [-1, 512, 64, 64],
             'up1.conv.double_conv.0': [-1, 1024, 56, 56],
             'up1.conv.double_conv.1': [-1, 512, 54, 54],
             'up1.conv.double_conv.2': [-1, 512, 54, 54],
             'up1.conv.double_conv.3': [-1, 512, 52, 52],
             'up2.up': [-1, 512, 52, 52],
             'up2.relu': [-1, 256, 104, 104],
             'up2.center_crop': [-1, 256, 136, 136],
             'up2.conv.double_conv.0': [-1, 512, 104, 104],
             'up2.conv.double_conv.1': [-1, 256, 102, 102],
             'up2.conv.double_conv.2': [-1, 256, 102, 102],
             'up2.conv.double_conv.3': [-1, 256, 100, 100],
             'up3.up': [-1, 256, 100, 100],
             'up3.relu': [-1, 128, 200, 200],
             'up3.center_crop': [-1, 128, 280, 280],
             'up3.conv.double_conv.0': [-1, 256, 200, 200],
             'up3.conv.double_conv.1': [-1, 128, 198, 198],
             'up3.conv.double_conv.2': [-1, 128, 198, 198],
             'up3.conv.double_conv.3': [-1, 128, 196, 196],
             'up4.up': [-1, 128, 196, 196],
             'up4.relu': [-1, 64, 392, 392],
             'up4.center_crop': [-1, 64, 568, 568],
             'up4.conv.double_conv.0': [-1, 128, 392, 392],
             'up4.conv.double_conv.1': [-1, 64, 390, 390],
             'up4.conv.double_conv.2': [-1, 64, 390, 390],
             'up4.conv.double_conv.3': [-1, 64, 388, 388],
             'outc.conv': [-1, 64, 388, 388],
            'OUTPUT': [-1, 64, 388, 388]
        }

        self.out_shapes = {
             'INPUT': [-1, 1, 572, 572],
             'inc.double_conv.0': [-1, 64, 570, 570],
             'inc.double_conv.1': [-1, 64, 570, 570],
             'inc.double_conv.2': [-1, 64, 568, 568],
             'inc.double_conv.3': [-1, 64, 568, 568],
             'down1.maxpool_conv.0': [-1, 64, 284, 284],
             'down1.maxpool_conv.1.double_conv.0': [-1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.1': [-1, 128, 282, 282],
             'down1.maxpool_conv.1.double_conv.2': [-1, 128, 280, 280],
             'down1.maxpool_conv.1.double_conv.3': [-1, 128, 280, 280],
             'down2.maxpool_conv.0': [-1, 128, 140, 140],
             'down2.maxpool_conv.1.double_conv.0': [-1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.1': [-1, 256, 138, 138],
             'down2.maxpool_conv.1.double_conv.2': [-1, 256, 136, 136],
             'down2.maxpool_conv.1.double_conv.3': [-1, 256, 136, 136],
             'down3.maxpool_conv.0': [-1, 256, 68, 68],
             'down3.maxpool_conv.1.double_conv.0': [-1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.1': [-1, 512, 66, 66],
             'down3.maxpool_conv.1.double_conv.2': [-1, 512, 64, 64],
             'down3.maxpool_conv.1.double_conv.3': [-1, 512, 64, 64],
             'down4.maxpool_conv.0': [-1, 512, 32, 32],
             'down4.maxpool_conv.1.double_conv.0': [-1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.1': [-1, 1024, 30, 30],
             'down4.maxpool_conv.1.double_conv.2': [-1, 1024, 28, 28],
             'down4.maxpool_conv.1.double_conv.3': [-1, 1024, 28, 28],
             'up1.up': [-1, 512, 56, 56],
             'up1.relu': [-1, 512, 56, 56],
             'up1.center_crop': [-1, 512, 56, 56],
             'up1.conv.double_conv.0': [-1, 512, 54, 54],
             'up1.conv.double_conv.1': [-1, 512, 54, 54],
             'up1.conv.double_conv.2': [-1, 512, 52, 52],
             'up1.conv.double_conv.3': [-1, 512, 52, 52],
             'up2.up': [-1, 256, 104, 104],
             'up2.relu': [-1, 256, 104, 104],
             'up2.center_crop': [-1, 256, 104, 104],
             'up2.conv.double_conv.0': [-1, 256, 102, 102],
             'up2.conv.double_conv.1': [-1, 256, 102, 102],
             'up2.conv.double_conv.2': [-1, 256, 100, 100],
             'up2.conv.double_conv.3': [-1, 256, 100, 100],
             'up3.up': [-1, 128, 200, 200],
             'up3.relu': [-1, 128, 200, 200],
             'up3.center_crop': [-1, 128, 200, 200],
             'up3.conv.double_conv.0': [-1, 128, 198, 198],
             'up3.conv.double_conv.1': [-1, 128, 198, 198],
             'up3.conv.double_conv.2': [-1, 128, 196, 196],
             'up3.conv.double_conv.3': [-1, 128, 196, 196],
             'up4.up': [-1, 64, 392, 392],
             'up4.relu': [-1, 64, 392, 392],
             'up4.center_crop': [-1, 64, 392, 392],
             'up4.conv.double_conv.0': [-1, 64, 390, 390],
             'up4.conv.double_conv.1': [-1, 64, 390, 390],
             'up4.conv.double_conv.2': [-1, 64, 388, 388],
             'up4.conv.double_conv.3': [-1, 64, 388, 388],
             'outc.conv': [-1, 2, 388, 388],
            'OUTPUT': [-1, 2, 388, 388]
        }


        self.orders = {
            "inc.double_conv.0": ["INPUT", "inc.double_conv.1"],
            "inc.double_conv.1": ["inc.double_conv.0", "inc.double_conv.2"],
            "inc.double_conv.2": ["inc.double_conv.1", "inc.double_conv.3"],
            "inc.double_conv.3": ["inc.double_conv.2", ["down1.maxpool_conv.0", "up4.center_crop"]],

            "down1.maxpool_conv.0": ["inc.double_conv.3", "down1.maxpool_conv.1.double_conv.0"],
            "down1.maxpool_conv.1.double_conv.0": ["down1.maxpool_conv.0", "down1.maxpool_conv.1.double_conv.1"],
            "down1.maxpool_conv.1.double_conv.1": ["down1.maxpool_conv.1.double_conv.0",
                                                   "down1.maxpool_conv.1.double_conv.2"],
            "down1.maxpool_conv.1.double_conv.2": ["down1.maxpool_conv.1.double_conv.1",
                                                   "down1.maxpool_conv.1.double_conv.3"],
            "down1.maxpool_conv.1.double_conv.3": ["down1.maxpool_conv.1.double_conv.2",
                                                   ["down2.maxpool_conv.0", "up3.center_crop"]],

            "down2.maxpool_conv.0": ["down1.maxpool_conv.1.double_conv.3", "down2.maxpool_conv.1.double_conv.0"],
            "down2.maxpool_conv.1.double_conv.0": ["down2.maxpool_conv.0", "down2.maxpool_conv.1.double_conv.1"],
            "down2.maxpool_conv.1.double_conv.1": ["down2.maxpool_conv.1.double_conv.0",
                                                   "down2.maxpool_conv.1.double_conv.2"],
            "down2.maxpool_conv.1.double_conv.2": ["down2.maxpool_conv.1.double_conv.1",
                                                   "down2.maxpool_conv.1.double_conv.3"],
            "down2.maxpool_conv.1.double_conv.3": ["down2.maxpool_conv.1.double_conv.2",
                                                   ["down3.maxpool_conv.0", "up2.center_crop"]],

            "down3.maxpool_conv.0": ["down2.maxpool_conv.1.double_conv.3", "down3.maxpool_conv.1.double_conv.0"],
            "down3.maxpool_conv.1.double_conv.0": ["down3.maxpool_conv.0", "down3.maxpool_conv.1.double_conv.1"],
            "down3.maxpool_conv.1.double_conv.1": ["down3.maxpool_conv.1.double_conv.0",
                                                   "down3.maxpool_conv.1.double_conv.2"],
            "down3.maxpool_conv.1.double_conv.2": ["down3.maxpool_conv.1.double_conv.1",
                                                   "down3.maxpool_conv.1.double_conv.3"],
            "down3.maxpool_conv.1.double_conv.3": ["down3.maxpool_conv.1.double_conv.2",
                                                   ["down4.maxpool_conv.0", "up1.center_crop"]],

            "down4.maxpool_conv.0": ["down3.maxpool_conv.1.double_conv.3", "down4.maxpool_conv.1.double_conv.0"],
            "down4.maxpool_conv.1.double_conv.0": ["down4.maxpool_conv.0", "down4.maxpool_conv.1.double_conv.1"],
            "down4.maxpool_conv.1.double_conv.1": ["down4.maxpool_conv.1.double_conv.0",
                                                   "down4.maxpool_conv.1.double_conv.2"],
            "down4.maxpool_conv.1.double_conv.2": ["down4.maxpool_conv.1.double_conv.1",
                                                   "down4.maxpool_conv.1.double_conv.3"],
            "down4.maxpool_conv.1.double_conv.3": ["down4.maxpool_conv.1.double_conv.2", "up1.up"],

            "up1.up": ["down4.maxpool_conv.1.double_conv.3", "up1.relu"],
            "up1.relu": ["up1.up", "up1.conv.double_conv.0"],
            "up1.center_crop": ["down3.maxpool_conv.1.double_conv.3", "up1.conv.double_conv.0"],
            "up1.conv.double_conv.0": [["up1.relu", "up1.center_crop"], "up1.conv.double_conv.1"],
            "up1.conv.double_conv.1": ["up1.conv.double_conv.0", "up1.conv.double_conv.2"],
            "up1.conv.double_conv.2": ["up1.conv.double_conv.1", "up1.conv.double_conv.3"],
            "up1.conv.double_conv.3": ["up1.conv.double_conv.2", "up2.up"],

            "up2.up": ["up1.conv.double_conv.3", "up2.relu"],
            "up2.relu": ["up2.up", "up2.conv.double_conv.0"],
            "up2.center_crop": ["down2.maxpool_conv.1.double_conv.3", "up2.conv.double_conv.0"],
            "up2.conv.double_conv.0": [["up2.relu", "up2.center_crop"], "up2.conv.double_conv.1"],
            "up2.conv.double_conv.1": ["up2.conv.double_conv.0", "up2.conv.double_conv.2"],
            "up2.conv.double_conv.2": ["up2.conv.double_conv.1", "up2.conv.double_conv.3"],
            "up2.conv.double_conv.3": ["up2.conv.double_conv.2", "up3.up"],

            "up3.up": ["up2.conv.double_conv.3", "up3.relu"],
            "up3.relu": ["up3.up", "up3.conv.double_conv.0"],
            "up3.center_crop": ["down1.maxpool_conv.1.double_conv.3", "up3.conv.double_conv.0"],
            "up3.conv.double_conv.0": [["up3.relu", "up3.center_crop"], "up3.conv.double_conv.1"],
            "up3.conv.double_conv.1": ["up3.conv.double_conv.0", "up3.conv.double_conv.2"],
            "up3.conv.double_conv.2": ["up3.conv.double_conv.1", "up3.conv.double_conv.3"],
            "up3.conv.double_conv.3": ["up3.conv.double_conv.2", "up4.up"],

            "up4.up": ["up3.conv.double_conv.3", "up4.relu"],
            "up4.relu": ["up4.up", "up4.conv.double_conv.0"],
            "up4.center_crop": ["inc.double_conv.3", "up4.conv.double_conv.0"],
            "up4.conv.double_conv.0": [["up4.relu", "up4.center_crop"], "up4.conv.double_conv.1"],
            "up4.conv.double_conv.1": ["up4.conv.double_conv.0", "up4.conv.double_conv.2"],
            "up4.conv.double_conv.2": ["up4.conv.double_conv.1", "up4.conv.double_conv.3"],
            "up4.conv.double_conv.3": ["up4.conv.double_conv.2", "outc.conv"],

            "outc.conv": ["up4.conv.double_conv.3", "OUTPUT"],
        }

        self.layer_names = {
            "inc": self.inc,
            "inc.double_conv": self.inc.double_conv,
            "inc.double_conv.0": self.inc.double_conv[0],
            "inc.double_conv.1": self.inc.double_conv[1],
            "inc.double_conv.2": self.inc.double_conv[2],
            "inc.double_conv.3": self.inc.double_conv[3],
            "down1": self.down1,
            "down1.maxpool_conv": self.down1.maxpool_conv,
            "down1.maxpool_conv.0": self.down1.maxpool_conv[0],
            "down1.maxpool_conv.1": self.down1.maxpool_conv[1],
            "down1.maxpool_conv.1.double_conv": self.down1.maxpool_conv[1].double_conv,
            "down1.maxpool_conv.1.double_conv.0": self.down1.maxpool_conv[1].double_conv[0],
            "down1.maxpool_conv.1.double_conv.1": self.down1.maxpool_conv[1].double_conv[1],
            "down1.maxpool_conv.1.double_conv.2": self.down1.maxpool_conv[1].double_conv[2],
            "down1.maxpool_conv.1.double_conv.3": self.down1.maxpool_conv[1].double_conv[3],
            "down2": self.down2,
            "down2.maxpool_conv": self.down2.maxpool_conv,
            "down2.maxpool_conv.0": self.down2.maxpool_conv[0],
            "down2.maxpool_conv.1": self.down2.maxpool_conv[1],
            "down2.maxpool_conv.1.double_conv": self.down2.maxpool_conv[1].double_conv,
            "down2.maxpool_conv.1.double_conv.0": self.down2.maxpool_conv[1].double_conv[0],
            "down2.maxpool_conv.1.double_conv.1": self.down2.maxpool_conv[1].double_conv[1],
            "down2.maxpool_conv.1.double_conv.2": self.down2.maxpool_conv[1].double_conv[2],
            "down2.maxpool_conv.1.double_conv.3": self.down2.maxpool_conv[1].double_conv[3],
            "down3": self.down3,
            "down3.maxpool_conv": self.down3.maxpool_conv,
            "down3.maxpool_conv.0": self.down3.maxpool_conv[0],
            "down3.maxpool_conv.1": self.down3.maxpool_conv[1],
            "down3.maxpool_conv.1.double_conv": self.down3.maxpool_conv[1].double_conv,
            "down3.maxpool_conv.1.double_conv.0": self.down3.maxpool_conv[1].double_conv[0],
            "down3.maxpool_conv.1.double_conv.1": self.down3.maxpool_conv[1].double_conv[1],
            "down3.maxpool_conv.1.double_conv.2": self.down3.maxpool_conv[1].double_conv[2],
            "down3.maxpool_conv.1.double_conv.3": self.down3.maxpool_conv[1].double_conv[3],
            "down4": self.down4,
            "down4.maxpool_conv": self.down4.maxpool_conv,
            "down4.maxpool_conv.0": self.down4.maxpool_conv[0],
            "down4.maxpool_conv.1": self.down4.maxpool_conv[1],
            "down4.maxpool_conv.1.double_conv": self.down4.maxpool_conv[1].double_conv,
            "down4.maxpool_conv.1.double_conv.0": self.down4.maxpool_conv[1].double_conv[0],
            "down4.maxpool_conv.1.double_conv.1": self.down4.maxpool_conv[1].double_conv[1],
            "down4.maxpool_conv.1.double_conv.2": self.down4.maxpool_conv[1].double_conv[2],
            "down4.maxpool_conv.1.double_conv.3": self.down4.maxpool_conv[1].double_conv[3],
            "up1": self.up1,
            "up1.center_crop": self.up1.center_crop,
            "up1.conv": self.up1.conv,
            "up1.conv.double_conv": self.up1.conv.double_conv,
            "up1.conv.double_conv.0": self.up1.conv.double_conv[0],
            "up1.conv.double_conv.1": self.up1.conv.double_conv[1],
            "up1.conv.double_conv.2": self.up1.conv.double_conv[2],
            "up1.conv.double_conv.3": self.up1.conv.double_conv[3],
            "up1.up": self.up1.up,
            "up1.relu": self.up1.relu,
            "up2": self.up2,
            "up2.center_crop": self.up2.center_crop,
            "up2.conv": self.up2.conv,
            "up2.conv.double_conv": self.up2.conv.double_conv,
            "up2.conv.double_conv.0": self.up2.conv.double_conv[0],
            "up2.conv.double_conv.1": self.up2.conv.double_conv[1],
            "up2.conv.double_conv.2": self.up2.conv.double_conv[2],
            "up2.conv.double_conv.3": self.up2.conv.double_conv[3],
            "up2.up": self.up2.up,
            "up2.relu": self.up2.relu,
            "up3": self.up3,
            "up3.center_crop": self.up3.center_crop,
            "up3.conv": self.up3.conv,
            "up3.conv.double_conv": self.up3.conv.double_conv,
            "up3.conv.double_conv.0": self.up3.conv.double_conv[0],
            "up3.conv.double_conv.1": self.up3.conv.double_conv[1],
            "up3.conv.double_conv.2": self.up3.conv.double_conv[2],
            "up3.conv.double_conv.3": self.up3.conv.double_conv[3],
            "up3.up": self.up3.up,
            "up3.relu": self.up3.relu,
            "up4": self.up4,
            "up4.center_crop": self.up4.center_crop,
            "up4.conv": self.up4.conv,
            "up4.conv.double_conv": self.up4.conv.double_conv,
            "up4.conv.double_conv.0": self.up4.conv.double_conv[0],
            "up4.conv.double_conv.1": self.up4.conv.double_conv[1],
            "up4.conv.double_conv.2": self.up4.conv.double_conv[2],
            "up4.conv.double_conv.3": self.up4.conv.double_conv[3],
            "up4.up": self.up4.up,
            "up4.relu": self.up4.relu,
            "outc": self.outc,
            "outc.conv": self.outc.conv,
        }

        self.origin_layer_names = {
            "inc": self.inc,
            "inc.double_conv": self.inc.double_conv,
            "inc.double_conv.0": self.inc.double_conv[0],
            "inc.double_conv.1": self.inc.double_conv[1],
            "inc.double_conv.2": self.inc.double_conv[2],
            "inc.double_conv.3": self.inc.double_conv[3],
            "down1": self.down1,
            "down1.maxpool_conv": self.down1.maxpool_conv,
            "down1.maxpool_conv.0": self.down1.maxpool_conv[0],
            "down1.maxpool_conv.1": self.down1.maxpool_conv[1],
            "down1.maxpool_conv.1.double_conv": self.down1.maxpool_conv[1].double_conv,
            "down1.maxpool_conv.1.double_conv.0": self.down1.maxpool_conv[1].double_conv[0],
            "down1.maxpool_conv.1.double_conv.1": self.down1.maxpool_conv[1].double_conv[1],
            "down1.maxpool_conv.1.double_conv.2": self.down1.maxpool_conv[1].double_conv[2],
            "down1.maxpool_conv.1.double_conv.3": self.down1.maxpool_conv[1].double_conv[3],
            "down2": self.down2,
            "down2.maxpool_conv": self.down2.maxpool_conv,
            "down2.maxpool_conv.0": self.down2.maxpool_conv[0],
            "down2.maxpool_conv.1": self.down2.maxpool_conv[1],
            "down2.maxpool_conv.1.double_conv": self.down2.maxpool_conv[1].double_conv,
            "down2.maxpool_conv.1.double_conv.0": self.down2.maxpool_conv[1].double_conv[0],
            "down2.maxpool_conv.1.double_conv.1": self.down2.maxpool_conv[1].double_conv[1],
            "down2.maxpool_conv.1.double_conv.2": self.down2.maxpool_conv[1].double_conv[2],
            "down2.maxpool_conv.1.double_conv.3": self.down2.maxpool_conv[1].double_conv[3],
            "down3": self.down3,
            "down3.maxpool_conv": self.down3.maxpool_conv,
            "down3.maxpool_conv.0": self.down3.maxpool_conv[0],
            "down3.maxpool_conv.1": self.down3.maxpool_conv[1],
            "down3.maxpool_conv.1.double_conv": self.down3.maxpool_conv[1].double_conv,
            "down3.maxpool_conv.1.double_conv.0": self.down3.maxpool_conv[1].double_conv[0],
            "down3.maxpool_conv.1.double_conv.1": self.down3.maxpool_conv[1].double_conv[1],
            "down3.maxpool_conv.1.double_conv.2": self.down3.maxpool_conv[1].double_conv[2],
            "down3.maxpool_conv.1.double_conv.3": self.down3.maxpool_conv[1].double_conv[3],
            "down4": self.down4,
            "down4.maxpool_conv": self.down4.maxpool_conv,
            "down4.maxpool_conv.0": self.down4.maxpool_conv[0],
            "down4.maxpool_conv.1": self.down4.maxpool_conv[1],
            "down4.maxpool_conv.1.double_conv": self.down4.maxpool_conv[1].double_conv,
            "down4.maxpool_conv.1.double_conv.0": self.down4.maxpool_conv[1].double_conv[0],
            "down4.maxpool_conv.1.double_conv.1": self.down4.maxpool_conv[1].double_conv[1],
            "down4.maxpool_conv.1.double_conv.2": self.down4.maxpool_conv[1].double_conv[2],
            "down4.maxpool_conv.1.double_conv.3": self.down4.maxpool_conv[1].double_conv[3],
            "up1": self.up1,
            "up1.center_crop": self.up1.center_crop,
            "up1.conv": self.up1.conv,
            "up1.conv.double_conv": self.up1.conv.double_conv,
            "up1.conv.double_conv.0": self.up1.conv.double_conv[0],
            "up1.conv.double_conv.1": self.up1.conv.double_conv[1],
            "up1.conv.double_conv.2": self.up1.conv.double_conv[2],
            "up1.conv.double_conv.3": self.up1.conv.double_conv[3],
            "up1.up": self.up1.up,
            "up1.relu": self.up1.relu,
            "up2": self.up2,
            "up2.center_crop": self.up2.center_crop,
            "up2.conv": self.up2.conv,
            "up2.conv.double_conv": self.up2.conv.double_conv,
            "up2.conv.double_conv.0": self.up2.conv.double_conv[0],
            "up2.conv.double_conv.1": self.up2.conv.double_conv[1],
            "up2.conv.double_conv.2": self.up2.conv.double_conv[2],
            "up2.conv.double_conv.3": self.up2.conv.double_conv[3],
            "up2.up": self.up2.up,
            "up2.relu": self.up2.relu,
            "up3": self.up3,
            "up3.center_crop": self.up3.center_crop,
            "up3.conv": self.up3.conv,
            "up3.conv.double_conv": self.up3.conv.double_conv,
            "up3.conv.double_conv.0": self.up3.conv.double_conv[0],
            "up3.conv.double_conv.1": self.up3.conv.double_conv[1],
            "up3.conv.double_conv.2": self.up3.conv.double_conv[2],
            "up3.conv.double_conv.3": self.up3.conv.double_conv[3],
            "up3.up": self.up3.up,
            "up3.relu": self.up3.relu,
            "up4": self.up4,
            "up4.center_crop": self.up4.center_crop,
            "up4.conv": self.up4.conv,
            "up4.conv.double_conv": self.up4.conv.double_conv,
            "up4.conv.double_conv.0": self.up4.conv.double_conv[0],
            "up4.conv.double_conv.1": self.up4.conv.double_conv[1],
            "up4.conv.double_conv.2": self.up4.conv.double_conv[2],
            "up4.conv.double_conv.3": self.up4.conv.double_conv[3],
            "up4.up": self.up4.up,
            "up4.relu": self.up4.relu,
            "outc": self.outc,
            "outc.conv": self.outc.conv,
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

        self.static_info = {'layer_names': 84,
                            'orders': 53,
                            'out_shapes': 55,
                            'Basic_OPS': 53,
                            'Cascade_OPs': 31
                            }

    def set_layers(self, layer_name, new_layer):
        if 'inc' == layer_name:
            self.inc = new_layer
            self.layer_names["inc"] = new_layer
            self.origin_layer_names["inc"] = new_layer
        elif 'inc.double_conv' == layer_name:
            self.inc.double_conv = new_layer
            self.layer_names["inc.double_conv"] = new_layer
            self.origin_layer_names["inc.double_conv"] = new_layer
        elif 'inc.double_conv.0' == layer_name:
            self.inc.double_conv[0] = new_layer
            self.layer_names["inc.double_conv.0"] = new_layer
            self.origin_layer_names["inc.double_conv.0"] = new_layer
        elif 'inc.double_conv.1' == layer_name:
            self.inc.double_conv[1] = new_layer
            self.layer_names["inc.double_conv.1"] = new_layer
            self.origin_layer_names["inc.double_conv.1"] = new_layer
        elif 'inc.double_conv.2' == layer_name:
            self.inc.double_conv[2] = new_layer
            self.layer_names["inc.double_conv.2"] = new_layer
            self.origin_layer_names["inc.double_conv.2"] = new_layer
        elif 'inc.double_conv.3' == layer_name:
            self.inc.double_conv[3] = new_layer
            self.layer_names["inc.double_conv.3"] = new_layer
            self.origin_layer_names["inc.double_conv.3"] = new_layer
        elif 'down1' == layer_name:
            self.down1 = new_layer
            self.layer_names["down1"] = new_layer
            self.origin_layer_names["down1"] = new_layer
        elif 'down1.maxpool_conv' == layer_name:
            self.down1.maxpool_conv = new_layer
            self.layer_names["down1.maxpool_conv"] = new_layer
            self.origin_layer_names["down1.maxpool_conv"] = new_layer
        elif 'down1.maxpool_conv.0' == layer_name:
            self.down1.maxpool_conv[0] = new_layer
            self.layer_names["down1.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.0"] = new_layer
        elif 'down1.maxpool_conv.1' == layer_name:
            self.down1.maxpool_conv[1] = new_layer
            self.layer_names["down1.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv' == layer_name:
            self.down1.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.0' == layer_name:
            self.down1.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.1' == layer_name:
            self.down1.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.2' == layer_name:
            self.down1.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down1.maxpool_conv.1.double_conv.3' == layer_name:
            self.down1.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down1.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down1.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down2' == layer_name:
            self.down2 = new_layer
            self.layer_names["down2"] = new_layer
            self.origin_layer_names["down2"] = new_layer
        elif 'down2.maxpool_conv' == layer_name:
            self.down2.maxpool_conv = new_layer
            self.layer_names["down2.maxpool_conv"] = new_layer
            self.origin_layer_names["down2.maxpool_conv"] = new_layer
        elif 'down2.maxpool_conv.0' == layer_name:
            self.down2.maxpool_conv[0] = new_layer
            self.layer_names["down2.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.0"] = new_layer
        elif 'down2.maxpool_conv.1' == layer_name:
            self.down2.maxpool_conv[1] = new_layer
            self.layer_names["down2.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv' == layer_name:
            self.down2.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.0' == layer_name:
            self.down2.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.1' == layer_name:
            self.down2.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.2' == layer_name:
            self.down2.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down2.maxpool_conv.1.double_conv.3' == layer_name:
            self.down2.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down2.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down2.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down3' == layer_name:
            self.down3 = new_layer
            self.layer_names["down3"] = new_layer
            self.origin_layer_names["down3"] = new_layer
        elif 'down3.maxpool_conv' == layer_name:
            self.down3.maxpool_conv = new_layer
            self.layer_names["down3.maxpool_conv"] = new_layer
            self.origin_layer_names["down3.maxpool_conv"] = new_layer
        elif 'down3.maxpool_conv.0' == layer_name:
            self.down3.maxpool_conv[0] = new_layer
            self.layer_names["down3.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.0"] = new_layer
        elif 'down3.maxpool_conv.1' == layer_name:
            self.down3.maxpool_conv[1] = new_layer
            self.layer_names["down3.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv' == layer_name:
            self.down3.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.0' == layer_name:
            self.down3.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.1' == layer_name:
            self.down3.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.2' == layer_name:
            self.down3.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down3.maxpool_conv.1.double_conv.3' == layer_name:
            self.down3.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down3.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down3.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'down4' == layer_name:
            self.down4 = new_layer
            self.layer_names["down4"] = new_layer
            self.origin_layer_names["down4"] = new_layer
        elif 'down4.maxpool_conv' == layer_name:
            self.down4.maxpool_conv = new_layer
            self.layer_names["down4.maxpool_conv"] = new_layer
            self.origin_layer_names["down4.maxpool_conv"] = new_layer
        elif 'down4.maxpool_conv.0' == layer_name:
            self.down4.maxpool_conv[0] = new_layer
            self.layer_names["down4.maxpool_conv.0"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.0"] = new_layer
        elif 'down4.maxpool_conv.1' == layer_name:
            self.down4.maxpool_conv[1] = new_layer
            self.layer_names["down4.maxpool_conv.1"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv' == layer_name:
            self.down4.maxpool_conv[1].double_conv = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.0' == layer_name:
            self.down4.maxpool_conv[1].double_conv[0] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.0"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.0"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.1' == layer_name:
            self.down4.maxpool_conv[1].double_conv[1] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.1"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.1"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.2' == layer_name:
            self.down4.maxpool_conv[1].double_conv[2] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.2"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.2"] = new_layer
        elif 'down4.maxpool_conv.1.double_conv.3' == layer_name:
            self.down4.maxpool_conv[1].double_conv[3] = new_layer
            self.layer_names["down4.maxpool_conv.1.double_conv.3"] = new_layer
            self.origin_layer_names["down4.maxpool_conv.1.double_conv.3"] = new_layer
        elif 'up1' == layer_name:
            self.up1 = new_layer
            self.layer_names["up1"] = new_layer
            self.origin_layer_names["up1"] = new_layer
        elif 'up1.center_crop' == layer_name:
            self.up1.center_crop = new_layer
            self.layer_names["up1.center_crop"] = new_layer
            self.origin_layer_names["up1.center_crop"] = new_layer
        elif 'up1.conv' == layer_name:
            self.up1.conv = new_layer
            self.layer_names["up1.conv"] = new_layer
            self.origin_layer_names["up1.conv"] = new_layer
        elif 'up1.conv.double_conv' == layer_name:
            self.up1.conv.double_conv = new_layer
            self.layer_names["up1.conv.double_conv"] = new_layer
            self.origin_layer_names["up1.conv.double_conv"] = new_layer
        elif 'up1.conv.double_conv.0' == layer_name:
            self.up1.conv.double_conv[0] = new_layer
            self.layer_names["up1.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.0"] = new_layer
        elif 'up1.conv.double_conv.1' == layer_name:
            self.up1.conv.double_conv[1] = new_layer
            self.layer_names["up1.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.1"] = new_layer
        elif 'up1.conv.double_conv.2' == layer_name:
            self.up1.conv.double_conv[2] = new_layer
            self.layer_names["up1.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.2"] = new_layer
        elif 'up1.conv.double_conv.3' == layer_name:
            self.up1.conv.double_conv[3] = new_layer
            self.layer_names["up1.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up1.conv.double_conv.3"] = new_layer
        elif 'up1.up' == layer_name:
            self.up1.up = new_layer
            self.layer_names["up1.up"] = new_layer
            self.origin_layer_names["up1.up"] = new_layer
        elif 'up1.relu' == layer_name:
            self.up1.relu = new_layer
            self.layer_names["up1.relu"] = new_layer
            self.origin_layer_names["up1.relu"] = new_layer
        elif 'up2' == layer_name:
            self.up2 = new_layer
            self.layer_names["up2"] = new_layer
            self.origin_layer_names["up2"] = new_layer
        elif 'up2.center_crop' == layer_name:
            self.up2.center_crop = new_layer
            self.layer_names["up2.center_crop"] = new_layer
            self.origin_layer_names["up2.center_crop"] = new_layer
        elif 'up2.conv' == layer_name:
            self.up2.conv = new_layer
            self.layer_names["up2.conv"] = new_layer
            self.origin_layer_names["up2.conv"] = new_layer
        elif 'up2.conv.double_conv' == layer_name:
            self.up2.conv.double_conv = new_layer
            self.layer_names["up2.conv.double_conv"] = new_layer
            self.origin_layer_names["up2.conv.double_conv"] = new_layer
        elif 'up2.conv.double_conv.0' == layer_name:
            self.up2.conv.double_conv[0] = new_layer
            self.layer_names["up2.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.0"] = new_layer
        elif 'up2.conv.double_conv.1' == layer_name:
            self.up2.conv.double_conv[1] = new_layer
            self.layer_names["up2.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.1"] = new_layer
        elif 'up2.conv.double_conv.2' == layer_name:
            self.up2.conv.double_conv[2] = new_layer
            self.layer_names["up2.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.2"] = new_layer
        elif 'up2.conv.double_conv.3' == layer_name:
            self.up2.conv.double_conv[3] = new_layer
            self.layer_names["up2.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up2.conv.double_conv.3"] = new_layer
        elif 'up2.up' == layer_name:
            self.up2.up = new_layer
            self.layer_names["up2.up"] = new_layer
            self.origin_layer_names["up2.up"] = new_layer
        elif 'up2.relu' == layer_name:
            self.up2.relu = new_layer
            self.layer_names["up2.relu"] = new_layer
            self.origin_layer_names["up2.relu"] = new_layer
        elif 'up3' == layer_name:
            self.up3 = new_layer
            self.layer_names["up3"] = new_layer
            self.origin_layer_names["up3"] = new_layer
        elif 'up3.center_crop' == layer_name:
            self.up3.center_crop = new_layer
            self.layer_names["up3.center_crop"] = new_layer
            self.origin_layer_names["up3.center_crop"] = new_layer
        elif 'up3.conv' == layer_name:
            self.up3.conv = new_layer
            self.layer_names["up3.conv"] = new_layer
            self.origin_layer_names["up3.conv"] = new_layer
        elif 'up3.conv.double_conv' == layer_name:
            self.up3.conv.double_conv = new_layer
            self.layer_names["up3.conv.double_conv"] = new_layer
            self.origin_layer_names["up3.conv.double_conv"] = new_layer
        elif 'up3.conv.double_conv.0' == layer_name:
            self.up3.conv.double_conv[0] = new_layer
            self.layer_names["up3.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.0"] = new_layer
        elif 'up3.conv.double_conv.1' == layer_name:
            self.up3.conv.double_conv[1] = new_layer
            self.layer_names["up3.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.1"] = new_layer
        elif 'up3.conv.double_conv.2' == layer_name:
            self.up3.conv.double_conv[2] = new_layer
            self.layer_names["up3.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.2"] = new_layer
        elif 'up3.conv.double_conv.3' == layer_name:
            self.up3.conv.double_conv[3] = new_layer
            self.layer_names["up3.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up3.conv.double_conv.3"] = new_layer
        elif 'up3.up' == layer_name:
            self.up3.up = new_layer
            self.layer_names["up3.up"] = new_layer
            self.origin_layer_names["up3.up"] = new_layer
        elif 'up3.relu' == layer_name:
            self.up3.relu = new_layer
            self.layer_names["up3.relu"] = new_layer
            self.origin_layer_names["up3.relu"] = new_layer
        elif 'up4' == layer_name:
            self.up4 = new_layer
            self.layer_names["up4"] = new_layer
            self.origin_layer_names["up4"] = new_layer
        elif 'up4.center_crop' == layer_name:
            self.up4.center_crop = new_layer
            self.layer_names["up4.center_crop"] = new_layer
            self.origin_layer_names["up4.center_crop"] = new_layer
        elif 'up4.conv' == layer_name:
            self.up4.conv = new_layer
            self.layer_names["up4.conv"] = new_layer
            self.origin_layer_names["up4.conv"] = new_layer
        elif 'up4.conv.double_conv' == layer_name:
            self.up4.conv.double_conv = new_layer
            self.layer_names["up4.conv.double_conv"] = new_layer
            self.origin_layer_names["up4.conv.double_conv"] = new_layer
        elif 'up4.conv.double_conv.0' == layer_name:
            self.up4.conv.double_conv[0] = new_layer
            self.layer_names["up4.conv.double_conv.0"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.0"] = new_layer
        elif 'up4.conv.double_conv.1' == layer_name:
            self.up4.conv.double_conv[1] = new_layer
            self.layer_names["up4.conv.double_conv.1"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.1"] = new_layer
        elif 'up4.conv.double_conv.2' == layer_name:
            self.up4.conv.double_conv[2] = new_layer
            self.layer_names["up4.conv.double_conv.2"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.2"] = new_layer
        elif 'up4.conv.double_conv.3' == layer_name:
            self.up4.conv.double_conv[3] = new_layer
            self.layer_names["up4.conv.double_conv.3"] = new_layer
            self.origin_layer_names["up4.conv.double_conv.3"] = new_layer
        elif 'up4.up' == layer_name:
            self.up4.up = new_layer
            self.layer_names["up4.up"] = new_layer
            self.origin_layer_names["up4.up"] = new_layer
        elif 'up4.relu' == layer_name:
            self.up4.relu = new_layer
            self.layer_names["up4.relu"] = new_layer
            self.origin_layer_names["up4.relu"] = new_layer
        elif 'outc' == layer_name:
            self.outc = new_layer
            self.layer_names["outc"] = new_layer
            self.origin_layer_names["outc"] = new_layer
        elif 'outc.conv' == layer_name:
            self.outc.conv = new_layer
            self.layer_names["outc.conv"] = new_layer
            self.origin_layer_names["outc.conv"] = new_layer

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


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

    def construct(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class MyLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = F.ReduceMean()
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()

    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mindspore.float32)
        weights = self.cast(weights, mindspore.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError


class CrossEntropyWithLogits(MyLoss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        _, _, _, c = F.Shape()(label)
        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, c)), self.reshape_fn(label, (-1, c))))
        return self.get_loss(loss)


class Losser(nn.Cell):  # deprecated since we are no longer needing to use this for gradient descent
    def __init__(self, network, criterion):
        super(Losser, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


def loser(logits, label, network):
    logits = network(logits)
    logits = F.Transpose()(logits, (0, 2, 3, 1))
    logits = F.Cast()(logits, mindspore.float32)
    label = F.Transpose()(label, (0, 2, 3, 1))
    _, _, _, c = F.Shape()(label)
    loss = F.ReduceMean()(
        nn.SoftmaxCrossEntropyWithLogits()(F.Reshape()(logits, (-1, c)), F.Reshape()(label, (-1, c))))
    return get_loss(loss)


def get_loss(x, weights=1.0):
    """
    Computes the weighted loss
    Args:
        weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
            inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
    """
    input_dtype = x.dtype
    x = F.Cast()(x, mindspore.float32)
    weights = F.Cast()(weights, mindspore.float32)
    x = F.Mul()(weights, x)
    if True and True:
        x = F.ReduceMean()(x, get_axis(x))
    # if True and not True:
    #     x = self.reduce_sum(x, self.get_axis(x))
    x = F.Cast()(x, input_dtype)
    return x

def get_axis(x):
    shape = F2.shape(x)
    length = F2.tuple_len(shape)
    perm = F2.make_range(0, length)
    return perm



if __name__ == '__main__':
    input_size=(1, 1, 572, 572)
    data=mindspore.Tensor(np.random.randn(*input_size), mindspore.float32)
    model=UNetMedical(n_channels=1, n_classes=2)
    losser = CrossEntropyWithLogits()
    t = np.random.randn(1, 1, 572, 572)
    a = np.random.randn(1, 2, 388, 388)
    t = mindspore.Tensor(t, dtype=mindspore.float32)
    a = mindspore.Tensor(a, dtype=mindspore.float32)


    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=1e-2, weight_decay=0.0005,loss_scale=1024.0)


    def forward_fn(data, label):
        logits = model(data)
        loss = losser(logits, label)
        return loss, logits


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer(grads))
        return loss


    print("================================================================")
    loss_ms = train_step(t, a)
    print("loss_ms", loss_ms)
    print("================================================================")

