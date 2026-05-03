from mindspore import nn, ops
import mindspore
import numpy as np
from models.SSD.ssd_utils import MultiBox,class_loss

ssd_vgg_bn = False


def _make_layer(channels):
    in_channels = channels[0]
    layers = []
    for out_channels in channels[1:]:
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3))
        if ssd_vgg_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.SequentialCell(layers)


class SSDWithVGG16(nn.Cell):
    def __init__(self):
        super(SSDWithVGG16, self).__init__()
        self.b1 = _make_layer([3, 64, 64])
        self.b2 = _make_layer([64, 128, 128])
        self.b3 = _make_layer([128, 256, 256, 256])
        self.b4 = _make_layer([256, 512, 512, 512])
        self.b5 = _make_layer([512, 512, 512, 512])
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='SAME')


        #after vgg16 backbone
        self.b6_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6, pad_mode='pad')
        self.b6_2 = nn.Dropout()  # p=0.5

        self.b7_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.b7_2 = nn.Dropout()  # p=0.5

        # Extra Feature Layers: block8~11
        self.b8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=1, pad_mode='pad')
        self.b8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, pad_mode='valid')

        self.b9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=1, pad_mode='pad')
        self.b9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, pad_mode='valid')

        self.b10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        self.b11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        self.multibox = MultiBox(81, [512, 1024, 512, 256, 256, 256], [3, 6, 6, 6, 6, 6], 7308)

        self.layer_names = {
            "b1": self.b1,
            "b1.0": self.b1[0],
            "b1.1": self.b1[1],
            "b1.2": self.b1[2],
            "b1.3": self.b1[3],
            "b2": self.b2,
            "b2.0": self.b2[0],
            "b2.1": self.b2[1],
            "b2.2": self.b2[2],
            "b2.3": self.b2[3],
            "b3": self.b3,
            "b3.0": self.b3[0],
            "b3.1": self.b3[1],
            "b3.2": self.b3[2],
            "b3.3": self.b3[3],
            "b3.4": self.b3[4],
            "b3.5": self.b3[5],
            "b4": self.b4,
            "b4.0": self.b4[0],
            "b4.1": self.b4[1],
            "b4.2": self.b4[2],
            "b4.3": self.b4[3],
            "b4.4": self.b4[4],
            "b4.5": self.b4[5],
            "b5": self.b5,
            "b5.0": self.b5[0],
            "b5.1": self.b5[1],
            "b5.2": self.b5[2],
            "b5.3": self.b5[3],
            "b5.4": self.b5[4],
            "b5.5": self.b5[5],
            "m1": self.m1,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
            "m5": self.m5,
            "b6_1": self.b6_1,
            "b6_2": self.b6_2,
            "b7_1": self.b7_1,
            "b7_2": self.b7_2,
            "b8_1": self.b8_1,
            "b8_2": self.b8_2,
            "b9_1": self.b9_1,
            "b9_2": self.b9_2,
            "b10_1": self.b10_1,
            "b10_2": self.b10_2,
            "b11_1": self.b11_1,
            "b11_2": self.b11_2,
        }

        self.origin_layer_names = {
            "b1": self.b1,
            "b1.0": self.b1[0],
            "b1.1": self.b1[1],
            "b1.2": self.b1[2],
            "b1.3": self.b1[3],
            "b2": self.b2,
            "b2.0": self.b2[0],
            "b2.1": self.b2[1],
            "b2.2": self.b2[2],
            "b2.3": self.b2[3],
            "b3": self.b3,
            "b3.0": self.b3[0],
            "b3.1": self.b3[1],
            "b3.2": self.b3[2],
            "b3.3": self.b3[3],
            "b3.4": self.b3[4],
            "b3.5": self.b3[5],
            "b4": self.b4,
            "b4.0": self.b4[0],
            "b4.1": self.b4[1],
            "b4.2": self.b4[2],
            "b4.3": self.b4[3],
            "b4.4": self.b4[4],
            "b4.5": self.b4[5],
            "b5": self.b5,
            "b5.0": self.b5[0],
            "b5.1": self.b5[1],
            "b5.2": self.b5[2],
            "b5.3": self.b5[3],
            "b5.4": self.b5[4],
            "b5.5": self.b5[5],
            "m1": self.m1,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
            "m5": self.m5,
            "b6_1": self.b6_1,
            "b6_2": self.b6_2,
            "b7_1": self.b7_1,
            "b7_2": self.b7_2,
            "b8_1": self.b8_1,
            "b8_2": self.b8_2,
            "b9_1": self.b9_1,
            "b9_2": self.b9_2,
            "b10_1": self.b10_1,
            "b10_2": self.b10_2,
            "b11_1": self.b11_1,
            "b11_2": self.b11_2, }

        self.in_shapes = {
            'INPUT': [1, 3, 300, 300],
            'b1.0': [1, 3, 300, 300],
             'b1.1': [1, 64, 300, 300],
             'b1.2': [1, 64, 300, 300],
             'b1.3': [1, 64, 300, 300],
             'b2.0': [1, 64, 150, 150],
             'b2.1': [1, 128, 150, 150],
             'b2.2': [1, 128, 150, 150],
             'b2.3': [1, 128, 150, 150],
             'b3.0': [1, 128, 75, 75],
             'b3.1': [1, 256, 75, 75],
             'b3.2': [1, 256, 75, 75],
             'b3.3': [1, 256, 75, 75],
             'b3.4': [1, 256, 75, 75],
             'b3.5': [1, 256, 75, 75],
             'b4.0': [1, 256, 38, 38],
             'b4.1': [1, 512, 38, 38],
             'b4.2': [1, 512, 38, 38],
             'b4.3': [1, 512, 38, 38],
             'b4.4': [1, 512, 38, 38],
             'b4.5': [1, 512, 38, 38],
             'b5.0': [1, 512, 19, 19],
             'b5.1': [1, 512, 19, 19],
             'b5.2': [1, 512, 19, 19],
             'b5.3': [1, 512, 19, 19],
             'b5.4': [1, 512, 19, 19],
             'b5.5': [1, 512, 19, 19],
             'm1': [1, 64, 300, 300],
             'm2': [1, 128, 150, 150],
             'm3': [1, 256, 75, 75],
             'm4': [1, 512, 38, 38],
             'm5': [1, 512, 19, 19],
             'b6_1': [1, 512, 19, 19],
             'b6_2': [1, 1024, 19, 19],
             'b7_1': [1, 1024, 19, 19],
             'b7_2': [1, 1024, 19, 19],
             'b8_1': [1, 1024, 19, 19],
             'b8_2': [1, 256, 21, 21],
             'b9_1': [1, 512, 10, 10],
             'b9_2': [1, 128, 12, 12],
             'b10_1': [1, 256, 5, 5],
             'b10_2': [1, 128, 5, 5],
             'b11_1': [1, 256, 3, 3],
             'b11_2': [1, 128, 3, 3],
            'OUTPUT1': [1, 512, 38, 38],
            'OUTPUT2': [1, 1024, 19, 19],
            'OUTPUT3': [1, 512, 10, 10],
            'OUTPUT4': [1, 256, 5, 5],
            'OUTPUT5': [1, 256, 3, 3],
            'OUTPUT6': [1, 128, 3, 3],
            'OUTPUT7': [1, 256, 1, 1],
        }

        self.out_shapes = {
            'INPUT': [1, 3,300, 300],
            'b1.0': [1, 64, 300, 300],
             'b1.1': [1, 64, 300, 300],
             'b1.2': [1, 64, 300, 300],
             'b1.3': [1, 64, 300, 300],
             'b2.0': [1, 128, 150, 150],
             'b2.1': [1, 128, 150, 150],
             'b2.2': [1, 128, 150, 150],
             'b2.3': [1, 128, 150, 150],
             'b3.0': [1, 256, 75, 75],
             'b3.1': [1, 256, 75, 75],
             'b3.2': [1, 256, 75, 75],
             'b3.3': [1, 256, 75, 75],
             'b3.4': [1, 256, 75, 75],
             'b3.5': [1, 256, 75, 75],
             'b4.0': [1, 512, 38, 38],
             'b4.1': [1, 512, 38, 38],
             'b4.2': [1, 512, 38, 38],
             'b4.3': [1, 512, 38, 38],
             'b4.4': [1, 512, 38, 38],
             'b4.5': [1, 512, 38, 38],
             'b5.0': [1, 512, 19, 19],
             'b5.1': [1, 512, 19, 19],
             'b5.2': [1, 512, 19, 19],
             'b5.3': [1, 512, 19, 19],
             'b5.4': [1, 512, 19, 19],
             'b5.5': [1, 512, 19, 19],
             'm1': [1, 64, 150, 150],
             'm2': [1, 128, 75, 75],
             'm3': [1, 256, 38, 38],
             'm4': [1, 512, 19, 19],
             'm5': [1, 512, 19, 19],
             'b6_1': [1, 1024, 19, 19],
             'b6_2': [1, 1024, 19, 19],
             'b7_1': [1, 1024, 19, 19],
             'b7_2': [1, 1024, 19, 19],
             'b8_1': [1, 256, 21, 21],
             'b8_2': [1, 512, 10, 10],
             'b9_1': [1, 128, 12, 12],
             'b9_2': [1, 256, 5, 5],
             'b10_1': [1, 128, 5, 5],
             'b10_2': [1, 256, 3, 3],
             'b11_1': [1, 128, 3, 3],
             'b11_2': [1, 256, 1, 1],
            'OUTPUT1': [1, 512, 38, 38],
            'OUTPUT2': [1, 1024, 19, 19],
            'OUTPUT3': [1, 512, 10, 10],
            'OUTPUT4': [1, 256, 5, 5],
            'OUTPUT5': [1, 256, 3, 3],
            'OUTPUT6': [1, 128, 3, 3],
            'OUTPUT7': [1, 256, 1, 1],
        }



        self.orders = {
            'b1.0': ['INPUT', 'b1.1'],
            'b1.1': ['b1.0', 'b1.2'],
            'b1.2': ['b1.1', 'b1.3'],
            'b1.3': ['b1.2', 'm1'],
            'm1': ['b1.3', 'b2.0'],
            'b2.0': ['m1', 'b2.1'],
            'b2.1': ['b2.0', 'b2.2'],
            'b2.2': ['b2.1', 'b2.3'],
            'b2.3': ['b2.2', 'm2'],
            'm2': ['b2.3', 'b3.0'],
            'b3.0': ['m2', 'b3.1'],
            'b3.1': ['b3.0', 'b3.2'],
            'b3.2': ['b3.1', 'b3.3'],
            'b3.3': ['b3.2', 'b3.4'],
            'b3.4': ['b3.3', 'b3.5'],
            'b3.5': ['b3.4', 'm3'],
            'm3': ['b3.5', 'b4.0'],
            'b4.0': ['m3', 'b4.1'],
            'b4.1': ['b4.0', 'b4.2'],
            'b4.2': ['b4.1', 'b4.3'],
            'b4.3': ['b4.2', 'b4.4'],
            'b4.4': ['b4.3', 'b4.5'],
            'b4.5': ['b4.4', ['m4', 'OUTPUT1']],
            'm4': ['b4.5', 'b5.0'],
            'b5.0': ['m4', 'b5.1'],
            'b5.1': ['b5.0', 'b5.2'],
            'b5.2': ['b5.1', 'b5.3'],
            'b5.3': ['b5.2', 'b5.4'],
            'b5.4': ['b5.3', 'b5.5'],
            'b5.5': ['b5.4', 'm5'],
            'm5': ['b5.5', 'b6_1'],
            'b6_1': ['m5', 'b6_2'],
            'b6_2': ['b6_1', 'b7_1'],
            'b7_1': ['b6_2', 'b7_2'],
            'b7_2': ['b7_1', ['b8_1', 'OUTPUT2']],
            'b8_1': ['b7_2', 'b8_2'],
            'b8_2': ['b8_1', ['b9_1', 'OUTPUT3']],
            'b9_1': ['b8_2', 'b9_2'],
            'b9_2': ['b9_1', ['b10_1', 'OUTPUT4']],
            'b10_1': ['b9_2', 'b10_2'],
            'b10_2': ['b10_1', ['b11_1', 'OUTPUT5']],
            'b11_1': ['b10_2', ['b11_2', 'OUTPUT6']],
            'b11_2': ['b11_1', 'OUTPUT7'],
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

    def construct(self, x):
        # block1
        x = self.b1(x)
        x = self.m1(x)
        # block2
        x = self.b2(x)
        x = self.m2(x)
        # block3
        x = self.b3(x)
        x = self.m3(x)
        # block4

        x = self.b4(x)
        block4 = x
        x = self.m4(x)
        # block5
        x = self.b5(x)
        x = self.m5(x)

        # SSD blocks: block6~7
        x = self.b6_1(x)  # 1024
        x = self.b6_2(x)

        x = self.b7_1(x)  # 1024
        x = self.b7_2(x)
        block7 = x

        # Extra Feature Layers: block8~11
        x = self.b8_1(x)  # 256
        x = self.b8_2(x)  # 512
        block8 = x

        x = self.b9_1(x)  # 128
        x = self.b9_2(x)  # 256
        block9 = x

        x = self.b10_1(x)  # 128
        x = self.b10_2(x)  # 256
        block10 = x

        x = self.b11_1(x)  # 128
        x = self.b11_2(x)  # 256
        block11 = x

        multi_feature = (block4, block7, block8, block9, block10, block11)

        pred_loc, pred_label = self.multibox(multi_feature)

        if not self.training:
            pred_label = ops.Sigmoid()(pred_label)

        pred_loc = ops.cast(pred_loc, mindspore.float32)
        pred_label = ops.cast(pred_label, mindspore.float32)

        return pred_loc, pred_label

    def set_layers(self, layer_name, new_layer):
        if 'b1' == layer_name:
            self.b1 = new_layer
            self.layer_names["b1"] = new_layer
            self.origin_layer_names["b1"] = new_layer
        elif 'b1.0' == layer_name:
            self.b1[0] = new_layer
            self.layer_names["b1.0"] = new_layer
            self.origin_layer_names["b1.0"] = new_layer
        elif 'b1.1' == layer_name:
            self.b1[1] = new_layer
            self.layer_names["b1.1"] = new_layer
            self.origin_layer_names["b1.1"] = new_layer
        elif 'b1.2' == layer_name:
            self.b1[2] = new_layer
            self.layer_names["b1.2"] = new_layer
            self.origin_layer_names["b1.2"] = new_layer
        elif 'b1.3' == layer_name:
            self.b1[3] = new_layer
            self.layer_names["b1.3"] = new_layer
            self.origin_layer_names["b1.3"] = new_layer
        elif 'b2' == layer_name:
            self.b2 = new_layer
            self.layer_names["b2"] = new_layer
            self.origin_layer_names["b2"] = new_layer
        elif 'b2.0' == layer_name:
            self.b2[0] = new_layer
            self.layer_names["b2.0"] = new_layer
            self.origin_layer_names["b2.0"] = new_layer
        elif 'b2.1' == layer_name:
            self.b2[1] = new_layer
            self.layer_names["b2.1"] = new_layer
            self.origin_layer_names["b2.1"] = new_layer
        elif 'b2.2' == layer_name:
            self.b2[2] = new_layer
            self.layer_names["b2.2"] = new_layer
            self.origin_layer_names["b2.2"] = new_layer
        elif 'b2.3' == layer_name:
            self.b2[3] = new_layer
            self.layer_names["b2.3"] = new_layer
            self.origin_layer_names["b2.3"] = new_layer
        elif 'b3' == layer_name:
            self.b3 = new_layer
            self.layer_names["b3"] = new_layer
            self.origin_layer_names["b3"] = new_layer
        elif 'b3.0' == layer_name:
            self.b3[0] = new_layer
            self.layer_names["b3.0"] = new_layer
            self.origin_layer_names["b3.0"] = new_layer
        elif 'b3.1' == layer_name:
            self.b3[1] = new_layer
            self.layer_names["b3.1"] = new_layer
            self.origin_layer_names["b3.1"] = new_layer
        elif 'b3.2' == layer_name:
            self.b3[2] = new_layer
            self.layer_names["b3.2"] = new_layer
            self.origin_layer_names["b3.2"] = new_layer
        elif 'b3.3' == layer_name:
            self.b3[3] = new_layer
            self.layer_names["b3.3"] = new_layer
            self.origin_layer_names["b3.3"] = new_layer
        elif 'b3.4' == layer_name:
            self.b3[4] = new_layer
            self.layer_names["b3.4"] = new_layer
            self.origin_layer_names["b3.4"] = new_layer
        elif 'b3.5' == layer_name:
            self.b3[5] = new_layer
            self.layer_names["b3.5"] = new_layer
            self.origin_layer_names["b3.5"] = new_layer
        elif 'b4' == layer_name:
            self.b4 = new_layer
            self.layer_names["b4"] = new_layer
            self.origin_layer_names["b4"] = new_layer
        elif 'b4.0' == layer_name:
            self.b4[0] = new_layer
            self.layer_names["b4.0"] = new_layer
            self.origin_layer_names["b4.0"] = new_layer
        elif 'b4.1' == layer_name:
            self.b4[1] = new_layer
            self.layer_names["b4.1"] = new_layer
            self.origin_layer_names["b4.1"] = new_layer
        elif 'b4.2' == layer_name:
            self.b4[2] = new_layer
            self.layer_names["b4.2"] = new_layer
            self.origin_layer_names["b4.2"] = new_layer
        elif 'b4.3' == layer_name:
            self.b4[3] = new_layer
            self.layer_names["b4.3"] = new_layer
            self.origin_layer_names["b4.3"] = new_layer
        elif 'b4.4' == layer_name:
            self.b4[4] = new_layer
            self.layer_names["b4.4"] = new_layer
            self.origin_layer_names["b4.4"] = new_layer
        elif 'b4.5' == layer_name:
            self.b4[5] = new_layer
            self.layer_names["b4.5"] = new_layer
            self.origin_layer_names["b4.5"] = new_layer
        elif 'b5' == layer_name:
            self.b5 = new_layer
            self.layer_names["b5"] = new_layer
            self.origin_layer_names["b5"] = new_layer
        elif 'b5.0' == layer_name:
            self.b5[0] = new_layer
            self.layer_names["b5.0"] = new_layer
            self.origin_layer_names["b5.0"] = new_layer
        elif 'b5.1' == layer_name:
            self.b5[1] = new_layer
            self.layer_names["b5.1"] = new_layer
            self.origin_layer_names["b5.1"] = new_layer
        elif 'b5.2' == layer_name:
            self.b5[2] = new_layer
            self.layer_names["b5.2"] = new_layer
            self.origin_layer_names["b5.2"] = new_layer
        elif 'b5.3' == layer_name:
            self.b5[3] = new_layer
            self.layer_names["b5.3"] = new_layer
            self.origin_layer_names["b5.3"] = new_layer
        elif 'b5.4' == layer_name:
            self.b5[4] = new_layer
            self.layer_names["b5.4"] = new_layer
            self.origin_layer_names["b5.4"] = new_layer
        elif 'b5.5' == layer_name:
            self.b5[5] = new_layer
            self.layer_names["b5.5"] = new_layer
            self.origin_layer_names["b5.5"] = new_layer
        elif 'm1' == layer_name:
            self.m1 = new_layer
            self.layer_names["m1"] = new_layer
            self.origin_layer_names["m1"] = new_layer
        elif 'm2' == layer_name:
            self.m2 = new_layer
            self.layer_names["m2"] = new_layer
            self.origin_layer_names["m2"] = new_layer
        elif 'm3' == layer_name:
            self.m3 = new_layer
            self.layer_names["m3"] = new_layer
            self.origin_layer_names["m3"] = new_layer
        elif 'm4' == layer_name:
            self.m4 = new_layer
            self.layer_names["m4"] = new_layer
            self.origin_layer_names["m4"] = new_layer
        elif 'm5' == layer_name:
            self.m5 = new_layer
            self.layer_names["m5"] = new_layer
            self.origin_layer_names["m5"] = new_layer


    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

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

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c


if __name__ == '__main__':
    image = mindspore.Tensor(np.random.rand(25, 3, 300, 300), mindspore.float32)
    num_matched_boxes = mindspore.Tensor([[33]], mindspore.int32)
    gt_label = mindspore.Tensor(np.random.randn(1, 1917), mindspore.int32)  # np.load("./official/gt_label.npy")
    get_loc = mindspore.Tensor(np.random.randn(1, 1917, 4),mindspore.float32)  # mindspore.Tensor(np.load("./official/get_loc.npy"), mindspore.float32)

    model = SSDWithVGG16()
    result1, result2 = model(image)
    print(result1.shape)
    print(result2.shape)


    # # Define the learning rate
    # lr = 1e-4
    #
    # # Define the optimizer
    # opt = nn.Momentum(filter(lambda x: x.requires_grad, model.get_parameters()), lr,0.9, 0.00015, float(1024))
    #
    #
    # # Define the forward procedure
    # def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
    #
    #     pred_loc, pred_label = model(x)
    #
    #     mask = ops.less(0, gt_label).astype(mindspore.float32)
    #     num_matched_boxes = ops.sum(num_matched_boxes.astype(mindspore.float32))
    #
    #     # Positioning loss
    #     mask_loc = ops.tile(ops.expand_dims(mask, -1), (1, 1, 4))
    #     smooth_l1 = nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
    #     loss_loc = ops.sum(ops.sum(smooth_l1, -1), -1)
    #
    #     # Category loss
    #     loss_cls = class_loss(pred_label, gt_label)
    #     loss_cls = ops.sum(loss_cls, (1, 2))
    #
    #     return ops.sum((loss_cls + loss_loc) / num_matched_boxes)
    #
    #
    # grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    #
    # # Gradient updates
    # def train_step(x, gt_loc, gt_label, num_matched_boxes):
    #     loss, grads = grad_fn(x, gt_loc, gt_label, num_matched_boxes)
    #     opt(grads)
    #     return loss
    #
    #
    # for epoch in range(5):
    #     model.set_train(True)
    #     model.set_train(True)
    #     loss = train_step(image, get_loc, gt_label, num_matched_boxes)
    #     print("loss: " + str(loss))









