import os
import time

import mindspore
from PIL import Image
from pycocotools.coco import COCO
import cv2
from mindspore import nn, ops
from models.yolov3_darknet53.model_utils.config import config as default_config, get_config
import mindspore as ms
import numpy as np
from models.yolov3_darknet53.src.lr_scheduler import get_lr
from models.yolov3_darknet53.src.transforms import reshape_fn, MultiScaleTrans
import mindspore.dataset as ds
from models.yolov3_darknet53.src.util import DetectionEngine
from models.yolov3_darknet53.model_utils.config import config
from mindspore.ops import functional as F
from mindspore.ops import composite as C

group_size = 1
per_batch_size = 16
steps_per_epoch = 1
min_keypoints_per_image = 10
log_interval = 1


class ResidualBlock(nn.Cell):
    """
    DarkNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tensor, output tensor.
    Examples:
        ResidualBlock(3, 208)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualBlock, self).__init__()
        out_chls = out_channels // 2
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = ops.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    pad_mode = 'same'
    padding = 0

    return nn.SequentialCell(
        [nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channels, momentum=0.1),
         nn.ReLU()]
    )


class DarkNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 detect=False):
        super(DarkNet, self).__init__()

        self.outchannel = out_channels[-1]
        self.detect = detect

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 5:
            raise ValueError("the length of layer_num, inchannel, outchannel list must be 5!")
        self.conv0 = conv_block(3,
                                in_channels[0],
                                kernel_size=3,
                                stride=1)
        self.conv1 = conv_block(in_channels[0],
                                out_channels[0],
                                kernel_size=3,
                                stride=2)
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=out_channels[0],
                                       out_channel=out_channels[0])
        self.conv2 = conv_block(in_channels[1],
                                out_channels[1],
                                kernel_size=3,
                                stride=2)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=out_channels[1],
                                       out_channel=out_channels[1])
        self.conv3 = conv_block(in_channels[2],
                                out_channels[2],
                                kernel_size=3,
                                stride=2)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=out_channels[2],
                                       out_channel=out_channels[2])
        self.conv4 = conv_block(in_channels[3],
                                out_channels[3],
                                kernel_size=3,
                                stride=2)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=out_channels[3],
                                       out_channel=out_channels[3])
        self.conv5 = conv_block(in_channels[4],
                                out_channels[4],
                                kernel_size=3,
                                stride=2)
        self.layer5 = self._make_layer(block,
                                       layer_nums[4],
                                       in_channel=out_channels[4],
                                       out_channel=out_channels[4])

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)
        c3 = self.layer1(c2)
        c4 = self.conv2(c3)
        c5 = self.layer2(c4)
        c6 = self.conv3(c5)
        c7 = self.layer3(c6)
        c8 = self.conv4(c7)
        c9 = self.layer4(c8)
        c10 = self.conv5(c9)
        c11 = self.layer5(c10)
        if self.detect:
            return c7, c9, c11

        return c11

    def get_out_channels(self):
        return self.outchannel


def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.9,
                  eps=1e-5,
                  pad_mode="same"):
    """Get a conv2d batchnorm and relu layer"""
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, momentum=momentum, eps=eps),
         nn.LeakyReLU(alpha)]
    )


class YOLOv3(nn.Cell):
    def __init__(self, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.backbone = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2] // 2, ksize=1)
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2] + backbone_shape[-3],
                                    out_chls=backbone_shape[-3],
                                    out_channels=out_channel)

        self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3] // 2, ksize=1)
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3] + backbone_shape[-4],
                                    out_chls=backbone_shape[-4],
                                    out_channels=out_channel)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        # feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        # feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        img_hight = ops.Shape()(x)[2]
        img_width = ops.Shape()(x)[3]
        feature_map1, feature_map2, feature_map3 = self.backbone(x)
        con1, big_object_output = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        ups1 = ops.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con1)
        con1 = self.concat((ups1, feature_map2))
        con2, medium_object_output = self.backblock1(con1)

        con2 = self.conv2(con2)
        ups2 = ops.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con2)
        con3 = self.concat((ups2, feature_map1))
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv3.

    Args:
        in_channels: Integer. Input channel.
        out_chls: Integer. Middle channel.
        out_channels: Integer. Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(1024, 512, 255)

    """

    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class DetectionBlock(nn.Cell):
    """
     YOLOv3 detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: Configuration.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32,config=config)
     """

    def __init__(self, scale, config=None, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        self.conf_training = is_training

    def construct(self, x, input_shape):
        num_batch = ops.Shape()(x)[0]
        grid_size = ops.Shape()(x)[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = ops.Reshape()(x, (num_batch, self.num_anchors_per_scale, self.num_attrib,
                                       grid_size[0], grid_size[1]))
        prediction = ops.Transpose()(prediction, (0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = ops.Cast()(ops.tuple_to_array(range_x), ms.float32)
        grid_y = ops.Cast()(ops.tuple_to_array(range_y), ms.float32)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.sigmoid(box_xy) + grid) / ops.Cast()(ops.tuple_to_array((grid_size[1],grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = ops.Exp()(box_wh) * self.anchors / input_shape

        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)


        if self.conf_training:
            return prediction, box_xy, box_wh#grid,

        return self.concat((box_xy, box_wh, box_confidence, box_probs))




class YOLOV3DarkNet53(nn.Cell):
    def __init__(self, is_training, config=default_config):
        super(YOLOV3DarkNet53, self).__init__()
        self.config = config
        self.keep_detect = self.config.keep_detect
        self.tenser_to_array = ops.TupleToArray()

        # YOLOv3 network
        self.feature_map = YOLOv3(backbone=DarkNet(ResidualBlock, self.config.backbone_layers,
                                                   self.config.backbone_input_shape,
                                                   self.config.backbone_shape,
                                                   detect=True),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training, config=self.config)
        self.detect_2 = DetectionBlock('m', is_training=is_training, config=self.config)
        self.detect_3 = DetectionBlock('s', is_training=is_training, config=self.config)

        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None


        self.layer_names = {
            "feature_map": self.feature_map,
            "feature_map.backbone": self.feature_map.backbone,
            "feature_map.backbone.conv0": self.feature_map.backbone.conv0,
            "feature_map.backbone.conv0.0": self.feature_map.backbone.conv0[0],
            "feature_map.backbone.conv0.1": self.feature_map.backbone.conv0[1],
            "feature_map.backbone.conv0.2": self.feature_map.backbone.conv0[2],
            "feature_map.backbone.conv1": self.feature_map.backbone.conv1,
            "feature_map.backbone.conv1.0": self.feature_map.backbone.conv1[0],
            "feature_map.backbone.conv1.1": self.feature_map.backbone.conv1[1],
            "feature_map.backbone.conv1.2": self.feature_map.backbone.conv1[2],
            "feature_map.backbone.layer1": self.feature_map.backbone.layer1,
            "feature_map.backbone.layer1.0": self.feature_map.backbone.layer1[0],
            "feature_map.backbone.layer1.0.conv1": self.feature_map.backbone.layer1[0].conv1,
            "feature_map.backbone.layer1.0.conv1.0": self.feature_map.backbone.layer1[0].conv1[0],
            "feature_map.backbone.layer1.0.conv1.1": self.feature_map.backbone.layer1[0].conv1[1],
            "feature_map.backbone.layer1.0.conv1.2": self.feature_map.backbone.layer1[0].conv1[2],
            "feature_map.backbone.layer1.0.conv2": self.feature_map.backbone.layer1[0].conv2,
            "feature_map.backbone.layer1.0.conv2.0": self.feature_map.backbone.layer1[0].conv2[0],
            "feature_map.backbone.layer1.0.conv2.1": self.feature_map.backbone.layer1[0].conv2[1],
            "feature_map.backbone.layer1.0.conv2.2": self.feature_map.backbone.layer1[0].conv2[2],
            "feature_map.backbone.conv2": self.feature_map.backbone.conv2,
            "feature_map.backbone.conv2.0": self.feature_map.backbone.conv2[0],
            "feature_map.backbone.conv2.1": self.feature_map.backbone.conv2[1],
            "feature_map.backbone.conv2.2": self.feature_map.backbone.conv2[2],
            "feature_map.backbone.layer2": self.feature_map.backbone.layer2,
            "feature_map.backbone.layer2.0": self.feature_map.backbone.layer2[0],
            "feature_map.backbone.layer2.0.conv1": self.feature_map.backbone.layer2[0].conv1,
            "feature_map.backbone.layer2.0.conv1.0": self.feature_map.backbone.layer2[0].conv1[0],
            "feature_map.backbone.layer2.0.conv1.1": self.feature_map.backbone.layer2[0].conv1[1],
            "feature_map.backbone.layer2.0.conv1.2": self.feature_map.backbone.layer2[0].conv1[2],
            "feature_map.backbone.layer2.0.conv2": self.feature_map.backbone.layer2[0].conv2,
            "feature_map.backbone.layer2.0.conv2.0": self.feature_map.backbone.layer2[0].conv2[0],
            "feature_map.backbone.layer2.0.conv2.1": self.feature_map.backbone.layer2[0].conv2[1],
            "feature_map.backbone.layer2.0.conv2.2": self.feature_map.backbone.layer2[0].conv2[2],
            "feature_map.backbone.layer2.1": self.feature_map.backbone.layer2[1],
            "feature_map.backbone.layer2.1.conv1": self.feature_map.backbone.layer2[1].conv1,
            "feature_map.backbone.layer2.1.conv1.0": self.feature_map.backbone.layer2[1].conv1[0],
            "feature_map.backbone.layer2.1.conv1.1": self.feature_map.backbone.layer2[1].conv1[1],
            "feature_map.backbone.layer2.1.conv1.2": self.feature_map.backbone.layer2[1].conv1[2],
            "feature_map.backbone.layer2.1.conv2": self.feature_map.backbone.layer2[1].conv2,
            "feature_map.backbone.layer2.1.conv2.0": self.feature_map.backbone.layer2[1].conv2[0],
            "feature_map.backbone.layer2.1.conv2.1": self.feature_map.backbone.layer2[1].conv2[1],
            "feature_map.backbone.layer2.1.conv2.2": self.feature_map.backbone.layer2[1].conv2[2],
            "feature_map.backbone.conv3": self.feature_map.backbone.conv3,
            "feature_map.backbone.conv3.0": self.feature_map.backbone.conv3[0],
            "feature_map.backbone.conv3.1": self.feature_map.backbone.conv3[1],
            "feature_map.backbone.conv3.2": self.feature_map.backbone.conv3[2],
            "feature_map.backbone.layer3": self.feature_map.backbone.layer3,
            "feature_map.backbone.layer3.0": self.feature_map.backbone.layer3[0],
            "feature_map.backbone.layer3.0.conv1": self.feature_map.backbone.layer3[0].conv1,
            "feature_map.backbone.layer3.0.conv1.0": self.feature_map.backbone.layer3[0].conv1[0],
            "feature_map.backbone.layer3.0.conv1.1": self.feature_map.backbone.layer3[0].conv1[1],
            "feature_map.backbone.layer3.0.conv1.2": self.feature_map.backbone.layer3[0].conv1[2],
            "feature_map.backbone.layer3.0.conv2": self.feature_map.backbone.layer3[0].conv2,
            "feature_map.backbone.layer3.0.conv2.0": self.feature_map.backbone.layer3[0].conv2[0],
            "feature_map.backbone.layer3.0.conv2.1": self.feature_map.backbone.layer3[0].conv2[1],
            "feature_map.backbone.layer3.0.conv2.2": self.feature_map.backbone.layer3[0].conv2[2],
            "feature_map.backbone.layer3.1": self.feature_map.backbone.layer3[1],
            "feature_map.backbone.layer3.1.conv1": self.feature_map.backbone.layer3[1].conv1,
            "feature_map.backbone.layer3.1.conv1.0": self.feature_map.backbone.layer3[1].conv1[0],
            "feature_map.backbone.layer3.1.conv1.1": self.feature_map.backbone.layer3[1].conv1[1],
            "feature_map.backbone.layer3.1.conv1.2": self.feature_map.backbone.layer3[1].conv1[2],
            "feature_map.backbone.layer3.1.conv2": self.feature_map.backbone.layer3[1].conv2,
            "feature_map.backbone.layer3.1.conv2.0": self.feature_map.backbone.layer3[1].conv2[0],
            "feature_map.backbone.layer3.1.conv2.1": self.feature_map.backbone.layer3[1].conv2[1],
            "feature_map.backbone.layer3.1.conv2.2": self.feature_map.backbone.layer3[1].conv2[2],
            "feature_map.backbone.layer3.2": self.feature_map.backbone.layer3[2],
            "feature_map.backbone.layer3.2.conv1": self.feature_map.backbone.layer3[2].conv1,
            "feature_map.backbone.layer3.2.conv1.0": self.feature_map.backbone.layer3[2].conv1[0],
            "feature_map.backbone.layer3.2.conv1.1": self.feature_map.backbone.layer3[2].conv1[1],
            "feature_map.backbone.layer3.2.conv1.2": self.feature_map.backbone.layer3[2].conv1[2],
            "feature_map.backbone.layer3.2.conv2": self.feature_map.backbone.layer3[2].conv2,
            "feature_map.backbone.layer3.2.conv2.0": self.feature_map.backbone.layer3[2].conv2[0],
            "feature_map.backbone.layer3.2.conv2.1": self.feature_map.backbone.layer3[2].conv2[1],
            "feature_map.backbone.layer3.2.conv2.2": self.feature_map.backbone.layer3[2].conv2[2],
            "feature_map.backbone.layer3.3": self.feature_map.backbone.layer3[3],
            "feature_map.backbone.layer3.3.conv1": self.feature_map.backbone.layer3[3].conv1,
            "feature_map.backbone.layer3.3.conv1.0": self.feature_map.backbone.layer3[3].conv1[0],
            "feature_map.backbone.layer3.3.conv1.1": self.feature_map.backbone.layer3[3].conv1[1],
            "feature_map.backbone.layer3.3.conv1.2": self.feature_map.backbone.layer3[3].conv1[2],
            "feature_map.backbone.layer3.3.conv2": self.feature_map.backbone.layer3[3].conv2,
            "feature_map.backbone.layer3.3.conv2.0": self.feature_map.backbone.layer3[3].conv2[0],
            "feature_map.backbone.layer3.3.conv2.1": self.feature_map.backbone.layer3[3].conv2[1],
            "feature_map.backbone.layer3.3.conv2.2": self.feature_map.backbone.layer3[3].conv2[2],
            "feature_map.backbone.layer3.4": self.feature_map.backbone.layer3[4],
            "feature_map.backbone.layer3.4.conv1": self.feature_map.backbone.layer3[4].conv1,
            "feature_map.backbone.layer3.4.conv1.0": self.feature_map.backbone.layer3[4].conv1[0],
            "feature_map.backbone.layer3.4.conv1.1": self.feature_map.backbone.layer3[4].conv1[1],
            "feature_map.backbone.layer3.4.conv1.2": self.feature_map.backbone.layer3[4].conv1[2],
            "feature_map.backbone.layer3.4.conv2": self.feature_map.backbone.layer3[4].conv2,
            "feature_map.backbone.layer3.4.conv2.0": self.feature_map.backbone.layer3[4].conv2[0],
            "feature_map.backbone.layer3.4.conv2.1": self.feature_map.backbone.layer3[4].conv2[1],
            "feature_map.backbone.layer3.4.conv2.2": self.feature_map.backbone.layer3[4].conv2[2],
            "feature_map.backbone.layer3.5": self.feature_map.backbone.layer3[5],
            "feature_map.backbone.layer3.5.conv1": self.feature_map.backbone.layer3[5].conv1,
            "feature_map.backbone.layer3.5.conv1.0": self.feature_map.backbone.layer3[5].conv1[0],
            "feature_map.backbone.layer3.5.conv1.1": self.feature_map.backbone.layer3[5].conv1[1],
            "feature_map.backbone.layer3.5.conv1.2": self.feature_map.backbone.layer3[5].conv1[2],
            "feature_map.backbone.layer3.5.conv2": self.feature_map.backbone.layer3[5].conv2,
            "feature_map.backbone.layer3.5.conv2.0": self.feature_map.backbone.layer3[5].conv2[0],
            "feature_map.backbone.layer3.5.conv2.1": self.feature_map.backbone.layer3[5].conv2[1],
            "feature_map.backbone.layer3.5.conv2.2": self.feature_map.backbone.layer3[5].conv2[2],
            "feature_map.backbone.layer3.6": self.feature_map.backbone.layer3[6],
            "feature_map.backbone.layer3.6.conv1": self.feature_map.backbone.layer3[6].conv1,
            "feature_map.backbone.layer3.6.conv1.0": self.feature_map.backbone.layer3[6].conv1[0],
            "feature_map.backbone.layer3.6.conv1.1": self.feature_map.backbone.layer3[6].conv1[1],
            "feature_map.backbone.layer3.6.conv1.2": self.feature_map.backbone.layer3[6].conv1[2],
            "feature_map.backbone.layer3.6.conv2": self.feature_map.backbone.layer3[6].conv2,
            "feature_map.backbone.layer3.6.conv2.0": self.feature_map.backbone.layer3[6].conv2[0],
            "feature_map.backbone.layer3.6.conv2.1": self.feature_map.backbone.layer3[6].conv2[1],
            "feature_map.backbone.layer3.6.conv2.2": self.feature_map.backbone.layer3[6].conv2[2],
            "feature_map.backbone.layer3.7": self.feature_map.backbone.layer3[7],
            "feature_map.backbone.layer3.7.conv1": self.feature_map.backbone.layer3[7].conv1,
            "feature_map.backbone.layer3.7.conv1.0": self.feature_map.backbone.layer3[7].conv1[0],
            "feature_map.backbone.layer3.7.conv1.1": self.feature_map.backbone.layer3[7].conv1[1],
            "feature_map.backbone.layer3.7.conv1.2": self.feature_map.backbone.layer3[7].conv1[2],
            "feature_map.backbone.layer3.7.conv2": self.feature_map.backbone.layer3[7].conv2,
            "feature_map.backbone.layer3.7.conv2.0": self.feature_map.backbone.layer3[7].conv2[0],
            "feature_map.backbone.layer3.7.conv2.1": self.feature_map.backbone.layer3[7].conv2[1],
            "feature_map.backbone.layer3.7.conv2.2": self.feature_map.backbone.layer3[7].conv2[2],
            "feature_map.backbone.conv4": self.feature_map.backbone.conv4,
            "feature_map.backbone.conv4.0": self.feature_map.backbone.conv4[0],
            "feature_map.backbone.conv4.1": self.feature_map.backbone.conv4[1],
            "feature_map.backbone.conv4.2": self.feature_map.backbone.conv4[2],
            "feature_map.backbone.layer4": self.feature_map.backbone.layer4,
            "feature_map.backbone.layer4.0": self.feature_map.backbone.layer4[0],
            "feature_map.backbone.layer4.0.conv1": self.feature_map.backbone.layer4[0].conv1,
            "feature_map.backbone.layer4.0.conv1.0": self.feature_map.backbone.layer4[0].conv1[0],
            "feature_map.backbone.layer4.0.conv1.1": self.feature_map.backbone.layer4[0].conv1[1],
            "feature_map.backbone.layer4.0.conv1.2": self.feature_map.backbone.layer4[0].conv1[2],
            "feature_map.backbone.layer4.0.conv2": self.feature_map.backbone.layer4[0].conv2,
            "feature_map.backbone.layer4.0.conv2.0": self.feature_map.backbone.layer4[0].conv2[0],
            "feature_map.backbone.layer4.0.conv2.1": self.feature_map.backbone.layer4[0].conv2[1],
            "feature_map.backbone.layer4.0.conv2.2": self.feature_map.backbone.layer4[0].conv2[2],
            "feature_map.backbone.layer4.1": self.feature_map.backbone.layer4[1],
            "feature_map.backbone.layer4.1.conv1": self.feature_map.backbone.layer4[1].conv1,
            "feature_map.backbone.layer4.1.conv1.0": self.feature_map.backbone.layer4[1].conv1[0],
            "feature_map.backbone.layer4.1.conv1.1": self.feature_map.backbone.layer4[1].conv1[1],
            "feature_map.backbone.layer4.1.conv1.2": self.feature_map.backbone.layer4[1].conv1[2],
            "feature_map.backbone.layer4.1.conv2": self.feature_map.backbone.layer4[1].conv2,
            "feature_map.backbone.layer4.1.conv2.0": self.feature_map.backbone.layer4[1].conv2[0],
            "feature_map.backbone.layer4.1.conv2.1": self.feature_map.backbone.layer4[1].conv2[1],
            "feature_map.backbone.layer4.1.conv2.2": self.feature_map.backbone.layer4[1].conv2[2],
            "feature_map.backbone.layer4.2": self.feature_map.backbone.layer4[2],
            "feature_map.backbone.layer4.2.conv1": self.feature_map.backbone.layer4[2].conv1,
            "feature_map.backbone.layer4.2.conv1.0": self.feature_map.backbone.layer4[2].conv1[0],
            "feature_map.backbone.layer4.2.conv1.1": self.feature_map.backbone.layer4[2].conv1[1],
            "feature_map.backbone.layer4.2.conv1.2": self.feature_map.backbone.layer4[2].conv1[2],
            "feature_map.backbone.layer4.2.conv2": self.feature_map.backbone.layer4[2].conv2,
            "feature_map.backbone.layer4.2.conv2.0": self.feature_map.backbone.layer4[2].conv2[0],
            "feature_map.backbone.layer4.2.conv2.1": self.feature_map.backbone.layer4[2].conv2[1],
            "feature_map.backbone.layer4.2.conv2.2": self.feature_map.backbone.layer4[2].conv2[2],
            "feature_map.backbone.layer4.3": self.feature_map.backbone.layer4[3],
            "feature_map.backbone.layer4.3.conv1": self.feature_map.backbone.layer4[3].conv1,
            "feature_map.backbone.layer4.3.conv1.0": self.feature_map.backbone.layer4[3].conv1[0],
            "feature_map.backbone.layer4.3.conv1.1": self.feature_map.backbone.layer4[3].conv1[1],
            "feature_map.backbone.layer4.3.conv1.2": self.feature_map.backbone.layer4[3].conv1[2],
            "feature_map.backbone.layer4.3.conv2": self.feature_map.backbone.layer4[3].conv2,
            "feature_map.backbone.layer4.3.conv2.0": self.feature_map.backbone.layer4[3].conv2[0],
            "feature_map.backbone.layer4.3.conv2.1": self.feature_map.backbone.layer4[3].conv2[1],
            "feature_map.backbone.layer4.3.conv2.2": self.feature_map.backbone.layer4[3].conv2[2],
            "feature_map.backbone.layer4.4": self.feature_map.backbone.layer4[4],
            "feature_map.backbone.layer4.4.conv1": self.feature_map.backbone.layer4[4].conv1,
            "feature_map.backbone.layer4.4.conv1.0": self.feature_map.backbone.layer4[4].conv1[0],
            "feature_map.backbone.layer4.4.conv1.1": self.feature_map.backbone.layer4[4].conv1[1],
            "feature_map.backbone.layer4.4.conv1.2": self.feature_map.backbone.layer4[4].conv1[2],
            "feature_map.backbone.layer4.4.conv2": self.feature_map.backbone.layer4[4].conv2,
            "feature_map.backbone.layer4.4.conv2.0": self.feature_map.backbone.layer4[4].conv2[0],
            "feature_map.backbone.layer4.4.conv2.1": self.feature_map.backbone.layer4[4].conv2[1],
            "feature_map.backbone.layer4.4.conv2.2": self.feature_map.backbone.layer4[4].conv2[2],
            "feature_map.backbone.layer4.5": self.feature_map.backbone.layer4[5],
            "feature_map.backbone.layer4.5.conv1": self.feature_map.backbone.layer4[5].conv1,
            "feature_map.backbone.layer4.5.conv1.0": self.feature_map.backbone.layer4[5].conv1[0],
            "feature_map.backbone.layer4.5.conv1.1": self.feature_map.backbone.layer4[5].conv1[1],
            "feature_map.backbone.layer4.5.conv1.2": self.feature_map.backbone.layer4[5].conv1[2],
            "feature_map.backbone.layer4.5.conv2": self.feature_map.backbone.layer4[5].conv2,
            "feature_map.backbone.layer4.5.conv2.0": self.feature_map.backbone.layer4[5].conv2[0],
            "feature_map.backbone.layer4.5.conv2.1": self.feature_map.backbone.layer4[5].conv2[1],
            "feature_map.backbone.layer4.5.conv2.2": self.feature_map.backbone.layer4[5].conv2[2],
            "feature_map.backbone.layer4.6": self.feature_map.backbone.layer4[6],
            "feature_map.backbone.layer4.6.conv1": self.feature_map.backbone.layer4[6].conv1,
            "feature_map.backbone.layer4.6.conv1.0": self.feature_map.backbone.layer4[6].conv1[0],
            "feature_map.backbone.layer4.6.conv1.1": self.feature_map.backbone.layer4[6].conv1[1],
            "feature_map.backbone.layer4.6.conv1.2": self.feature_map.backbone.layer4[6].conv1[2],
            "feature_map.backbone.layer4.6.conv2": self.feature_map.backbone.layer4[6].conv2,
            "feature_map.backbone.layer4.6.conv2.0": self.feature_map.backbone.layer4[6].conv2[0],
            "feature_map.backbone.layer4.6.conv2.1": self.feature_map.backbone.layer4[6].conv2[1],
            "feature_map.backbone.layer4.6.conv2.2": self.feature_map.backbone.layer4[6].conv2[2],
            "feature_map.backbone.layer4.7": self.feature_map.backbone.layer4[7],
            "feature_map.backbone.layer4.7.conv1": self.feature_map.backbone.layer4[7].conv1,
            "feature_map.backbone.layer4.7.conv1.0": self.feature_map.backbone.layer4[7].conv1[0],
            "feature_map.backbone.layer4.7.conv1.1": self.feature_map.backbone.layer4[7].conv1[1],
            "feature_map.backbone.layer4.7.conv1.2": self.feature_map.backbone.layer4[7].conv1[2],
            "feature_map.backbone.layer4.7.conv2": self.feature_map.backbone.layer4[7].conv2,
            "feature_map.backbone.layer4.7.conv2.0": self.feature_map.backbone.layer4[7].conv2[0],
            "feature_map.backbone.layer4.7.conv2.1": self.feature_map.backbone.layer4[7].conv2[1],
            "feature_map.backbone.layer4.7.conv2.2": self.feature_map.backbone.layer4[7].conv2[2],
            "feature_map.backbone.conv5": self.feature_map.backbone.conv5,
            "feature_map.backbone.conv5.0": self.feature_map.backbone.conv5[0],
            "feature_map.backbone.conv5.1": self.feature_map.backbone.conv5[1],
            "feature_map.backbone.conv5.2": self.feature_map.backbone.conv5[2],
            "feature_map.backbone.layer5": self.feature_map.backbone.layer5,
            "feature_map.backbone.layer5.0": self.feature_map.backbone.layer5[0],
            "feature_map.backbone.layer5.0.conv1": self.feature_map.backbone.layer5[0].conv1,
            "feature_map.backbone.layer5.0.conv1.0": self.feature_map.backbone.layer5[0].conv1[0],
            "feature_map.backbone.layer5.0.conv1.1": self.feature_map.backbone.layer5[0].conv1[1],
            "feature_map.backbone.layer5.0.conv1.2": self.feature_map.backbone.layer5[0].conv1[2],
            "feature_map.backbone.layer5.0.conv2": self.feature_map.backbone.layer5[0].conv2,
            "feature_map.backbone.layer5.0.conv2.0": self.feature_map.backbone.layer5[0].conv2[0],
            "feature_map.backbone.layer5.0.conv2.1": self.feature_map.backbone.layer5[0].conv2[1],
            "feature_map.backbone.layer5.0.conv2.2": self.feature_map.backbone.layer5[0].conv2[2],
            "feature_map.backbone.layer5.1": self.feature_map.backbone.layer5[1],
            "feature_map.backbone.layer5.1.conv1": self.feature_map.backbone.layer5[1].conv1,
            "feature_map.backbone.layer5.1.conv1.0": self.feature_map.backbone.layer5[1].conv1[0],
            "feature_map.backbone.layer5.1.conv1.1": self.feature_map.backbone.layer5[1].conv1[1],
            "feature_map.backbone.layer5.1.conv1.2": self.feature_map.backbone.layer5[1].conv1[2],
            "feature_map.backbone.layer5.1.conv2": self.feature_map.backbone.layer5[1].conv2,
            "feature_map.backbone.layer5.1.conv2.0": self.feature_map.backbone.layer5[1].conv2[0],
            "feature_map.backbone.layer5.1.conv2.1": self.feature_map.backbone.layer5[1].conv2[1],
            "feature_map.backbone.layer5.1.conv2.2": self.feature_map.backbone.layer5[1].conv2[2],
            "feature_map.backbone.layer5.2": self.feature_map.backbone.layer5[2],
            "feature_map.backbone.layer5.2.conv1": self.feature_map.backbone.layer5[2].conv1,
            "feature_map.backbone.layer5.2.conv1.0": self.feature_map.backbone.layer5[2].conv1[0],
            "feature_map.backbone.layer5.2.conv1.1": self.feature_map.backbone.layer5[2].conv1[1],
            "feature_map.backbone.layer5.2.conv1.2": self.feature_map.backbone.layer5[2].conv1[2],
            "feature_map.backbone.layer5.2.conv2": self.feature_map.backbone.layer5[2].conv2,
            "feature_map.backbone.layer5.2.conv2.0": self.feature_map.backbone.layer5[2].conv2[0],
            "feature_map.backbone.layer5.2.conv2.1": self.feature_map.backbone.layer5[2].conv2[1],
            "feature_map.backbone.layer5.2.conv2.2": self.feature_map.backbone.layer5[2].conv2[2],
            "feature_map.backbone.layer5.3": self.feature_map.backbone.layer5[3],
            "feature_map.backbone.layer5.3.conv1": self.feature_map.backbone.layer5[3].conv1,
            "feature_map.backbone.layer5.3.conv1.0": self.feature_map.backbone.layer5[3].conv1[0],
            "feature_map.backbone.layer5.3.conv1.1": self.feature_map.backbone.layer5[3].conv1[1],
            "feature_map.backbone.layer5.3.conv1.2": self.feature_map.backbone.layer5[3].conv1[2],
            "feature_map.backbone.layer5.3.conv2": self.feature_map.backbone.layer5[3].conv2,
            "feature_map.backbone.layer5.3.conv2.0": self.feature_map.backbone.layer5[3].conv2[0],
            "feature_map.backbone.layer5.3.conv2.1": self.feature_map.backbone.layer5[3].conv2[1],
            "feature_map.backbone.layer5.3.conv2.2": self.feature_map.backbone.layer5[3].conv2[2],
            # "feature_map.backblock0": self.feature_map.backblock0,
            # "feature_map.backblock0.conv0": self.feature_map.backblock0.conv0,
            # "feature_map.backblock0.conv0.0": self.feature_map.backblock0.conv0[0],
            # "feature_map.backblock0.conv0.1": self.feature_map.backblock0.conv0[1],
            # "feature_map.backblock0.conv0.2": self.feature_map.backblock0.conv0[2],
            # "feature_map.backblock0.conv1": self.feature_map.backblock0.conv1,
            # "feature_map.backblock0.conv1.0": self.feature_map.backblock0.conv1[0],
            # "feature_map.backblock0.conv1.1": self.feature_map.backblock0.conv1[1],
            # "feature_map.backblock0.conv1.2": self.feature_map.backblock0.conv1[2],
            # "feature_map.backblock0.conv2": self.feature_map.backblock0.conv2,
            # "feature_map.backblock0.conv2.0": self.feature_map.backblock0.conv2[0],
            # "feature_map.backblock0.conv2.1": self.feature_map.backblock0.conv2[1],
            # "feature_map.backblock0.conv2.2": self.feature_map.backblock0.conv2[2],
            # "feature_map.backblock0.conv3": self.feature_map.backblock0.conv3,
            # "feature_map.backblock0.conv3.0": self.feature_map.backblock0.conv3[0],
            # "feature_map.backblock0.conv3.1": self.feature_map.backblock0.conv3[1],
            # "feature_map.backblock0.conv3.2": self.feature_map.backblock0.conv3[2],
            # "feature_map.backblock0.conv4": self.feature_map.backblock0.conv4,
            # "feature_map.backblock0.conv4.0": self.feature_map.backblock0.conv4[0],
            # "feature_map.backblock0.conv4.1": self.feature_map.backblock0.conv4[1],
            # "feature_map.backblock0.conv4.2": self.feature_map.backblock0.conv4[2],
            # "feature_map.backblock0.conv5": self.feature_map.backblock0.conv5,
            # "feature_map.backblock0.conv5.0": self.feature_map.backblock0.conv5[0],
            # "feature_map.backblock0.conv5.1": self.feature_map.backblock0.conv5[1],
            # "feature_map.backblock0.conv5.2": self.feature_map.backblock0.conv5[2],
            # "feature_map.backblock0.conv6": self.feature_map.backblock0.conv6,
            # "feature_map.conv1": self.feature_map.conv1,
            # "feature_map.conv1.0": self.feature_map.conv1[0],
            # "feature_map.conv1.1": self.feature_map.conv1[1],
            # "feature_map.conv1.2": self.feature_map.conv1[2],
            # "feature_map.backblock1": self.feature_map.backblock1,
            # "feature_map.backblock1.conv0": self.feature_map.backblock1.conv0,
            # "feature_map.backblock1.conv0.0": self.feature_map.backblock1.conv0[0],
            # "feature_map.backblock1.conv0.1": self.feature_map.backblock1.conv0[1],
            # "feature_map.backblock1.conv0.2": self.feature_map.backblock1.conv0[2],
            # "feature_map.backblock1.conv1": self.feature_map.backblock1.conv1,
            # "feature_map.backblock1.conv1.0": self.feature_map.backblock1.conv1[0],
            # "feature_map.backblock1.conv1.1": self.feature_map.backblock1.conv1[1],
            # "feature_map.backblock1.conv1.2": self.feature_map.backblock1.conv1[2],
            # "feature_map.backblock1.conv2": self.feature_map.backblock1.conv2,
            # "feature_map.backblock1.conv2.0": self.feature_map.backblock1.conv2[0],
            # "feature_map.backblock1.conv2.1": self.feature_map.backblock1.conv2[1],
            # "feature_map.backblock1.conv2.2": self.feature_map.backblock1.conv2[2],
            # "feature_map.backblock1.conv3": self.feature_map.backblock1.conv3,
            # "feature_map.backblock1.conv3.0": self.feature_map.backblock1.conv3[0],
            # "feature_map.backblock1.conv3.1": self.feature_map.backblock1.conv3[1],
            # "feature_map.backblock1.conv3.2": self.feature_map.backblock1.conv3[2],
            # "feature_map.backblock1.conv4": self.feature_map.backblock1.conv4,
            # "feature_map.backblock1.conv4.0": self.feature_map.backblock1.conv4[0],
            # "feature_map.backblock1.conv4.1": self.feature_map.backblock1.conv4[1],
            # "feature_map.backblock1.conv4.2": self.feature_map.backblock1.conv4[2],
            # "feature_map.backblock1.conv5": self.feature_map.backblock1.conv5,
            # "feature_map.backblock1.conv5.0": self.feature_map.backblock1.conv5[0],
            # "feature_map.backblock1.conv5.1": self.feature_map.backblock1.conv5[1],
            # "feature_map.backblock1.conv5.2": self.feature_map.backblock1.conv5[2],
            # "feature_map.backblock1.conv6": self.feature_map.backblock1.conv6,
            # "feature_map.conv2": self.feature_map.conv2,
            # "feature_map.conv2.0": self.feature_map.conv2[0],
            # "feature_map.conv2.1": self.feature_map.conv2[1],
            # "feature_map.conv2.2": self.feature_map.conv2[2],
            # "feature_map.backblock2": self.feature_map.backblock2,
            # "feature_map.backblock2.conv0": self.feature_map.backblock2.conv0,
            # "feature_map.backblock2.conv0.0": self.feature_map.backblock2.conv0[0],
            # "feature_map.backblock2.conv0.1": self.feature_map.backblock2.conv0[1],
            # "feature_map.backblock2.conv0.2": self.feature_map.backblock2.conv0[2],
            # "feature_map.backblock2.conv1": self.feature_map.backblock2.conv1,
            # "feature_map.backblock2.conv1.0": self.feature_map.backblock2.conv1[0],
            # "feature_map.backblock2.conv1.1": self.feature_map.backblock2.conv1[1],
            # "feature_map.backblock2.conv1.2": self.feature_map.backblock2.conv1[2],
            # "feature_map.backblock2.conv2": self.feature_map.backblock2.conv2,
            # "feature_map.backblock2.conv2.0": self.feature_map.backblock2.conv2[0],
            # "feature_map.backblock2.conv2.1": self.feature_map.backblock2.conv2[1],
            # "feature_map.backblock2.conv2.2": self.feature_map.backblock2.conv2[2],
            # "feature_map.backblock2.conv3": self.feature_map.backblock2.conv3,
            # "feature_map.backblock2.conv3.0": self.feature_map.backblock2.conv3[0],
            # "feature_map.backblock2.conv3.1": self.feature_map.backblock2.conv3[1],
            # "feature_map.backblock2.conv3.2": self.feature_map.backblock2.conv3[2],
            # "feature_map.backblock2.conv4": self.feature_map.backblock2.conv4,
            # "feature_map.backblock2.conv4.0": self.feature_map.backblock2.conv4[0],
            # "feature_map.backblock2.conv4.1": self.feature_map.backblock2.conv4[1],
            # "feature_map.backblock2.conv4.2": self.feature_map.backblock2.conv4[2],
            # "feature_map.backblock2.conv5": self.feature_map.backblock2.conv5,
            # "feature_map.backblock2.conv5.0": self.feature_map.backblock2.conv5[0],
            # "feature_map.backblock2.conv5.1": self.feature_map.backblock2.conv5[1],
            # "feature_map.backblock2.conv5.2": self.feature_map.backblock2.conv5[2],
            # "feature_map.backblock2.conv6": self.feature_map.backblock2.conv6,
            # "detect_1": self.detect_1,
            # "detect_1.sigmoid": self.detect_1.sigmoid,
            # "detect_2": self.detect_2,
            # "detect_2.sigmoid": self.detect_2.sigmoid,
            # "detect_3": self.detect_3,
            # "detect_3.sigmoid": self.detect_3.sigmoid,

        }
        self.origin_layer_names={
            "feature_map": self.feature_map,
            "feature_map.backbone": self.feature_map.backbone,
            "feature_map.backbone.conv0": self.feature_map.backbone.conv0,
            "feature_map.backbone.conv0.0": self.feature_map.backbone.conv0[0],
            "feature_map.backbone.conv0.1": self.feature_map.backbone.conv0[1],
            "feature_map.backbone.conv0.2": self.feature_map.backbone.conv0[2],
            "feature_map.backbone.conv1": self.feature_map.backbone.conv1,
            "feature_map.backbone.conv1.0": self.feature_map.backbone.conv1[0],
            "feature_map.backbone.conv1.1": self.feature_map.backbone.conv1[1],
            "feature_map.backbone.conv1.2": self.feature_map.backbone.conv1[2],
            "feature_map.backbone.layer1": self.feature_map.backbone.layer1,
            "feature_map.backbone.layer1.0": self.feature_map.backbone.layer1[0],
            "feature_map.backbone.layer1.0.conv1": self.feature_map.backbone.layer1[0].conv1,
            "feature_map.backbone.layer1.0.conv1.0": self.feature_map.backbone.layer1[0].conv1[0],
            "feature_map.backbone.layer1.0.conv1.1": self.feature_map.backbone.layer1[0].conv1[1],
            "feature_map.backbone.layer1.0.conv1.2": self.feature_map.backbone.layer1[0].conv1[2],
            "feature_map.backbone.layer1.0.conv2": self.feature_map.backbone.layer1[0].conv2,
            "feature_map.backbone.layer1.0.conv2.0": self.feature_map.backbone.layer1[0].conv2[0],
            "feature_map.backbone.layer1.0.conv2.1": self.feature_map.backbone.layer1[0].conv2[1],
            "feature_map.backbone.layer1.0.conv2.2": self.feature_map.backbone.layer1[0].conv2[2],
            "feature_map.backbone.conv2": self.feature_map.backbone.conv2,
            "feature_map.backbone.conv2.0": self.feature_map.backbone.conv2[0],
            "feature_map.backbone.conv2.1": self.feature_map.backbone.conv2[1],
            "feature_map.backbone.conv2.2": self.feature_map.backbone.conv2[2],
            "feature_map.backbone.layer2": self.feature_map.backbone.layer2,
            "feature_map.backbone.layer2.0": self.feature_map.backbone.layer2[0],
            "feature_map.backbone.layer2.0.conv1": self.feature_map.backbone.layer2[0].conv1,
            "feature_map.backbone.layer2.0.conv1.0": self.feature_map.backbone.layer2[0].conv1[0],
            "feature_map.backbone.layer2.0.conv1.1": self.feature_map.backbone.layer2[0].conv1[1],
            "feature_map.backbone.layer2.0.conv1.2": self.feature_map.backbone.layer2[0].conv1[2],
            "feature_map.backbone.layer2.0.conv2": self.feature_map.backbone.layer2[0].conv2,
            "feature_map.backbone.layer2.0.conv2.0": self.feature_map.backbone.layer2[0].conv2[0],
            "feature_map.backbone.layer2.0.conv2.1": self.feature_map.backbone.layer2[0].conv2[1],
            "feature_map.backbone.layer2.0.conv2.2": self.feature_map.backbone.layer2[0].conv2[2],
            "feature_map.backbone.layer2.1": self.feature_map.backbone.layer2[1],
            "feature_map.backbone.layer2.1.conv1": self.feature_map.backbone.layer2[1].conv1,
            "feature_map.backbone.layer2.1.conv1.0": self.feature_map.backbone.layer2[1].conv1[0],
            "feature_map.backbone.layer2.1.conv1.1": self.feature_map.backbone.layer2[1].conv1[1],
            "feature_map.backbone.layer2.1.conv1.2": self.feature_map.backbone.layer2[1].conv1[2],
            "feature_map.backbone.layer2.1.conv2": self.feature_map.backbone.layer2[1].conv2,
            "feature_map.backbone.layer2.1.conv2.0": self.feature_map.backbone.layer2[1].conv2[0],
            "feature_map.backbone.layer2.1.conv2.1": self.feature_map.backbone.layer2[1].conv2[1],
            "feature_map.backbone.layer2.1.conv2.2": self.feature_map.backbone.layer2[1].conv2[2],
            "feature_map.backbone.conv3": self.feature_map.backbone.conv3,
            "feature_map.backbone.conv3.0": self.feature_map.backbone.conv3[0],
            "feature_map.backbone.conv3.1": self.feature_map.backbone.conv3[1],
            "feature_map.backbone.conv3.2": self.feature_map.backbone.conv3[2],
            "feature_map.backbone.layer3": self.feature_map.backbone.layer3,
            "feature_map.backbone.layer3.0": self.feature_map.backbone.layer3[0],
            "feature_map.backbone.layer3.0.conv1": self.feature_map.backbone.layer3[0].conv1,
            "feature_map.backbone.layer3.0.conv1.0": self.feature_map.backbone.layer3[0].conv1[0],
            "feature_map.backbone.layer3.0.conv1.1": self.feature_map.backbone.layer3[0].conv1[1],
            "feature_map.backbone.layer3.0.conv1.2": self.feature_map.backbone.layer3[0].conv1[2],
            "feature_map.backbone.layer3.0.conv2": self.feature_map.backbone.layer3[0].conv2,
            "feature_map.backbone.layer3.0.conv2.0": self.feature_map.backbone.layer3[0].conv2[0],
            "feature_map.backbone.layer3.0.conv2.1": self.feature_map.backbone.layer3[0].conv2[1],
            "feature_map.backbone.layer3.0.conv2.2": self.feature_map.backbone.layer3[0].conv2[2],
            "feature_map.backbone.layer3.1": self.feature_map.backbone.layer3[1],
            "feature_map.backbone.layer3.1.conv1": self.feature_map.backbone.layer3[1].conv1,
            "feature_map.backbone.layer3.1.conv1.0": self.feature_map.backbone.layer3[1].conv1[0],
            "feature_map.backbone.layer3.1.conv1.1": self.feature_map.backbone.layer3[1].conv1[1],
            "feature_map.backbone.layer3.1.conv1.2": self.feature_map.backbone.layer3[1].conv1[2],
            "feature_map.backbone.layer3.1.conv2": self.feature_map.backbone.layer3[1].conv2,
            "feature_map.backbone.layer3.1.conv2.0": self.feature_map.backbone.layer3[1].conv2[0],
            "feature_map.backbone.layer3.1.conv2.1": self.feature_map.backbone.layer3[1].conv2[1],
            "feature_map.backbone.layer3.1.conv2.2": self.feature_map.backbone.layer3[1].conv2[2],
            "feature_map.backbone.layer3.2": self.feature_map.backbone.layer3[2],
            "feature_map.backbone.layer3.2.conv1": self.feature_map.backbone.layer3[2].conv1,
            "feature_map.backbone.layer3.2.conv1.0": self.feature_map.backbone.layer3[2].conv1[0],
            "feature_map.backbone.layer3.2.conv1.1": self.feature_map.backbone.layer3[2].conv1[1],
            "feature_map.backbone.layer3.2.conv1.2": self.feature_map.backbone.layer3[2].conv1[2],
            "feature_map.backbone.layer3.2.conv2": self.feature_map.backbone.layer3[2].conv2,
            "feature_map.backbone.layer3.2.conv2.0": self.feature_map.backbone.layer3[2].conv2[0],
            "feature_map.backbone.layer3.2.conv2.1": self.feature_map.backbone.layer3[2].conv2[1],
            "feature_map.backbone.layer3.2.conv2.2": self.feature_map.backbone.layer3[2].conv2[2],
            "feature_map.backbone.layer3.3": self.feature_map.backbone.layer3[3],
            "feature_map.backbone.layer3.3.conv1": self.feature_map.backbone.layer3[3].conv1,
            "feature_map.backbone.layer3.3.conv1.0": self.feature_map.backbone.layer3[3].conv1[0],
            "feature_map.backbone.layer3.3.conv1.1": self.feature_map.backbone.layer3[3].conv1[1],
            "feature_map.backbone.layer3.3.conv1.2": self.feature_map.backbone.layer3[3].conv1[2],
            "feature_map.backbone.layer3.3.conv2": self.feature_map.backbone.layer3[3].conv2,
            "feature_map.backbone.layer3.3.conv2.0": self.feature_map.backbone.layer3[3].conv2[0],
            "feature_map.backbone.layer3.3.conv2.1": self.feature_map.backbone.layer3[3].conv2[1],
            "feature_map.backbone.layer3.3.conv2.2": self.feature_map.backbone.layer3[3].conv2[2],
            "feature_map.backbone.layer3.4": self.feature_map.backbone.layer3[4],
            "feature_map.backbone.layer3.4.conv1": self.feature_map.backbone.layer3[4].conv1,
            "feature_map.backbone.layer3.4.conv1.0": self.feature_map.backbone.layer3[4].conv1[0],
            "feature_map.backbone.layer3.4.conv1.1": self.feature_map.backbone.layer3[4].conv1[1],
            "feature_map.backbone.layer3.4.conv1.2": self.feature_map.backbone.layer3[4].conv1[2],
            "feature_map.backbone.layer3.4.conv2": self.feature_map.backbone.layer3[4].conv2,
            "feature_map.backbone.layer3.4.conv2.0": self.feature_map.backbone.layer3[4].conv2[0],
            "feature_map.backbone.layer3.4.conv2.1": self.feature_map.backbone.layer3[4].conv2[1],
            "feature_map.backbone.layer3.4.conv2.2": self.feature_map.backbone.layer3[4].conv2[2],
            "feature_map.backbone.layer3.5": self.feature_map.backbone.layer3[5],
            "feature_map.backbone.layer3.5.conv1": self.feature_map.backbone.layer3[5].conv1,
            "feature_map.backbone.layer3.5.conv1.0": self.feature_map.backbone.layer3[5].conv1[0],
            "feature_map.backbone.layer3.5.conv1.1": self.feature_map.backbone.layer3[5].conv1[1],
            "feature_map.backbone.layer3.5.conv1.2": self.feature_map.backbone.layer3[5].conv1[2],
            "feature_map.backbone.layer3.5.conv2": self.feature_map.backbone.layer3[5].conv2,
            "feature_map.backbone.layer3.5.conv2.0": self.feature_map.backbone.layer3[5].conv2[0],
            "feature_map.backbone.layer3.5.conv2.1": self.feature_map.backbone.layer3[5].conv2[1],
            "feature_map.backbone.layer3.5.conv2.2": self.feature_map.backbone.layer3[5].conv2[2],
            "feature_map.backbone.layer3.6": self.feature_map.backbone.layer3[6],
            "feature_map.backbone.layer3.6.conv1": self.feature_map.backbone.layer3[6].conv1,
            "feature_map.backbone.layer3.6.conv1.0": self.feature_map.backbone.layer3[6].conv1[0],
            "feature_map.backbone.layer3.6.conv1.1": self.feature_map.backbone.layer3[6].conv1[1],
            "feature_map.backbone.layer3.6.conv1.2": self.feature_map.backbone.layer3[6].conv1[2],
            "feature_map.backbone.layer3.6.conv2": self.feature_map.backbone.layer3[6].conv2,
            "feature_map.backbone.layer3.6.conv2.0": self.feature_map.backbone.layer3[6].conv2[0],
            "feature_map.backbone.layer3.6.conv2.1": self.feature_map.backbone.layer3[6].conv2[1],
            "feature_map.backbone.layer3.6.conv2.2": self.feature_map.backbone.layer3[6].conv2[2],
            "feature_map.backbone.layer3.7": self.feature_map.backbone.layer3[7],
            "feature_map.backbone.layer3.7.conv1": self.feature_map.backbone.layer3[7].conv1,
            "feature_map.backbone.layer3.7.conv1.0": self.feature_map.backbone.layer3[7].conv1[0],
            "feature_map.backbone.layer3.7.conv1.1": self.feature_map.backbone.layer3[7].conv1[1],
            "feature_map.backbone.layer3.7.conv1.2": self.feature_map.backbone.layer3[7].conv1[2],
            "feature_map.backbone.layer3.7.conv2": self.feature_map.backbone.layer3[7].conv2,
            "feature_map.backbone.layer3.7.conv2.0": self.feature_map.backbone.layer3[7].conv2[0],
            "feature_map.backbone.layer3.7.conv2.1": self.feature_map.backbone.layer3[7].conv2[1],
            "feature_map.backbone.layer3.7.conv2.2": self.feature_map.backbone.layer3[7].conv2[2],
            "feature_map.backbone.conv4": self.feature_map.backbone.conv4,
            "feature_map.backbone.conv4.0": self.feature_map.backbone.conv4[0],
            "feature_map.backbone.conv4.1": self.feature_map.backbone.conv4[1],
            "feature_map.backbone.conv4.2": self.feature_map.backbone.conv4[2],
            "feature_map.backbone.layer4": self.feature_map.backbone.layer4,
            "feature_map.backbone.layer4.0": self.feature_map.backbone.layer4[0],
            "feature_map.backbone.layer4.0.conv1": self.feature_map.backbone.layer4[0].conv1,
            "feature_map.backbone.layer4.0.conv1.0": self.feature_map.backbone.layer4[0].conv1[0],
            "feature_map.backbone.layer4.0.conv1.1": self.feature_map.backbone.layer4[0].conv1[1],
            "feature_map.backbone.layer4.0.conv1.2": self.feature_map.backbone.layer4[0].conv1[2],
            "feature_map.backbone.layer4.0.conv2": self.feature_map.backbone.layer4[0].conv2,
            "feature_map.backbone.layer4.0.conv2.0": self.feature_map.backbone.layer4[0].conv2[0],
            "feature_map.backbone.layer4.0.conv2.1": self.feature_map.backbone.layer4[0].conv2[1],
            "feature_map.backbone.layer4.0.conv2.2": self.feature_map.backbone.layer4[0].conv2[2],
            "feature_map.backbone.layer4.1": self.feature_map.backbone.layer4[1],
            "feature_map.backbone.layer4.1.conv1": self.feature_map.backbone.layer4[1].conv1,
            "feature_map.backbone.layer4.1.conv1.0": self.feature_map.backbone.layer4[1].conv1[0],
            "feature_map.backbone.layer4.1.conv1.1": self.feature_map.backbone.layer4[1].conv1[1],
            "feature_map.backbone.layer4.1.conv1.2": self.feature_map.backbone.layer4[1].conv1[2],
            "feature_map.backbone.layer4.1.conv2": self.feature_map.backbone.layer4[1].conv2,
            "feature_map.backbone.layer4.1.conv2.0": self.feature_map.backbone.layer4[1].conv2[0],
            "feature_map.backbone.layer4.1.conv2.1": self.feature_map.backbone.layer4[1].conv2[1],
            "feature_map.backbone.layer4.1.conv2.2": self.feature_map.backbone.layer4[1].conv2[2],
            "feature_map.backbone.layer4.2": self.feature_map.backbone.layer4[2],
            "feature_map.backbone.layer4.2.conv1": self.feature_map.backbone.layer4[2].conv1,
            "feature_map.backbone.layer4.2.conv1.0": self.feature_map.backbone.layer4[2].conv1[0],
            "feature_map.backbone.layer4.2.conv1.1": self.feature_map.backbone.layer4[2].conv1[1],
            "feature_map.backbone.layer4.2.conv1.2": self.feature_map.backbone.layer4[2].conv1[2],
            "feature_map.backbone.layer4.2.conv2": self.feature_map.backbone.layer4[2].conv2,
            "feature_map.backbone.layer4.2.conv2.0": self.feature_map.backbone.layer4[2].conv2[0],
            "feature_map.backbone.layer4.2.conv2.1": self.feature_map.backbone.layer4[2].conv2[1],
            "feature_map.backbone.layer4.2.conv2.2": self.feature_map.backbone.layer4[2].conv2[2],
            "feature_map.backbone.layer4.3": self.feature_map.backbone.layer4[3],
            "feature_map.backbone.layer4.3.conv1": self.feature_map.backbone.layer4[3].conv1,
            "feature_map.backbone.layer4.3.conv1.0": self.feature_map.backbone.layer4[3].conv1[0],
            "feature_map.backbone.layer4.3.conv1.1": self.feature_map.backbone.layer4[3].conv1[1],
            "feature_map.backbone.layer4.3.conv1.2": self.feature_map.backbone.layer4[3].conv1[2],
            "feature_map.backbone.layer4.3.conv2": self.feature_map.backbone.layer4[3].conv2,
            "feature_map.backbone.layer4.3.conv2.0": self.feature_map.backbone.layer4[3].conv2[0],
            "feature_map.backbone.layer4.3.conv2.1": self.feature_map.backbone.layer4[3].conv2[1],
            "feature_map.backbone.layer4.3.conv2.2": self.feature_map.backbone.layer4[3].conv2[2],
            "feature_map.backbone.layer4.4": self.feature_map.backbone.layer4[4],
            "feature_map.backbone.layer4.4.conv1": self.feature_map.backbone.layer4[4].conv1,
            "feature_map.backbone.layer4.4.conv1.0": self.feature_map.backbone.layer4[4].conv1[0],
            "feature_map.backbone.layer4.4.conv1.1": self.feature_map.backbone.layer4[4].conv1[1],
            "feature_map.backbone.layer4.4.conv1.2": self.feature_map.backbone.layer4[4].conv1[2],
            "feature_map.backbone.layer4.4.conv2": self.feature_map.backbone.layer4[4].conv2,
            "feature_map.backbone.layer4.4.conv2.0": self.feature_map.backbone.layer4[4].conv2[0],
            "feature_map.backbone.layer4.4.conv2.1": self.feature_map.backbone.layer4[4].conv2[1],
            "feature_map.backbone.layer4.4.conv2.2": self.feature_map.backbone.layer4[4].conv2[2],
            "feature_map.backbone.layer4.5": self.feature_map.backbone.layer4[5],
            "feature_map.backbone.layer4.5.conv1": self.feature_map.backbone.layer4[5].conv1,
            "feature_map.backbone.layer4.5.conv1.0": self.feature_map.backbone.layer4[5].conv1[0],
            "feature_map.backbone.layer4.5.conv1.1": self.feature_map.backbone.layer4[5].conv1[1],
            "feature_map.backbone.layer4.5.conv1.2": self.feature_map.backbone.layer4[5].conv1[2],
            "feature_map.backbone.layer4.5.conv2": self.feature_map.backbone.layer4[5].conv2,
            "feature_map.backbone.layer4.5.conv2.0": self.feature_map.backbone.layer4[5].conv2[0],
            "feature_map.backbone.layer4.5.conv2.1": self.feature_map.backbone.layer4[5].conv2[1],
            "feature_map.backbone.layer4.5.conv2.2": self.feature_map.backbone.layer4[5].conv2[2],
            "feature_map.backbone.layer4.6": self.feature_map.backbone.layer4[6],
            "feature_map.backbone.layer4.6.conv1": self.feature_map.backbone.layer4[6].conv1,
            "feature_map.backbone.layer4.6.conv1.0": self.feature_map.backbone.layer4[6].conv1[0],
            "feature_map.backbone.layer4.6.conv1.1": self.feature_map.backbone.layer4[6].conv1[1],
            "feature_map.backbone.layer4.6.conv1.2": self.feature_map.backbone.layer4[6].conv1[2],
            "feature_map.backbone.layer4.6.conv2": self.feature_map.backbone.layer4[6].conv2,
            "feature_map.backbone.layer4.6.conv2.0": self.feature_map.backbone.layer4[6].conv2[0],
            "feature_map.backbone.layer4.6.conv2.1": self.feature_map.backbone.layer4[6].conv2[1],
            "feature_map.backbone.layer4.6.conv2.2": self.feature_map.backbone.layer4[6].conv2[2],
            "feature_map.backbone.layer4.7": self.feature_map.backbone.layer4[7],
            "feature_map.backbone.layer4.7.conv1": self.feature_map.backbone.layer4[7].conv1,
            "feature_map.backbone.layer4.7.conv1.0": self.feature_map.backbone.layer4[7].conv1[0],
            "feature_map.backbone.layer4.7.conv1.1": self.feature_map.backbone.layer4[7].conv1[1],
            "feature_map.backbone.layer4.7.conv1.2": self.feature_map.backbone.layer4[7].conv1[2],
            "feature_map.backbone.layer4.7.conv2": self.feature_map.backbone.layer4[7].conv2,
            "feature_map.backbone.layer4.7.conv2.0": self.feature_map.backbone.layer4[7].conv2[0],
            "feature_map.backbone.layer4.7.conv2.1": self.feature_map.backbone.layer4[7].conv2[1],
            "feature_map.backbone.layer4.7.conv2.2": self.feature_map.backbone.layer4[7].conv2[2],
            "feature_map.backbone.conv5": self.feature_map.backbone.conv5,
            "feature_map.backbone.conv5.0": self.feature_map.backbone.conv5[0],
            "feature_map.backbone.conv5.1": self.feature_map.backbone.conv5[1],
            "feature_map.backbone.conv5.2": self.feature_map.backbone.conv5[2],
            "feature_map.backbone.layer5": self.feature_map.backbone.layer5,
            "feature_map.backbone.layer5.0": self.feature_map.backbone.layer5[0],
            "feature_map.backbone.layer5.0.conv1": self.feature_map.backbone.layer5[0].conv1,
            "feature_map.backbone.layer5.0.conv1.0": self.feature_map.backbone.layer5[0].conv1[0],
            "feature_map.backbone.layer5.0.conv1.1": self.feature_map.backbone.layer5[0].conv1[1],
            "feature_map.backbone.layer5.0.conv1.2": self.feature_map.backbone.layer5[0].conv1[2],
            "feature_map.backbone.layer5.0.conv2": self.feature_map.backbone.layer5[0].conv2,
            "feature_map.backbone.layer5.0.conv2.0": self.feature_map.backbone.layer5[0].conv2[0],
            "feature_map.backbone.layer5.0.conv2.1": self.feature_map.backbone.layer5[0].conv2[1],
            "feature_map.backbone.layer5.0.conv2.2": self.feature_map.backbone.layer5[0].conv2[2],
            "feature_map.backbone.layer5.1": self.feature_map.backbone.layer5[1],
            "feature_map.backbone.layer5.1.conv1": self.feature_map.backbone.layer5[1].conv1,
            "feature_map.backbone.layer5.1.conv1.0": self.feature_map.backbone.layer5[1].conv1[0],
            "feature_map.backbone.layer5.1.conv1.1": self.feature_map.backbone.layer5[1].conv1[1],
            "feature_map.backbone.layer5.1.conv1.2": self.feature_map.backbone.layer5[1].conv1[2],
            "feature_map.backbone.layer5.1.conv2": self.feature_map.backbone.layer5[1].conv2,
            "feature_map.backbone.layer5.1.conv2.0": self.feature_map.backbone.layer5[1].conv2[0],
            "feature_map.backbone.layer5.1.conv2.1": self.feature_map.backbone.layer5[1].conv2[1],
            "feature_map.backbone.layer5.1.conv2.2": self.feature_map.backbone.layer5[1].conv2[2],
            "feature_map.backbone.layer5.2": self.feature_map.backbone.layer5[2],
            "feature_map.backbone.layer5.2.conv1": self.feature_map.backbone.layer5[2].conv1,
            "feature_map.backbone.layer5.2.conv1.0": self.feature_map.backbone.layer5[2].conv1[0],
            "feature_map.backbone.layer5.2.conv1.1": self.feature_map.backbone.layer5[2].conv1[1],
            "feature_map.backbone.layer5.2.conv1.2": self.feature_map.backbone.layer5[2].conv1[2],
            "feature_map.backbone.layer5.2.conv2": self.feature_map.backbone.layer5[2].conv2,
            "feature_map.backbone.layer5.2.conv2.0": self.feature_map.backbone.layer5[2].conv2[0],
            "feature_map.backbone.layer5.2.conv2.1": self.feature_map.backbone.layer5[2].conv2[1],
            "feature_map.backbone.layer5.2.conv2.2": self.feature_map.backbone.layer5[2].conv2[2],
            "feature_map.backbone.layer5.3": self.feature_map.backbone.layer5[3],
            "feature_map.backbone.layer5.3.conv1": self.feature_map.backbone.layer5[3].conv1,
            "feature_map.backbone.layer5.3.conv1.0": self.feature_map.backbone.layer5[3].conv1[0],
            "feature_map.backbone.layer5.3.conv1.1": self.feature_map.backbone.layer5[3].conv1[1],
            "feature_map.backbone.layer5.3.conv1.2": self.feature_map.backbone.layer5[3].conv1[2],
            "feature_map.backbone.layer5.3.conv2": self.feature_map.backbone.layer5[3].conv2,
            "feature_map.backbone.layer5.3.conv2.0": self.feature_map.backbone.layer5[3].conv2[0],
            "feature_map.backbone.layer5.3.conv2.1": self.feature_map.backbone.layer5[3].conv2[1],
            "feature_map.backbone.layer5.3.conv2.2": self.feature_map.backbone.layer5[3].conv2[2],
            # "feature_map.backblock0": self.feature_map.backblock0,
            # "feature_map.backblock0.conv0": self.feature_map.backblock0.conv0,
            # "feature_map.backblock0.conv0.0": self.feature_map.backblock0.conv0[0],
            # "feature_map.backblock0.conv0.1": self.feature_map.backblock0.conv0[1],
            # "feature_map.backblock0.conv0.2": self.feature_map.backblock0.conv0[2],
            # "feature_map.backblock0.conv1": self.feature_map.backblock0.conv1,
            # "feature_map.backblock0.conv1.0": self.feature_map.backblock0.conv1[0],
            # "feature_map.backblock0.conv1.1": self.feature_map.backblock0.conv1[1],
            # "feature_map.backblock0.conv1.2": self.feature_map.backblock0.conv1[2],
            # "feature_map.backblock0.conv2": self.feature_map.backblock0.conv2,
            # "feature_map.backblock0.conv2.0": self.feature_map.backblock0.conv2[0],
            # "feature_map.backblock0.conv2.1": self.feature_map.backblock0.conv2[1],
            # "feature_map.backblock0.conv2.2": self.feature_map.backblock0.conv2[2],
            # "feature_map.backblock0.conv3": self.feature_map.backblock0.conv3,
            # "feature_map.backblock0.conv3.0": self.feature_map.backblock0.conv3[0],
            # "feature_map.backblock0.conv3.1": self.feature_map.backblock0.conv3[1],
            # "feature_map.backblock0.conv3.2": self.feature_map.backblock0.conv3[2],
            # "feature_map.backblock0.conv4": self.feature_map.backblock0.conv4,
            # "feature_map.backblock0.conv4.0": self.feature_map.backblock0.conv4[0],
            # "feature_map.backblock0.conv4.1": self.feature_map.backblock0.conv4[1],
            # "feature_map.backblock0.conv4.2": self.feature_map.backblock0.conv4[2],
            # "feature_map.backblock0.conv5": self.feature_map.backblock0.conv5,
            # "feature_map.backblock0.conv5.0": self.feature_map.backblock0.conv5[0],
            # "feature_map.backblock0.conv5.1": self.feature_map.backblock0.conv5[1],
            # "feature_map.backblock0.conv5.2": self.feature_map.backblock0.conv5[2],
            # "feature_map.backblock0.conv6": self.feature_map.backblock0.conv6,
            # "feature_map.conv1": self.feature_map.conv1,
            # "feature_map.conv1.0": self.feature_map.conv1[0],
            # "feature_map.conv1.1": self.feature_map.conv1[1],
            # "feature_map.conv1.2": self.feature_map.conv1[2],
            # "feature_map.backblock1": self.feature_map.backblock1,
            # "feature_map.backblock1.conv0": self.feature_map.backblock1.conv0,
            # "feature_map.backblock1.conv0.0": self.feature_map.backblock1.conv0[0],
            # "feature_map.backblock1.conv0.1": self.feature_map.backblock1.conv0[1],
            # "feature_map.backblock1.conv0.2": self.feature_map.backblock1.conv0[2],
            # "feature_map.backblock1.conv1": self.feature_map.backblock1.conv1,
            # "feature_map.backblock1.conv1.0": self.feature_map.backblock1.conv1[0],
            # "feature_map.backblock1.conv1.1": self.feature_map.backblock1.conv1[1],
            # "feature_map.backblock1.conv1.2": self.feature_map.backblock1.conv1[2],
            # "feature_map.backblock1.conv2": self.feature_map.backblock1.conv2,
            # "feature_map.backblock1.conv2.0": self.feature_map.backblock1.conv2[0],
            # "feature_map.backblock1.conv2.1": self.feature_map.backblock1.conv2[1],
            # "feature_map.backblock1.conv2.2": self.feature_map.backblock1.conv2[2],
            # "feature_map.backblock1.conv3": self.feature_map.backblock1.conv3,
            # "feature_map.backblock1.conv3.0": self.feature_map.backblock1.conv3[0],
            # "feature_map.backblock1.conv3.1": self.feature_map.backblock1.conv3[1],
            # "feature_map.backblock1.conv3.2": self.feature_map.backblock1.conv3[2],
            # "feature_map.backblock1.conv4": self.feature_map.backblock1.conv4,
            # "feature_map.backblock1.conv4.0": self.feature_map.backblock1.conv4[0],
            # "feature_map.backblock1.conv4.1": self.feature_map.backblock1.conv4[1],
            # "feature_map.backblock1.conv4.2": self.feature_map.backblock1.conv4[2],
            # "feature_map.backblock1.conv5": self.feature_map.backblock1.conv5,
            # "feature_map.backblock1.conv5.0": self.feature_map.backblock1.conv5[0],
            # "feature_map.backblock1.conv5.1": self.feature_map.backblock1.conv5[1],
            # "feature_map.backblock1.conv5.2": self.feature_map.backblock1.conv5[2],
            # "feature_map.backblock1.conv6": self.feature_map.backblock1.conv6,
            # "feature_map.conv2": self.feature_map.conv2,
            # "feature_map.conv2.0": self.feature_map.conv2[0],
            # "feature_map.conv2.1": self.feature_map.conv2[1],
            # "feature_map.conv2.2": self.feature_map.conv2[2],
            # "feature_map.backblock2": self.feature_map.backblock2,
            # "feature_map.backblock2.conv0": self.feature_map.backblock2.conv0,
            # "feature_map.backblock2.conv0.0": self.feature_map.backblock2.conv0[0],
            # "feature_map.backblock2.conv0.1": self.feature_map.backblock2.conv0[1],
            # "feature_map.backblock2.conv0.2": self.feature_map.backblock2.conv0[2],
            # "feature_map.backblock2.conv1": self.feature_map.backblock2.conv1,
            # "feature_map.backblock2.conv1.0": self.feature_map.backblock2.conv1[0],
            # "feature_map.backblock2.conv1.1": self.feature_map.backblock2.conv1[1],
            # "feature_map.backblock2.conv1.2": self.feature_map.backblock2.conv1[2],
            # "feature_map.backblock2.conv2": self.feature_map.backblock2.conv2,
            # "feature_map.backblock2.conv2.0": self.feature_map.backblock2.conv2[0],
            # "feature_map.backblock2.conv2.1": self.feature_map.backblock2.conv2[1],
            # "feature_map.backblock2.conv2.2": self.feature_map.backblock2.conv2[2],
            # "feature_map.backblock2.conv3": self.feature_map.backblock2.conv3,
            # "feature_map.backblock2.conv3.0": self.feature_map.backblock2.conv3[0],
            # "feature_map.backblock2.conv3.1": self.feature_map.backblock2.conv3[1],
            # "feature_map.backblock2.conv3.2": self.feature_map.backblock2.conv3[2],
            # "feature_map.backblock2.conv4": self.feature_map.backblock2.conv4,
            # "feature_map.backblock2.conv4.0": self.feature_map.backblock2.conv4[0],
            # "feature_map.backblock2.conv4.1": self.feature_map.backblock2.conv4[1],
            # "feature_map.backblock2.conv4.2": self.feature_map.backblock2.conv4[2],
            # "feature_map.backblock2.conv5": self.feature_map.backblock2.conv5,
            # "feature_map.backblock2.conv5.0": self.feature_map.backblock2.conv5[0],
            # "feature_map.backblock2.conv5.1": self.feature_map.backblock2.conv5[1],
            # "feature_map.backblock2.conv5.2": self.feature_map.backblock2.conv5[2],
            # "feature_map.backblock2.conv6": self.feature_map.backblock2.conv6,
            # "detect_1": self.detect_1,
            # "detect_1.sigmoid": self.detect_1.sigmoid,
            # "detect_2": self.detect_2,
            # "detect_2.sigmoid": self.detect_2.sigmoid,
            # "detect_3": self.detect_3,
            # "detect_3.sigmoid": self.detect_3.sigmoid,

        }


        self.in_shapes = {
            "INPUT": [-1,3,416,416],
            'feature_map.backbone.layer4.2.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.3.conv1.0': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv3.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.1.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.1.conv1.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv5.0': [-1, 256, 26, 26], 'feature_map.backblock0.conv6': [-1, 1024, 13, 13],
            'feature_map.backbone.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.layer4.0.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.6.conv1.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv2.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer4.6.conv1.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer5.1.conv1.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer4.4.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.3.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer2.1.conv2.2': [-1, 128, 104, 104],
            # 'feature_map.backblock0.conv4.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv5.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.2.conv1.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.7.conv1.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv1.0': [-1, 256, 26, 26], 'feature_map.backblock1.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv1.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv5.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer2.0.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.layer4.7.conv1.1': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv0.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.0': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv1.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv1.2': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv0.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.0': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv4.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.3.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.5.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer2.1.conv1.0': [-1, 128, 104, 104],
            'feature_map.backbone.conv3.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer5.3.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer1.0.conv1.2': [-1, 32, 208, 208],
            'feature_map.backbone.layer5.2.conv1.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.2.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer5.2.conv2.0': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.2.conv1.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer2.1.conv1.2': [-1, 64, 104, 104],
            # 'feature_map.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer1.0.conv1.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer2.1.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.conv3.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.4.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.6.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer5.2.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.3.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.5.conv1.0': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv5.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv1.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv1.1': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv5.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.0.conv1.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer4.4.conv1.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.0.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv5.2': [-1, 256, 52, 52], 'feature_map.backblock2.conv1.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.2.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.0.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer2.0.conv2.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer3.3.conv2.1': [-1, 256, 52, 52],
            # 'feature_map.backblock0.conv3.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.4.conv1.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.layer5.3.conv2.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock1.conv1.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer2.0.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.layer5.1.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.0.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer1.0.conv2.0': [-1, 32, 208, 208],
            'feature_map.backbone.layer3.6.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv4.0': [-1, 256, 52, 52], 'feature_map.backblock2.conv6': [-1, 256, 52, 52],
            'feature_map.backbone.layer1.0.conv2.2': [-1, 64, 208, 208],
            'feature_map.backbone.layer4.2.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer5.0.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.5.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.2.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.2.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer5.3.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.2.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer5.0.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.6.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.3.conv2.0': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv5.1': [-1, 256, 52, 52], 'detect_3.sigmoid': [-1, 255, 52, 52],
            # 'feature_map.backblock1.conv4.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.2.conv1.2': [-1, 512, 13, 13],
            # 'feature_map.backblock1.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.conv4.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer2.1.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer4.6.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv1.0': [-1, 32, 416, 416],
            'feature_map.backbone.layer4.7.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv1.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv5.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock2.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.6.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.2': [-1, 512, 26, 26],
            # 'feature_map.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv4.2': [-1, 128, 52, 52], 'feature_map.backblock0.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.7.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.6.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv2.1': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv0.2': [-1, 128, 52, 52],
            'feature_map.backbone.conv3.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer5.3.conv1.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv1.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer2.1.conv2.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer5.1.conv2.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock1.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.0.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer2.0.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.conv1.2': [-1, 64, 208, 208],
            'feature_map.backbone.layer3.3.conv2.2': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv2.2': [-1, 128, 52, 52], 'feature_map.backblock0.conv0.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.1.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.backblock0.conv5.0': [-1, 512, 13, 13],
            'feature_map.backbone.conv0.1': [-1, 32, 416, 416],
            'feature_map.backbone.layer4.6.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.1.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.5.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.7.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.3.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.1.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv4.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv5.0': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv4.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer1.0.conv2.1': [-1, 64, 208, 208],
            # 'feature_map.backblock1.conv3.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer5.0.conv2.0': [-1, 512, 13, 13],
            # 'feature_map.backblock1.conv3.1': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv0.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.5.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.conv1.2': [-1, 256, 13, 13],
            # 'feature_map.backblock1.conv6': [-1, 512, 26, 26], 'feature_map.backblock0.conv4.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.7.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.2.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.0.conv2.1': [-1, 256, 52, 52],
            # 'feature_map.backblock0.conv2.2': [-1, 512, 13, 13], 'feature_map.backblock2.conv3.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer1.0.conv1.1': [-1, 32, 208, 208],
            # 'feature_map.backblock2.conv0.0': [-1, 128, 26, 26], 'feature_map.backblock0.conv3.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.4.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv0.0': [-1, 3, 416, 416],
            'feature_map.backbone.layer3.5.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.0.conv1.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv0.0': [-1, 256, 13, 13], 'feature_map.backblock1.conv4.1': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv1.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.5.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer2.0.conv1.2': [-1, 64, 104, 104],
            # 'feature_map.backblock0.conv0.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.6.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.1.conv1.1': [-1, 512, 13, 13],
            # 'detect_1.sigmoid': [-1, 255, 13, 13],
            'feature_map.backbone.layer3.7.conv1.0': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv3.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.7.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer2.0.conv1.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer4.3.conv1.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv3.1': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv5.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv4.1': [-1, 512, 26, 26], 'feature_map.backbone.conv0.2': [-1, 32, 416, 416],
            'feature_map.backbone.layer3.0.conv1.0': [-1, 256, 52, 52],
            # 'feature_map.backblock0.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.1.conv2.1': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv2.1': [-1, 128, 52, 52], 'feature_map.backblock0.conv4.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.3.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv2.2': [-1, 512, 26, 26],
            # 'feature_map.conv1.1': [-1, 256, 13, 13],
            'feature_map.backbone.layer4.1.conv1.1': [-1, 256, 26, 26],
            'INPUT': [-1, 3, 416, 416],
            'OUTPUT1': [-1, 256, 52, 52],
            'OUTPUT2': [-1, 512, 26, 26],
            'OUTPUT3': [-1, 1024, 13, 13]
            # 'OUTPUT1': [-1, 256, 26, 26],
            # 'OUTPUT2': [-1, 26, 26, 3, 2],
            # 'OUTPUT3': [-1, 52, 52, 3, 2]
        }
        self.out_shapes = {
            'INPUT': [-1, 3, 416, 416],
            'feature_map.backbone.conv0.0': [-1, 32, 416, 416],
            'feature_map.backbone.conv0.1': [-1, 32, 416, 416],
            'feature_map.backbone.conv0.2': [-1, 32, 416, 416],
            'feature_map.backbone.conv1.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv1.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv1.2': [-1, 64, 208, 208],
            'feature_map.backbone.layer1.0.conv1.0': [-1, 32, 208, 208],
            'feature_map.backbone.layer1.0.conv1.1': [-1, 32, 208, 208],
            'feature_map.backbone.layer1.0.conv1.2': [-1, 32, 208, 208],
            'feature_map.backbone.layer1.0.conv2.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer1.0.conv2.1': [-1, 64, 208, 208],
            'feature_map.backbone.layer1.0.conv2.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv2.0': [-1, 128, 104, 104],
            'feature_map.backbone.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.0.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv2.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.0.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.0.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.1.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv2.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.1.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.1.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.conv3.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv3.1': [-1, 256, 52, 52],
            'feature_map.backbone.conv3.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.3.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.3.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.3.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.5.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.5.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.5.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.conv4.0': [-1, 512, 26, 26],
            'feature_map.backbone.conv4.1': [-1, 512, 26, 26],
            'feature_map.backbone.conv4.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.6.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.6.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.6.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv5.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv5.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv5.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv2.2': [-1, 1024, 13, 13],
            'OUTPUT1': [-1, 256, 52, 52],
            'OUTPUT2': [-1, 512, 26, 26],
            'OUTPUT3': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv0.0': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv0.1': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv0.2': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv1.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv1.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv1.2': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv2.0': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv2.1': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv2.2': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv3.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv3.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv3.2': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv4.0': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv4.1': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv4.2': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv5.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv5.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv5.2': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv6': [-1, 255, 13, 13],
            # 'feature_map.conv1.0': [-1, 256, 13, 13],
            # 'feature_map.conv1.1': [-1, 256, 13, 13],
            # 'feature_map.conv1.2': [-1, 256, 13, 13],
            # 'feature_map.backblock1.conv0.0': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv0.1': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv0.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv1.0': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv1.1': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv1.2': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv2.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv3.0': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv3.1': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv3.2': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv4.0': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv4.1': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv4.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv5.0': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv5.1': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv5.2': [-1, 512, 26, 26],
            # 'feature_map.backblock1.conv6': [-1, 255, 26, 26],
            # 'feature_map.conv2.0': [-1, 128, 26, 26],
            # 'feature_map.conv2.1': [-1, 128, 26, 26],
            # 'feature_map.conv2.2': [-1, 128, 26, 26],
            # 'feature_map.backblock2.conv0.0': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv0.1': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv0.2': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv1.0': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv1.1': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv1.2': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv2.0': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv2.1': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv2.2': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv3.0': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv3.1': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv3.2': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv4.0': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv4.1': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv4.2': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv5.0': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv5.1': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv5.2': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv6': [-1, 255, 52, 52],
            # 'detect_1.sigmoid': [-1, 13, 13, 3, 2],
            # 'detect_2.sigmoid': [-1, 26, 26, 3, 2],
            # 'detect_3.sigmoid': [-1, 52, 52, 3, 2],
            # 'OUTPUT1': [-1, 13, 13, 3, 2],
            # 'OUTPUT2': [-1, 26, 26, 3, 2],
            # 'OUTPUT3': [-1, 52, 52, 3, 2]
        }
        self.orders = {
            'feature_map.backbone.conv0.0': ["INPUT", "feature_map.backbone.conv0.1"],
            'feature_map.backbone.conv0.1': ["feature_map.backbone.conv0.0", "feature_map.backbone.conv0.2"],
            'feature_map.backbone.conv0.2': ["feature_map.backbone.conv0.1", "feature_map.backbone.conv1.0"],
            'feature_map.backbone.conv1.0': ["feature_map.backbone.conv0.2", "feature_map.backbone.conv1.1"],
            'feature_map.backbone.conv1.1': ["feature_map.backbone.conv1.0", "feature_map.backbone.conv1.2"],
            'feature_map.backbone.conv1.2': ["feature_map.backbone.conv1.1", "feature_map.backbone.layer1.0.conv1.0"],
            'feature_map.backbone.layer1.0.conv1.0': ["feature_map.backbone.conv1.2",
                                                      "feature_map.backbone.layer1.0.conv1.1"],
            'feature_map.backbone.layer1.0.conv1.1': ["feature_map.backbone.layer1.0.conv1.0",
                                                      "feature_map.backbone.layer1.0.conv1.2"],
            'feature_map.backbone.layer1.0.conv1.2': ["feature_map.backbone.layer1.0.conv1.1",
                                                      "feature_map.backbone.layer1.0.conv2.0"],
            'feature_map.backbone.layer1.0.conv2.0': ["feature_map.backbone.layer1.0.conv1.2",
                                                      "feature_map.backbone.layer1.0.conv2.1"],
            'feature_map.backbone.layer1.0.conv2.1': ["feature_map.backbone.layer1.0.conv2.0",
                                                      "feature_map.backbone.layer1.0.conv2.2"],
            'feature_map.backbone.layer1.0.conv2.2': ["feature_map.backbone.layer1.0.conv2.1",
                                                      "feature_map.backbone.conv2.0"],
            'feature_map.backbone.conv2.0': ["feature_map.backbone.layer1.0.conv2.2", "feature_map.backbone.conv2.1"],
            'feature_map.backbone.conv2.1': ["feature_map.backbone.conv2.0", "feature_map.backbone.conv2.2"],
            'feature_map.backbone.conv2.2': ["feature_map.backbone.conv2.1", "feature_map.backbone.layer2.0.conv1.0"],
            'feature_map.backbone.layer2.0.conv1.0': ["feature_map.backbone.conv2.2",
                                                      "feature_map.backbone.layer2.0.conv1.1"],
            'feature_map.backbone.layer2.0.conv1.1': ["feature_map.backbone.layer2.0.conv1.0",
                                                      "feature_map.backbone.layer2.0.conv1.2"],
            'feature_map.backbone.layer2.0.conv1.2': ["feature_map.backbone.layer2.0.conv1.1",
                                                      "feature_map.backbone.layer2.0.conv2.0"],
            'feature_map.backbone.layer2.0.conv2.0': ["feature_map.backbone.layer2.0.conv1.2",
                                                      "feature_map.backbone.layer2.0.conv2.1"],
            'feature_map.backbone.layer2.0.conv2.1': ["feature_map.backbone.layer2.0.conv2.0",
                                                      "feature_map.backbone.layer2.0.conv2.2"],
            'feature_map.backbone.layer2.0.conv2.2': ["feature_map.backbone.layer2.0.conv2.1",
                                                      "feature_map.backbone.layer2.1.conv1.0"],
            'feature_map.backbone.layer2.1.conv1.0': ["feature_map.backbone.layer2.0.conv2.2",
                                                      "feature_map.backbone.layer2.1.conv1.1"],
            'feature_map.backbone.layer2.1.conv1.1': ["feature_map.backbone.layer2.1.conv1.0",
                                                      "feature_map.backbone.layer2.1.conv1.2"],
            'feature_map.backbone.layer2.1.conv1.2': ["feature_map.backbone.layer2.1.conv1.1",
                                                      "feature_map.backbone.layer2.1.conv2.0"],
            'feature_map.backbone.layer2.1.conv2.0': ["feature_map.backbone.layer2.1.conv1.2",
                                                      "feature_map.backbone.layer2.1.conv2.1"],
            'feature_map.backbone.layer2.1.conv2.1': ["feature_map.backbone.layer2.1.conv2.0",
                                                      "feature_map.backbone.layer2.1.conv2.2"],
            'feature_map.backbone.layer2.1.conv2.2': ["feature_map.backbone.layer2.1.conv2.1",
                                                      "feature_map.backbone.conv3.0"],
            'feature_map.backbone.conv3.0': ["feature_map.backbone.layer2.1.conv2.2", "feature_map.backbone.conv3.1"],
            'feature_map.backbone.conv3.1': ["feature_map.backbone.conv3.0", "feature_map.backbone.conv3.2"],
            'feature_map.backbone.conv3.2': ["feature_map.backbone.conv3.1", "feature_map.backbone.layer3.0.conv1.0"],
            'feature_map.backbone.layer3.0.conv1.0': ["feature_map.backbone.conv3.2",
                                                      "feature_map.backbone.layer3.0.conv1.1"],
            'feature_map.backbone.layer3.0.conv1.1': ["feature_map.backbone.layer3.0.conv1.0",
                                                      "feature_map.backbone.layer3.0.conv1.2"],
            'feature_map.backbone.layer3.0.conv1.2': ["feature_map.backbone.layer3.0.conv1.1",
                                                      "feature_map.backbone.layer3.0.conv2.0"],
            'feature_map.backbone.layer3.0.conv2.0': ["feature_map.backbone.layer3.0.conv1.2",
                                                      "feature_map.backbone.layer3.0.conv2.1"],
            'feature_map.backbone.layer3.0.conv2.1': ["feature_map.backbone.layer3.0.conv2.0",
                                                      "feature_map.backbone.layer3.0.conv2.2"],
            'feature_map.backbone.layer3.0.conv2.2': ["feature_map.backbone.layer3.0.conv2.1",
                                                      "feature_map.backbone.layer3.1.conv1.0"],
            'feature_map.backbone.layer3.1.conv1.0': ["feature_map.backbone.layer3.0.conv2.2",
                                                      "feature_map.backbone.layer3.1.conv1.1"],
            'feature_map.backbone.layer3.1.conv1.1': ["feature_map.backbone.layer3.1.conv1.0",
                                                      "feature_map.backbone.layer3.1.conv1.2"],
            'feature_map.backbone.layer3.1.conv1.2': ["feature_map.backbone.layer3.1.conv1.1",
                                                      "feature_map.backbone.layer3.1.conv2.0"],
            'feature_map.backbone.layer3.1.conv2.0': ["feature_map.backbone.layer3.1.conv1.2",
                                                      "feature_map.backbone.layer3.1.conv2.1"],
            'feature_map.backbone.layer3.1.conv2.1': ["feature_map.backbone.layer3.1.conv2.0",
                                                      "feature_map.backbone.layer3.1.conv2.2"],
            'feature_map.backbone.layer3.1.conv2.2': ["feature_map.backbone.layer3.1.conv2.1",
                                                      "feature_map.backbone.layer3.2.conv1.0"],
            'feature_map.backbone.layer3.2.conv1.0': ["feature_map.backbone.layer3.1.conv2.2",
                                                      "feature_map.backbone.layer3.2.conv1.1"],
            'feature_map.backbone.layer3.2.conv1.1': ["feature_map.backbone.layer3.2.conv1.0",
                                                      "feature_map.backbone.layer3.2.conv1.2"],
            'feature_map.backbone.layer3.2.conv1.2': ["feature_map.backbone.layer3.2.conv1.1",
                                                      "feature_map.backbone.layer3.2.conv2.0"],
            'feature_map.backbone.layer3.2.conv2.0': ["feature_map.backbone.layer3.2.conv1.2",
                                                      "feature_map.backbone.layer3.2.conv2.1"],
            'feature_map.backbone.layer3.2.conv2.1': ["feature_map.backbone.layer3.2.conv2.0",
                                                      "feature_map.backbone.layer3.2.conv2.2"],
            'feature_map.backbone.layer3.2.conv2.2': ["feature_map.backbone.layer3.2.conv2.1",
                                                      "feature_map.backbone.layer3.3.conv1.0"],
            'feature_map.backbone.layer3.3.conv1.0': ["feature_map.backbone.layer3.2.conv2.2",
                                                      "feature_map.backbone.layer3.3.conv1.1"],
            'feature_map.backbone.layer3.3.conv1.1': ["feature_map.backbone.layer3.3.conv1.0",
                                                      "feature_map.backbone.layer3.3.conv1.2"],
            'feature_map.backbone.layer3.3.conv1.2': ["feature_map.backbone.layer3.3.conv1.1",
                                                      "feature_map.backbone.layer3.3.conv2.0"],
            'feature_map.backbone.layer3.3.conv2.0': ["feature_map.backbone.layer3.3.conv1.2",
                                                      "feature_map.backbone.layer3.3.conv2.1"],
            'feature_map.backbone.layer3.3.conv2.1': ["feature_map.backbone.layer3.3.conv2.0",
                                                      "feature_map.backbone.layer3.3.conv2.2"],
            'feature_map.backbone.layer3.3.conv2.2': ["feature_map.backbone.layer3.3.conv2.1",
                                                      "feature_map.backbone.layer3.4.conv1.0"],
            'feature_map.backbone.layer3.4.conv1.0': ["feature_map.backbone.layer3.3.conv2.2",
                                                      "feature_map.backbone.layer3.4.conv1.1"],
            'feature_map.backbone.layer3.4.conv1.1': ["feature_map.backbone.layer3.4.conv1.0",
                                                      "feature_map.backbone.layer3.4.conv1.2"],
            'feature_map.backbone.layer3.4.conv1.2': ["feature_map.backbone.layer3.4.conv1.1",
                                                      "feature_map.backbone.layer3.4.conv2.0"],
            'feature_map.backbone.layer3.4.conv2.0': ["feature_map.backbone.layer3.4.conv1.2",
                                                      "feature_map.backbone.layer3.4.conv2.1"],
            'feature_map.backbone.layer3.4.conv2.1': ["feature_map.backbone.layer3.4.conv2.0",
                                                      "feature_map.backbone.layer3.4.conv2.2"],
            'feature_map.backbone.layer3.4.conv2.2': ["feature_map.backbone.layer3.4.conv2.1",
                                                      "feature_map.backbone.layer3.5.conv1.0"],
            'feature_map.backbone.layer3.5.conv1.0': ["feature_map.backbone.layer3.4.conv2.2",
                                                      "feature_map.backbone.layer3.5.conv1.1"],
            'feature_map.backbone.layer3.5.conv1.1': ["feature_map.backbone.layer3.5.conv1.0",
                                                      "feature_map.backbone.layer3.5.conv1.2"],
            'feature_map.backbone.layer3.5.conv1.2': ["feature_map.backbone.layer3.5.conv1.1",
                                                      "feature_map.backbone.layer3.5.conv2.0"],
            'feature_map.backbone.layer3.5.conv2.0': ["feature_map.backbone.layer3.5.conv1.2",
                                                      "feature_map.backbone.layer3.5.conv2.1"],
            'feature_map.backbone.layer3.5.conv2.1': ["feature_map.backbone.layer3.5.conv2.0",
                                                      "feature_map.backbone.layer3.5.conv2.2"],
            'feature_map.backbone.layer3.5.conv2.2': ["feature_map.backbone.layer3.5.conv2.1",
                                                      "feature_map.backbone.layer3.6.conv1.0"],
            'feature_map.backbone.layer3.6.conv1.0': ["feature_map.backbone.layer3.5.conv2.2",
                                                      "feature_map.backbone.layer3.6.conv1.1"],
            'feature_map.backbone.layer3.6.conv1.1': ["feature_map.backbone.layer3.6.conv1.0",
                                                      "feature_map.backbone.layer3.6.conv1.2"],
            'feature_map.backbone.layer3.6.conv1.2': ["feature_map.backbone.layer3.6.conv1.1",
                                                      "feature_map.backbone.layer3.6.conv2.0"],
            'feature_map.backbone.layer3.6.conv2.0': ["feature_map.backbone.layer3.6.conv1.2",
                                                      "feature_map.backbone.layer3.6.conv2.1"],
            'feature_map.backbone.layer3.6.conv2.1': ["feature_map.backbone.layer3.6.conv2.0",
                                                      "feature_map.backbone.layer3.6.conv2.2"],
            'feature_map.backbone.layer3.6.conv2.2': ["feature_map.backbone.layer3.6.conv2.1",
                                                      "feature_map.backbone.layer3.7.conv1.0"],
            'feature_map.backbone.layer3.7.conv1.0': ["feature_map.backbone.layer3.6.conv2.2",
                                                      "feature_map.backbone.layer3.7.conv1.1"],
            'feature_map.backbone.layer3.7.conv1.1': ["feature_map.backbone.layer3.7.conv1.0",
                                                      "feature_map.backbone.layer3.7.conv1.2"],
            'feature_map.backbone.layer3.7.conv1.2': ["feature_map.backbone.layer3.7.conv1.1",
                                                      "feature_map.backbone.layer3.7.conv2.0"],
            'feature_map.backbone.layer3.7.conv2.0': ["feature_map.backbone.layer3.7.conv1.2",
                                                      "feature_map.backbone.layer3.7.conv2.1"],
            'feature_map.backbone.layer3.7.conv2.1': ["feature_map.backbone.layer3.7.conv2.0",
                                                      "feature_map.backbone.layer3.7.conv2.2"],
            'feature_map.backbone.layer3.7.conv2.2': ["feature_map.backbone.layer3.7.conv2.1",
                                                      ["feature_map.backbone.conv4.0",
                                                       "OUTPUT1"]],  # layer3输出接到最后一个yoloblock前
            'feature_map.backbone.conv4.0': ["feature_map.backbone.layer3.7.conv2.2", "feature_map.backbone.conv4.1"],
            'feature_map.backbone.conv4.1': ["feature_map.backbone.conv4.0", "feature_map.backbone.conv4.2"],
            'feature_map.backbone.conv4.2': ["feature_map.backbone.conv4.1", "feature_map.backbone.layer4.0.conv1.0"],
            'feature_map.backbone.layer4.0.conv1.0': ["feature_map.backbone.conv4.2",
                                                      "feature_map.backbone.layer4.0.conv1.1"],
            'feature_map.backbone.layer4.0.conv1.1': ["feature_map.backbone.layer4.0.conv1.0",
                                                      "feature_map.backbone.layer4.0.conv1.2"],
            'feature_map.backbone.layer4.0.conv1.2': ["feature_map.backbone.layer4.0.conv1.1",
                                                      "feature_map.backbone.layer4.0.conv2.0"],
            'feature_map.backbone.layer4.0.conv2.0': ["feature_map.backbone.layer4.0.conv1.2",
                                                      "feature_map.backbone.layer4.0.conv2.1"],
            'feature_map.backbone.layer4.0.conv2.1': ["feature_map.backbone.layer4.0.conv2.0",
                                                      "feature_map.backbone.layer4.0.conv2.2"],
            'feature_map.backbone.layer4.0.conv2.2': ["feature_map.backbone.layer4.0.conv2.1",
                                                      "feature_map.backbone.layer4.1.conv1.0"],
            'feature_map.backbone.layer4.1.conv1.0': ["feature_map.backbone.layer4.0.conv2.2",
                                                      "feature_map.backbone.layer4.1.conv1.1"],
            'feature_map.backbone.layer4.1.conv1.1': ["feature_map.backbone.layer4.1.conv1.0",
                                                      "feature_map.backbone.layer4.1.conv1.2"],
            'feature_map.backbone.layer4.1.conv1.2': ["feature_map.backbone.layer4.1.conv1.1",
                                                      "feature_map.backbone.layer4.1.conv2.0"],
            'feature_map.backbone.layer4.1.conv2.0': ["feature_map.backbone.layer4.1.conv1.2",
                                                      "feature_map.backbone.layer4.1.conv2.1"],
            'feature_map.backbone.layer4.1.conv2.1': ["feature_map.backbone.layer4.1.conv2.0",
                                                      "feature_map.backbone.layer4.1.conv2.2"],
            'feature_map.backbone.layer4.1.conv2.2': ["feature_map.backbone.layer4.1.conv2.1",
                                                      "feature_map.backbone.layer4.2.conv1.0"],
            'feature_map.backbone.layer4.2.conv1.0': ["feature_map.backbone.layer4.1.conv2.2",
                                                      "feature_map.backbone.layer4.2.conv1.1"],
            'feature_map.backbone.layer4.2.conv1.1': ["feature_map.backbone.layer4.2.conv1.0",
                                                      "feature_map.backbone.layer4.2.conv1.2"],
            'feature_map.backbone.layer4.2.conv1.2': ["feature_map.backbone.layer4.2.conv1.1",
                                                      "feature_map.backbone.layer4.2.conv2.0"],
            'feature_map.backbone.layer4.2.conv2.0': ["feature_map.backbone.layer4.2.conv1.2",
                                                      "feature_map.backbone.layer4.2.conv2.1"],
            'feature_map.backbone.layer4.2.conv2.1': ["feature_map.backbone.layer4.2.conv2.0",
                                                      "feature_map.backbone.layer4.2.conv2.2"],
            'feature_map.backbone.layer4.2.conv2.2': ["feature_map.backbone.layer4.2.conv2.1",
                                                      "feature_map.backbone.layer4.3.conv1.0"],
            'feature_map.backbone.layer4.3.conv1.0': ["feature_map.backbone.layer4.2.conv2.2",
                                                      "feature_map.backbone.layer4.3.conv1.1"],
            'feature_map.backbone.layer4.3.conv1.1': ["feature_map.backbone.layer4.3.conv1.0",
                                                      "feature_map.backbone.layer4.3.conv1.2"],
            'feature_map.backbone.layer4.3.conv1.2': ["feature_map.backbone.layer4.3.conv1.1",
                                                      "feature_map.backbone.layer4.3.conv2.0"],
            'feature_map.backbone.layer4.3.conv2.0': ["feature_map.backbone.layer4.3.conv1.2",
                                                      "feature_map.backbone.layer4.3.conv2.1"],
            'feature_map.backbone.layer4.3.conv2.1': ["feature_map.backbone.layer4.3.conv2.0",
                                                      "feature_map.backbone.layer4.3.conv2.2"],
            'feature_map.backbone.layer4.3.conv2.2': ["feature_map.backbone.layer4.3.conv2.1",
                                                      "feature_map.backbone.layer4.4.conv1.0"],
            'feature_map.backbone.layer4.4.conv1.0': ["feature_map.backbone.layer4.3.conv2.2",
                                                      "feature_map.backbone.layer4.4.conv1.1"],
            'feature_map.backbone.layer4.4.conv1.1': ["feature_map.backbone.layer4.4.conv1.0",
                                                      "feature_map.backbone.layer4.4.conv1.2"],
            'feature_map.backbone.layer4.4.conv1.2': ["feature_map.backbone.layer4.4.conv1.1",
                                                      "feature_map.backbone.layer4.4.conv2.0"],
            'feature_map.backbone.layer4.4.conv2.0': ["feature_map.backbone.layer4.4.conv1.2",
                                                      "feature_map.backbone.layer4.4.conv2.1"],
            'feature_map.backbone.layer4.4.conv2.1': ["feature_map.backbone.layer4.4.conv2.0",
                                                      "feature_map.backbone.layer4.4.conv2.2"],
            'feature_map.backbone.layer4.4.conv2.2': ["feature_map.backbone.layer4.4.conv2.1",
                                                      "feature_map.backbone.layer4.5.conv1.0"],
            'feature_map.backbone.layer4.5.conv1.0': ["feature_map.backbone.layer4.4.conv2.2",
                                                      "feature_map.backbone.layer4.5.conv1.1"],
            'feature_map.backbone.layer4.5.conv1.1': ["feature_map.backbone.layer4.5.conv1.0",
                                                      "feature_map.backbone.layer4.5.conv1.2"],
            'feature_map.backbone.layer4.5.conv1.2': ["feature_map.backbone.layer4.5.conv1.1",
                                                      "feature_map.backbone.layer4.5.conv2.0"],
            'feature_map.backbone.layer4.5.conv2.0': ["feature_map.backbone.layer4.5.conv1.2",
                                                      "feature_map.backbone.layer4.5.conv2.1"],
            'feature_map.backbone.layer4.5.conv2.1': ["feature_map.backbone.layer4.5.conv2.0",
                                                      "feature_map.backbone.layer4.5.conv2.2"],
            'feature_map.backbone.layer4.5.conv2.2': ["feature_map.backbone.layer4.5.conv2.1",
                                                      "feature_map.backbone.layer4.6.conv1.0"],
            'feature_map.backbone.layer4.6.conv1.0': ["feature_map.backbone.layer4.5.conv2.2",
                                                      "feature_map.backbone.layer4.6.conv1.1"],
            'feature_map.backbone.layer4.6.conv1.1': ["feature_map.backbone.layer4.6.conv1.0",
                                                      "feature_map.backbone.layer4.6.conv1.2"],
            'feature_map.backbone.layer4.6.conv1.2': ["feature_map.backbone.layer4.6.conv1.1",
                                                      "feature_map.backbone.layer4.6.conv2.0"],
            'feature_map.backbone.layer4.6.conv2.0': ["feature_map.backbone.layer4.6.conv1.2",
                                                      "feature_map.backbone.layer4.6.conv2.1"],
            'feature_map.backbone.layer4.6.conv2.1': ["feature_map.backbone.layer4.6.conv2.0",
                                                      "feature_map.backbone.layer4.6.conv2.2"],
            'feature_map.backbone.layer4.6.conv2.2': ["feature_map.backbone.layer4.6.conv2.1",
                                                      "feature_map.backbone.layer4.7.conv1.0"],
            'feature_map.backbone.layer4.7.conv1.0': ["feature_map.backbone.layer4.6.conv2.2",
                                                      "feature_map.backbone.layer4.7.conv1.1"],
            'feature_map.backbone.layer4.7.conv1.1': ["feature_map.backbone.layer4.7.conv1.0",
                                                      "feature_map.backbone.layer4.7.conv1.2"],
            'feature_map.backbone.layer4.7.conv1.2': ["feature_map.backbone.layer4.7.conv1.1",
                                                      "feature_map.backbone.layer4.7.conv2.0"],
            'feature_map.backbone.layer4.7.conv2.0': ["feature_map.backbone.layer4.7.conv1.2",
                                                      "feature_map.backbone.layer4.7.conv2.1"],
            'feature_map.backbone.layer4.7.conv2.1': ["feature_map.backbone.layer4.7.conv2.0",
                                                      "feature_map.backbone.layer4.7.conv2.2"],
            'feature_map.backbone.layer4.7.conv2.2': ["feature_map.backbone.layer4.7.conv2.1",
                                                      ["feature_map.backbone.conv5.0",
                                                       "OUTPUT2"]],  # layer4接第二个yoloblock开头
            'feature_map.backbone.conv5.0': ["feature_map.backbone.layer4.7.conv2.2", "feature_map.backbone.conv5.1"],
            'feature_map.backbone.conv5.1': ["feature_map.backbone.conv5.0", "feature_map.backbone.conv5.2"],
            'feature_map.backbone.conv5.2': ["feature_map.backbone.conv5.1", "feature_map.backbone.layer5.0.conv1.0"],
            'feature_map.backbone.layer5.0.conv1.0': ["feature_map.backbone.conv5.2",
                                                      "feature_map.backbone.layer5.0.conv1.1"],
            'feature_map.backbone.layer5.0.conv1.1': ["feature_map.backbone.layer5.0.conv1.0",
                                                      "feature_map.backbone.layer5.0.conv1.2"],
            'feature_map.backbone.layer5.0.conv1.2': ["feature_map.backbone.layer5.0.conv1.1",
                                                      "feature_map.backbone.layer5.0.conv2.0"],
            'feature_map.backbone.layer5.0.conv2.0': ["feature_map.backbone.layer5.0.conv1.2",
                                                      "feature_map.backbone.layer5.0.conv2.1"],
            'feature_map.backbone.layer5.0.conv2.1': ["feature_map.backbone.layer5.0.conv2.0",
                                                      "feature_map.backbone.layer5.0.conv2.2"],
            'feature_map.backbone.layer5.0.conv2.2': ["feature_map.backbone.layer5.0.conv2.1",
                                                      "feature_map.backbone.layer5.1.conv1.0"],
            'feature_map.backbone.layer5.1.conv1.0': ["feature_map.backbone.layer5.0.conv2.2",
                                                      "feature_map.backbone.layer5.1.conv1.1"],
            'feature_map.backbone.layer5.1.conv1.1': ["feature_map.backbone.layer5.1.conv1.0",
                                                      "feature_map.backbone.layer5.1.conv1.2"],
            'feature_map.backbone.layer5.1.conv1.2': ["feature_map.backbone.layer5.1.conv1.1",
                                                      "feature_map.backbone.layer5.1.conv2.0"],
            'feature_map.backbone.layer5.1.conv2.0': ["feature_map.backbone.layer5.1.conv1.2",
                                                      "feature_map.backbone.layer5.1.conv2.1"],
            'feature_map.backbone.layer5.1.conv2.1': ["feature_map.backbone.layer5.1.conv2.0",
                                                      "feature_map.backbone.layer5.1.conv2.2"],
            'feature_map.backbone.layer5.1.conv2.2': ["feature_map.backbone.layer5.1.conv2.1",
                                                      "feature_map.backbone.layer5.2.conv1.0"],
            'feature_map.backbone.layer5.2.conv1.0': ["feature_map.backbone.layer5.1.conv2.2",
                                                      "feature_map.backbone.layer5.2.conv1.1"],
            'feature_map.backbone.layer5.2.conv1.1': ["feature_map.backbone.layer5.2.conv1.0",
                                                      "feature_map.backbone.layer5.2.conv1.2"],
            'feature_map.backbone.layer5.2.conv1.2': ["feature_map.backbone.layer5.2.conv1.1",
                                                      "feature_map.backbone.layer5.2.conv2.0"],
            'feature_map.backbone.layer5.2.conv2.0': ["feature_map.backbone.layer5.2.conv1.2",
                                                      "feature_map.backbone.layer5.2.conv2.1"],
            'feature_map.backbone.layer5.2.conv2.1': ["feature_map.backbone.layer5.2.conv2.0",
                                                      "feature_map.backbone.layer5.2.conv2.2"],
            'feature_map.backbone.layer5.2.conv2.2': ["feature_map.backbone.layer5.2.conv2.1",
                                                      "feature_map.backbone.layer5.3.conv1.0"],
            'feature_map.backbone.layer5.3.conv1.0': ["feature_map.backbone.layer5.2.conv2.2",
                                                      "feature_map.backbone.layer5.3.conv1.1"],
            'feature_map.backbone.layer5.3.conv1.1': ["feature_map.backbone.layer5.3.conv1.0",
                                                      "feature_map.backbone.layer5.3.conv1.2"],
            'feature_map.backbone.layer5.3.conv1.2': ["feature_map.backbone.layer5.3.conv1.1",
                                                      "feature_map.backbone.layer5.3.conv2.0"],
            'feature_map.backbone.layer5.3.conv2.0': ["feature_map.backbone.layer5.3.conv1.2",
                                                      "feature_map.backbone.layer5.3.conv2.1"],
            'feature_map.backbone.layer5.3.conv2.1': ["feature_map.backbone.layer5.3.conv2.0",
                                                      "feature_map.backbone.layer5.3.conv2.2"],
            'feature_map.backbone.layer5.3.conv2.2': ["feature_map.backbone.layer5.3.conv2.1",
                                                      'OUTPUT3'],  # layer5直接往下接第一个yoloblock

            # 'feature_map.backblock0.conv0.0': ["feature_map.backbone.layer5.3.conv2.2",
            #                                    "feature_map.backblock0.conv0.1"],
            # 'feature_map.backblock0.conv0.1': ["feature_map.backblock0.conv0.0", "feature_map.backblock0.conv0.2"],
            # 'feature_map.backblock0.conv0.2': ["feature_map.backblock0.conv0.1", "feature_map.backblock0.conv1.0"],
            # 'feature_map.backblock0.conv1.0': ["feature_map.backblock0.conv0.2", "feature_map.backblock0.conv1.1"],
            # 'feature_map.backblock0.conv1.1': ["feature_map.backblock0.conv1.0", "feature_map.backblock0.conv1.2"],
            # 'feature_map.backblock0.conv1.2': ["feature_map.backblock0.conv1.1", "feature_map.backblock0.conv2.0"],
            # 'feature_map.backblock0.conv2.0': ["feature_map.backblock0.conv1.2", "feature_map.backblock0.conv2.1"],
            # 'feature_map.backblock0.conv2.1': ["feature_map.backblock0.conv2.0", "feature_map.backblock0.conv2.2"],
            # 'feature_map.backblock0.conv2.2': ["feature_map.backblock0.conv2.1", "feature_map.backblock0.conv3.0"],
            # 'feature_map.backblock0.conv3.0': ["feature_map.backblock0.conv2.2", "feature_map.backblock0.conv3.1"],
            # 'feature_map.backblock0.conv3.1': ["feature_map.backblock0.conv3.0", "feature_map.backblock0.conv3.2"],
            # 'feature_map.backblock0.conv3.2': ["feature_map.backblock0.conv3.1", "feature_map.backblock0.conv4.0"],
            # 'feature_map.backblock0.conv4.0': ["feature_map.backblock0.conv3.2", "feature_map.backblock0.conv4.1"],
            # 'feature_map.backblock0.conv4.1': ["feature_map.backblock0.conv4.0", "feature_map.backblock0.conv4.2"],
            # 'feature_map.backblock0.conv4.2': ["feature_map.backblock0.conv4.1",
            #                                    ["feature_map.backblock0.conv5.0", "feature_map.conv1.0"]],
            # # 第一个yoloblock的分支处，接第二个yoloblock前的cbr
            # 'feature_map.backblock0.conv5.0': ["feature_map.backblock0.conv4.2", "feature_map.backblock0.conv5.1"],
            # 'feature_map.backblock0.conv5.1': ["feature_map.backblock0.conv5.0", "feature_map.backblock0.conv5.2"],
            # 'feature_map.backblock0.conv5.2': ["feature_map.backblock0.conv5.1", "feature_map.backblock0.conv6"],
            # 'feature_map.backblock0.conv6': ["feature_map.backblock0.conv5.2", "detect_1.sigmoid"],
            # 'feature_map.conv1.0': ["feature_map.backblock0.conv4.2", "feature_map.conv1.1"],  # 从第一个yoloblock分支直接接下来
            # 'feature_map.conv1.1': ["feature_map.conv1.0", "feature_map.conv1.2"],
            # 'feature_map.conv1.2': ["feature_map.conv1.1", "feature_map.backblock1.conv0.0"],
            # 'feature_map.backblock1.conv0.0': [["feature_map.conv1.2", "feature_map.backbone.layer4.7.conv2.2"],
            #                                    "feature_map.backblock1.conv0.1"],  # 第二个yoloblock的开头
            # 'feature_map.backblock1.conv0.1': ["feature_map.backblock1.conv0.0", "feature_map.backblock1.conv0.2"],
            # 'feature_map.backblock1.conv0.2': ["feature_map.backblock1.conv0.1", "feature_map.backblock1.conv1.0"],
            # 'feature_map.backblock1.conv1.0': ["feature_map.backblock1.conv0.2", "feature_map.backblock1.conv1.1"],
            # 'feature_map.backblock1.conv1.1': ["feature_map.backblock1.conv1.0", "feature_map.backblock1.conv1.2"],
            # 'feature_map.backblock1.conv1.2': ["feature_map.backblock1.conv1.1", "feature_map.backblock1.conv2.0"],
            # 'feature_map.backblock1.conv2.0': ["feature_map.backblock1.conv1.2", "feature_map.backblock1.conv2.1"],
            # 'feature_map.backblock1.conv2.1': ["feature_map.backblock1.conv2.0", "feature_map.backblock1.conv2.2"],
            # 'feature_map.backblock1.conv2.2': ["feature_map.backblock1.conv2.1", "feature_map.backblock1.conv3.0"],
            # 'feature_map.backblock1.conv3.0': ["feature_map.backblock1.conv2.2", "feature_map.backblock1.conv3.1"],
            # 'feature_map.backblock1.conv3.1': ["feature_map.backblock1.conv3.0", "feature_map.backblock1.conv3.2"],
            # 'feature_map.backblock1.conv3.2': ["feature_map.backblock1.conv3.1", "feature_map.backblock1.conv4.0"],
            # 'feature_map.backblock1.conv4.0': ["feature_map.backblock1.conv3.2", "feature_map.backblock1.conv4.1"],
            # 'feature_map.backblock1.conv4.1': ["feature_map.backblock1.conv4.0", "feature_map.backblock1.conv4.2"],
            # 'feature_map.backblock1.conv4.2': ["feature_map.backblock1.conv4.1",
            #                                    ["feature_map.backblock1.conv5.0", "feature_map.conv2.0"]],
            # # 第二个yoloblock的分支处，接第二个yoloblock前的cbr
            # 'feature_map.backblock1.conv5.0': ["feature_map.backblock1.conv4.2", "feature_map.backblock1.conv5.1"],
            # 'feature_map.backblock1.conv5.1': ["feature_map.backblock1.conv5.0", "feature_map.backblock1.conv5.2"],
            # 'feature_map.backblock1.conv5.2': ["feature_map.backblock1.conv5.1", "feature_map.backblock1.conv6"],
            # 'feature_map.backblock1.conv6': ["feature_map.backblock1.conv5.2", "detect_2.sigmoid"],
            # 'feature_map.conv2.0': ["feature_map.backblock1.conv4.2", "feature_map.conv2.1"],  # 从第二个yoloblock分支直接接下来
            # 'feature_map.conv2.1': ["feature_map.conv2.0", "feature_map.conv2.2"],
            # 'feature_map.conv2.2': ["feature_map.conv2.1", "feature_map.backblock2.conv0.0"],
            # 'feature_map.backblock2.conv0.0': [["feature_map.conv2.2", "feature_map.backbone.layer3.7.conv2.2"],
            #                                    "feature_map.backblock2.conv0.1"],  # 第三个yoloblock的开头
            # 'feature_map.backblock2.conv0.1': ["feature_map.backblock2.conv0.0", "feature_map.backblock2.conv0.2"],
            # 'feature_map.backblock2.conv0.2': ["feature_map.backblock2.conv0.1", "feature_map.backblock2.conv1.0"],
            # 'feature_map.backblock2.conv1.0': ["feature_map.backblock2.conv0.2", "feature_map.backblock2.conv1.1"],
            # 'feature_map.backblock2.conv1.1': ["feature_map.backblock2.conv1.0", "feature_map.backblock2.conv1.2"],
            # 'feature_map.backblock2.conv1.2': ["feature_map.backblock2.conv1.1", "feature_map.backblock2.conv2.0"],
            # 'feature_map.backblock2.conv2.0': ["feature_map.backblock2.conv1.2", "feature_map.backblock2.conv2.1"],
            # 'feature_map.backblock2.conv2.1': ["feature_map.backblock2.conv2.0", "feature_map.backblock2.conv2.2"],
            # 'feature_map.backblock2.conv2.2': ["feature_map.backblock2.conv2.1", "feature_map.backblock2.conv3.0"],
            # 'feature_map.backblock2.conv3.0': ["feature_map.backblock2.conv2.2", "feature_map.backblock2.conv3.1"],
            # 'feature_map.backblock2.conv3.1': ["feature_map.backblock2.conv3.0", "feature_map.backblock2.conv3.2"],
            # 'feature_map.backblock2.conv3.2': ["feature_map.backblock2.conv3.1", "feature_map.backblock2.conv4.0"],
            # 'feature_map.backblock2.conv4.0': ["feature_map.backblock2.conv3.2", "feature_map.backblock2.conv4.1"],
            # 'feature_map.backblock2.conv4.1': ["feature_map.backblock2.conv4.0", "feature_map.backblock2.conv4.2"],
            # 'feature_map.backblock2.conv4.2': ["feature_map.backblock2.conv4.1", "feature_map.backblock2.conv5.0"],
            # 'feature_map.backblock2.conv5.0': ["feature_map.backblock2.conv4.2", "feature_map.backblock2.conv5.1"],
            # 'feature_map.backblock2.conv5.1': ["feature_map.backblock2.conv5.0", "feature_map.backblock2.conv5.2"],
            # 'feature_map.backblock2.conv5.2': ["feature_map.backblock2.conv5.1", "feature_map.backblock2.conv6"],
            # 'feature_map.backblock2.conv6': ["feature_map.backblock2.conv5.2", "detect_3.sigmoid"],
            # 'detect_1.sigmoid': ["feature_map.backblock0.conv6", "OUTPUT1"],  # 从第一个yoloblock分支接出来
            # 'detect_2.sigmoid': ["feature_map.backblock1.conv6", "OUTPUT2"],  # 从第二个yoloblock分支接出来
            # 'detect_3.sigmoid': ["feature_map.backblock2.conv6", "OUTPUT3"],
        }





    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):
        if 'feature_map' == layer_name:
            self.feature_map = new_layer
            self.layer_names["feature_map"] = new_layer
            self.origin_layer_names["feature_map"] = new_layer
        elif 'feature_map.backbone' == layer_name:
            self.feature_map.backbone = new_layer
            self.layer_names["feature_map.backbone"] = new_layer
            self.origin_layer_names["feature_map.backbone"] = new_layer
        elif 'feature_map.backbone.conv0' == layer_name:
            self.feature_map.backbone.conv0 = new_layer
            self.layer_names["feature_map.backbone.conv0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv0"] = new_layer
        elif 'feature_map.backbone.conv0.0' == layer_name:
            self.feature_map.backbone.conv0[0] = new_layer
            self.layer_names["feature_map.backbone.conv0.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv0.0"] = new_layer
        elif 'feature_map.backbone.conv0.1' == layer_name:
            self.feature_map.backbone.conv0[1] = new_layer
            self.layer_names["feature_map.backbone.conv0.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv0.1"] = new_layer
        elif 'feature_map.backbone.conv0.2' == layer_name:
            self.feature_map.backbone.conv0[2] = new_layer
            self.layer_names["feature_map.backbone.conv0.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv0.2"] = new_layer
        elif 'feature_map.backbone.conv1' == layer_name:
            self.feature_map.backbone.conv1 = new_layer
            self.layer_names["feature_map.backbone.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv1"] = new_layer
        elif 'feature_map.backbone.conv1.0' == layer_name:
            self.feature_map.backbone.conv1[0] = new_layer
            self.layer_names["feature_map.backbone.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv1.0"] = new_layer
        elif 'feature_map.backbone.conv1.1' == layer_name:
            self.feature_map.backbone.conv1[1] = new_layer
            self.layer_names["feature_map.backbone.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv1.1"] = new_layer
        elif 'feature_map.backbone.conv1.2' == layer_name:
            self.feature_map.backbone.conv1[2] = new_layer
            self.layer_names["feature_map.backbone.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer1' == layer_name:
            self.feature_map.backbone.layer1 = new_layer
            self.layer_names["feature_map.backbone.layer1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1"] = new_layer
        elif 'feature_map.backbone.layer1.0' == layer_name:
            self.feature_map.backbone.layer1[0] = new_layer
            self.layer_names["feature_map.backbone.layer1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv1' == layer_name:
            self.feature_map.backbone.layer1[0].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv2' == layer_name:
            self.feature_map.backbone.layer1[0].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer1.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.2"] = new_layer
        elif 'feature_map.backbone.conv2' == layer_name:
            self.feature_map.backbone.conv2 = new_layer
            self.layer_names["feature_map.backbone.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv2"] = new_layer
        elif 'feature_map.backbone.conv2.0' == layer_name:
            self.feature_map.backbone.conv2[0] = new_layer
            self.layer_names["feature_map.backbone.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv2.0"] = new_layer
        elif 'feature_map.backbone.conv2.1' == layer_name:
            self.feature_map.backbone.conv2[1] = new_layer
            self.layer_names["feature_map.backbone.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv2.1"] = new_layer
        elif 'feature_map.backbone.conv2.2' == layer_name:
            self.feature_map.backbone.conv2[2] = new_layer
            self.layer_names["feature_map.backbone.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer2' == layer_name:
            self.feature_map.backbone.layer2 = new_layer
            self.layer_names["feature_map.backbone.layer2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2"] = new_layer
        elif 'feature_map.backbone.layer2.0' == layer_name:
            self.feature_map.backbone.layer2[0] = new_layer
            self.layer_names["feature_map.backbone.layer2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer2.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer2.1' == layer_name:
            self.feature_map.backbone.layer2[1] = new_layer
            self.layer_names["feature_map.backbone.layer2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer2.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.2"] = new_layer
        elif 'feature_map.backbone.conv3' == layer_name:
            self.feature_map.backbone.conv3 = new_layer
            self.layer_names["feature_map.backbone.conv3"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv3"] = new_layer
        elif 'feature_map.backbone.conv3.0' == layer_name:
            self.feature_map.backbone.conv3[0] = new_layer
            self.layer_names["feature_map.backbone.conv3.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv3.0"] = new_layer
        elif 'feature_map.backbone.conv3.1' == layer_name:
            self.feature_map.backbone.conv3[1] = new_layer
            self.layer_names["feature_map.backbone.conv3.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv3.1"] = new_layer
        elif 'feature_map.backbone.conv3.2' == layer_name:
            self.feature_map.backbone.conv3[2] = new_layer
            self.layer_names["feature_map.backbone.conv3.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv3.2"] = new_layer
        elif 'feature_map.backbone.layer3' == layer_name:
            self.feature_map.backbone.layer3 = new_layer
            self.layer_names["feature_map.backbone.layer3"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3"] = new_layer
        elif 'feature_map.backbone.layer3.0' == layer_name:
            self.feature_map.backbone.layer3[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.1' == layer_name:
            self.feature_map.backbone.layer3[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.2' == layer_name:
            self.feature_map.backbone.layer3[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.3' == layer_name:
            self.feature_map.backbone.layer3[3] = new_layer
            self.layer_names["feature_map.backbone.layer3.3"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.4' == layer_name:
            self.feature_map.backbone.layer3[4] = new_layer
            self.layer_names["feature_map.backbone.layer3.4"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.5' == layer_name:
            self.feature_map.backbone.layer3[5] = new_layer
            self.layer_names["feature_map.backbone.layer3.5"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.6' == layer_name:
            self.feature_map.backbone.layer3[6] = new_layer
            self.layer_names["feature_map.backbone.layer3.6"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer3.7' == layer_name:
            self.feature_map.backbone.layer3[7] = new_layer
            self.layer_names["feature_map.backbone.layer3.7"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer3.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.2"] = new_layer
        elif 'feature_map.backbone.conv4' == layer_name:
            self.feature_map.backbone.conv4 = new_layer
            self.layer_names["feature_map.backbone.conv4"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv4"] = new_layer
        elif 'feature_map.backbone.conv4.0' == layer_name:
            self.feature_map.backbone.conv4[0] = new_layer
            self.layer_names["feature_map.backbone.conv4.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv4.0"] = new_layer
        elif 'feature_map.backbone.conv4.1' == layer_name:
            self.feature_map.backbone.conv4[1] = new_layer
            self.layer_names["feature_map.backbone.conv4.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv4.1"] = new_layer
        elif 'feature_map.backbone.conv4.2' == layer_name:
            self.feature_map.backbone.conv4[2] = new_layer
            self.layer_names["feature_map.backbone.conv4.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv4.2"] = new_layer
        elif 'feature_map.backbone.layer4' == layer_name:
            self.feature_map.backbone.layer4 = new_layer
            self.layer_names["feature_map.backbone.layer4"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4"] = new_layer
        elif 'feature_map.backbone.layer4.0' == layer_name:
            self.feature_map.backbone.layer4[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.1' == layer_name:
            self.feature_map.backbone.layer4[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.2' == layer_name:
            self.feature_map.backbone.layer4[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.3' == layer_name:
            self.feature_map.backbone.layer4[3] = new_layer
            self.layer_names["feature_map.backbone.layer4.3"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.4' == layer_name:
            self.feature_map.backbone.layer4[4] = new_layer
            self.layer_names["feature_map.backbone.layer4.4"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.5' == layer_name:
            self.feature_map.backbone.layer4[5] = new_layer
            self.layer_names["feature_map.backbone.layer4.5"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.6' == layer_name:
            self.feature_map.backbone.layer4[6] = new_layer
            self.layer_names["feature_map.backbone.layer4.6"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer4.7' == layer_name:
            self.feature_map.backbone.layer4[7] = new_layer
            self.layer_names["feature_map.backbone.layer4.7"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer4.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.2"] = new_layer
        elif 'feature_map.backbone.conv5' == layer_name:
            self.feature_map.backbone.conv5 = new_layer
            self.layer_names["feature_map.backbone.conv5"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv5"] = new_layer
        elif 'feature_map.backbone.conv5.0' == layer_name:
            self.feature_map.backbone.conv5[0] = new_layer
            self.layer_names["feature_map.backbone.conv5.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv5.0"] = new_layer
        elif 'feature_map.backbone.conv5.1' == layer_name:
            self.feature_map.backbone.conv5[1] = new_layer
            self.layer_names["feature_map.backbone.conv5.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv5.1"] = new_layer
        elif 'feature_map.backbone.conv5.2' == layer_name:
            self.feature_map.backbone.conv5[2] = new_layer
            self.layer_names["feature_map.backbone.conv5.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.conv5.2"] = new_layer
        elif 'feature_map.backbone.layer5' == layer_name:
            self.feature_map.backbone.layer5 = new_layer
            self.layer_names["feature_map.backbone.layer5"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5"] = new_layer
        elif 'feature_map.backbone.layer5.0' == layer_name:
            self.feature_map.backbone.layer5[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer5.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer5.1' == layer_name:
            self.feature_map.backbone.layer5[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer5.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer5.2' == layer_name:
            self.feature_map.backbone.layer5[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer5.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.2"] = new_layer
        elif 'feature_map.backbone.layer5.3' == layer_name:
            self.feature_map.backbone.layer5[3] = new_layer
            self.layer_names["feature_map.backbone.layer5.3"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1 = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.0"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.1"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.2"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2 = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[0] = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.0"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[1] = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.1"] = new_layer
        elif 'feature_map.backbone.layer5.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[2] = new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.2"] = new_layer
        elif 'feature_map.backblock0' == layer_name:
            self.feature_map.backblock0 = new_layer
            self.layer_names["feature_map.backblock0"] = new_layer
            self.origin_layer_names["feature_map.backblock0"] = new_layer
        elif 'feature_map.backblock0.conv0' == layer_name:
            self.feature_map.backblock0.conv0 = new_layer
            self.layer_names["feature_map.backblock0.conv0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv0"] = new_layer
        elif 'feature_map.backblock0.conv0.0' == layer_name:
            self.feature_map.backblock0.conv0[0] = new_layer
            self.layer_names["feature_map.backblock0.conv0.0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.0"] = new_layer
        elif 'feature_map.backblock0.conv0.1' == layer_name:
            self.feature_map.backblock0.conv0[1] = new_layer
            self.layer_names["feature_map.backblock0.conv0.1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.1"] = new_layer
        elif 'feature_map.backblock0.conv0.2' == layer_name:
            self.feature_map.backblock0.conv0[2] = new_layer
            self.layer_names["feature_map.backblock0.conv0.2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.2"] = new_layer
        elif 'feature_map.backblock0.conv1' == layer_name:
            self.feature_map.backblock0.conv1 = new_layer
            self.layer_names["feature_map.backblock0.conv1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv1"] = new_layer
        elif 'feature_map.backblock0.conv1.0' == layer_name:
            self.feature_map.backblock0.conv1[0] = new_layer
            self.layer_names["feature_map.backblock0.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.0"] = new_layer
        elif 'feature_map.backblock0.conv1.1' == layer_name:
            self.feature_map.backblock0.conv1[1] = new_layer
            self.layer_names["feature_map.backblock0.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.1"] = new_layer
        elif 'feature_map.backblock0.conv1.2' == layer_name:
            self.feature_map.backblock0.conv1[2] = new_layer
            self.layer_names["feature_map.backblock0.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.2"] = new_layer
        elif 'feature_map.backblock0.conv2' == layer_name:
            self.feature_map.backblock0.conv2 = new_layer
            self.layer_names["feature_map.backblock0.conv2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv2"] = new_layer
        elif 'feature_map.backblock0.conv2.0' == layer_name:
            self.feature_map.backblock0.conv2[0] = new_layer
            self.layer_names["feature_map.backblock0.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.0"] = new_layer
        elif 'feature_map.backblock0.conv2.1' == layer_name:
            self.feature_map.backblock0.conv2[1] = new_layer
            self.layer_names["feature_map.backblock0.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.1"] = new_layer
        elif 'feature_map.backblock0.conv2.2' == layer_name:
            self.feature_map.backblock0.conv2[2] = new_layer
            self.layer_names["feature_map.backblock0.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.2"] = new_layer
        elif 'feature_map.backblock0.conv3' == layer_name:
            self.feature_map.backblock0.conv3 = new_layer
            self.layer_names["feature_map.backblock0.conv3"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv3"] = new_layer
        elif 'feature_map.backblock0.conv3.0' == layer_name:
            self.feature_map.backblock0.conv3[0] = new_layer
            self.layer_names["feature_map.backblock0.conv3.0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.0"] = new_layer
        elif 'feature_map.backblock0.conv3.1' == layer_name:
            self.feature_map.backblock0.conv3[1] = new_layer
            self.layer_names["feature_map.backblock0.conv3.1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.1"] = new_layer
        elif 'feature_map.backblock0.conv3.2' == layer_name:
            self.feature_map.backblock0.conv3[2] = new_layer
            self.layer_names["feature_map.backblock0.conv3.2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.2"] = new_layer
        elif 'feature_map.backblock0.conv4' == layer_name:
            self.feature_map.backblock0.conv4 = new_layer
            self.layer_names["feature_map.backblock0.conv4"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv4"] = new_layer
        elif 'feature_map.backblock0.conv4.0' == layer_name:
            self.feature_map.backblock0.conv4[0] = new_layer
            self.layer_names["feature_map.backblock0.conv4.0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.0"] = new_layer
        elif 'feature_map.backblock0.conv4.1' == layer_name:
            self.feature_map.backblock0.conv4[1] = new_layer
            self.layer_names["feature_map.backblock0.conv4.1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.1"] = new_layer
        elif 'feature_map.backblock0.conv4.2' == layer_name:
            self.feature_map.backblock0.conv4[2] = new_layer
            self.layer_names["feature_map.backblock0.conv4.2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.2"] = new_layer
        elif 'feature_map.backblock0.conv5' == layer_name:
            self.feature_map.backblock0.conv5 = new_layer
            self.layer_names["feature_map.backblock0.conv5"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv5"] = new_layer
        elif 'feature_map.backblock0.conv5.0' == layer_name:
            self.feature_map.backblock0.conv5[0] = new_layer
            self.layer_names["feature_map.backblock0.conv5.0"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.0"] = new_layer
        elif 'feature_map.backblock0.conv5.1' == layer_name:
            self.feature_map.backblock0.conv5[1] = new_layer
            self.layer_names["feature_map.backblock0.conv5.1"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.1"] = new_layer
        elif 'feature_map.backblock0.conv5.2' == layer_name:
            self.feature_map.backblock0.conv5[2] = new_layer
            self.layer_names["feature_map.backblock0.conv5.2"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.2"] = new_layer
        elif 'feature_map.backblock0.conv6' == layer_name:
            self.feature_map.backblock0.conv6 = new_layer
            self.layer_names["feature_map.backblock0.conv6"] = new_layer
            self.origin_layer_names["feature_map.backblock0.conv6"] = new_layer
        elif 'feature_map.conv1' == layer_name:
            self.feature_map.conv1 = new_layer
            self.layer_names["feature_map.conv1"] = new_layer
            self.origin_layer_names["feature_map.conv1"] = new_layer
        elif 'feature_map.conv1.0' == layer_name:
            self.feature_map.conv1[0] = new_layer
            self.layer_names["feature_map.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.conv1.0"] = new_layer
        elif 'feature_map.conv1.1' == layer_name:
            self.feature_map.conv1[1] = new_layer
            self.layer_names["feature_map.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.conv1.1"] = new_layer
        elif 'feature_map.conv1.2' == layer_name:
            self.feature_map.conv1[2] = new_layer
            self.layer_names["feature_map.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.conv1.2"] = new_layer
        elif 'feature_map.backblock1' == layer_name:
            self.feature_map.backblock1 = new_layer
            self.layer_names["feature_map.backblock1"] = new_layer
            self.origin_layer_names["feature_map.backblock1"] = new_layer
        elif 'feature_map.backblock1.conv0' == layer_name:
            self.feature_map.backblock1.conv0 = new_layer
            self.layer_names["feature_map.backblock1.conv0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv0"] = new_layer
        elif 'feature_map.backblock1.conv0.0' == layer_name:
            self.feature_map.backblock1.conv0[0] = new_layer
            self.layer_names["feature_map.backblock1.conv0.0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.0"] = new_layer
        elif 'feature_map.backblock1.conv0.1' == layer_name:
            self.feature_map.backblock1.conv0[1] = new_layer
            self.layer_names["feature_map.backblock1.conv0.1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.1"] = new_layer
        elif 'feature_map.backblock1.conv0.2' == layer_name:
            self.feature_map.backblock1.conv0[2] = new_layer
            self.layer_names["feature_map.backblock1.conv0.2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.2"] = new_layer
        elif 'feature_map.backblock1.conv1' == layer_name:
            self.feature_map.backblock1.conv1 = new_layer
            self.layer_names["feature_map.backblock1.conv1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv1"] = new_layer
        elif 'feature_map.backblock1.conv1.0' == layer_name:
            self.feature_map.backblock1.conv1[0] = new_layer
            self.layer_names["feature_map.backblock1.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.0"] = new_layer
        elif 'feature_map.backblock1.conv1.1' == layer_name:
            self.feature_map.backblock1.conv1[1] = new_layer
            self.layer_names["feature_map.backblock1.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.1"] = new_layer
        elif 'feature_map.backblock1.conv1.2' == layer_name:
            self.feature_map.backblock1.conv1[2] = new_layer
            self.layer_names["feature_map.backblock1.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.2"] = new_layer
        elif 'feature_map.backblock1.conv2' == layer_name:
            self.feature_map.backblock1.conv2 = new_layer
            self.layer_names["feature_map.backblock1.conv2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv2"] = new_layer
        elif 'feature_map.backblock1.conv2.0' == layer_name:
            self.feature_map.backblock1.conv2[0] = new_layer
            self.layer_names["feature_map.backblock1.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.0"] = new_layer
        elif 'feature_map.backblock1.conv2.1' == layer_name:
            self.feature_map.backblock1.conv2[1] = new_layer
            self.layer_names["feature_map.backblock1.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.1"] = new_layer
        elif 'feature_map.backblock1.conv2.2' == layer_name:
            self.feature_map.backblock1.conv2[2] = new_layer
            self.layer_names["feature_map.backblock1.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.2"] = new_layer
        elif 'feature_map.backblock1.conv3' == layer_name:
            self.feature_map.backblock1.conv3 = new_layer
            self.layer_names["feature_map.backblock1.conv3"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv3"] = new_layer
        elif 'feature_map.backblock1.conv3.0' == layer_name:
            self.feature_map.backblock1.conv3[0] = new_layer
            self.layer_names["feature_map.backblock1.conv3.0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.0"] = new_layer
        elif 'feature_map.backblock1.conv3.1' == layer_name:
            self.feature_map.backblock1.conv3[1] = new_layer
            self.layer_names["feature_map.backblock1.conv3.1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.1"] = new_layer
        elif 'feature_map.backblock1.conv3.2' == layer_name:
            self.feature_map.backblock1.conv3[2] = new_layer
            self.layer_names["feature_map.backblock1.conv3.2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.2"] = new_layer
        elif 'feature_map.backblock1.conv4' == layer_name:
            self.feature_map.backblock1.conv4 = new_layer
            self.layer_names["feature_map.backblock1.conv4"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv4"] = new_layer
        elif 'feature_map.backblock1.conv4.0' == layer_name:
            self.feature_map.backblock1.conv4[0] = new_layer
            self.layer_names["feature_map.backblock1.conv4.0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.0"] = new_layer
        elif 'feature_map.backblock1.conv4.1' == layer_name:
            self.feature_map.backblock1.conv4[1] = new_layer
            self.layer_names["feature_map.backblock1.conv4.1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.1"] = new_layer
        elif 'feature_map.backblock1.conv4.2' == layer_name:
            self.feature_map.backblock1.conv4[2] = new_layer
            self.layer_names["feature_map.backblock1.conv4.2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.2"] = new_layer
        elif 'feature_map.backblock1.conv5' == layer_name:
            self.feature_map.backblock1.conv5 = new_layer
            self.layer_names["feature_map.backblock1.conv5"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv5"] = new_layer
        elif 'feature_map.backblock1.conv5.0' == layer_name:
            self.feature_map.backblock1.conv5[0] = new_layer
            self.layer_names["feature_map.backblock1.conv5.0"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.0"] = new_layer
        elif 'feature_map.backblock1.conv5.1' == layer_name:
            self.feature_map.backblock1.conv5[1] = new_layer
            self.layer_names["feature_map.backblock1.conv5.1"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.1"] = new_layer
        elif 'feature_map.backblock1.conv5.2' == layer_name:
            self.feature_map.backblock1.conv5[2] = new_layer
            self.layer_names["feature_map.backblock1.conv5.2"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.2"] = new_layer
        elif 'feature_map.backblock1.conv6' == layer_name:
            self.feature_map.backblock1.conv6 = new_layer
            self.layer_names["feature_map.backblock1.conv6"] = new_layer
            self.origin_layer_names["feature_map.backblock1.conv6"] = new_layer
        elif 'feature_map.conv2' == layer_name:
            self.feature_map.conv2 = new_layer
            self.layer_names["feature_map.conv2"] = new_layer
            self.origin_layer_names["feature_map.conv2"] = new_layer
        elif 'feature_map.conv2.0' == layer_name:
            self.feature_map.conv2[0] = new_layer
            self.layer_names["feature_map.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.conv2.0"] = new_layer
        elif 'feature_map.conv2.1' == layer_name:
            self.feature_map.conv2[1] = new_layer
            self.layer_names["feature_map.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.conv2.1"] = new_layer
        elif 'feature_map.conv2.2' == layer_name:
            self.feature_map.conv2[2] = new_layer
            self.layer_names["feature_map.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.conv2.2"] = new_layer
        elif 'feature_map.backblock2' == layer_name:
            self.feature_map.backblock2 = new_layer
            self.layer_names["feature_map.backblock2"] = new_layer
            self.origin_layer_names["feature_map.backblock2"] = new_layer
        elif 'feature_map.backblock2.conv0' == layer_name:
            self.feature_map.backblock2.conv0 = new_layer
            self.layer_names["feature_map.backblock2.conv0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv0"] = new_layer
        elif 'feature_map.backblock2.conv0.0' == layer_name:
            self.feature_map.backblock2.conv0[0] = new_layer
            self.layer_names["feature_map.backblock2.conv0.0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.0"] = new_layer
        elif 'feature_map.backblock2.conv0.1' == layer_name:
            self.feature_map.backblock2.conv0[1] = new_layer
            self.layer_names["feature_map.backblock2.conv0.1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.1"] = new_layer
        elif 'feature_map.backblock2.conv0.2' == layer_name:
            self.feature_map.backblock2.conv0[2] = new_layer
            self.layer_names["feature_map.backblock2.conv0.2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.2"] = new_layer
        elif 'feature_map.backblock2.conv1' == layer_name:
            self.feature_map.backblock2.conv1 = new_layer
            self.layer_names["feature_map.backblock2.conv1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv1"] = new_layer
        elif 'feature_map.backblock2.conv1.0' == layer_name:
            self.feature_map.backblock2.conv1[0] = new_layer
            self.layer_names["feature_map.backblock2.conv1.0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.0"] = new_layer
        elif 'feature_map.backblock2.conv1.1' == layer_name:
            self.feature_map.backblock2.conv1[1] = new_layer
            self.layer_names["feature_map.backblock2.conv1.1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.1"] = new_layer
        elif 'feature_map.backblock2.conv1.2' == layer_name:
            self.feature_map.backblock2.conv1[2] = new_layer
            self.layer_names["feature_map.backblock2.conv1.2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.2"] = new_layer
        elif 'feature_map.backblock2.conv2' == layer_name:
            self.feature_map.backblock2.conv2 = new_layer
            self.layer_names["feature_map.backblock2.conv2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv2"] = new_layer
        elif 'feature_map.backblock2.conv2.0' == layer_name:
            self.feature_map.backblock2.conv2[0] = new_layer
            self.layer_names["feature_map.backblock2.conv2.0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.0"] = new_layer
        elif 'feature_map.backblock2.conv2.1' == layer_name:
            self.feature_map.backblock2.conv2[1] = new_layer
            self.layer_names["feature_map.backblock2.conv2.1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.1"] = new_layer
        elif 'feature_map.backblock2.conv2.2' == layer_name:
            self.feature_map.backblock2.conv2[2] = new_layer
            self.layer_names["feature_map.backblock2.conv2.2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.2"] = new_layer
        elif 'feature_map.backblock2.conv3' == layer_name:
            self.feature_map.backblock2.conv3 = new_layer
            self.layer_names["feature_map.backblock2.conv3"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv3"] = new_layer
        elif 'feature_map.backblock2.conv3.0' == layer_name:
            self.feature_map.backblock2.conv3[0] = new_layer
            self.layer_names["feature_map.backblock2.conv3.0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.0"] = new_layer
        elif 'feature_map.backblock2.conv3.1' == layer_name:
            self.feature_map.backblock2.conv3[1] = new_layer
            self.layer_names["feature_map.backblock2.conv3.1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.1"] = new_layer
        elif 'feature_map.backblock2.conv3.2' == layer_name:
            self.feature_map.backblock2.conv3[2] = new_layer
            self.layer_names["feature_map.backblock2.conv3.2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.2"] = new_layer
        elif 'feature_map.backblock2.conv4' == layer_name:
            self.feature_map.backblock2.conv4 = new_layer
            self.layer_names["feature_map.backblock2.conv4"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv4"] = new_layer
        elif 'feature_map.backblock2.conv4.0' == layer_name:
            self.feature_map.backblock2.conv4[0] = new_layer
            self.layer_names["feature_map.backblock2.conv4.0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.0"] = new_layer
        elif 'feature_map.backblock2.conv4.1' == layer_name:
            self.feature_map.backblock2.conv4[1] = new_layer
            self.layer_names["feature_map.backblock2.conv4.1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.1"] = new_layer
        elif 'feature_map.backblock2.conv4.2' == layer_name:
            self.feature_map.backblock2.conv4[2] = new_layer
            self.layer_names["feature_map.backblock2.conv4.2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.2"] = new_layer
        elif 'feature_map.backblock2.conv5' == layer_name:
            self.feature_map.backblock2.conv5 = new_layer
            self.layer_names["feature_map.backblock2.conv5"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv5"] = new_layer
        elif 'feature_map.backblock2.conv5.0' == layer_name:
            self.feature_map.backblock2.conv5[0] = new_layer
            self.layer_names["feature_map.backblock2.conv5.0"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.0"] = new_layer
        elif 'feature_map.backblock2.conv5.1' == layer_name:
            self.feature_map.backblock2.conv5[1] = new_layer
            self.layer_names["feature_map.backblock2.conv5.1"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.1"] = new_layer
        elif 'feature_map.backblock2.conv5.2' == layer_name:
            self.feature_map.backblock2.conv5[2] = new_layer
            self.layer_names["feature_map.backblock2.conv5.2"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.2"] = new_layer
        elif 'feature_map.backblock2.conv6' == layer_name:
            self.feature_map.backblock2.conv6 = new_layer
            self.layer_names["feature_map.backblock2.conv6"] = new_layer
            self.origin_layer_names["feature_map.backblock2.conv6"] = new_layer
        elif 'detect_1' == layer_name:
            self.detect_1 = new_layer
            self.layer_names["detect_1"] = new_layer
            self.origin_layer_names["detect_1"] = new_layer
        elif 'detect_1.sigmoid' == layer_name:
            self.detect_1.sigmoid = new_layer
            self.layer_names["detect_1.sigmoid"] = new_layer
            self.origin_layer_names["detect_1.sigmoid"] = new_layer
        elif 'detect_2' == layer_name:
            self.detect_2 = new_layer
            self.layer_names["detect_2"] = new_layer
            self.origin_layer_names["detect_2"] = new_layer
        elif 'detect_2.sigmoid' == layer_name:
            self.detect_2.sigmoid = new_layer
            self.layer_names["detect_2.sigmoid"] = new_layer
            self.origin_layer_names["detect_2.sigmoid"] = new_layer
        elif 'detect_3' == layer_name:
            self.detect_3 = new_layer
            self.layer_names["detect_3"] = new_layer
            self.origin_layer_names["detect_3"] = new_layer
        elif 'detect_3.sigmoid' == layer_name:
            self.detect_3.sigmoid = new_layer
            self.layer_names["detect_3.sigmoid"] = new_layer
            self.origin_layer_names["detect_3.sigmoid"] = new_layer

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
        input_shape = ops.shape(x)[2:4]
        input_shape = ops.cast(self.tenser_to_array(input_shape), ms.float32)
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        if not self.keep_detect:
            return big_object_output, medium_object_output, small_object_output
        output_big = self.detect_1(big_object_output, input_shape)
        output_me = self.detect_2(medium_object_output, input_shape)
        output_small = self.detect_3(small_object_output, input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def has_valid_annotation(anno):
    """Check annotation file."""
    # if it's empty, there is no annotation
    if not anno:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCOYoloDataset:
    """YOLOV3 Dataset for COCO."""

    def __init__(self, root, ann_file, remove_images_without_annotations=True,
                 filter_crowd_anno=True, is_training=True):
        self.coco = COCO(ann_file)
        self.root = root
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.filter_crowd_anno = filter_crowd_anno
        self.is_training = is_training

        # filter images without any annotations
        if remove_images_without_annotations:
            img_ids = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    img_ids.append(img_id)
            self.img_ids = img_ids

        self.categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}

        self.cat_ids_to_continuous_ids = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.continuous_ids_cat_ids = {
            v: k for k, v in self.cat_ids_to_continuous_ids.items()
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (img, target) (tuple): target is a dictionary contains "bbox", "segmentation" or "keypoints",
                generated by the image's annotation. img is a PIL image.
        """
        coco = self.coco
        img_id = self.img_ids[index]
        img_path = coco.loadImgs(img_id)[0]["file_name"]
        if not self.is_training:
            img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
            return img, img_id
        img = np.fromfile(os.path.join(self.root, img_path), dtype="int8")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        # filter crowd annotations
        if self.filter_crowd_anno:
            annos = [anno for anno in target if anno["iscrowd"] == 0]
        else:
            annos = [anno for anno in target]

        target = {}
        boxes = [anno["bbox"] for anno in annos]
        target["bboxes"] = boxes

        classes = [anno["category_id"] for anno in annos]
        classes = [self.cat_ids_to_continuous_ids[cl] for cl in classes]
        target["labels"] = classes

        bboxes = target['bboxes']
        labels = target['labels']
        out_target = []
        for bbox, label in zip(bboxes, labels):
            tmp = []
            # convert to [x_min y_min x_max y_max]
            bbox = self._convetTopDown(bbox)
            tmp.extend(bbox)
            tmp.append(int(label))
            # tmp [x_min y_min x_max y_max, label]
            out_target.append(tmp)
        return img, out_target, [], [], [], [], [], []

    def __len__(self):
        return len(self.img_ids)

    def _convetTopDown(self, bbox):
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]
        return [x_min, y_min, x_min + w, y_min + h]


def create_yolo_dataset(image_dir, anno_path, batch_size, device_num, rank,
                        config=None, is_training=True, shuffle=True):
    """Create dataset for YOLOV3."""
    cv2.setNumThreads(0)

    if is_training:
        filter_crowd = True
        remove_empty_anno = True
    else:
        filter_crowd = False
        remove_empty_anno = False

    yolo_dataset = COCOYoloDataset(root=image_dir, ann_file=anno_path, filter_crowd_anno=filter_crowd,
                                   remove_images_without_annotations=remove_empty_anno, is_training=is_training)
    hwc_to_chw = ds.vision.HWC2CHW()

    config.dataset_size = len(yolo_dataset)

    if is_training:
        multi_scale_trans = MultiScaleTrans(config, device_num)
        dataset_column_names = ["image", "annotation", "bbox1", "bbox2", "bbox3",
                                "gt_box1", "gt_box2", "gt_box3"]
        dataset = ds.GeneratorDataset(yolo_dataset, column_names=dataset_column_names)
        dataset = dataset.map(operations=ds.vision.Decode(), input_columns=["image"])
        dataset = dataset.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                                num_parallel_workers=min(32, 2), drop_remainder=True)
    else:
        dataset = ds.GeneratorDataset(yolo_dataset, column_names=["image", "img_id"])
        compose_map_func = (lambda image, img_id: reshape_fn(image, img_id, config))
        dataset = dataset.map(operations=compose_map_func, input_columns=["image", "img_id"],
                              output_columns=["image", "image_shape", "img_id"],
                              num_parallel_workers=8)
        dataset = dataset.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=8)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()
        self.min = ops.Minimum()
        self.max = ops.Maximum()

    def construct(self, box1, box2):
        # box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        # box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        # convert to topLeft and rightDown
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / ops.scalar_to_tensor(2.0)  # topLeft
        box1_maxs = box1_xy + box1_wh / ops.scalar_to_tensor(2.0)  # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / ops.scalar_to_tensor(2.0)
        box2_maxs = box2_xy + box2_wh / ops.scalar_to_tensor(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, ops.scalar_to_tensor(0.0))
        # ops.squeeze: for effiecient slice
        intersect_area = ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


class XYLoss(nn.Cell):
    """Loss for x and y."""

    def __init__(self):
        super(XYLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_xy, true_xy):
        xy_loss = object_mask * box_loss_scale * self.cross_entropy(predict_xy, true_xy)
        xy_loss = self.reduce_sum(xy_loss, ())
        return xy_loss


class WHLoss(nn.Cell):
    """Loss for w and h."""

    def __init__(self):
        super(WHLoss, self).__init__()
        self.square = ops.Square()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_wh, true_wh):
        wh_loss = object_mask * box_loss_scale * 0.5 * ops.Square()(true_wh - predict_wh)
        wh_loss = self.reduce_sum(wh_loss, ())
        return wh_loss


class ConfidenceLoss(nn.Cell):
    """Loss for confidence."""

    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, predict_confidence, ignore_mask):
        confidence_loss = self.cross_entropy(predict_confidence, object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        confidence_loss = self.reduce_sum(confidence_loss, ())
        return confidence_loss


class ClassLoss(nn.Cell):
    """Loss for classification."""

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, predict_class, class_probs):
        class_loss = object_mask * self.cross_entropy(predict_class, class_probs)
        class_loss = self.reduce_sum(class_loss, ())
        return class_loss


def yololossblock(grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape, scale):
    object_mask = y_true[:, :, :, :, 4:5]
    class_probs = y_true[:, :, :, :, 5:]
    grid_shape = ops.Shape()(prediction)[1:3]
    grid_shape = ops.Cast()(ops.tuple_to_array(grid_shape[::-1]), ms.float32)
    pred_boxes = ops.Concat(axis=-1)((pred_xy, pred_wh))
    true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
    true_wh = y_true[:, :, :, :, 2:4]
    true_wh = ops.Select()(ops.Equal()(true_wh, 0.0), ops.Fill()(ops.DType()(true_wh),
                                                                 ops.Shape()(true_wh), 1.0), true_wh)
    if scale == 's':
        # anchor mask
        idx = (0, 1, 2)
    elif scale == 'm':
        idx = (3, 4, 5)
    else:
        idx = (6, 7, 8)
    true_wh = ops.Log()(true_wh / ms.Tensor([config.anchor_scales[i] for i in idx], ms.float32) * input_shape)
    # 2-w*h for large picture, use small scale, since small obj need more precise
    box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]
    gt_shape = ops.Shape()(gt_box)
    gt_box = ops.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))
    # add one more dimension for broadcast
    iou = Iou()(ops.ExpandDims()(pred_boxes, -2), gt_box)
    # gt_box is x,y,h,w after normalize
    # [batch, grid[0], grid[1], num_anchor, num_gt]
    best_iou = ops.ReduceMax(keep_dims=False)(iou, -1)
    # [batch, grid[0], grid[1], num_anchor]
    # ignore_mask IOU too small
    ignore_mask = best_iou < ms.Tensor(config.ignore_threshold, ms.float32)
    ignore_mask = ops.Cast()(ignore_mask, ms.float32)
    ignore_mask = ops.ExpandDims()(ignore_mask, -1)
    # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
    # so we turn off its gradient
    ignore_mask = ops.stop_gradient(ignore_mask)
    xy_loss = XYLoss()(object_mask, box_loss_scale, prediction[:, :, :, :, :2], true_xy)
    wh_loss = WHLoss()(object_mask, box_loss_scale, prediction[:, :, :, :, 2:4], true_wh)
    confidence_loss = ConfidenceLoss()(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
    class_loss = ClassLoss()(object_mask, prediction[:, :, :, :, 5:], class_probs)
    loss = xy_loss + wh_loss + confidence_loss + class_loss
    batch_size = ops.Shape()(prediction)[0]
    return loss / batch_size


class YoloLossBlock(nn.Cell):
    def __init__(self, scale, config=None):
        super(YoloLossBlock, self).__init__()
        self.config = config
        if scale == 's':
            # anchor mask
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = ms.Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = ops.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidenceLoss = ConfidenceLoss()
        self.classLoss = ClassLoss()

    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]

        grid_shape = ops.Shape()(prediction)[1:3]
        grid_shape = ops.Cast()(ops.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = ops.Select()(ops.Equal()(true_wh, 0.0), ops.Fill()(ops.DType()(true_wh),
                                                                     ops.Shape()(true_wh), 1.0), true_wh)

        true_wh = ops.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = ops.Shape()(gt_box)
        gt_box = ops.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(ops.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ops.Cast()(ignore_mask, ms.float32)
        ignore_mask = ops.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = ops.stop_gradient(ignore_mask)
        xy_loss = self.xy_loss(object_mask, box_loss_scale, prediction[:, :, :, :, :2], true_xy)
        wh_loss = self.wh_loss(object_mask, box_loss_scale, prediction[:, :, :, :, 2:4], true_wh)
        confidence_loss = self.confidenceLoss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.classLoss(object_mask, prediction[:, :, :, :, 5:], class_probs)
        loss = xy_loss + wh_loss + confidence_loss + class_loss
        batch_size = ops.Shape()(prediction)[0]
        return loss / batch_size


config = get_config()


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


def loser(x, yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    input_shape = ops.shape(x)[2:4]
    input_shape = ops.cast(ops.TupleToArray()(input_shape), ms.float32)
    loss_l = YoloLossBlock('l', config)(*yolo_out[0], y_true_0, gt_0, input_shape)
    loss_m = YoloLossBlock('m', config)(*yolo_out[1], y_true_1, gt_1, input_shape)
    loss_s = YoloLossBlock('s', config)(*yolo_out[2], y_true_2, gt_2, input_shape)
    return loss_l + loss_m + loss_s


def conver_testing_shape(args):
    """Convert testing shape to list."""
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape


def load_parameters(network, file_name):
    config.logger.info("yolov3 pretrained network model: %s", file_name)
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)
    config.logger.info('load_model %s success', file_name)


def set_eval(network):
    network.yolo_network.detect_1.conf_training = False
    network.yolo_network.detect_2.conf_training = False
    network.yolo_network.detect_3.conf_training = False
    return network


def settrain(network):
    network.yolo_network.detect_1.conf_training = True
    network.yolo_network.detect_2.conf_training = True
    network.yolo_network.detect_3.conf_training = True
    return network


class losser(nn.Cell):
    def __init__(self, network, config=default_config):
        super(losser, self).__init__()
        self.yolo_network = network
        self.config = config
        self.tenser_to_array = ops.TupleToArray()
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        # print("===================")
        # print(x.shape)
        # print(y_true_0.shape)
        # print(y_true_1.shape)
        # print(y_true_2.shape)
        # print(gt_0.shape)
        # print(gt_1.shape)
        # print(gt_2.shape)
        input_shape = ops.shape(x)[2:4]
        input_shape = ops.cast(self.tenser_to_array(input_shape), ms.float32)
        yolo_out = self.yolo_network(x)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        return loss_l + loss_m + loss_s


def set_graph_kernel_context():
    if ms.get_context("device_target") == "GPU":
        ms.set_context(enable_graph_kernel=True)
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion "
                                          "--enable_trans_op_optimize "
                                          "--disable_cluster_ops=ReduceMax,Reshape "
                                          "--enable_expand_ops=Conv2D")


def keep_loss_fp32(network):
    """Keep loss of network with float32"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, (YoloLossBlock,)):
            cell.to_float(ms.float32)


def initialize(device):
    device_id = int(os.getenv('DEVICE_ID', '0'))
    #ms.set_context(device_target=device, save_graphs=False, device_id=device_id)
    set_graph_kernel_context()


if __name__ == '__main__':
    eval_flag = False
    device = "CPU"
    initialize(device)
    # config.data_root = os.path.join(config.data_dir, 'val2014')
    # config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2014.json')
    network = YOLOV3DarkNet53(is_training=True)

    image = mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32)
    result1, result2, result3 = network(image)
    # input_shape = mindspore.Tensor((416,416), ms.float32)
    #
    # detect_1 = DetectionBlock('l', is_training=True, config=config)
    # detect_2 = DetectionBlock('m', is_training=True, config=config)
    # detect_3 = DetectionBlock('s', is_training=True, config=config)
    #
    #
    # image = mindspore.Tensor(np.random.randn(1,3,416,416),mindspore.float32)
    # result1,result2,result3= network.feature_map(image)
    # result1_detect=detect_1(result1, input_shape)

    #network_withloss = losser(network, config)

    # images = mindspore.Tensor(np.random.randn(1, 3, 416, 416), dtype=mindspore.float32)
    # y_true_0 = mindspore.Tensor(np.random.randn(1, 13, 13, 3, 85), dtype=mindspore.float32)
    # y_true_1 = mindspore.Tensor(np.random.randn(1, 26, 26, 3, 85), dtype=mindspore.float32)
    # y_true_2 = mindspore.Tensor(np.random.randn(1, 52, 52, 3, 85), dtype=mindspore.float32)
    # gt_0 = mindspore.Tensor(np.random.randn(1, 50, 4), dtype=mindspore.float32)
    # gt_1 = mindspore.Tensor(np.random.randn(1, 50, 4), dtype=mindspore.float32)
    # gt_2 = mindspore.Tensor(np.random.randn(1, 50, 4), dtype=mindspore.float32)
    # in_shape = images.shape[2:4]
    # in_shape = mindspore.Tensor(tuple(in_shape), dtype=mindspore.float32)
    # result = network(images)
    #result=network_withloss(images, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)


    # train_root = os.path.join("/root/datasets/coco2014", 'val2014')
    # # set dataset path temporarily to val set to reduce time cost
    # test_root = os.path.join("/root/datasets/coco2014", 'val2014')
    # # dataset = create_yolo_dataset(image_dir=train_root,
    # #                               anno_path=os.path.join(config.data_dir, 'annotations/instances_val2014.json'),
    # #                               is_training=True,
    # #                               batch_size=8, device_num=1, shuffle=False,
    # #                               rank=0, config=config)
    # # dataiter = dataset.create_dict_iterator(output_numpy=True)
    # t_end = time.time()
    # first_step = True
    # lr = get_lr(config)
    #
    #
    # opt = nn.SGD(params=get_param_groups(network), momentum=config.momentum, learning_rate=ms.Tensor(lr),
    #              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    # if config.testing_shape:
    #     config.test_img_shape = conver_testing_shape(config)
    # # testset = create_yolo_dataset(test_root, os.path.join(config.data_dir, 'annotations/instances_val2014.json'),
    # #                               is_training=False,
    # #                               batch_size=16, device_num=1,
    # #                               rank=0, shuffle=False, config=config)
    # # testdata = testset.create_dict_iterator(num_epochs=1)
    # network = losser(network, config)
    #
    # images = mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32)
    #
    # def forward_fn(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    #     loss = network(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
    #     return loss
    #
    #
    # grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    #
    # def train_step(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    #     (loss), grads = grad_fn(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
    #     loss = mindspore.ops.depend(loss, opt(grads))
    #     return loss
    #
    #
    # batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2=mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32),\
    #                                                                                             mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32),\
    #                                                                                             mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32),\
    #                                                                                             mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32),\
    #                                                                                             mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32),\
    #                                                                                             mindspore.Tensor(np.random.randn(1, 3, 416, 416), mindspore.float32)
    #
    #
    # loss_ms = train_step(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2)
    #
    #
    #
    #
    #





    # # keep_loss_fp32(network)
    # for epoch_idx in range(config.max_epoch):
    #     network.set_train(True)
    #     for step_idx, data in enumerate(dataiter):
    #         # network = settrain(network)
    #         network.set_train(True)
    #         images = data["image"]
    #         input_shape = images.shape[2:4]
    #         print('iter[{}], shape{}'.format(step_idx, input_shape[0]))
    #         images = ms.Tensor.from_numpy(images)
    #         batch_y_true_0 = ms.Tensor.from_numpy(data['bbox1'])
    #         batch_y_true_1 = ms.Tensor.from_numpy(data['bbox2'])
    #         batch_y_true_2 = ms.Tensor.from_numpy(data['bbox3'])
    #         batch_gt_box0 = ms.Tensor.from_numpy(data['gt_box1'])
    #         batch_gt_box1 = ms.Tensor.from_numpy(data['gt_box2'])
    #         batch_gt_box2 = ms.Tensor.from_numpy(data['gt_box3'])
    #
    #
    #         def forward_fn(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    #             loss = network(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
    #             return loss
    #
    #
    #         grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    #
    #         def train_step(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    #             (loss), grads = grad_fn(x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2)
    #             loss = mindspore.ops.depend(loss, opt(grads))
    #             return loss
    #
    #
    #         print("================================================================")
    #         # print(loser(data[0], data[1]))
    #         loss_ms = train_step(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
    #                              batch_gt_box2)
    #         print("loss_ms", loss_ms)
    #         print("================================================================")
    #         if (epoch_idx * steps_per_epoch + step_idx) % log_interval == 0:
    #             time_used = time.time() - t_end
    #             if first_step:
    #                 fps = per_batch_size * group_size / time_used
    #                 per_step_time = time_used * 1000
    #                 first_step = False
    #             else:
    #                 fps = per_batch_size * log_interval * group_size / time_used
    #                 per_step_time = time_used / log_interval * 1000
    #             print('epoch[{}], iter[{}], fps:{:.2f} imgs/sec,'
    #                   'per step time: {}ms'.format(epoch_idx + 1, step_idx + 1
    #                                                , fps, per_step_time))
    #             t_end = time.time()
    #         if eval_flag:
    #             start_time = time.time()
    #             network = set_eval(network)
    #             network.set_train(False)
    #             config.outputs_dir = os.path.join(config.log_path)
    #             detection = DetectionEngine(config)
    #             print('Start inference....')
    #             for i, data in enumerate(testdata):
    #                 image = data["image"]
    #                 image_shape = data["image_shape"]
    #                 image_id = data["img_id"]
    #                 output_big, output_me, output_small = network.yolo_network(image)
    #                 output_big = output_big.asnumpy()
    #                 output_me = output_me.asnumpy()
    #                 output_small = output_small.asnumpy()
    #                 image_id = image_id.asnumpy()
    #                 image_shape = image_shape.asnumpy()
    #                 detection.detect([output_small, output_me, output_big], per_batch_size, image_shape, image_id)
    #                 if i % 50 == 0:
    #                     print('Processing... {:.2f}% '.format(i / testset.get_dataset_size() * 100))
    #
    #
    #             def com_mAP(detection):
    #                 print('Calculating mAP...')
    #                 detection.do_nms_for_results()
    #                 result_file_path = detection.write_result()
    #                 print('result file path: %s', result_file_path)
    #                 eval_result = detection.get_eval_result()
    #                 cost_time = time.time() - start_time
    #                 eval_print_str = '\n=============coco eval result=========\n' + eval_result
    #                 print(eval_print_str)
    #                 print('testing cost time %.2f h', cost_time / 3600.)
    #
    #
    #             com_mAP(detection)
