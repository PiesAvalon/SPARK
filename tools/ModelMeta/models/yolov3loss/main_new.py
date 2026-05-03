import datetime
import multiprocessing
import os
import math
import random
import sys
import time
from collections import defaultdict, Counter
import mindspore
import mindspore.dataset.vision as CV
import cv2
import numpy as np
import numpy.random
from PIL import Image
from mindspore.common import set_seed
from mindspore import context, nn, ops
from mindspore.common.tensor import Tensor
from mindspore.nn import Momentum
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from models.yolov3loss.src.transforms import reshape_fn, MultiScaleTrans
from models.yolov3loss.model_utils.config import config
from models.yolov3loss.model_utils.config import config as default_config
from models.yolov3loss.src.distributed_sampler import DistributedSampler
from models.yolov3loss.src.logger import get_logger
from mindspore.profiler.profiling import Profiler
from mindspore.ops import operations as P
import mindspore as ms
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.dataset as de
from models.yolov3loss.src.yolo_dataset import has_valid_annotation

set_seed(1)
from mindspore import log as logger


def set_default():
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.t_max:
        config.t_max = config.max_epoch
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.data_root = os.path.join(config.data_dir, 'val2017')
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    config.data_val_root = os.path.join(config.data_dir, 'val2017')
    config.ann_val_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False, device_id=device_id)
    if config.need_profiler:
        profiler = Profiler(output_path=config.outputs_dir, is_detail=True, is_show_op_path=True)
    else:
        profiler = None
    config.rank = 0
    config.group_size = 1
    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1
    config.outputs_dir = os.path.join(config.ckpt_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    return profiler


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma ** milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def multi_step_lr(lr, milestones, steps_per_epoch, max_epoch, gamma=0.1):
    return warmup_step_lr(lr, milestones, steps_per_epoch, 0, max_epoch, gamma=gamma)


def step_lr(lr, epoch_size, steps_per_epoch, max_epoch, gamma=0.1):
    lr_epochs = []
    for i in range(1, max_epoch):
        if i % epoch_size == 0:
            lr_epochs.append(i)
    return multi_step_lr(lr, lr_epochs, steps_per_epoch, max_epoch, gamma=gamma)


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_v2(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate V2."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    last_lr = 0
    last_epoch_v1 = 0

    t_max_v2 = int(max_epoch * 1 / 3)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            if i < total_steps * 2 / 3:
                lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
                last_lr = lr
                last_epoch_v1 = last_epoch
            else:
                base_lr = last_lr
                last_epoch = last_epoch - last_epoch_v1
                lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max_v2)) / 2

        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_sample(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Warmup cosine annealing learning rate."""
    start_sample_epoch = 60
    step_sample = 2
    tobe_sampled_epoch = 60
    end_sampled_epoch = start_sample_epoch + step_sample * tobe_sampled_epoch
    max_sampled_epoch = max_epoch + tobe_sampled_epoch
    t_max = max_sampled_epoch

    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    total_sampled_steps = int(max_sampled_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []

    for i in range(total_sampled_steps):
        last_epoch = i // steps_per_epoch
        if last_epoch in range(start_sample_epoch, end_sampled_epoch, step_sample):
            continue
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    assert total_steps == len(lr_each_step)
    return np.array(lr_each_step).astype(np.float32)


def get_lr(args):
    """generate learning rate."""
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(args.lr,
                                        args.steps_per_epoch,
                                        args.warmup_epochs,
                                        args.max_epoch,
                                        args.t_max,
                                        args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_V2':
        lr = warmup_cosine_annealing_lr_v2(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.t_max,
                                           args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_sample':
        lr = warmup_cosine_annealing_lr_sample(args.lr,
                                               args.steps_per_epoch,
                                               args.warmup_epochs,
                                               args.max_epoch,
                                               args.t_max,
                                               args.eta_min)
    else:
        raise NotImplementedError(args.lr_scheduler)
    return lr


class DetectionBlock(nn.Cell):
    def __init__(self, scale, config=default_config):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
            self.scale_x_y = 1.2
            self.offset_x_y = 0.1
        elif scale == 'm':
            idx = (3, 4, 5)
            self.scale_x_y = 1.1
            self.offset_x_y = 0.05
        elif scale == 'l':
            idx = (6, 7, 8)
            self.scale_x_y = 1.05
            self.offset_x_y = 0.025
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.conf_training=True
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=-1)

    def construct(self, x, input_shape):
        """construct method"""
        num_batch = P.Shape()(x)[0]
        grid_size = P.Shape()(x)[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = P.Reshape()(x, (num_batch,
                                     self.num_anchors_per_scale,
                                     self.num_attrib,
                                     grid_size[0],
                                     grid_size[1]))
        prediction = P.Transpose()(prediction, (0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = P.Cast()(F.tuple_to_array(range_x), ms.float32)
        grid_y = P.Cast()(F.tuple_to_array(range_y), ms.float32)
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
        box_xy = (self.scale_x_y * self.sigmoid(box_xy) - self.offset_x_y + grid) / \
                 P.Cast()(F.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = P.Exp()(box_wh) * self.anchors / input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)
        if self.conf_training:
            return prediction, box_xy, box_wh



        return self.concat((box_xy, box_wh, box_confidence, box_probs))


class YoloWithLossCell(nn.Cell):
    def __init__(self, network):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = default_config
        self.loss_coff = default_config.detect_head_loss_coff
        self.loss_l_coff = int(self.loss_coff[0])
        self.loss_m_coff = int(self.loss_coff[1])
        self.loss_s_coff = int(self.loss_coff[2])
        self.loss_big = YoloLossBlock_ms('l', self.config)
        self.loss_me = YoloLossBlock_ms('m', self.config)
        self.loss_small = YoloLossBlock_ms('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
        yolo_out = self.yolo_network(x, input_shape)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        return loss_l * self.loss_l_coff + loss_m * self.loss_m_coff + loss_s * self.loss_s_coff


def _conv_bn_leakyrelu(in_channel,
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


class YoloBlock(nn.Cell):
    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_leakyrelu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_leakyrelu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_leakyrelu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_leakyrelu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_leakyrelu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_leakyrelu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        """construct method"""
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class YOLOv4(nn.Cell):
    def __init__(self, backbone_shape, backbone, out_channel):
        super(YOLOv4, self).__init__()
        self.out_channel = out_channel
        self.backbone = backbone

        self.conv1 = _conv_bn_leakyrelu(1024, 512, ksize=1)
        self.conv2 = _conv_bn_leakyrelu(512, 1024, ksize=3)
        self.conv3 = _conv_bn_leakyrelu(1024, 512, ksize=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, pad_mode='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, pad_mode='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, pad_mode='same')
        self.conv4 = _conv_bn_leakyrelu(2048, 512, ksize=1)

        self.conv5 = _conv_bn_leakyrelu(512, 1024, ksize=3)
        self.conv6 = _conv_bn_leakyrelu(1024, 512, ksize=1)
        self.conv7 = _conv_bn_leakyrelu(512, 256, ksize=1)

        self.conv8 = _conv_bn_leakyrelu(512, 256, ksize=1)
        self.backblock0 = YoloBlock(backbone_shape[-2], out_chls=backbone_shape[-3], out_channels=out_channel)

        self.conv9 = _conv_bn_leakyrelu(256, 128, ksize=1)
        self.conv10 = _conv_bn_leakyrelu(256, 128, ksize=1)
        self.conv11 = _conv_bn_leakyrelu(128, 256, ksize=3, stride=2)
        self.conv12 = _conv_bn_leakyrelu(256, 512, ksize=3, stride=2)

        self.backblock1 = YoloBlock(backbone_shape[-3], out_chls=backbone_shape[-4], out_channels=out_channel)
        self.backblock2 = YoloBlock(backbone_shape[-2], out_chls=backbone_shape[-3], out_channels=out_channel)
        self.backblock3 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """
        input_shape of x is (batch_size, 3, h, w)
        feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        """
        img_hight = P.Shape()(x)[2]
        img_width = P.Shape()(x)[3]

        # input=(1,3,608,608)
        # feature_map1=(1,256,76,76)
        # feature_map2=(1,512,38,38)
        # feature_map3=(1,1024,19,19)
        feature_map1, feature_map2, feature_map3 = self.backbone(x)

        con1 = self.conv1(feature_map3)
        con2 = self.conv2(con1)
        con3 = self.conv3(con2)

        m1 = self.maxpool1(con3)
        m2 = self.maxpool2(con3)
        m3 = self.maxpool3(con3)
        spp = self.concat((m3, m2, m1, con3))
        con4 = self.conv4(spp)

        con5 = self.conv5(con4)
        con6 = self.conv6(con5)
        con7 = self.conv7(con6)

        ups1 = P.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con7)
        con8 = self.conv8(feature_map2)
        con9 = self.concat((ups1, con8))
        con10, _ = self.backblock0(con9)
        con11 = self.conv9(con10)
        ups2 = P.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con11)
        con12 = self.conv10(feature_map1)
        con13 = self.concat((ups2, con12))
        con14, small_object_output = self.backblock1(con13)

        con15 = self.conv11(con14)
        con16 = self.concat((con15, con10))
        con17, medium_object_output = self.backblock2(con16)

        con18 = self.conv12(con17)
        con19 = self.concat((con18, con6))
        _, big_object_output = self.backblock3(con19)
        return big_object_output, medium_object_output, small_object_output


class Mish(nn.Cell):
    """Mish activation method"""

    def __init__(self):
        super(Mish, self).__init__()
        self.mul = P.Mul()
        self.tanh = P.Tanh()
        self.softplus = P.Softplus()

    def construct(self, input_x):
        res1 = self.softplus(input_x)
        tanh = self.tanh(res1)
        output = self.mul(input_x, tanh)

        return output


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    """Get a conv2d batchnorm and relu layer"""
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
         nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
         Mish()
         ]
    )


class CspDarkNet53(nn.Cell):
    """
    DarkNet V1 network.

    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        num_classes: Integer. Class number. Default:100.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3,f4,f5).

    Examples:
        DarkNet(ResidualBlock)
    """

    def __init__(self,
                 block,
                 detect=False):
        super(CspDarkNet53, self).__init__()
        self.outchannel = 1024
        self.detect = detect
        self.concat = P.Concat(axis=1)
        self.add = P.Add()
        self.conv0 = conv_block(3, 32, kernel_size=3, stride=1)
        self.conv1 = conv_block(32, 64, kernel_size=3, stride=2)
        self.conv2 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv3 = conv_block(64, 32, kernel_size=1, stride=1)
        self.conv4 = conv_block(32, 64, kernel_size=3, stride=1)
        self.conv5 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv6 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv7 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv8 = conv_block(64, 128, kernel_size=3, stride=2)
        self.conv9 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv10 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv11 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv12 = conv_block(128, 128, kernel_size=1, stride=1)
        self.conv13 = conv_block(128, 256, kernel_size=3, stride=2)
        self.conv14 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv15 = conv_block(128, 128, kernel_size=1, stride=1)
        self.conv16 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv17 = conv_block(256, 256, kernel_size=1, stride=1)
        self.conv18 = conv_block(256, 512, kernel_size=3, stride=2)
        self.conv19 = conv_block(512, 256, kernel_size=1, stride=1)
        self.conv20 = conv_block(256, 256, kernel_size=1, stride=1)
        self.conv21 = conv_block(512, 256, kernel_size=1, stride=1)
        self.conv22 = conv_block(512, 512, kernel_size=1, stride=1)
        self.conv23 = conv_block(512, 1024, kernel_size=3, stride=2)
        self.conv24 = conv_block(1024, 512, kernel_size=1, stride=1)
        self.conv25 = conv_block(512, 512, kernel_size=1, stride=1)
        self.conv26 = conv_block(1024, 512, kernel_size=1, stride=1)
        self.conv27 = conv_block(1024, 1024, kernel_size=1, stride=1)
        self.layer2 = self._make_layer(block, 2, in_channel=64, out_channel=64)
        self.layer3 = self._make_layer(block, 8, in_channel=128, out_channel=128)
        self.layer4 = self._make_layer(block, 8, in_channel=256, out_channel=256)
        self.layer5 = self._make_layer(block, 4, in_channel=512, out_channel=512)

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        """
        Make Layer for DarkNet.

        :param block: Cell. DarkNet block.
        :param layer_num: Integer. Layer number.
        :param in_channel: Integer. Input channel.
        :param out_channel: Integer. Output channel.
        :return: SequentialCell, the output layer.

        Examples:
            _make_layer(ConvBlock, 1, 128, 256)
        """
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct method"""
        c1 = self.conv0(x)
        c2 = self.conv1(c1)  # route
        c3 = self.conv2(c2)
        c4 = self.conv3(c3)
        c5 = self.conv4(c4)
        c6 = self.add(c3, c5)
        c7 = self.conv5(c6)
        c8 = self.conv6(c2)
        c9 = self.concat((c7, c8))
        c10 = self.conv7(c9)
        c11 = self.conv8(c10)  # route
        c12 = self.conv9(c11)
        c13 = self.layer2(c12)
        c14 = self.conv10(c13)
        c15 = self.conv11(c11)
        c16 = self.concat((c14, c15))
        c17 = self.conv12(c16)
        c18 = self.conv13(c17)  # route
        c19 = self.conv14(c18)
        c20 = self.layer3(c19)
        c21 = self.conv15(c20)
        c22 = self.conv16(c18)
        c23 = self.concat((c21, c22))
        c24 = self.conv17(c23)  # output1
        c25 = self.conv18(c24)  # route
        c26 = self.conv19(c25)
        c27 = self.layer4(c26)
        c28 = self.conv20(c27)
        c29 = self.conv21(c25)
        c30 = self.concat((c28, c29))
        c31 = self.conv22(c30)  # output2
        c32 = self.conv23(c31)  # route
        c33 = self.conv24(c32)
        c34 = self.layer5(c33)
        c35 = self.conv25(c34)
        c36 = self.conv26(c32)
        c37 = self.concat((c35, c36))
        c38 = self.conv27(c37)  # output3

        if self.detect:
            return c24, c31, c38

        return c38

    def get_out_channels(self):
        return self.outchannel


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

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualBlock, self).__init__()
        out_chls = out_channels
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


class YOLOV4CspDarkNet53_ms(nn.Cell):
    def __init__(self):
        super(YOLOV4CspDarkNet53_ms, self).__init__()
        self.config = default_config
        self.keep_detect = self.config.keep_detect
        self.input_shape = mindspore.Tensor(tuple(default_config.test_img_shape), dtype=mindspore.float32)
        # YOLOv4 network
        self.feature_map = YOLOv4(backbone=CspDarkNet53(ResidualBlock, detect=True),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l')
        self.detect_2 = DetectionBlock('m')
        self.detect_3 = DetectionBlock('s')

        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None

        self.origin_layer_names = {
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
            "feature_map.backbone.conv2": self.feature_map.backbone.conv2,
            "feature_map.backbone.conv2.0": self.feature_map.backbone.conv2[0],
            "feature_map.backbone.conv2.1": self.feature_map.backbone.conv2[1],
            "feature_map.backbone.conv2.2": self.feature_map.backbone.conv2[2],
            "feature_map.backbone.conv3": self.feature_map.backbone.conv3,
            "feature_map.backbone.conv3.0": self.feature_map.backbone.conv3[0],
            "feature_map.backbone.conv3.1": self.feature_map.backbone.conv3[1],
            "feature_map.backbone.conv3.2": self.feature_map.backbone.conv3[2],
            "feature_map.backbone.conv4": self.feature_map.backbone.conv4,
            "feature_map.backbone.conv4.0": self.feature_map.backbone.conv4[0],
            "feature_map.backbone.conv4.1": self.feature_map.backbone.conv4[1],
            "feature_map.backbone.conv4.2": self.feature_map.backbone.conv4[2],
            "feature_map.backbone.conv5": self.feature_map.backbone.conv5,
            "feature_map.backbone.conv5.0": self.feature_map.backbone.conv5[0],
            "feature_map.backbone.conv5.1": self.feature_map.backbone.conv5[1],
            "feature_map.backbone.conv5.2": self.feature_map.backbone.conv5[2],
            "feature_map.backbone.conv6": self.feature_map.backbone.conv6,
            "feature_map.backbone.conv6.0": self.feature_map.backbone.conv6[0],
            "feature_map.backbone.conv6.1": self.feature_map.backbone.conv6[1],
            "feature_map.backbone.conv6.2": self.feature_map.backbone.conv6[2],
            "feature_map.backbone.conv7": self.feature_map.backbone.conv7,
            "feature_map.backbone.conv7.0": self.feature_map.backbone.conv7[0],
            "feature_map.backbone.conv7.1": self.feature_map.backbone.conv7[1],
            "feature_map.backbone.conv7.2": self.feature_map.backbone.conv7[2],
            "feature_map.backbone.conv8": self.feature_map.backbone.conv8,
            "feature_map.backbone.conv8.0": self.feature_map.backbone.conv8[0],
            "feature_map.backbone.conv8.1": self.feature_map.backbone.conv8[1],
            "feature_map.backbone.conv8.2": self.feature_map.backbone.conv8[2],
            "feature_map.backbone.conv9": self.feature_map.backbone.conv9,
            "feature_map.backbone.conv9.0": self.feature_map.backbone.conv9[0],
            "feature_map.backbone.conv9.1": self.feature_map.backbone.conv9[1],
            "feature_map.backbone.conv9.2": self.feature_map.backbone.conv9[2],
            "feature_map.backbone.conv10": self.feature_map.backbone.conv10,
            "feature_map.backbone.conv10.0": self.feature_map.backbone.conv10[0],
            "feature_map.backbone.conv10.1": self.feature_map.backbone.conv10[1],
            "feature_map.backbone.conv10.2": self.feature_map.backbone.conv10[2],
            "feature_map.backbone.conv11": self.feature_map.backbone.conv11,
            "feature_map.backbone.conv11.0": self.feature_map.backbone.conv11[0],
            "feature_map.backbone.conv11.1": self.feature_map.backbone.conv11[1],
            "feature_map.backbone.conv11.2": self.feature_map.backbone.conv11[2],
            "feature_map.backbone.conv12": self.feature_map.backbone.conv12,
            "feature_map.backbone.conv12.0": self.feature_map.backbone.conv12[0],
            "feature_map.backbone.conv12.1": self.feature_map.backbone.conv12[1],
            "feature_map.backbone.conv12.2": self.feature_map.backbone.conv12[2],
            "feature_map.backbone.conv13": self.feature_map.backbone.conv13,
            "feature_map.backbone.conv13.0": self.feature_map.backbone.conv13[0],
            "feature_map.backbone.conv13.1": self.feature_map.backbone.conv13[1],
            "feature_map.backbone.conv13.2": self.feature_map.backbone.conv13[2],
            "feature_map.backbone.conv14": self.feature_map.backbone.conv14,
            "feature_map.backbone.conv14.0": self.feature_map.backbone.conv14[0],
            "feature_map.backbone.conv14.1": self.feature_map.backbone.conv14[1],
            "feature_map.backbone.conv14.2": self.feature_map.backbone.conv14[2],
            "feature_map.backbone.conv15": self.feature_map.backbone.conv15,
            "feature_map.backbone.conv15.0": self.feature_map.backbone.conv15[0],
            "feature_map.backbone.conv15.1": self.feature_map.backbone.conv15[1],
            "feature_map.backbone.conv15.2": self.feature_map.backbone.conv15[2],
            "feature_map.backbone.conv16": self.feature_map.backbone.conv16,
            "feature_map.backbone.conv16.0": self.feature_map.backbone.conv16[0],
            "feature_map.backbone.conv16.1": self.feature_map.backbone.conv16[1],
            "feature_map.backbone.conv16.2": self.feature_map.backbone.conv16[2],
            "feature_map.backbone.conv17": self.feature_map.backbone.conv17,
            "feature_map.backbone.conv17.0": self.feature_map.backbone.conv17[0],
            "feature_map.backbone.conv17.1": self.feature_map.backbone.conv17[1],
            "feature_map.backbone.conv17.2": self.feature_map.backbone.conv17[2],
            "feature_map.backbone.conv18": self.feature_map.backbone.conv18,
            "feature_map.backbone.conv18.0": self.feature_map.backbone.conv18[0],
            "feature_map.backbone.conv18.1": self.feature_map.backbone.conv18[1],
            "feature_map.backbone.conv18.2": self.feature_map.backbone.conv18[2],
            "feature_map.backbone.conv19": self.feature_map.backbone.conv19,
            "feature_map.backbone.conv19.0": self.feature_map.backbone.conv19[0],
            "feature_map.backbone.conv19.1": self.feature_map.backbone.conv19[1],
            "feature_map.backbone.conv19.2": self.feature_map.backbone.conv19[2],
            "feature_map.backbone.conv20": self.feature_map.backbone.conv20,
            "feature_map.backbone.conv20.0": self.feature_map.backbone.conv20[0],
            "feature_map.backbone.conv20.1": self.feature_map.backbone.conv20[1],
            "feature_map.backbone.conv20.2": self.feature_map.backbone.conv20[2],
            "feature_map.backbone.conv21": self.feature_map.backbone.conv21,
            "feature_map.backbone.conv21.0": self.feature_map.backbone.conv21[0],
            "feature_map.backbone.conv21.1": self.feature_map.backbone.conv21[1],
            "feature_map.backbone.conv21.2": self.feature_map.backbone.conv21[2],
            "feature_map.backbone.conv22": self.feature_map.backbone.conv22,
            "feature_map.backbone.conv22.0": self.feature_map.backbone.conv22[0],
            "feature_map.backbone.conv22.1": self.feature_map.backbone.conv22[1],
            "feature_map.backbone.conv22.2": self.feature_map.backbone.conv22[2],
            "feature_map.backbone.conv23": self.feature_map.backbone.conv23,
            "feature_map.backbone.conv23.0": self.feature_map.backbone.conv23[0],
            "feature_map.backbone.conv23.1": self.feature_map.backbone.conv23[1],
            "feature_map.backbone.conv23.2": self.feature_map.backbone.conv23[2],
            "feature_map.backbone.conv24": self.feature_map.backbone.conv24,
            "feature_map.backbone.conv24.0": self.feature_map.backbone.conv24[0],
            "feature_map.backbone.conv24.1": self.feature_map.backbone.conv24[1],
            "feature_map.backbone.conv24.2": self.feature_map.backbone.conv24[2],
            "feature_map.backbone.conv25": self.feature_map.backbone.conv25,
            "feature_map.backbone.conv25.0": self.feature_map.backbone.conv25[0],
            "feature_map.backbone.conv25.1": self.feature_map.backbone.conv25[1],
            "feature_map.backbone.conv25.2": self.feature_map.backbone.conv25[2],
            "feature_map.backbone.conv26": self.feature_map.backbone.conv26,
            "feature_map.backbone.conv26.0": self.feature_map.backbone.conv26[0],
            "feature_map.backbone.conv26.1": self.feature_map.backbone.conv26[1],
            "feature_map.backbone.conv26.2": self.feature_map.backbone.conv26[2],
            "feature_map.backbone.conv27": self.feature_map.backbone.conv27,
            "feature_map.backbone.conv27.0": self.feature_map.backbone.conv27[0],
            "feature_map.backbone.conv27.1": self.feature_map.backbone.conv27[1],
            "feature_map.backbone.conv27.2": self.feature_map.backbone.conv27[2],
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
            # "feature_map.conv1": self.feature_map.conv1,
            # "feature_map.conv1.0": self.feature_map.conv1[0],
            # "feature_map.conv1.1": self.feature_map.conv1[1],
            # "feature_map.conv1.2": self.feature_map.conv1[2],
            # "feature_map.conv2": self.feature_map.conv2,
            # "feature_map.conv2.0": self.feature_map.conv2[0],
            # "feature_map.conv2.1": self.feature_map.conv2[1],
            # "feature_map.conv2.2": self.feature_map.conv2[2],
            # "feature_map.conv3": self.feature_map.conv3,
            # "feature_map.conv3.0": self.feature_map.conv3[0],
            # "feature_map.conv3.1": self.feature_map.conv3[1],
            # "feature_map.conv3.2": self.feature_map.conv3[2],
            # "feature_map.maxpool1": self.feature_map.maxpool1,
            # "feature_map.maxpool2": self.feature_map.maxpool2,
            # "feature_map.maxpool3": self.feature_map.maxpool3,
            # "feature_map.conv4": self.feature_map.conv4,
            # "feature_map.conv4.0": self.feature_map.conv4[0],
            # "feature_map.conv4.1": self.feature_map.conv4[1],
            # "feature_map.conv4.2": self.feature_map.conv4[2],
            # "feature_map.conv5": self.feature_map.conv5,
            # "feature_map.conv5.0": self.feature_map.conv5[0],
            # "feature_map.conv5.1": self.feature_map.conv5[1],
            # "feature_map.conv5.2": self.feature_map.conv5[2],
            # "feature_map.conv6": self.feature_map.conv6,
            # "feature_map.conv6.0": self.feature_map.conv6[0],
            # "feature_map.conv6.1": self.feature_map.conv6[1],
            # "feature_map.conv6.2": self.feature_map.conv6[2],
            # "feature_map.conv7": self.feature_map.conv7,
            # "feature_map.conv7.0": self.feature_map.conv7[0],
            # "feature_map.conv7.1": self.feature_map.conv7[1],
            # "feature_map.conv7.2": self.feature_map.conv7[2],
            # "feature_map.conv8": self.feature_map.conv8,
            # "feature_map.conv8.0": self.feature_map.conv8[0],
            # "feature_map.conv8.1": self.feature_map.conv8[1],
            # "feature_map.conv8.2": self.feature_map.conv8[2],
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
            # # "feature_map.backblock0.conv5": self.feature_map.backblock0.conv5,
            # # "feature_map.backblock0.conv5.0": self.feature_map.backblock0.conv5[0],
            # # "feature_map.backblock0.conv5.1": self.feature_map.backblock0.conv5[1],
            # # "feature_map.backblock0.conv5.2": self.feature_map.backblock0.conv5[2],
            # # "feature_map.backblock0.conv6": self.feature_map.backblock0.conv6,
            # "feature_map.conv9": self.feature_map.conv9,
            # "feature_map.conv9.0": self.feature_map.conv9[0],
            # "feature_map.conv9.1": self.feature_map.conv9[1],
            # "feature_map.conv9.2": self.feature_map.conv9[2],
            # "feature_map.conv10": self.feature_map.conv10,
            # "feature_map.conv10.0": self.feature_map.conv10[0],
            # "feature_map.conv10.1": self.feature_map.conv10[1],
            # "feature_map.conv10.2": self.feature_map.conv10[2],
            # "feature_map.conv11": self.feature_map.conv11,
            # "feature_map.conv11.0": self.feature_map.conv11[0],
            # "feature_map.conv11.1": self.feature_map.conv11[1],
            # "feature_map.conv11.2": self.feature_map.conv11[2],
            # "feature_map.conv12": self.feature_map.conv12,
            # "feature_map.conv12.0": self.feature_map.conv12[0],
            # "feature_map.conv12.1": self.feature_map.conv12[1],
            # "feature_map.conv12.2": self.feature_map.conv12[2],
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
            # "feature_map.backblock3": self.feature_map.backblock3,
            # "feature_map.backblock3.conv0": self.feature_map.backblock3.conv0,
            # "feature_map.backblock3.conv0.0": self.feature_map.backblock3.conv0[0],
            # "feature_map.backblock3.conv0.1": self.feature_map.backblock3.conv0[1],
            # "feature_map.backblock3.conv0.2": self.feature_map.backblock3.conv0[2],
            # "feature_map.backblock3.conv1": self.feature_map.backblock3.conv1,
            # "feature_map.backblock3.conv1.0": self.feature_map.backblock3.conv1[0],
            # "feature_map.backblock3.conv1.1": self.feature_map.backblock3.conv1[1],
            # "feature_map.backblock3.conv1.2": self.feature_map.backblock3.conv1[2],
            # "feature_map.backblock3.conv2": self.feature_map.backblock3.conv2,
            # "feature_map.backblock3.conv2.0": self.feature_map.backblock3.conv2[0],
            # "feature_map.backblock3.conv2.1": self.feature_map.backblock3.conv2[1],
            # "feature_map.backblock3.conv2.2": self.feature_map.backblock3.conv2[2],
            # "feature_map.backblock3.conv3": self.feature_map.backblock3.conv3,
            # "feature_map.backblock3.conv3.0": self.feature_map.backblock3.conv3[0],
            # "feature_map.backblock3.conv3.1": self.feature_map.backblock3.conv3[1],
            # "feature_map.backblock3.conv3.2": self.feature_map.backblock3.conv3[2],
            # "feature_map.backblock3.conv4": self.feature_map.backblock3.conv4,
            # "feature_map.backblock3.conv4.0": self.feature_map.backblock3.conv4[0],
            # "feature_map.backblock3.conv4.1": self.feature_map.backblock3.conv4[1],
            # "feature_map.backblock3.conv4.2": self.feature_map.backblock3.conv4[2],
            # "feature_map.backblock3.conv5": self.feature_map.backblock3.conv5,
            # "feature_map.backblock3.conv5.0": self.feature_map.backblock3.conv5[0],
            # "feature_map.backblock3.conv5.1": self.feature_map.backblock3.conv5[1],
            # "feature_map.backblock3.conv5.2": self.feature_map.backblock3.conv5[2],
            # "feature_map.backblock3.conv6": self.feature_map.backblock3.conv6,
            # "detect_1": self.detect_1,
            # "detect_1.sigmoid": self.detect_1.sigmoid,
            # "detect_2": self.detect_2,
            # "detect_2.sigmoid": self.detect_2.sigmoid,
            # "detect_3": self.detect_3,
            # "detect_3.sigmoid": self.detect_3.sigmoid,
        }
        self.layer_names = {"feature_map": self.feature_map,
                            "feature_map.backbone": self.feature_map.backbone,
                            "feature_map.backbone.conv0": self.feature_map.backbone.conv0,
                            "feature_map.backbone.conv0.0": self.feature_map.backbone.conv0[0],
                            "feature_map.backbone.conv0.1": self.feature_map.backbone.conv0[1],
                            "feature_map.backbone.conv0.2": self.feature_map.backbone.conv0[2],
                            "feature_map.backbone.conv1": self.feature_map.backbone.conv1,
                            "feature_map.backbone.conv1.0": self.feature_map.backbone.conv1[0],
                            "feature_map.backbone.conv1.1": self.feature_map.backbone.conv1[1],
                            "feature_map.backbone.conv1.2": self.feature_map.backbone.conv1[2],
                            "feature_map.backbone.conv2": self.feature_map.backbone.conv2,
                            "feature_map.backbone.conv2.0": self.feature_map.backbone.conv2[0],
                            "feature_map.backbone.conv2.1": self.feature_map.backbone.conv2[1],
                            "feature_map.backbone.conv2.2": self.feature_map.backbone.conv2[2],
                            "feature_map.backbone.conv3": self.feature_map.backbone.conv3,
                            "feature_map.backbone.conv3.0": self.feature_map.backbone.conv3[0],
                            "feature_map.backbone.conv3.1": self.feature_map.backbone.conv3[1],
                            "feature_map.backbone.conv3.2": self.feature_map.backbone.conv3[2],
                            "feature_map.backbone.conv4": self.feature_map.backbone.conv4,
                            "feature_map.backbone.conv4.0": self.feature_map.backbone.conv4[0],
                            "feature_map.backbone.conv4.1": self.feature_map.backbone.conv4[1],
                            "feature_map.backbone.conv4.2": self.feature_map.backbone.conv4[2],
                            "feature_map.backbone.conv5": self.feature_map.backbone.conv5,
                            "feature_map.backbone.conv5.0": self.feature_map.backbone.conv5[0],
                            "feature_map.backbone.conv5.1": self.feature_map.backbone.conv5[1],
                            "feature_map.backbone.conv5.2": self.feature_map.backbone.conv5[2],
                            "feature_map.backbone.conv6": self.feature_map.backbone.conv6,
                            "feature_map.backbone.conv6.0": self.feature_map.backbone.conv6[0],
                            "feature_map.backbone.conv6.1": self.feature_map.backbone.conv6[1],
                            "feature_map.backbone.conv6.2": self.feature_map.backbone.conv6[2],
                            "feature_map.backbone.conv7": self.feature_map.backbone.conv7,
                            "feature_map.backbone.conv7.0": self.feature_map.backbone.conv7[0],
                            "feature_map.backbone.conv7.1": self.feature_map.backbone.conv7[1],
                            "feature_map.backbone.conv7.2": self.feature_map.backbone.conv7[2],
                            "feature_map.backbone.conv8": self.feature_map.backbone.conv8,
                            "feature_map.backbone.conv8.0": self.feature_map.backbone.conv8[0],
                            "feature_map.backbone.conv8.1": self.feature_map.backbone.conv8[1],
                            "feature_map.backbone.conv8.2": self.feature_map.backbone.conv8[2],
                            "feature_map.backbone.conv9": self.feature_map.backbone.conv9,
                            "feature_map.backbone.conv9.0": self.feature_map.backbone.conv9[0],
                            "feature_map.backbone.conv9.1": self.feature_map.backbone.conv9[1],
                            "feature_map.backbone.conv9.2": self.feature_map.backbone.conv9[2],
                            "feature_map.backbone.conv10": self.feature_map.backbone.conv10,
                            "feature_map.backbone.conv10.0": self.feature_map.backbone.conv10[0],
                            "feature_map.backbone.conv10.1": self.feature_map.backbone.conv10[1],
                            "feature_map.backbone.conv10.2": self.feature_map.backbone.conv10[2],
                            "feature_map.backbone.conv11": self.feature_map.backbone.conv11,
                            "feature_map.backbone.conv11.0": self.feature_map.backbone.conv11[0],
                            "feature_map.backbone.conv11.1": self.feature_map.backbone.conv11[1],
                            "feature_map.backbone.conv11.2": self.feature_map.backbone.conv11[2],
                            "feature_map.backbone.conv12": self.feature_map.backbone.conv12,
                            "feature_map.backbone.conv12.0": self.feature_map.backbone.conv12[0],
                            "feature_map.backbone.conv12.1": self.feature_map.backbone.conv12[1],
                            "feature_map.backbone.conv12.2": self.feature_map.backbone.conv12[2],
                            "feature_map.backbone.conv13": self.feature_map.backbone.conv13,
                            "feature_map.backbone.conv13.0": self.feature_map.backbone.conv13[0],
                            "feature_map.backbone.conv13.1": self.feature_map.backbone.conv13[1],
                            "feature_map.backbone.conv13.2": self.feature_map.backbone.conv13[2],
                            "feature_map.backbone.conv14": self.feature_map.backbone.conv14,
                            "feature_map.backbone.conv14.0": self.feature_map.backbone.conv14[0],
                            "feature_map.backbone.conv14.1": self.feature_map.backbone.conv14[1],
                            "feature_map.backbone.conv14.2": self.feature_map.backbone.conv14[2],
                            "feature_map.backbone.conv15": self.feature_map.backbone.conv15,
                            "feature_map.backbone.conv15.0": self.feature_map.backbone.conv15[0],
                            "feature_map.backbone.conv15.1": self.feature_map.backbone.conv15[1],
                            "feature_map.backbone.conv15.2": self.feature_map.backbone.conv15[2],
                            "feature_map.backbone.conv16": self.feature_map.backbone.conv16,
                            "feature_map.backbone.conv16.0": self.feature_map.backbone.conv16[0],
                            "feature_map.backbone.conv16.1": self.feature_map.backbone.conv16[1],
                            "feature_map.backbone.conv16.2": self.feature_map.backbone.conv16[2],
                            "feature_map.backbone.conv17": self.feature_map.backbone.conv17,
                            "feature_map.backbone.conv17.0": self.feature_map.backbone.conv17[0],
                            "feature_map.backbone.conv17.1": self.feature_map.backbone.conv17[1],
                            "feature_map.backbone.conv17.2": self.feature_map.backbone.conv17[2],
                            "feature_map.backbone.conv18": self.feature_map.backbone.conv18,
                            "feature_map.backbone.conv18.0": self.feature_map.backbone.conv18[0],
                            "feature_map.backbone.conv18.1": self.feature_map.backbone.conv18[1],
                            "feature_map.backbone.conv18.2": self.feature_map.backbone.conv18[2],
                            "feature_map.backbone.conv19": self.feature_map.backbone.conv19,
                            "feature_map.backbone.conv19.0": self.feature_map.backbone.conv19[0],
                            "feature_map.backbone.conv19.1": self.feature_map.backbone.conv19[1],
                            "feature_map.backbone.conv19.2": self.feature_map.backbone.conv19[2],
                            "feature_map.backbone.conv20": self.feature_map.backbone.conv20,
                            "feature_map.backbone.conv20.0": self.feature_map.backbone.conv20[0],
                            "feature_map.backbone.conv20.1": self.feature_map.backbone.conv20[1],
                            "feature_map.backbone.conv20.2": self.feature_map.backbone.conv20[2],
                            "feature_map.backbone.conv21": self.feature_map.backbone.conv21,
                            "feature_map.backbone.conv21.0": self.feature_map.backbone.conv21[0],
                            "feature_map.backbone.conv21.1": self.feature_map.backbone.conv21[1],
                            "feature_map.backbone.conv21.2": self.feature_map.backbone.conv21[2],
                            "feature_map.backbone.conv22": self.feature_map.backbone.conv22,
                            "feature_map.backbone.conv22.0": self.feature_map.backbone.conv22[0],
                            "feature_map.backbone.conv22.1": self.feature_map.backbone.conv22[1],
                            "feature_map.backbone.conv22.2": self.feature_map.backbone.conv22[2],
                            "feature_map.backbone.conv23": self.feature_map.backbone.conv23,
                            "feature_map.backbone.conv23.0": self.feature_map.backbone.conv23[0],
                            "feature_map.backbone.conv23.1": self.feature_map.backbone.conv23[1],
                            "feature_map.backbone.conv23.2": self.feature_map.backbone.conv23[2],
                            "feature_map.backbone.conv24": self.feature_map.backbone.conv24,
                            "feature_map.backbone.conv24.0": self.feature_map.backbone.conv24[0],
                            "feature_map.backbone.conv24.1": self.feature_map.backbone.conv24[1],
                            "feature_map.backbone.conv24.2": self.feature_map.backbone.conv24[2],
                            "feature_map.backbone.conv25": self.feature_map.backbone.conv25,
                            "feature_map.backbone.conv25.0": self.feature_map.backbone.conv25[0],
                            "feature_map.backbone.conv25.1": self.feature_map.backbone.conv25[1],
                            "feature_map.backbone.conv25.2": self.feature_map.backbone.conv25[2],
                            "feature_map.backbone.conv26": self.feature_map.backbone.conv26,
                            "feature_map.backbone.conv26.0": self.feature_map.backbone.conv26[0],
                            "feature_map.backbone.conv26.1": self.feature_map.backbone.conv26[1],
                            "feature_map.backbone.conv26.2": self.feature_map.backbone.conv26[2],
                            "feature_map.backbone.conv27": self.feature_map.backbone.conv27,
                            "feature_map.backbone.conv27.0": self.feature_map.backbone.conv27[0],
                            "feature_map.backbone.conv27.1": self.feature_map.backbone.conv27[1],
                            "feature_map.backbone.conv27.2": self.feature_map.backbone.conv27[2],
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
                            # "feature_map.conv1": self.feature_map.conv1,
                            # "feature_map.conv1.0": self.feature_map.conv1[0],
                            # "feature_map.conv1.1": self.feature_map.conv1[1],
                            # "feature_map.conv1.2": self.feature_map.conv1[2],
                            # "feature_map.conv2": self.feature_map.conv2,
                            # "feature_map.conv2.0": self.feature_map.conv2[0],
                            # "feature_map.conv2.1": self.feature_map.conv2[1],
                            # "feature_map.conv2.2": self.feature_map.conv2[2],
                            # "feature_map.conv3": self.feature_map.conv3,
                            # "feature_map.conv3.0": self.feature_map.conv3[0],
                            # "feature_map.conv3.1": self.feature_map.conv3[1],
                            # "feature_map.conv3.2": self.feature_map.conv3[2],
                            # "feature_map.maxpool1": self.feature_map.maxpool1,
                            # "feature_map.maxpool2": self.feature_map.maxpool2,
                            # "feature_map.maxpool3": self.feature_map.maxpool3,
                            # "feature_map.conv4": self.feature_map.conv4,
                            # "feature_map.conv4.0": self.feature_map.conv4[0],
                            # "feature_map.conv4.1": self.feature_map.conv4[1],
                            # "feature_map.conv4.2": self.feature_map.conv4[2],
                            # "feature_map.conv5": self.feature_map.conv5,
                            # "feature_map.conv5.0": self.feature_map.conv5[0],
                            # "feature_map.conv5.1": self.feature_map.conv5[1],
                            # "feature_map.conv5.2": self.feature_map.conv5[2],
                            # "feature_map.conv6": self.feature_map.conv6,
                            # "feature_map.conv6.0": self.feature_map.conv6[0],
                            # "feature_map.conv6.1": self.feature_map.conv6[1],
                            # "feature_map.conv6.2": self.feature_map.conv6[2],
                            # "feature_map.conv7": self.feature_map.conv7,
                            # "feature_map.conv7.0": self.feature_map.conv7[0],
                            # "feature_map.conv7.1": self.feature_map.conv7[1],
                            # "feature_map.conv7.2": self.feature_map.conv7[2],
                            # "feature_map.conv8": self.feature_map.conv8,
                            # "feature_map.conv8.0": self.feature_map.conv8[0],
                            # "feature_map.conv8.1": self.feature_map.conv8[1],
                            # "feature_map.conv8.2": self.feature_map.conv8[2],
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
                            # # "feature_map.backblock0.conv5":self.feature_map.backblock0.conv5,
                            # # "feature_map.backblock0.conv5.0":self.feature_map.backblock0.conv5[0],
                            # # "feature_map.backblock0.conv5.1":self.feature_map.backblock0.conv5[1],
                            # # "feature_map.backblock0.conv5.2":self.feature_map.backblock0.conv5[2],
                            # # "feature_map.backblock0.conv6":self.feature_map.backblock0.conv6,
                            # "feature_map.conv9": self.feature_map.conv9,
                            # "feature_map.conv9.0": self.feature_map.conv9[0],
                            # "feature_map.conv9.1": self.feature_map.conv9[1],
                            # "feature_map.conv9.2": self.feature_map.conv9[2],
                            # "feature_map.conv10": self.feature_map.conv10,
                            # "feature_map.conv10.0": self.feature_map.conv10[0],
                            # "feature_map.conv10.1": self.feature_map.conv10[1],
                            # "feature_map.conv10.2": self.feature_map.conv10[2],
                            # "feature_map.conv11": self.feature_map.conv11,
                            # "feature_map.conv11.0": self.feature_map.conv11[0],
                            # "feature_map.conv11.1": self.feature_map.conv11[1],
                            # "feature_map.conv11.2": self.feature_map.conv11[2],
                            # "feature_map.conv12": self.feature_map.conv12,
                            # "feature_map.conv12.0": self.feature_map.conv12[0],
                            # "feature_map.conv12.1": self.feature_map.conv12[1],
                            # "feature_map.conv12.2": self.feature_map.conv12[2],
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
                            # "feature_map.backblock3": self.feature_map.backblock3,
                            # "feature_map.backblock3.conv0": self.feature_map.backblock3.conv0,
                            # "feature_map.backblock3.conv0.0": self.feature_map.backblock3.conv0[0],
                            # "feature_map.backblock3.conv0.1": self.feature_map.backblock3.conv0[1],
                            # "feature_map.backblock3.conv0.2": self.feature_map.backblock3.conv0[2],
                            # "feature_map.backblock3.conv1": self.feature_map.backblock3.conv1,
                            # "feature_map.backblock3.conv1.0": self.feature_map.backblock3.conv1[0],
                            # "feature_map.backblock3.conv1.1": self.feature_map.backblock3.conv1[1],
                            # "feature_map.backblock3.conv1.2": self.feature_map.backblock3.conv1[2],
                            # "feature_map.backblock3.conv2": self.feature_map.backblock3.conv2,
                            # "feature_map.backblock3.conv2.0": self.feature_map.backblock3.conv2[0],
                            # "feature_map.backblock3.conv2.1": self.feature_map.backblock3.conv2[1],
                            # "feature_map.backblock3.conv2.2": self.feature_map.backblock3.conv2[2],
                            # "feature_map.backblock3.conv3": self.feature_map.backblock3.conv3,
                            # "feature_map.backblock3.conv3.0": self.feature_map.backblock3.conv3[0],
                            # "feature_map.backblock3.conv3.1": self.feature_map.backblock3.conv3[1],
                            # "feature_map.backblock3.conv3.2": self.feature_map.backblock3.conv3[2],
                            # "feature_map.backblock3.conv4": self.feature_map.backblock3.conv4,
                            # "feature_map.backblock3.conv4.0": self.feature_map.backblock3.conv4[0],
                            # "feature_map.backblock3.conv4.1": self.feature_map.backblock3.conv4[1],
                            # "feature_map.backblock3.conv4.2": self.feature_map.backblock3.conv4[2],
                            # "feature_map.backblock3.conv5": self.feature_map.backblock3.conv5,
                            # "feature_map.backblock3.conv5.0": self.feature_map.backblock3.conv5[0],
                            # "feature_map.backblock3.conv5.1": self.feature_map.backblock3.conv5[1],
                            # "feature_map.backblock3.conv5.2": self.feature_map.backblock3.conv5[2],
                            # "feature_map.backblock3.conv6": self.feature_map.backblock3.conv6,
                            # "detect_1": self.detect_1,
                            # "detect_1.sigmoid": self.detect_1.sigmoid,
                            # "detect_2": self.detect_2,
                            # "detect_2.sigmoid": self.detect_2.sigmoid,
                            # "detect_3": self.detect_3,
                            # "detect_3.sigmoid": self.detect_3.sigmoid,
                            }

        self.in_shapes = {
            'INPUT': [-1, 3, 416, 416],
            'feature_map.backbone.layer3.6.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv6': [-1, 256, 52, 52],
            # 'feature_map.conv12.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.5.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv8.1': [-1, 128, 104, 104],
            # 'feature_map.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.conv21.0': [-1, 512, 26, 26],
            'feature_map.backbone.conv15.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.conv20.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer2.0.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.conv16.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv23.2': [-1, 1024, 13, 13],
            # 'detect_1.sigmoid': [-1, 255, 52, 52],
            'feature_map.backbone.layer4.0.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.1.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.maxpool2': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv3.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv3.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv14.0': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv8.2': [-1, 128, 104, 104],
            'feature_map.backbone.conv11.1': [-1, 64, 104, 104],
            'feature_map.backbone.conv9.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer4.0.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.conv3.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.3.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.1.conv1.0': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv3.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv18.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer5.0.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.4.conv1.1': [-1, 256, 26, 26],
            # 'feature_map.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.2': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv0.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv22.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv0.0': [-1, 3, 416, 416],
            'feature_map.backbone.layer4.6.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.1.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.2': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv4.1': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv0.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.1.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv13.0': [-1, 128, 104, 104],
            'feature_map.backbone.conv23.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.1.conv1.2': [-1, 128, 52, 52],
            # 'feature_map.backblock3.conv3.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv0.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv27.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer2.1.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer4.6.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv3.1': [-1, 32, 208, 208],
            'feature_map.backbone.layer3.0.conv2.1': [-1, 128, 52, 52],
            # 'feature_map.conv11.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv18.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer2.0.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.conv15.0': [-1, 128, 52, 52],
            # 'feature_map.backblock3.conv5.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv26.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv1.0': [-1, 32, 416, 416],
            # 'detect_2.sigmoid': [-1, 255, 26, 26],
            # 'feature_map.backblock0.conv1.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv1.1': [-1, 256, 26, 26],
            # 'feature_map.conv4.0': [-1, 512, 13, 13],
            'feature_map.backbone.conv7.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv7.1': [-1, 64, 208, 208],
            'feature_map.backbone.layer3.3.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer2.0.conv2.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer3.6.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv10.0': [-1, 64, 104, 104],
            'feature_map.backbone.conv26.1': [-1, 512, 13, 13],
            # 'feature_map.conv6.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock0.conv0.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv13.2': [-1, 256, 52, 52],
            # 'feature_map.backblock0.conv4.2': [-1, 256, 26, 26],
            # 'detect_3.sigmoid': [-1, 255, 13, 13],
            'feature_map.backbone.layer3.1.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv5.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv10.1': [-1, 64, 104, 104],
            'feature_map.backbone.conv10.2': [-1, 64, 104, 104],
            'feature_map.backbone.conv23.1': [-1, 1024, 13, 13],
            # 'feature_map.conv3.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.6.conv2.1': [-1, 128, 52, 52],
            # 'feature_map.backblock3.conv6': [-1, 1024, 13, 13],
            'feature_map.backbone.conv11.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer4.3.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.3.conv2.2': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv4.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.7.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv6.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer4.7.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer2.1.conv2.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer3.0.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.2.conv1.2': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv4.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv1.0': [-1, 256, 26, 26],
            # 'feature_map.conv9.1': [-1, 128, 26, 26],
            # 'feature_map.backblock3.conv1.2': [-1, 1024, 13, 13],
            'feature_map.backbone.conv5.2': [-1, 64, 208, 208],
            # 'feature_map.conv5.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer2.1.conv1.0': [-1, 64, 104, 104],
            # 'feature_map.conv4.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.7.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.2.conv1.2': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv4.1': [-1, 256, 26, 26],
            # 'feature_map.maxpool3': [-1, 512, 13, 13],
            'feature_map.backbone.conv7.2': [-1, 64, 208, 208],
            # 'feature_map.backblock1.conv5.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv20.2': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv0.2': [-1, 128, 52, 52],
            # 'feature_map.conv7.1': [-1, 256, 13, 13],
            'feature_map.backbone.conv4.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv17.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv2.0': [-1, 128, 52, 52],
            # 'feature_map.conv10.1': [-1, 128, 52, 52],
            # 'feature_map.maxpool1': [-1, 512, 13, 13],
            # 'feature_map.conv9.2': [-1, 128, 26, 26],
            'feature_map.backbone.conv27.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv25.1': [-1, 512, 13, 13],
            # 'feature_map.conv12.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.3.conv2.1': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv1.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.0.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv24.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv9.2': [-1, 64, 104, 104],
            # 'feature_map.backblock1.conv0.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.3.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv1.1': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv2.2': [-1, 128, 52, 52],
            # 'feature_map.conv11.1': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv3.1': [-1, 256, 52, 52],
            # 'feature_map.backblock3.conv5.0': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv17.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv21.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv27.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.4.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv2.0': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv6': [-1, 512, 26, 26],
            # 'feature_map.conv1.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.2.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.0.conv1.2': [-1, 256, 26, 26],
            # 'feature_map.conv3.1': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv0.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.4.conv2.2': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.6.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.0.conv1.0': [-1, 512, 13, 13],
            # 'feature_map.conv2.1': [-1, 1024, 13, 13],
            # 'feature_map.conv4.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.2.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer2.1.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.conv5.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer4.6.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.2.conv1.0': [-1, 512, 13, 13],
            # 'feature_map.backblock1.conv1.2': [-1, 256, 52, 52],
            # 'feature_map.backblock2.conv3.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv2.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer3.5.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.backblock0.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.backblock3.conv1.0': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv3.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.4.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.1.conv2.1': [-1, 512, 13, 13],
            # 'feature_map.backblock0.conv3.1': [-1, 512, 26, 26],
            'feature_map.backbone.conv14.2': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv5.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.conv25.0': [-1, 512, 13, 13],
            'feature_map.backbone.conv24.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock1.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.2.conv2.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv19.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer5.1.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.2.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.4.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv2.2': [-1, 64, 208, 208],
            # 'feature_map.conv8.1': [-1, 256, 26, 26],
            'feature_map.backbone.conv9.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer4.7.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer2.0.conv2.0': [-1, 64, 104, 104],
            # 'feature_map.backblock2.conv1.0': [-1, 256, 26, 26],
            # 'feature_map.conv7.2': [-1, 256, 13, 13],
            'feature_map.backbone.conv19.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer2.1.conv2.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer4.6.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv19.2': [-1, 256, 26, 26],
            # 'feature_map.conv10.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv4.1': [-1, 64, 208, 208],
            'feature_map.backbone.layer3.3.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.2.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.conv4.0': [-1, 32, 208, 208],
            'feature_map.backbone.conv6.2': [-1, 64, 208, 208],
            # 'feature_map.backblock3.conv1.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv16.2': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv0.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.2.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.4.conv1.0': [-1, 256, 26, 26],
            # 'feature_map.conv12.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.0.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv17.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer4.1.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv1.1': [-1, 64, 208, 208],
            # 'feature_map.conv5.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.2': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.5.conv2.1': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv1.0': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv5.2': [-1, 256, 52, 52],
            'feature_map.backbone.conv12.1': [-1, 128, 104, 104],
            'feature_map.backbone.conv0.1': [-1, 32, 416, 416],
            'feature_map.backbone.conv12.2': [-1, 128, 104, 104],
            # 'feature_map.backblock0.conv1.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.3.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.3.conv1.1': [-1, 512, 13, 13],
            # 'feature_map.conv5.2': [-1, 1024, 13, 13],
            # 'feature_map.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.7.conv1.1': [-1, 128, 52, 52],
            # 'feature_map.conv8.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.2': [-1, 256, 26, 26],
            # 'feature_map.conv11.0': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv3.0': [-1, 128, 52, 52],
            # 'feature_map.conv6.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv15.2': [-1, 128, 52, 52],
            # 'feature_map.backblock0.conv4.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.7.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv22.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.7.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.2.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer2.0.conv2.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer4.1.conv1.1': [-1, 256, 26, 26],
            # 'feature_map.conv6.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.2.conv1.2': [-1, 128, 52, 52],
            # 'feature_map.backblock3.conv3.2': [-1, 1024, 13, 13],
            'feature_map.backbone.conv22.2': [-1, 512, 26, 26],
            # 'feature_map.conv9.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv25.2': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv6.1': [-1, 64, 208, 208],
            'feature_map.backbone.layer5.0.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.conv24.1': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv5.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer3.4.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv18.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer5.0.conv2.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.0': [-1, 512, 13, 13],
            # 'feature_map.backblock2.conv1.2': [-1, 512, 26, 26],
            # 'feature_map.backblock3.conv4.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv3.0': [-1, 64, 208, 208],
            # 'feature_map.backblock3.conv2.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv11.2': [-1, 64, 104, 104],
            # 'feature_map.backblock2.conv5.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.1.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.4.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer2.0.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer4.5.conv1.1': [-1, 256, 26, 26],
            # 'feature_map.conv8.0': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv12.0': [-1, 64, 104, 104],
            'feature_map.backbone.conv8.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv16.0': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv3.2': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv4.0': [-1, 256, 52, 52],
            # 'feature_map.backblock0.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.conv7.0': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv0.2': [-1, 512, 13, 13],
            # 'feature_map.backblock1.conv0.0': [-1, 128, 52, 52],
            # 'feature_map.backblock2.conv5.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv20.0': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv3.1': [-1, 512, 26, 26],
            'feature_map.backbone.conv26.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv3.2': [-1, 32, 208, 208],
            'feature_map.backbone.layer2.1.conv2.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer5.2.conv1.1': [-1, 512, 13, 13],
            # 'feature_map.conv10.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv14.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv1.2': [-1, 64, 208, 208],
            'feature_map.backbone.layer4.0.conv1.0': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv4.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.5.conv1.0': [-1, 256, 26, 26],
            # 'feature_map.backblock1.conv4.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.1.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.4.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.0.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock3.conv4.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer4.5.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.2.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv13.1': [-1, 256, 52, 52],
            'feature_map.backbone.conv21.1': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv5.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer3.3.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.1.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer3.5.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer5.2.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer3.5.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv2.1': [-1, 64, 208, 208],
            # 'feature_map.backblock2.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.2': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv0.0': [-1, 256, 26, 26],
            # 'feature_map.backblock3.conv0.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer4.2.conv2.2': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv4.0': [-1, 512, 26, 26],
            'feature_map.backbone.conv0.2': [-1, 32, 416, 416],
            'OUTPUT1': [-1, 256, 52, 52],
            'OUTPUT2': [-1, 512, 26, 26],
            'OUTPUT3': [-1, 1024, 13, 13]
            # 'INPUT': [-1, 3, 416, 416], 'OUTPUT1': [-1, 13, 13, 3, 2], 'OUTPUT2': [-1, 26, 26, 3, 2],
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
            'feature_map.backbone.conv2.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv2.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv2.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv3.0': [-1, 32, 208, 208],
            'feature_map.backbone.conv3.1': [-1, 32, 208, 208],
            'feature_map.backbone.conv3.2': [-1, 32, 208, 208],
            'feature_map.backbone.conv4.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv4.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv4.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv5.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv5.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv5.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv6.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv6.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv6.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv7.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv7.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv7.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv8.0': [-1, 128, 104, 104],
            'feature_map.backbone.conv8.1': [-1, 128, 104, 104],
            'feature_map.backbone.conv8.2': [-1, 128, 104, 104],
            'feature_map.backbone.conv9.0': [-1, 64, 104, 104],
            'feature_map.backbone.conv9.1': [-1, 64, 104, 104],
            'feature_map.backbone.conv9.2': [-1, 64, 104, 104],
            'feature_map.backbone.conv10.0': [-1, 64, 104, 104],
            'feature_map.backbone.conv10.1': [-1, 64, 104, 104],
            'feature_map.backbone.conv10.2': [-1, 64, 104, 104],
            'feature_map.backbone.conv11.0': [-1, 64, 104, 104],
            'feature_map.backbone.conv11.1': [-1, 64, 104, 104],
            'feature_map.backbone.conv11.2': [-1, 64, 104, 104],
            'feature_map.backbone.conv12.0': [-1, 128, 104, 104],
            'feature_map.backbone.conv12.1': [-1, 128, 104, 104],
            'feature_map.backbone.conv12.2': [-1, 128, 104, 104],
            'feature_map.backbone.conv13.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv13.1': [-1, 256, 52, 52],
            'feature_map.backbone.conv13.2': [-1, 256, 52, 52],
            'feature_map.backbone.conv14.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv14.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv14.2': [-1, 128, 52, 52],
            'feature_map.backbone.conv15.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv15.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv15.2': [-1, 128, 52, 52],
            'feature_map.backbone.conv16.0': [-1, 128, 52, 52],
            'feature_map.backbone.conv16.1': [-1, 128, 52, 52],
            'feature_map.backbone.conv16.2': [-1, 128, 52, 52],
            'feature_map.backbone.conv17.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv17.1': [-1, 256, 52, 52],
            'feature_map.backbone.conv17.2': [-1, 256, 52, 52],
            'feature_map.backbone.conv18.0': [-1, 512, 26, 26],
            'feature_map.backbone.conv18.1': [-1, 512, 26, 26],
            'feature_map.backbone.conv18.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv19.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv19.1': [-1, 256, 26, 26],
            'feature_map.backbone.conv19.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv20.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv20.1': [-1, 256, 26, 26],
            'feature_map.backbone.conv20.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv21.0': [-1, 256, 26, 26],
            'feature_map.backbone.conv21.1': [-1, 256, 26, 26],
            'feature_map.backbone.conv21.2': [-1, 256, 26, 26],
            'feature_map.backbone.conv22.0': [-1, 512, 26, 26],
            'feature_map.backbone.conv22.1': [-1, 512, 26, 26],
            'feature_map.backbone.conv22.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv23.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv23.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv23.2': [-1, 1024, 13, 13],
            'feature_map.backbone.conv24.0': [-1, 512, 13, 13],
            'feature_map.backbone.conv24.1': [-1, 512, 13, 13],
            'feature_map.backbone.conv24.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv25.0': [-1, 512, 13, 13],
            'feature_map.backbone.conv25.1': [-1, 512, 13, 13],
            'feature_map.backbone.conv25.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv26.0': [-1, 512, 13, 13],
            'feature_map.backbone.conv26.1': [-1, 512, 13, 13],
            'feature_map.backbone.conv26.2': [-1, 512, 13, 13],
            'feature_map.backbone.conv27.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv27.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv27.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer2.0.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv2.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv2.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv2.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv2.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv2.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv2.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer3.0.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv2.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv2.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv2.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer4.0.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer5.0.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv2.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv2.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv2.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv2.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv2.2': [-1, 512, 13, 13],
            'OUTPUT1': [-1, 256, 52, 52],
            'OUTPUT2': [-1, 512, 26, 26],
            'OUTPUT3': [-1, 1024, 13, 13]
            # 'feature_map.conv1.0': [-1, 512, 13, 13],
            # 'feature_map.conv1.1': [-1, 512, 13, 13],
            # 'feature_map.conv1.2': [-1, 512, 13, 13],
            # 'feature_map.conv2.0': [-1, 1024, 13, 13],
            # 'feature_map.conv2.1': [-1, 1024, 13, 13],
            # 'feature_map.conv2.2': [-1, 1024, 13, 13],
            # 'feature_map.conv3.0': [-1, 512, 13, 13],
            # 'feature_map.conv3.1': [-1, 512, 13, 13],
            # 'feature_map.conv3.2': [-1, 512, 13, 13],
            # 'feature_map.maxpool1': [-1, 512, 13, 13],
            # 'feature_map.maxpool2': [-1, 512, 13, 13],
            # 'feature_map.maxpool3': [-1, 512, 13, 13],
            # 'feature_map.conv4.0': [-1, 512, 13, 13],
            # 'feature_map.conv4.1': [-1, 512, 13, 13],
            # 'feature_map.conv4.2': [-1, 512, 13, 13],
            # 'feature_map.conv5.0': [-1, 1024, 13, 13],
            # 'feature_map.conv5.1': [-1, 1024, 13, 13],
            # 'feature_map.conv5.2': [-1, 1024, 13, 13],
            # 'feature_map.conv6.0': [-1, 512, 13, 13],
            # 'feature_map.conv6.1': [-1, 512, 13, 13],
            # 'feature_map.conv6.2': [-1, 512, 13, 13],
            # 'feature_map.conv7.0': [-1, 256, 13, 13],
            # 'feature_map.conv7.1': [-1, 256, 13, 13],
            # 'feature_map.conv7.2': [-1, 256, 13, 13],
            # 'feature_map.conv8.0': [-1, 256, 26, 26],
            # 'feature_map.conv8.1': [-1, 256, 26, 26],
            # 'feature_map.conv8.2': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv0.0': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv0.1': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv0.2': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv1.0': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv1.1': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv1.2': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv2.2': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv3.0': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv3.1': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv3.2': [-1, 512, 26, 26],
            # 'feature_map.backblock0.conv4.0': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv4.1': [-1, 256, 26, 26],
            # 'feature_map.backblock0.conv4.2': [-1, 256, 26, 26],
            # 'feature_map.conv9.0': [-1, 128, 26, 26],
            # 'feature_map.conv9.1': [-1, 128, 26, 26],
            # 'feature_map.conv9.2': [-1, 128, 26, 26],
            # 'feature_map.conv10.0': [-1, 128, 52, 52],
            # 'feature_map.conv10.1': [-1, 128, 52, 52],
            # 'feature_map.conv10.2': [-1, 128, 52, 52],
            # 'feature_map.conv11.0': [-1, 256, 26, 26],
            # 'feature_map.conv11.1': [-1, 256, 26, 26],
            # 'feature_map.conv11.2': [-1, 256, 26, 26],
            # 'feature_map.conv12.0': [-1, 512, 13, 13],
            # 'feature_map.conv12.1': [-1, 512, 13, 13],
            # 'feature_map.conv12.2': [-1, 512, 13, 13],
            # 'feature_map.backblock1.conv0.0': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv0.1': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv0.2': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv1.0': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv1.1': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv1.2': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv2.0': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv2.1': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv2.2': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv3.0': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv3.1': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv3.2': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv4.0': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv4.1': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv4.2': [-1, 128, 52, 52],
            # 'feature_map.backblock1.conv5.0': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv5.1': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv5.2': [-1, 256, 52, 52],
            # 'feature_map.backblock1.conv6': [-1, 255, 52, 52],
            # 'feature_map.backblock2.conv0.0': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv0.1': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv0.2': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv1.0': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv1.1': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv1.2': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv2.0': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv2.1': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv2.2': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv3.0': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv3.1': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv3.2': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv4.0': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv4.1': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv4.2': [-1, 256, 26, 26],
            # 'feature_map.backblock2.conv5.0': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv5.1': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv5.2': [-1, 512, 26, 26],
            # 'feature_map.backblock2.conv6': [-1, 255, 26, 26],
            # 'feature_map.backblock3.conv0.0': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv0.1': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv0.2': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv1.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv1.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv1.2': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv2.0': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv2.1': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv2.2': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv3.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv3.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv3.2': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv4.0': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv4.1': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv4.2': [-1, 512, 13, 13],
            # 'feature_map.backblock3.conv5.0': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv5.1': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv5.2': [-1, 1024, 13, 13],
            # 'feature_map.backblock3.conv6': [-1, 255, 13, 13],
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
            'feature_map.backbone.conv1.2': ["feature_map.backbone.conv1.1",
                                             ["feature_map.backbone.conv2.0", "feature_map.backbone.conv6.0"]],
            'feature_map.backbone.conv2.0': ["feature_map.backbone.conv1.2", "feature_map.backbone.conv2.1"],
            'feature_map.backbone.conv2.1': ["feature_map.backbone.conv2.0", "feature_map.backbone.conv2.2"],
            'feature_map.backbone.conv2.2': ["feature_map.backbone.conv2.1",
                                             ["feature_map.backbone.conv3.0", "feature_map.backbone.conv5.0"]],
            'feature_map.backbone.conv3.0': ["feature_map.backbone.conv2.2", "feature_map.backbone.conv3.1"],
            'feature_map.backbone.conv3.1': ["feature_map.backbone.conv3.0", "feature_map.backbone.conv3.2"],
            'feature_map.backbone.conv3.2': ["feature_map.backbone.conv3.1", "feature_map.backbone.conv4.0"],
            'feature_map.backbone.conv4.0': ["feature_map.backbone.conv3.2", "feature_map.backbone.conv4.1"],
            'feature_map.backbone.conv4.1': ["feature_map.backbone.conv4.0", "feature_map.backbone.conv4.2"],
            'feature_map.backbone.conv4.2': ["feature_map.backbone.conv4.1", "feature_map.backbone.conv5.0"],
            'feature_map.backbone.conv5.0': [["feature_map.backbone.conv4.2", "feature_map.backbone.conv2.2"],
                                             "feature_map.backbone.conv5.1"],
            'feature_map.backbone.conv5.1': ["feature_map.backbone.conv5.0", "feature_map.backbone.conv5.2"],
            'feature_map.backbone.conv5.2': ["feature_map.backbone.conv5.1", "feature_map.backbone.conv7.0"],
            'feature_map.backbone.conv6.0': ["feature_map.backbone.conv1.2", "feature_map.backbone.conv6.1"],
            'feature_map.backbone.conv6.1': ["feature_map.backbone.conv6.0", "feature_map.backbone.conv6.2"],
            'feature_map.backbone.conv6.2': ["feature_map.backbone.conv6.1", "feature_map.backbone.conv7.0"],
            'feature_map.backbone.conv7.0': [["feature_map.backbone.conv6.2", "feature_map.backbone.conv5.2"],
                                             "feature_map.backbone.conv7.1"],
            'feature_map.backbone.conv7.1': ["feature_map.backbone.conv7.0", "feature_map.backbone.conv7.2"],
            'feature_map.backbone.conv7.2': ["feature_map.backbone.conv7.1", "feature_map.backbone.conv8.0"],
            'feature_map.backbone.conv8.0': ["feature_map.backbone.conv7.2", "feature_map.backbone.conv8.1"],
            'feature_map.backbone.conv8.1': ["feature_map.backbone.conv8.0", "feature_map.backbone.conv8.2"],
            'feature_map.backbone.conv8.2': ["feature_map.backbone.conv8.1",
                                             ["feature_map.backbone.conv9.0", "feature_map.backbone.conv11.0"]],
            'feature_map.backbone.conv9.0': ["feature_map.backbone.conv8.2", "feature_map.backbone.conv9.1"],
            'feature_map.backbone.conv9.1': ["feature_map.backbone.conv9.0", "feature_map.backbone.conv9.2"],
            'feature_map.backbone.conv9.2': ["feature_map.backbone.conv9.1", "feature_map.backbone.layer2.0.conv1.0"],
            # 接layer2
            'feature_map.backbone.conv10.0': ["feature_map.backbone.layer2.1.conv2.2", "feature_map.backbone.conv10.1"],
            'feature_map.backbone.conv10.1': ["feature_map.backbone.conv10.0", "feature_map.backbone.conv10.2"],
            'feature_map.backbone.conv10.2': ["feature_map.backbone.conv10.1", "feature_map.backbone.conv12.0"],
            'feature_map.backbone.conv11.0': ["feature_map.backbone.conv8.2", "feature_map.backbone.conv11.1"],
            'feature_map.backbone.conv11.1': ["feature_map.backbone.conv11.0", "feature_map.backbone.conv11.2"],
            'feature_map.backbone.conv11.2': ["feature_map.backbone.conv11.1", "feature_map.backbone.conv12.0"],
            'feature_map.backbone.conv12.0': [["feature_map.backbone.conv11.2", "feature_map.backbone.conv10.2"],
                                              "feature_map.backbone.conv12.1"],
            'feature_map.backbone.conv12.1': ["feature_map.backbone.conv12.0", "feature_map.backbone.conv12.2"],
            'feature_map.backbone.conv12.2': ["feature_map.backbone.conv12.1", "feature_map.backbone.conv13.0"],
            'feature_map.backbone.conv13.0': ["feature_map.backbone.conv12.2", "feature_map.backbone.conv13.1"],
            'feature_map.backbone.conv13.1': ["feature_map.backbone.conv13.0", "feature_map.backbone.conv13.2"],
            'feature_map.backbone.conv13.2': ["feature_map.backbone.conv13.1",
                                              ["feature_map.backbone.conv14.0", "feature_map.backbone.conv16.0"]],
            'feature_map.backbone.conv14.0': ["feature_map.backbone.conv13.2", "feature_map.backbone.conv14.1"],
            'feature_map.backbone.conv14.1': ["feature_map.backbone.conv14.0", "feature_map.backbone.conv14.2"],
            'feature_map.backbone.conv14.2': ["feature_map.backbone.conv14.1", "feature_map.backbone.layer3.0.conv1.0"],
            'feature_map.backbone.conv15.0': ["feature_map.backbone.layer3.7.conv2.2", "feature_map.backbone.conv15.1"],
            'feature_map.backbone.conv15.1': ["feature_map.backbone.conv15.0", "feature_map.backbone.conv15.2"],
            'feature_map.backbone.conv15.2': ["feature_map.backbone.conv15.1", "feature_map.backbone.conv17.0"],
            'feature_map.backbone.conv16.0': ["feature_map.backbone.conv13.2", "feature_map.backbone.conv16.1"],
            'feature_map.backbone.conv16.1': ["feature_map.backbone.conv16.0", "feature_map.backbone.conv16.2"],
            'feature_map.backbone.conv16.2': ["feature_map.backbone.conv16.1", "feature_map.backbone.conv17.0"],
            'feature_map.backbone.conv17.0': [["feature_map.backbone.conv16.2", "feature_map.backbone.conv15.2"],
                                              "feature_map.backbone.conv17.1"],
            'feature_map.backbone.conv17.1': ["feature_map.backbone.conv17.0", "feature_map.backbone.conv17.2"],
            'feature_map.backbone.conv17.2': ["feature_map.backbone.conv17.1",
                                              ["OUTPUT1", "feature_map.backbone.conv18.0"]],
            # feature_map1的输出
            'feature_map.backbone.conv18.0': ["feature_map.backbone.conv17.2", "feature_map.backbone.conv18.1"],
            'feature_map.backbone.conv18.1': ["feature_map.backbone.conv18.0", "feature_map.backbone.conv18.2"],
            'feature_map.backbone.conv18.2': ["feature_map.backbone.conv18.1",
                                              ["feature_map.backbone.conv19.0", "feature_map.backbone.conv21.0"]],
            'feature_map.backbone.conv19.0': ["feature_map.backbone.conv18.2", "feature_map.backbone.conv19.1"],
            'feature_map.backbone.conv19.1': ["feature_map.backbone.conv19.0", "feature_map.backbone.conv19.2"],
            'feature_map.backbone.conv19.2': ["feature_map.backbone.conv19.1", ["feature_map.backbone.layer4.0.conv1.0",
                                                                                "feature_map.backbone.layer4.1.conv1.0"]],
            'feature_map.backbone.layer4.0.conv1.0': ["feature_map.backbone.conv19.2",
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
                                                      ["feature_map.backbone.layer4.1.conv1.0",
                                                       "feature_map.backbone.layer4.2.conv1.0"]],
            'feature_map.backbone.layer4.1.conv1.0': [
                ["feature_map.backbone.layer4.0.conv2.2", "feature_map.backbone.conv19.2"],
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
                                                      ["feature_map.backbone.layer4.2.conv1.0",
                                                       "feature_map.backbone.layer4.3.conv1.0"]],
            'feature_map.backbone.layer4.2.conv1.0': [
                ["feature_map.backbone.layer4.1.conv2.2", "feature_map.backbone.layer4.0.conv2.2"],
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
                                                      ["feature_map.backbone.layer4.3.conv1.0",
                                                       "feature_map.backbone.layer4.4.conv1.0"]],
            'feature_map.backbone.layer4.3.conv1.0': [
                ["feature_map.backbone.layer4.2.conv2.2", "feature_map.backbone.layer4.1.conv2.2"],
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
                                                      ["feature_map.backbone.layer4.4.conv1.0",
                                                       "feature_map.backbone.layer4.5.conv1.0"]],
            'feature_map.backbone.layer4.4.conv1.0': [
                ["feature_map.backbone.layer4.3.conv2.2", "feature_map.backbone.layer4.2.conv2.2"],
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
                                                      ["feature_map.backbone.layer4.5.conv1.0",
                                                       "feature_map.backbone.layer4.6.conv1.0"]],
            'feature_map.backbone.layer4.5.conv1.0': [
                ["feature_map.backbone.layer4.4.conv2.2", "feature_map.backbone.layer4.3.conv2.2"],
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
                                                      ["feature_map.backbone.layer4.6.conv1.0",
                                                       "feature_map.backbone.layer4.7.conv1.0"]],
            'feature_map.backbone.layer4.6.conv1.0': [
                ["feature_map.backbone.layer4.5.conv2.2", "feature_map.backbone.layer4.4.conv2.2"],
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
                                                      ["feature_map.backbone.layer4.7.conv1.0",
                                                       "feature_map.backbone.conv20.0"]],
            'feature_map.backbone.layer4.7.conv1.0': [
                ["feature_map.backbone.layer4.6.conv2.2", "feature_map.backbone.layer4.5.conv2.2"],
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
                                                      "feature_map.backbone.conv20.0"],
            'feature_map.backbone.conv20.0': [
                ["feature_map.backbone.layer4.7.conv2.2", "feature_map.backbone.layer4.6.conv2.2"],
                "feature_map.backbone.conv20.1"],
            'feature_map.backbone.conv20.1': ["feature_map.backbone.conv20.0", "feature_map.backbone.conv20.2"],
            'feature_map.backbone.conv20.2': ["feature_map.backbone.conv20.1", "feature_map.backbone.conv22.0"],
            'feature_map.backbone.conv21.0': ["feature_map.backbone.conv18.2", "feature_map.backbone.conv21.1"],
            'feature_map.backbone.conv21.1': ["feature_map.backbone.conv21.0", "feature_map.backbone.conv21.2"],
            'feature_map.backbone.conv21.2': ["feature_map.backbone.conv21.1", "feature_map.backbone.conv22.0"],
            'feature_map.backbone.conv22.0': [["feature_map.backbone.conv21.2", "feature_map.backbone.conv20.2"],
                                              "feature_map.backbone.conv22.1"],
            'feature_map.backbone.conv22.1': ["feature_map.backbone.conv22.0", "feature_map.backbone.conv22.2"],
            'feature_map.backbone.conv22.2': ["feature_map.backbone.conv22.1",
                                              ["feature_map.backbone.conv23.0", "OUTPUT2"]],
            # feature_map2的输出
            'feature_map.backbone.conv23.0': ["feature_map.backbone.conv22.2", "feature_map.backbone.conv23.1"],
            'feature_map.backbone.conv23.1': ["feature_map.backbone.conv23.0", "feature_map.backbone.conv23.2"],
            'feature_map.backbone.conv23.2': ["feature_map.backbone.conv23.1",
                                              ["feature_map.backbone.conv24.0", "feature_map.backbone.conv26.0"]],
            'feature_map.backbone.conv24.0': ["feature_map.backbone.conv23.2", "feature_map.backbone.conv24.1"],
            'feature_map.backbone.conv24.1': ["feature_map.backbone.conv24.0", "feature_map.backbone.conv24.2"],
            'feature_map.backbone.conv24.2': ["feature_map.backbone.conv24.1", "feature_map.backbone.layer5.0.conv1.0"],
            'feature_map.backbone.layer5.0.conv1.0': ["feature_map.backbone.conv24.2",
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
                                                      "feature_map.backbone.conv25.0"],
            'feature_map.backbone.conv25.0': ["feature_map.backbone.layer5.3.conv2.2", "feature_map.backbone.conv25.1"],
            'feature_map.backbone.conv25.1': ["feature_map.backbone.conv25.0", "feature_map.backbone.conv25.2"],
            'feature_map.backbone.conv25.2': ["feature_map.backbone.conv25.1", "feature_map.backbone.conv27.0"],
            'feature_map.backbone.conv26.0': ["feature_map.backbone.conv23.2", "feature_map.backbone.conv26.1"],
            'feature_map.backbone.conv26.1': ["feature_map.backbone.conv26.0", "feature_map.backbone.conv26.2"],
            'feature_map.backbone.conv26.2': ["feature_map.backbone.conv26.1", "feature_map.backbone.conv27.0"],
            'feature_map.backbone.conv27.0': [["feature_map.backbone.conv26.2", "feature_map.backbone.conv25.2"],
                                              "feature_map.backbone.conv27.1"],
            'feature_map.backbone.conv27.1': ["feature_map.backbone.conv27.0", "feature_map.backbone.conv27.2"],
            'feature_map.backbone.conv27.2': ["feature_map.backbone.conv27.1", "OUTPUT3"],
            # feature_map3的输出
            'feature_map.backbone.layer2.0.conv1.0': ["feature_map.backbone.conv9.2",
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
                                                      "feature_map.backbone.conv10.0"],
            'feature_map.backbone.layer3.0.conv1.0': ["feature_map.backbone.conv14.2",
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
                                                      "feature_map.backbone.conv15.0"],
            # 'feature_map.conv1.0': ["feature_map.backbone.conv27.2", "feature_map.conv1.1"],
            # 'feature_map.conv1.1': ["feature_map.conv1.0", "feature_map.conv1.2"],
            # 'feature_map.conv1.2': ["feature_map.conv1.1", "feature_map.conv2.0"],
            # 'feature_map.conv2.0': ["feature_map.conv1.2", "feature_map.conv2.1"],
            # 'feature_map.conv2.1': ["feature_map.conv2.0", "feature_map.conv2.2"],
            # 'feature_map.conv2.2': ["feature_map.conv2.1", "feature_map.conv3.0"],
            # 'feature_map.conv3.0': ["feature_map.conv2.2", "feature_map.conv3.1"],
            # 'feature_map.conv3.1': ["feature_map.conv3.0", "feature_map.conv3.2"],
            # 'feature_map.conv3.2': ["feature_map.conv3.1",
            #                         ["feature_map.maxpool1", "feature_map.maxpool2", "feature_map.maxpool3",
            #                          "feature_map.conv4.0"]],
            # 'feature_map.maxpool1': ["feature_map.conv3.2", "feature_map.conv4.0"],
            # 'feature_map.maxpool2': ["feature_map.conv3.2", "feature_map.conv4.0"],
            # 'feature_map.maxpool3': ["feature_map.conv3.2", "feature_map.conv4.0"],
            # 'feature_map.conv4.0': [
            #     ["feature_map.maxpool3", "feature_map.maxpool2", "feature_map.maxpool1", "feature_map.conv3.2"],
            #     "feature_map.conv4.1"],
            # 'feature_map.conv4.1': ["feature_map.conv4.0", "feature_map.conv4.2"],
            # 'feature_map.conv4.2': ["feature_map.conv4.1", "feature_map.conv5.0"],
            # 'feature_map.conv5.0': ["feature_map.conv4.2", "feature_map.conv5.1"],
            # 'feature_map.conv5.1': ["feature_map.conv5.0", "feature_map.conv5.2"],
            # 'feature_map.conv5.2': ["feature_map.conv5.1", "feature_map.conv6.0"],
            # 'feature_map.conv6.0': ["feature_map.conv5.2", "feature_map.conv6.1"],
            # 'feature_map.conv6.1': ["feature_map.conv6.0", "feature_map.conv6.2"],
            # 'feature_map.conv6.2': ["feature_map.conv6.1", ["feature_map.conv7.0", "feature_map.backblock3.conv0.0"]],
            # 'feature_map.conv7.0': ["feature_map.conv6.2", "feature_map.conv7.1"],
            # 'feature_map.conv7.1': ["feature_map.conv7.0", "feature_map.conv7.2"],
            # 'feature_map.conv7.2': ["feature_map.conv7.1", "feature_map.backblock0.conv0.0"],
            # 'feature_map.conv8.0': ["feature_map.backbone.conv22.2", "feature_map.conv8.1"],
            # 'feature_map.conv8.1': ["feature_map.conv8.0", "feature_map.conv8.2"],
            # 'feature_map.conv8.2': ["feature_map.conv8.1", "feature_map.backblock0.conv0.0"],
            # 'feature_map.backblock0.conv0.0': [["feature_map.conv8.2", "feature_map.conv7.2"],
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
            # 'feature_map.backblock0.conv4.2': [["feature_map.backblock0.conv4.1", "feature_map.backblock2.conv0.0"],
            #                                    "feature_map.conv9.0"],
            # 'feature_map.conv9.0': ["feature_map.backblock0.conv4.2", "feature_map.conv9.1"],
            # 'feature_map.conv9.1': ["feature_map.conv9.0", "feature_map.conv9.2"],
            # 'feature_map.conv9.2': ["feature_map.conv9.1", "feature_map.backblock1.conv0.0"],
            # 'feature_map.conv10.0': ["feature_map.backbone.conv17.2", "feature_map.conv10.1"],
            # 'feature_map.conv10.1': ["feature_map.conv10.0", "feature_map.conv10.2"],
            # 'feature_map.conv10.2': ["feature_map.conv10.1", "feature_map.backblock1.conv0.0"],
            # 'feature_map.backblock1.conv0.0': [["feature_map.conv10.2", "feature_map.conv9.2"],
            #                                    "feature_map.backblock1.conv0.1"],
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
            #                                    ["feature_map.backblock1.conv5.0", "feature_map.conv11.0"]],
            # 'feature_map.backblock1.conv5.0': ["feature_map.backblock1.conv4.2", "feature_map.backblock1.conv5.1"],
            # 'feature_map.backblock1.conv5.1': ["feature_map.backblock1.conv5.0", "feature_map.backblock1.conv5.2"],
            # 'feature_map.backblock1.conv5.2': ["feature_map.backblock1.conv5.1", "feature_map.backblock1.conv6"],
            # 'feature_map.backblock1.conv6': ["feature_map.backblock1.conv5.2", "detect_1.sigmoid"],  # 输出一
            # 'feature_map.conv11.0': ["feature_map.backblock1.conv4.2", "feature_map.conv11.1"],
            # 'feature_map.conv11.1': ["feature_map.conv11.0", "feature_map.conv11.2"],
            # 'feature_map.conv11.2': ["feature_map.conv11.1", "feature_map.backblock2.conv0.0"],
            # 'feature_map.backblock2.conv0.0': ["feature_map.conv11.2",
            #                                    ["feature_map.backblock2.conv0.1", "feature_map.backblock0.conv4.2"]],
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
            # 'feature_map.backblock2.conv4.2': ["feature_map.backblock2.conv4.1",
            #                                    ["feature_map.backblock2.conv5.0", "feature_map.conv12.0"]],
            # 'feature_map.backblock2.conv5.0': ["feature_map.backblock2.conv4.2", "feature_map.backblock2.conv5.1"],
            # 'feature_map.backblock2.conv5.1': ["feature_map.backblock2.conv5.0", "feature_map.backblock2.conv5.2"],
            # 'feature_map.backblock2.conv5.2': ["feature_map.backblock2.conv5.1", "feature_map.backblock2.conv6"],
            # 'feature_map.backblock2.conv6': ["feature_map.backblock2.conv5.2", "detect_2.sigmoid"],  # 输出二
            # 'feature_map.conv12.0': ["feature_map.backblock2.conv4.2", "feature_map.conv12.1"],
            # 'feature_map.conv12.1': ["feature_map.conv12.0", "feature_map.conv12.2"],
            # 'feature_map.conv12.2': ["feature_map.conv12.1", "feature_map.backblock3.conv0.0"],
            # 'feature_map.backblock3.conv0.0': [["feature_map.conv12.2", "feature_map.conv6.2"],
            #                                    "feature_map.backblock3.conv0.1"],
            # 'feature_map.backblock3.conv0.1': ["feature_map.backblock3.conv0.0", "feature_map.backblock3.conv0.2"],
            # 'feature_map.backblock3.conv0.2': ["feature_map.backblock3.conv0.1", "feature_map.backblock3.conv1.0"],
            # 'feature_map.backblock3.conv1.0': ["feature_map.backblock3.conv0.2", "feature_map.backblock3.conv1.1"],
            # 'feature_map.backblock3.conv1.1': ["feature_map.backblock3.conv1.0", "feature_map.backblock3.conv1.2"],
            # 'feature_map.backblock3.conv1.2': ["feature_map.backblock3.conv1.1", "feature_map.backblock3.conv2.0"],
            # 'feature_map.backblock3.conv2.0': ["feature_map.backblock3.conv1.2", "feature_map.backblock3.conv2.1"],
            # 'feature_map.backblock3.conv2.1': ["feature_map.backblock3.conv2.0", "feature_map.backblock3.conv2.2"],
            # 'feature_map.backblock3.conv2.2': ["feature_map.backblock3.conv2.1", "feature_map.backblock3.conv3.0"],
            # 'feature_map.backblock3.conv3.0': ["feature_map.backblock3.conv2.2", "feature_map.backblock3.conv3.1"],
            # 'feature_map.backblock3.conv3.1': ["feature_map.backblock3.conv3.0", "feature_map.backblock3.conv3.2"],
            # 'feature_map.backblock3.conv3.2': ["feature_map.backblock3.conv3.1", "feature_map.backblock3.conv4.0"],
            # 'feature_map.backblock3.conv4.0': ["feature_map.backblock3.conv3.2", "feature_map.backblock3.conv4.1"],
            # 'feature_map.backblock3.conv4.1': ["feature_map.backblock3.conv4.0", "feature_map.backblock3.conv4.2"],
            # 'feature_map.backblock3.conv4.2': ["feature_map.backblock3.conv4.1", "feature_map.backblock3.conv5.0"],
            # 'feature_map.backblock3.conv5.0': ["feature_map.backblock3.conv4.2", "feature_map.backblock3.conv5.1"],
            # 'feature_map.backblock3.conv5.1': ["feature_map.backblock3.conv5.0", "feature_map.backblock3.conv5.2"],
            # 'feature_map.backblock3.conv5.2': ["feature_map.backblock3.conv5.1", "feature_map.backblock3.conv6"],
            # 'feature_map.backblock3.conv6': ["feature_map.backblock3.conv5.2", "detect_3.sigmoid"],  # 输出三
            # 'detect_1.sigmoid': ["feature_map.backblock1.conv6", "OUTPUT1"],
            # 'detect_2.sigmoid': ["feature_map.backblock2.conv6", "OUTPUT2"],
            # 'detect_3.sigmoid': ["feature_map.backblock3.conv6", "OUTPUT3"],
        }


    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self,layer_name,new_layer):
        if 'feature_map' == layer_name:
            self.feature_map= new_layer
            self.layer_names["feature_map"]=new_layer
            self.origin_layer_names["feature_map"]=new_layer
        elif 'feature_map.backbone' == layer_name:
            self.feature_map.backbone= new_layer
            self.layer_names["feature_map.backbone"]=new_layer
            self.origin_layer_names["feature_map.backbone"]=new_layer
        elif 'feature_map.backbone.conv0' == layer_name:
            self.feature_map.backbone.conv0= new_layer
            self.layer_names["feature_map.backbone.conv0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0"]=new_layer
        elif 'feature_map.backbone.conv0.0' == layer_name:
            self.feature_map.backbone.conv0[0]= new_layer
            self.layer_names["feature_map.backbone.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0.0"]=new_layer
        elif 'feature_map.backbone.conv0.1' == layer_name:
            self.feature_map.backbone.conv0[1]= new_layer
            self.layer_names["feature_map.backbone.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0.1"]=new_layer
        elif 'feature_map.backbone.conv0.2' == layer_name:
            self.feature_map.backbone.conv0[2]= new_layer
            self.layer_names["feature_map.backbone.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0.2"]=new_layer
        elif 'feature_map.backbone.conv1' == layer_name:
            self.feature_map.backbone.conv1= new_layer
            self.layer_names["feature_map.backbone.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1"]=new_layer
        elif 'feature_map.backbone.conv1.0' == layer_name:
            self.feature_map.backbone.conv1[0]= new_layer
            self.layer_names["feature_map.backbone.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1.0"]=new_layer
        elif 'feature_map.backbone.conv1.1' == layer_name:
            self.feature_map.backbone.conv1[1]= new_layer
            self.layer_names["feature_map.backbone.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1.1"]=new_layer
        elif 'feature_map.backbone.conv1.2' == layer_name:
            self.feature_map.backbone.conv1[2]= new_layer
            self.layer_names["feature_map.backbone.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1.2"]=new_layer
        elif 'feature_map.backbone.conv2' == layer_name:
            self.feature_map.backbone.conv2= new_layer
            self.layer_names["feature_map.backbone.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2"]=new_layer
        elif 'feature_map.backbone.conv2.0' == layer_name:
            self.feature_map.backbone.conv2[0]= new_layer
            self.layer_names["feature_map.backbone.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2.0"]=new_layer
        elif 'feature_map.backbone.conv2.1' == layer_name:
            self.feature_map.backbone.conv2[1]= new_layer
            self.layer_names["feature_map.backbone.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2.1"]=new_layer
        elif 'feature_map.backbone.conv2.2' == layer_name:
            self.feature_map.backbone.conv2[2]= new_layer
            self.layer_names["feature_map.backbone.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv3' == layer_name:
            self.feature_map.backbone.conv3= new_layer
            self.layer_names["feature_map.backbone.conv3"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3"]=new_layer
        elif 'feature_map.backbone.conv3.0' == layer_name:
            self.feature_map.backbone.conv3[0]= new_layer
            self.layer_names["feature_map.backbone.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3.0"]=new_layer
        elif 'feature_map.backbone.conv3.1' == layer_name:
            self.feature_map.backbone.conv3[1]= new_layer
            self.layer_names["feature_map.backbone.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3.1"]=new_layer
        elif 'feature_map.backbone.conv3.2' == layer_name:
            self.feature_map.backbone.conv3[2]= new_layer
            self.layer_names["feature_map.backbone.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3.2"]=new_layer
        elif 'feature_map.backbone.conv4' == layer_name:
            self.feature_map.backbone.conv4= new_layer
            self.layer_names["feature_map.backbone.conv4"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4"]=new_layer
        elif 'feature_map.backbone.conv4.0' == layer_name:
            self.feature_map.backbone.conv4[0]= new_layer
            self.layer_names["feature_map.backbone.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4.0"]=new_layer
        elif 'feature_map.backbone.conv4.1' == layer_name:
            self.feature_map.backbone.conv4[1]= new_layer
            self.layer_names["feature_map.backbone.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4.1"]=new_layer
        elif 'feature_map.backbone.conv4.2' == layer_name:
            self.feature_map.backbone.conv4[2]= new_layer
            self.layer_names["feature_map.backbone.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4.2"]=new_layer
        elif 'feature_map.backbone.conv5' == layer_name:
            self.feature_map.backbone.conv5= new_layer
            self.layer_names["feature_map.backbone.conv5"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5"]=new_layer
        elif 'feature_map.backbone.conv5.0' == layer_name:
            self.feature_map.backbone.conv5[0]= new_layer
            self.layer_names["feature_map.backbone.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5.0"]=new_layer
        elif 'feature_map.backbone.conv5.1' == layer_name:
            self.feature_map.backbone.conv5[1]= new_layer
            self.layer_names["feature_map.backbone.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5.1"]=new_layer
        elif 'feature_map.backbone.conv5.2' == layer_name:
            self.feature_map.backbone.conv5[2]= new_layer
            self.layer_names["feature_map.backbone.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5.2"]=new_layer
        elif 'feature_map.backbone.conv6' == layer_name:
            self.feature_map.backbone.conv6= new_layer
            self.layer_names["feature_map.backbone.conv6"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv6"]=new_layer
        elif 'feature_map.backbone.conv6.0' == layer_name:
            self.feature_map.backbone.conv6[0]= new_layer
            self.layer_names["feature_map.backbone.conv6.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv6.0"]=new_layer
        elif 'feature_map.backbone.conv6.1' == layer_name:
            self.feature_map.backbone.conv6[1]= new_layer
            self.layer_names["feature_map.backbone.conv6.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv6.1"]=new_layer
        elif 'feature_map.backbone.conv6.2' == layer_name:
            self.feature_map.backbone.conv6[2]= new_layer
            self.layer_names["feature_map.backbone.conv6.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv6.2"]=new_layer
        elif 'feature_map.backbone.conv7' == layer_name:
            self.feature_map.backbone.conv7= new_layer
            self.layer_names["feature_map.backbone.conv7"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv7"]=new_layer
        elif 'feature_map.backbone.conv7.0' == layer_name:
            self.feature_map.backbone.conv7[0]= new_layer
            self.layer_names["feature_map.backbone.conv7.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv7.0"]=new_layer
        elif 'feature_map.backbone.conv7.1' == layer_name:
            self.feature_map.backbone.conv7[1]= new_layer
            self.layer_names["feature_map.backbone.conv7.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv7.1"]=new_layer
        elif 'feature_map.backbone.conv7.2' == layer_name:
            self.feature_map.backbone.conv7[2]= new_layer
            self.layer_names["feature_map.backbone.conv7.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv7.2"]=new_layer
        elif 'feature_map.backbone.conv8' == layer_name:
            self.feature_map.backbone.conv8= new_layer
            self.layer_names["feature_map.backbone.conv8"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv8"]=new_layer
        elif 'feature_map.backbone.conv8.0' == layer_name:
            self.feature_map.backbone.conv8[0]= new_layer
            self.layer_names["feature_map.backbone.conv8.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv8.0"]=new_layer
        elif 'feature_map.backbone.conv8.1' == layer_name:
            self.feature_map.backbone.conv8[1]= new_layer
            self.layer_names["feature_map.backbone.conv8.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv8.1"]=new_layer
        elif 'feature_map.backbone.conv8.2' == layer_name:
            self.feature_map.backbone.conv8[2]= new_layer
            self.layer_names["feature_map.backbone.conv8.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv8.2"]=new_layer
        elif 'feature_map.backbone.conv9' == layer_name:
            self.feature_map.backbone.conv9= new_layer
            self.layer_names["feature_map.backbone.conv9"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv9"]=new_layer
        elif 'feature_map.backbone.conv9.0' == layer_name:
            self.feature_map.backbone.conv9[0]= new_layer
            self.layer_names["feature_map.backbone.conv9.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv9.0"]=new_layer
        elif 'feature_map.backbone.conv9.1' == layer_name:
            self.feature_map.backbone.conv9[1]= new_layer
            self.layer_names["feature_map.backbone.conv9.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv9.1"]=new_layer
        elif 'feature_map.backbone.conv9.2' == layer_name:
            self.feature_map.backbone.conv9[2]= new_layer
            self.layer_names["feature_map.backbone.conv9.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv9.2"]=new_layer
        elif 'feature_map.backbone.conv10' == layer_name:
            self.feature_map.backbone.conv10= new_layer
            self.layer_names["feature_map.backbone.conv10"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv10"]=new_layer
        elif 'feature_map.backbone.conv10.0' == layer_name:
            self.feature_map.backbone.conv10[0]= new_layer
            self.layer_names["feature_map.backbone.conv10.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv10.0"]=new_layer
        elif 'feature_map.backbone.conv10.1' == layer_name:
            self.feature_map.backbone.conv10[1]= new_layer
            self.layer_names["feature_map.backbone.conv10.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv10.1"]=new_layer
        elif 'feature_map.backbone.conv10.2' == layer_name:
            self.feature_map.backbone.conv10[2]= new_layer
            self.layer_names["feature_map.backbone.conv10.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv10.2"]=new_layer
        elif 'feature_map.backbone.conv11' == layer_name:
            self.feature_map.backbone.conv11= new_layer
            self.layer_names["feature_map.backbone.conv11"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv11"]=new_layer
        elif 'feature_map.backbone.conv11.0' == layer_name:
            self.feature_map.backbone.conv11[0]= new_layer
            self.layer_names["feature_map.backbone.conv11.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv11.0"]=new_layer
        elif 'feature_map.backbone.conv11.1' == layer_name:
            self.feature_map.backbone.conv11[1]= new_layer
            self.layer_names["feature_map.backbone.conv11.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv11.1"]=new_layer
        elif 'feature_map.backbone.conv11.2' == layer_name:
            self.feature_map.backbone.conv11[2]= new_layer
            self.layer_names["feature_map.backbone.conv11.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv11.2"]=new_layer
        elif 'feature_map.backbone.conv12' == layer_name:
            self.feature_map.backbone.conv12= new_layer
            self.layer_names["feature_map.backbone.conv12"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv12"]=new_layer
        elif 'feature_map.backbone.conv12.0' == layer_name:
            self.feature_map.backbone.conv12[0]= new_layer
            self.layer_names["feature_map.backbone.conv12.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv12.0"]=new_layer
        elif 'feature_map.backbone.conv12.1' == layer_name:
            self.feature_map.backbone.conv12[1]= new_layer
            self.layer_names["feature_map.backbone.conv12.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv12.1"]=new_layer
        elif 'feature_map.backbone.conv12.2' == layer_name:
            self.feature_map.backbone.conv12[2]= new_layer
            self.layer_names["feature_map.backbone.conv12.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv12.2"]=new_layer
        elif 'feature_map.backbone.conv13' == layer_name:
            self.feature_map.backbone.conv13= new_layer
            self.layer_names["feature_map.backbone.conv13"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv13"]=new_layer
        elif 'feature_map.backbone.conv13.0' == layer_name:
            self.feature_map.backbone.conv13[0]= new_layer
            self.layer_names["feature_map.backbone.conv13.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv13.0"]=new_layer
        elif 'feature_map.backbone.conv13.1' == layer_name:
            self.feature_map.backbone.conv13[1]= new_layer
            self.layer_names["feature_map.backbone.conv13.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv13.1"]=new_layer
        elif 'feature_map.backbone.conv13.2' == layer_name:
            self.feature_map.backbone.conv13[2]= new_layer
            self.layer_names["feature_map.backbone.conv13.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv13.2"]=new_layer
        elif 'feature_map.backbone.conv14' == layer_name:
            self.feature_map.backbone.conv14= new_layer
            self.layer_names["feature_map.backbone.conv14"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv14"]=new_layer
        elif 'feature_map.backbone.conv14.0' == layer_name:
            self.feature_map.backbone.conv14[0]= new_layer
            self.layer_names["feature_map.backbone.conv14.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv14.0"]=new_layer
        elif 'feature_map.backbone.conv14.1' == layer_name:
            self.feature_map.backbone.conv14[1]= new_layer
            self.layer_names["feature_map.backbone.conv14.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv14.1"]=new_layer
        elif 'feature_map.backbone.conv14.2' == layer_name:
            self.feature_map.backbone.conv14[2]= new_layer
            self.layer_names["feature_map.backbone.conv14.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv14.2"]=new_layer
        elif 'feature_map.backbone.conv15' == layer_name:
            self.feature_map.backbone.conv15= new_layer
            self.layer_names["feature_map.backbone.conv15"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv15"]=new_layer
        elif 'feature_map.backbone.conv15.0' == layer_name:
            self.feature_map.backbone.conv15[0]= new_layer
            self.layer_names["feature_map.backbone.conv15.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv15.0"]=new_layer
        elif 'feature_map.backbone.conv15.1' == layer_name:
            self.feature_map.backbone.conv15[1]= new_layer
            self.layer_names["feature_map.backbone.conv15.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv15.1"]=new_layer
        elif 'feature_map.backbone.conv15.2' == layer_name:
            self.feature_map.backbone.conv15[2]= new_layer
            self.layer_names["feature_map.backbone.conv15.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv15.2"]=new_layer
        elif 'feature_map.backbone.conv16' == layer_name:
            self.feature_map.backbone.conv16= new_layer
            self.layer_names["feature_map.backbone.conv16"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv16"]=new_layer
        elif 'feature_map.backbone.conv16.0' == layer_name:
            self.feature_map.backbone.conv16[0]= new_layer
            self.layer_names["feature_map.backbone.conv16.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv16.0"]=new_layer
        elif 'feature_map.backbone.conv16.1' == layer_name:
            self.feature_map.backbone.conv16[1]= new_layer
            self.layer_names["feature_map.backbone.conv16.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv16.1"]=new_layer
        elif 'feature_map.backbone.conv16.2' == layer_name:
            self.feature_map.backbone.conv16[2]= new_layer
            self.layer_names["feature_map.backbone.conv16.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv16.2"]=new_layer
        elif 'feature_map.backbone.conv17' == layer_name:
            self.feature_map.backbone.conv17= new_layer
            self.layer_names["feature_map.backbone.conv17"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv17"]=new_layer
        elif 'feature_map.backbone.conv17.0' == layer_name:
            self.feature_map.backbone.conv17[0]= new_layer
            self.layer_names["feature_map.backbone.conv17.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv17.0"]=new_layer
        elif 'feature_map.backbone.conv17.1' == layer_name:
            self.feature_map.backbone.conv17[1]= new_layer
            self.layer_names["feature_map.backbone.conv17.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv17.1"]=new_layer
        elif 'feature_map.backbone.conv17.2' == layer_name:
            self.feature_map.backbone.conv17[2]= new_layer
            self.layer_names["feature_map.backbone.conv17.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv17.2"]=new_layer
        elif 'feature_map.backbone.conv18' == layer_name:
            self.feature_map.backbone.conv18= new_layer
            self.layer_names["feature_map.backbone.conv18"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv18"]=new_layer
        elif 'feature_map.backbone.conv18.0' == layer_name:
            self.feature_map.backbone.conv18[0]= new_layer
            self.layer_names["feature_map.backbone.conv18.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv18.0"]=new_layer
        elif 'feature_map.backbone.conv18.1' == layer_name:
            self.feature_map.backbone.conv18[1]= new_layer
            self.layer_names["feature_map.backbone.conv18.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv18.1"]=new_layer
        elif 'feature_map.backbone.conv18.2' == layer_name:
            self.feature_map.backbone.conv18[2]= new_layer
            self.layer_names["feature_map.backbone.conv18.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv18.2"]=new_layer
        elif 'feature_map.backbone.conv19' == layer_name:
            self.feature_map.backbone.conv19= new_layer
            self.layer_names["feature_map.backbone.conv19"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv19"]=new_layer
        elif 'feature_map.backbone.conv19.0' == layer_name:
            self.feature_map.backbone.conv19[0]= new_layer
            self.layer_names["feature_map.backbone.conv19.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv19.0"]=new_layer
        elif 'feature_map.backbone.conv19.1' == layer_name:
            self.feature_map.backbone.conv19[1]= new_layer
            self.layer_names["feature_map.backbone.conv19.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv19.1"]=new_layer
        elif 'feature_map.backbone.conv19.2' == layer_name:
            self.feature_map.backbone.conv19[2]= new_layer
            self.layer_names["feature_map.backbone.conv19.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv19.2"]=new_layer
        elif 'feature_map.backbone.conv20' == layer_name:
            self.feature_map.backbone.conv20= new_layer
            self.layer_names["feature_map.backbone.conv20"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv20"]=new_layer
        elif 'feature_map.backbone.conv20.0' == layer_name:
            self.feature_map.backbone.conv20[0]= new_layer
            self.layer_names["feature_map.backbone.conv20.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv20.0"]=new_layer
        elif 'feature_map.backbone.conv20.1' == layer_name:
            self.feature_map.backbone.conv20[1]= new_layer
            self.layer_names["feature_map.backbone.conv20.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv20.1"]=new_layer
        elif 'feature_map.backbone.conv20.2' == layer_name:
            self.feature_map.backbone.conv20[2]= new_layer
            self.layer_names["feature_map.backbone.conv20.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv20.2"]=new_layer
        elif 'feature_map.backbone.conv21' == layer_name:
            self.feature_map.backbone.conv21= new_layer
            self.layer_names["feature_map.backbone.conv21"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv21"]=new_layer
        elif 'feature_map.backbone.conv21.0' == layer_name:
            self.feature_map.backbone.conv21[0]= new_layer
            self.layer_names["feature_map.backbone.conv21.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv21.0"]=new_layer
        elif 'feature_map.backbone.conv21.1' == layer_name:
            self.feature_map.backbone.conv21[1]= new_layer
            self.layer_names["feature_map.backbone.conv21.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv21.1"]=new_layer
        elif 'feature_map.backbone.conv21.2' == layer_name:
            self.feature_map.backbone.conv21[2]= new_layer
            self.layer_names["feature_map.backbone.conv21.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv21.2"]=new_layer
        elif 'feature_map.backbone.conv22' == layer_name:
            self.feature_map.backbone.conv22= new_layer
            self.layer_names["feature_map.backbone.conv22"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv22"]=new_layer
        elif 'feature_map.backbone.conv22.0' == layer_name:
            self.feature_map.backbone.conv22[0]= new_layer
            self.layer_names["feature_map.backbone.conv22.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv22.0"]=new_layer
        elif 'feature_map.backbone.conv22.1' == layer_name:
            self.feature_map.backbone.conv22[1]= new_layer
            self.layer_names["feature_map.backbone.conv22.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv22.1"]=new_layer
        elif 'feature_map.backbone.conv22.2' == layer_name:
            self.feature_map.backbone.conv22[2]= new_layer
            self.layer_names["feature_map.backbone.conv22.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv22.2"]=new_layer
        elif 'feature_map.backbone.conv23' == layer_name:
            self.feature_map.backbone.conv23= new_layer
            self.layer_names["feature_map.backbone.conv23"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv23"]=new_layer
        elif 'feature_map.backbone.conv23.0' == layer_name:
            self.feature_map.backbone.conv23[0]= new_layer
            self.layer_names["feature_map.backbone.conv23.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv23.0"]=new_layer
        elif 'feature_map.backbone.conv23.1' == layer_name:
            self.feature_map.backbone.conv23[1]= new_layer
            self.layer_names["feature_map.backbone.conv23.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv23.1"]=new_layer
        elif 'feature_map.backbone.conv23.2' == layer_name:
            self.feature_map.backbone.conv23[2]= new_layer
            self.layer_names["feature_map.backbone.conv23.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv23.2"]=new_layer
        elif 'feature_map.backbone.conv24' == layer_name:
            self.feature_map.backbone.conv24= new_layer
            self.layer_names["feature_map.backbone.conv24"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv24"]=new_layer
        elif 'feature_map.backbone.conv24.0' == layer_name:
            self.feature_map.backbone.conv24[0]= new_layer
            self.layer_names["feature_map.backbone.conv24.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv24.0"]=new_layer
        elif 'feature_map.backbone.conv24.1' == layer_name:
            self.feature_map.backbone.conv24[1]= new_layer
            self.layer_names["feature_map.backbone.conv24.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv24.1"]=new_layer
        elif 'feature_map.backbone.conv24.2' == layer_name:
            self.feature_map.backbone.conv24[2]= new_layer
            self.layer_names["feature_map.backbone.conv24.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv24.2"]=new_layer
        elif 'feature_map.backbone.conv25' == layer_name:
            self.feature_map.backbone.conv25= new_layer
            self.layer_names["feature_map.backbone.conv25"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv25"]=new_layer
        elif 'feature_map.backbone.conv25.0' == layer_name:
            self.feature_map.backbone.conv25[0]= new_layer
            self.layer_names["feature_map.backbone.conv25.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv25.0"]=new_layer
        elif 'feature_map.backbone.conv25.1' == layer_name:
            self.feature_map.backbone.conv25[1]= new_layer
            self.layer_names["feature_map.backbone.conv25.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv25.1"]=new_layer
        elif 'feature_map.backbone.conv25.2' == layer_name:
            self.feature_map.backbone.conv25[2]= new_layer
            self.layer_names["feature_map.backbone.conv25.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv25.2"]=new_layer
        elif 'feature_map.backbone.conv26' == layer_name:
            self.feature_map.backbone.conv26= new_layer
            self.layer_names["feature_map.backbone.conv26"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv26"]=new_layer
        elif 'feature_map.backbone.conv26.0' == layer_name:
            self.feature_map.backbone.conv26[0]= new_layer
            self.layer_names["feature_map.backbone.conv26.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv26.0"]=new_layer
        elif 'feature_map.backbone.conv26.1' == layer_name:
            self.feature_map.backbone.conv26[1]= new_layer
            self.layer_names["feature_map.backbone.conv26.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv26.1"]=new_layer
        elif 'feature_map.backbone.conv26.2' == layer_name:
            self.feature_map.backbone.conv26[2]= new_layer
            self.layer_names["feature_map.backbone.conv26.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv26.2"]=new_layer
        elif 'feature_map.backbone.conv27' == layer_name:
            self.feature_map.backbone.conv27= new_layer
            self.layer_names["feature_map.backbone.conv27"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv27"]=new_layer
        elif 'feature_map.backbone.conv27.0' == layer_name:
            self.feature_map.backbone.conv27[0]= new_layer
            self.layer_names["feature_map.backbone.conv27.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv27.0"]=new_layer
        elif 'feature_map.backbone.conv27.1' == layer_name:
            self.feature_map.backbone.conv27[1]= new_layer
            self.layer_names["feature_map.backbone.conv27.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv27.1"]=new_layer
        elif 'feature_map.backbone.conv27.2' == layer_name:
            self.feature_map.backbone.conv27[2]= new_layer
            self.layer_names["feature_map.backbone.conv27.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv27.2"]=new_layer
        elif 'feature_map.backbone.layer2' == layer_name:
            self.feature_map.backbone.layer2= new_layer
            self.layer_names["feature_map.backbone.layer2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2"]=new_layer
        elif 'feature_map.backbone.layer2.0' == layer_name:
            self.feature_map.backbone.layer2[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer2.1' == layer_name:
            self.feature_map.backbone.layer2[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3' == layer_name:
            self.feature_map.backbone.layer3= new_layer
            self.layer_names["feature_map.backbone.layer3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3"]=new_layer
        elif 'feature_map.backbone.layer3.0' == layer_name:
            self.feature_map.backbone.layer3[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.1' == layer_name:
            self.feature_map.backbone.layer3[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.2' == layer_name:
            self.feature_map.backbone.layer3[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.3' == layer_name:
            self.feature_map.backbone.layer3[3]= new_layer
            self.layer_names["feature_map.backbone.layer3.3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.4' == layer_name:
            self.feature_map.backbone.layer3[4]= new_layer
            self.layer_names["feature_map.backbone.layer3.4"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.5' == layer_name:
            self.feature_map.backbone.layer3[5]= new_layer
            self.layer_names["feature_map.backbone.layer3.5"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.6' == layer_name:
            self.feature_map.backbone.layer3[6]= new_layer
            self.layer_names["feature_map.backbone.layer3.6"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.7' == layer_name:
            self.feature_map.backbone.layer3[7]= new_layer
            self.layer_names["feature_map.backbone.layer3.7"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4' == layer_name:
            self.feature_map.backbone.layer4= new_layer
            self.layer_names["feature_map.backbone.layer4"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4"]=new_layer
        elif 'feature_map.backbone.layer4.0' == layer_name:
            self.feature_map.backbone.layer4[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.1' == layer_name:
            self.feature_map.backbone.layer4[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.2' == layer_name:
            self.feature_map.backbone.layer4[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.3' == layer_name:
            self.feature_map.backbone.layer4[3]= new_layer
            self.layer_names["feature_map.backbone.layer4.3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.4' == layer_name:
            self.feature_map.backbone.layer4[4]= new_layer
            self.layer_names["feature_map.backbone.layer4.4"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.5' == layer_name:
            self.feature_map.backbone.layer4[5]= new_layer
            self.layer_names["feature_map.backbone.layer4.5"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.6' == layer_name:
            self.feature_map.backbone.layer4[6]= new_layer
            self.layer_names["feature_map.backbone.layer4.6"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.7' == layer_name:
            self.feature_map.backbone.layer4[7]= new_layer
            self.layer_names["feature_map.backbone.layer4.7"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5' == layer_name:
            self.feature_map.backbone.layer5= new_layer
            self.layer_names["feature_map.backbone.layer5"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5"]=new_layer
        elif 'feature_map.backbone.layer5.0' == layer_name:
            self.feature_map.backbone.layer5[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.1' == layer_name:
            self.feature_map.backbone.layer5[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.2' == layer_name:
            self.feature_map.backbone.layer5[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.3' == layer_name:
            self.feature_map.backbone.layer5[3]= new_layer
            self.layer_names["feature_map.backbone.layer5.3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.2"]=new_layer
        elif 'feature_map.conv1' == layer_name:
            self.feature_map.conv1= new_layer
            self.layer_names["feature_map.conv1"]=new_layer
            self.origin_layer_names["feature_map.conv1"]=new_layer
        elif 'feature_map.conv1.0' == layer_name:
            self.feature_map.conv1[0]= new_layer
            self.layer_names["feature_map.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.conv1.0"]=new_layer
        elif 'feature_map.conv1.1' == layer_name:
            self.feature_map.conv1[1]= new_layer
            self.layer_names["feature_map.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.conv1.1"]=new_layer
        elif 'feature_map.conv1.2' == layer_name:
            self.feature_map.conv1[2]= new_layer
            self.layer_names["feature_map.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.conv1.2"]=new_layer
        elif 'feature_map.conv2' == layer_name:
            self.feature_map.conv2= new_layer
            self.layer_names["feature_map.conv2"]=new_layer
            self.origin_layer_names["feature_map.conv2"]=new_layer
        elif 'feature_map.conv2.0' == layer_name:
            self.feature_map.conv2[0]= new_layer
            self.layer_names["feature_map.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.conv2.0"]=new_layer
        elif 'feature_map.conv2.1' == layer_name:
            self.feature_map.conv2[1]= new_layer
            self.layer_names["feature_map.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.conv2.1"]=new_layer
        elif 'feature_map.conv2.2' == layer_name:
            self.feature_map.conv2[2]= new_layer
            self.layer_names["feature_map.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.conv2.2"]=new_layer
        elif 'feature_map.conv3' == layer_name:
            self.feature_map.conv3= new_layer
            self.layer_names["feature_map.conv3"]=new_layer
            self.origin_layer_names["feature_map.conv3"]=new_layer
        elif 'feature_map.conv3.0' == layer_name:
            self.feature_map.conv3[0]= new_layer
            self.layer_names["feature_map.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.conv3.0"]=new_layer
        elif 'feature_map.conv3.1' == layer_name:
            self.feature_map.conv3[1]= new_layer
            self.layer_names["feature_map.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.conv3.1"]=new_layer
        elif 'feature_map.conv3.2' == layer_name:
            self.feature_map.conv3[2]= new_layer
            self.layer_names["feature_map.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.conv3.2"]=new_layer
        elif 'feature_map.maxpool1' == layer_name:
            self.feature_map.maxpool1= new_layer
            self.layer_names["feature_map.maxpool1"]=new_layer
            self.origin_layer_names["feature_map.maxpool1"]=new_layer
        elif 'feature_map.maxpool2' == layer_name:
            self.feature_map.maxpool2= new_layer
            self.layer_names["feature_map.maxpool2"]=new_layer
            self.origin_layer_names["feature_map.maxpool2"]=new_layer
        elif 'feature_map.maxpool3' == layer_name:
            self.feature_map.maxpool3= new_layer
            self.layer_names["feature_map.maxpool3"]=new_layer
            self.origin_layer_names["feature_map.maxpool3"]=new_layer
        elif 'feature_map.conv4' == layer_name:
            self.feature_map.conv4= new_layer
            self.layer_names["feature_map.conv4"]=new_layer
            self.origin_layer_names["feature_map.conv4"]=new_layer
        elif 'feature_map.conv4.0' == layer_name:
            self.feature_map.conv4[0]= new_layer
            self.layer_names["feature_map.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.conv4.0"]=new_layer
        elif 'feature_map.conv4.1' == layer_name:
            self.feature_map.conv4[1]= new_layer
            self.layer_names["feature_map.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.conv4.1"]=new_layer
        elif 'feature_map.conv4.2' == layer_name:
            self.feature_map.conv4[2]= new_layer
            self.layer_names["feature_map.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.conv4.2"]=new_layer
        elif 'feature_map.conv5' == layer_name:
            self.feature_map.conv5= new_layer
            self.layer_names["feature_map.conv5"]=new_layer
            self.origin_layer_names["feature_map.conv5"]=new_layer
        elif 'feature_map.conv5.0' == layer_name:
            self.feature_map.conv5[0]= new_layer
            self.layer_names["feature_map.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.conv5.0"]=new_layer
        elif 'feature_map.conv5.1' == layer_name:
            self.feature_map.conv5[1]= new_layer
            self.layer_names["feature_map.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.conv5.1"]=new_layer
        elif 'feature_map.conv5.2' == layer_name:
            self.feature_map.conv5[2]= new_layer
            self.layer_names["feature_map.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.conv5.2"]=new_layer
        elif 'feature_map.conv6' == layer_name:
            self.feature_map.conv6= new_layer
            self.layer_names["feature_map.conv6"]=new_layer
            self.origin_layer_names["feature_map.conv6"]=new_layer
        elif 'feature_map.conv6.0' == layer_name:
            self.feature_map.conv6[0]= new_layer
            self.layer_names["feature_map.conv6.0"]=new_layer
            self.origin_layer_names["feature_map.conv6.0"]=new_layer
        elif 'feature_map.conv6.1' == layer_name:
            self.feature_map.conv6[1]= new_layer
            self.layer_names["feature_map.conv6.1"]=new_layer
            self.origin_layer_names["feature_map.conv6.1"]=new_layer
        elif 'feature_map.conv6.2' == layer_name:
            self.feature_map.conv6[2]= new_layer
            self.layer_names["feature_map.conv6.2"]=new_layer
            self.origin_layer_names["feature_map.conv6.2"]=new_layer
        elif 'feature_map.conv7' == layer_name:
            self.feature_map.conv7= new_layer
            self.layer_names["feature_map.conv7"]=new_layer
            self.origin_layer_names["feature_map.conv7"]=new_layer
        elif 'feature_map.conv7.0' == layer_name:
            self.feature_map.conv7[0]= new_layer
            self.layer_names["feature_map.conv7.0"]=new_layer
            self.origin_layer_names["feature_map.conv7.0"]=new_layer
        elif 'feature_map.conv7.1' == layer_name:
            self.feature_map.conv7[1]= new_layer
            self.layer_names["feature_map.conv7.1"]=new_layer
            self.origin_layer_names["feature_map.conv7.1"]=new_layer
        elif 'feature_map.conv7.2' == layer_name:
            self.feature_map.conv7[2]= new_layer
            self.layer_names["feature_map.conv7.2"]=new_layer
            self.origin_layer_names["feature_map.conv7.2"]=new_layer
        elif 'feature_map.conv8' == layer_name:
            self.feature_map.conv8= new_layer
            self.layer_names["feature_map.conv8"]=new_layer
            self.origin_layer_names["feature_map.conv8"]=new_layer
        elif 'feature_map.conv8.0' == layer_name:
            self.feature_map.conv8[0]= new_layer
            self.layer_names["feature_map.conv8.0"]=new_layer
            self.origin_layer_names["feature_map.conv8.0"]=new_layer
        elif 'feature_map.conv8.1' == layer_name:
            self.feature_map.conv8[1]= new_layer
            self.layer_names["feature_map.conv8.1"]=new_layer
            self.origin_layer_names["feature_map.conv8.1"]=new_layer
        elif 'feature_map.conv8.2' == layer_name:
            self.feature_map.conv8[2]= new_layer
            self.layer_names["feature_map.conv8.2"]=new_layer
            self.origin_layer_names["feature_map.conv8.2"]=new_layer
        elif 'feature_map.backblock0' == layer_name:
            self.feature_map.backblock0= new_layer
            self.layer_names["feature_map.backblock0"]=new_layer
            self.origin_layer_names["feature_map.backblock0"]=new_layer
        elif 'feature_map.backblock0.conv0' == layer_name:
            self.feature_map.backblock0.conv0= new_layer
            self.layer_names["feature_map.backblock0.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0"]=new_layer
        elif 'feature_map.backblock0.conv0.0' == layer_name:
            self.feature_map.backblock0.conv0[0]= new_layer
            self.layer_names["feature_map.backblock0.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.0"]=new_layer
        elif 'feature_map.backblock0.conv0.1' == layer_name:
            self.feature_map.backblock0.conv0[1]= new_layer
            self.layer_names["feature_map.backblock0.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.1"]=new_layer
        elif 'feature_map.backblock0.conv0.2' == layer_name:
            self.feature_map.backblock0.conv0[2]= new_layer
            self.layer_names["feature_map.backblock0.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.2"]=new_layer
        elif 'feature_map.backblock0.conv1' == layer_name:
            self.feature_map.backblock0.conv1= new_layer
            self.layer_names["feature_map.backblock0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1"]=new_layer
        elif 'feature_map.backblock0.conv1.0' == layer_name:
            self.feature_map.backblock0.conv1[0]= new_layer
            self.layer_names["feature_map.backblock0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.0"]=new_layer
        elif 'feature_map.backblock0.conv1.1' == layer_name:
            self.feature_map.backblock0.conv1[1]= new_layer
            self.layer_names["feature_map.backblock0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.1"]=new_layer
        elif 'feature_map.backblock0.conv1.2' == layer_name:
            self.feature_map.backblock0.conv1[2]= new_layer
            self.layer_names["feature_map.backblock0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.2"]=new_layer
        elif 'feature_map.backblock0.conv2' == layer_name:
            self.feature_map.backblock0.conv2= new_layer
            self.layer_names["feature_map.backblock0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2"]=new_layer
        elif 'feature_map.backblock0.conv2.0' == layer_name:
            self.feature_map.backblock0.conv2[0]= new_layer
            self.layer_names["feature_map.backblock0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.0"]=new_layer
        elif 'feature_map.backblock0.conv2.1' == layer_name:
            self.feature_map.backblock0.conv2[1]= new_layer
            self.layer_names["feature_map.backblock0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.1"]=new_layer
        elif 'feature_map.backblock0.conv2.2' == layer_name:
            self.feature_map.backblock0.conv2[2]= new_layer
            self.layer_names["feature_map.backblock0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.2"]=new_layer
        elif 'feature_map.backblock0.conv3' == layer_name:
            self.feature_map.backblock0.conv3= new_layer
            self.layer_names["feature_map.backblock0.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3"]=new_layer
        elif 'feature_map.backblock0.conv3.0' == layer_name:
            self.feature_map.backblock0.conv3[0]= new_layer
            self.layer_names["feature_map.backblock0.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.0"]=new_layer
        elif 'feature_map.backblock0.conv3.1' == layer_name:
            self.feature_map.backblock0.conv3[1]= new_layer
            self.layer_names["feature_map.backblock0.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.1"]=new_layer
        elif 'feature_map.backblock0.conv3.2' == layer_name:
            self.feature_map.backblock0.conv3[2]= new_layer
            self.layer_names["feature_map.backblock0.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.2"]=new_layer
        elif 'feature_map.backblock0.conv4' == layer_name:
            self.feature_map.backblock0.conv4= new_layer
            self.layer_names["feature_map.backblock0.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4"]=new_layer
        elif 'feature_map.backblock0.conv4.0' == layer_name:
            self.feature_map.backblock0.conv4[0]= new_layer
            self.layer_names["feature_map.backblock0.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.0"]=new_layer
        elif 'feature_map.backblock0.conv4.1' == layer_name:
            self.feature_map.backblock0.conv4[1]= new_layer
            self.layer_names["feature_map.backblock0.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.1"]=new_layer
        elif 'feature_map.backblock0.conv4.2' == layer_name:
            self.feature_map.backblock0.conv4[2]= new_layer
            self.layer_names["feature_map.backblock0.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.2"]=new_layer
        # elif 'feature_map.backblock0.conv5' == layer_name:
        #     self.feature_map.backblock0.conv5= new_layer
        #     self.layer_names["feature_map.backblock0.conv5"]=new_layer
        #     self.origin_layer_names["feature_map.backblock0.conv5"]=new_layer
        # elif 'feature_map.backblock0.conv5.0' == layer_name:
        #     self.feature_map.backblock0.conv5[0]= new_layer
        #     self.layer_names["feature_map.backblock0.conv5.0"]=new_layer
        #     self.origin_layer_names["feature_map.backblock0.conv5.0"]=new_layer
        # elif 'feature_map.backblock0.conv5.1' == layer_name:
        #     self.feature_map.backblock0.conv5[1]= new_layer
        #     self.layer_names["feature_map.backblock0.conv5.1"]=new_layer
        #     self.origin_layer_names["feature_map.backblock0.conv5.1"]=new_layer
        # elif 'feature_map.backblock0.conv5.2' == layer_name:
        #     self.feature_map.backblock0.conv5[2]= new_layer
        #     self.layer_names["feature_map.backblock0.conv5.2"]=new_layer
        #     self.origin_layer_names["feature_map.backblock0.conv5.2"]=new_layer
        # elif 'feature_map.backblock0.conv6' == layer_name:
        #     self.feature_map.backblock0.conv6= new_layer
        #     self.layer_names["feature_map.backblock0.conv6"]=new_layer
        #     self.origin_layer_names["feature_map.backblock0.conv6"]=new_layer
        elif 'feature_map.conv9' == layer_name:
            self.feature_map.conv9= new_layer
            self.layer_names["feature_map.conv9"]=new_layer
            self.origin_layer_names["feature_map.conv9"]=new_layer
        elif 'feature_map.conv9.0' == layer_name:
            self.feature_map.conv9[0]= new_layer
            self.layer_names["feature_map.conv9.0"]=new_layer
            self.origin_layer_names["feature_map.conv9.0"]=new_layer
        elif 'feature_map.conv9.1' == layer_name:
            self.feature_map.conv9[1]= new_layer
            self.layer_names["feature_map.conv9.1"]=new_layer
            self.origin_layer_names["feature_map.conv9.1"]=new_layer
        elif 'feature_map.conv9.2' == layer_name:
            self.feature_map.conv9[2]= new_layer
            self.layer_names["feature_map.conv9.2"]=new_layer
            self.origin_layer_names["feature_map.conv9.2"]=new_layer
        elif 'feature_map.conv10' == layer_name:
            self.feature_map.conv10= new_layer
            self.layer_names["feature_map.conv10"]=new_layer
            self.origin_layer_names["feature_map.conv10"]=new_layer
        elif 'feature_map.conv10.0' == layer_name:
            self.feature_map.conv10[0]= new_layer
            self.layer_names["feature_map.conv10.0"]=new_layer
            self.origin_layer_names["feature_map.conv10.0"]=new_layer
        elif 'feature_map.conv10.1' == layer_name:
            self.feature_map.conv10[1]= new_layer
            self.layer_names["feature_map.conv10.1"]=new_layer
            self.origin_layer_names["feature_map.conv10.1"]=new_layer
        elif 'feature_map.conv10.2' == layer_name:
            self.feature_map.conv10[2]= new_layer
            self.layer_names["feature_map.conv10.2"]=new_layer
            self.origin_layer_names["feature_map.conv10.2"]=new_layer
        elif 'feature_map.conv11' == layer_name:
            self.feature_map.conv11= new_layer
            self.layer_names["feature_map.conv11"]=new_layer
            self.origin_layer_names["feature_map.conv11"]=new_layer
        elif 'feature_map.conv11.0' == layer_name:
            self.feature_map.conv11[0]= new_layer
            self.layer_names["feature_map.conv11.0"]=new_layer
            self.origin_layer_names["feature_map.conv11.0"]=new_layer
        elif 'feature_map.conv11.1' == layer_name:
            self.feature_map.conv11[1]= new_layer
            self.layer_names["feature_map.conv11.1"]=new_layer
            self.origin_layer_names["feature_map.conv11.1"]=new_layer
        elif 'feature_map.conv11.2' == layer_name:
            self.feature_map.conv11[2]= new_layer
            self.layer_names["feature_map.conv11.2"]=new_layer
            self.origin_layer_names["feature_map.conv11.2"]=new_layer
        elif 'feature_map.conv12' == layer_name:
            self.feature_map.conv12= new_layer
            self.layer_names["feature_map.conv12"]=new_layer
            self.origin_layer_names["feature_map.conv12"]=new_layer
        elif 'feature_map.conv12.0' == layer_name:
            self.feature_map.conv12[0]= new_layer
            self.layer_names["feature_map.conv12.0"]=new_layer
            self.origin_layer_names["feature_map.conv12.0"]=new_layer
        elif 'feature_map.conv12.1' == layer_name:
            self.feature_map.conv12[1]= new_layer
            self.layer_names["feature_map.conv12.1"]=new_layer
            self.origin_layer_names["feature_map.conv12.1"]=new_layer
        elif 'feature_map.conv12.2' == layer_name:
            self.feature_map.conv12[2]= new_layer
            self.layer_names["feature_map.conv12.2"]=new_layer
            self.origin_layer_names["feature_map.conv12.2"]=new_layer
        elif 'feature_map.backblock1' == layer_name:
            self.feature_map.backblock1= new_layer
            self.layer_names["feature_map.backblock1"]=new_layer
            self.origin_layer_names["feature_map.backblock1"]=new_layer
        elif 'feature_map.backblock1.conv0' == layer_name:
            self.feature_map.backblock1.conv0= new_layer
            self.layer_names["feature_map.backblock1.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0"]=new_layer
        elif 'feature_map.backblock1.conv0.0' == layer_name:
            self.feature_map.backblock1.conv0[0]= new_layer
            self.layer_names["feature_map.backblock1.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.0"]=new_layer
        elif 'feature_map.backblock1.conv0.1' == layer_name:
            self.feature_map.backblock1.conv0[1]= new_layer
            self.layer_names["feature_map.backblock1.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.1"]=new_layer
        elif 'feature_map.backblock1.conv0.2' == layer_name:
            self.feature_map.backblock1.conv0[2]= new_layer
            self.layer_names["feature_map.backblock1.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.2"]=new_layer
        elif 'feature_map.backblock1.conv1' == layer_name:
            self.feature_map.backblock1.conv1= new_layer
            self.layer_names["feature_map.backblock1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1"]=new_layer
        elif 'feature_map.backblock1.conv1.0' == layer_name:
            self.feature_map.backblock1.conv1[0]= new_layer
            self.layer_names["feature_map.backblock1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.0"]=new_layer
        elif 'feature_map.backblock1.conv1.1' == layer_name:
            self.feature_map.backblock1.conv1[1]= new_layer
            self.layer_names["feature_map.backblock1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.1"]=new_layer
        elif 'feature_map.backblock1.conv1.2' == layer_name:
            self.feature_map.backblock1.conv1[2]= new_layer
            self.layer_names["feature_map.backblock1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.2"]=new_layer
        elif 'feature_map.backblock1.conv2' == layer_name:
            self.feature_map.backblock1.conv2= new_layer
            self.layer_names["feature_map.backblock1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2"]=new_layer
        elif 'feature_map.backblock1.conv2.0' == layer_name:
            self.feature_map.backblock1.conv2[0]= new_layer
            self.layer_names["feature_map.backblock1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.0"]=new_layer
        elif 'feature_map.backblock1.conv2.1' == layer_name:
            self.feature_map.backblock1.conv2[1]= new_layer
            self.layer_names["feature_map.backblock1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.1"]=new_layer
        elif 'feature_map.backblock1.conv2.2' == layer_name:
            self.feature_map.backblock1.conv2[2]= new_layer
            self.layer_names["feature_map.backblock1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.2"]=new_layer
        elif 'feature_map.backblock1.conv3' == layer_name:
            self.feature_map.backblock1.conv3= new_layer
            self.layer_names["feature_map.backblock1.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3"]=new_layer
        elif 'feature_map.backblock1.conv3.0' == layer_name:
            self.feature_map.backblock1.conv3[0]= new_layer
            self.layer_names["feature_map.backblock1.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.0"]=new_layer
        elif 'feature_map.backblock1.conv3.1' == layer_name:
            self.feature_map.backblock1.conv3[1]= new_layer
            self.layer_names["feature_map.backblock1.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.1"]=new_layer
        elif 'feature_map.backblock1.conv3.2' == layer_name:
            self.feature_map.backblock1.conv3[2]= new_layer
            self.layer_names["feature_map.backblock1.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.2"]=new_layer
        elif 'feature_map.backblock1.conv4' == layer_name:
            self.feature_map.backblock1.conv4= new_layer
            self.layer_names["feature_map.backblock1.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4"]=new_layer
        elif 'feature_map.backblock1.conv4.0' == layer_name:
            self.feature_map.backblock1.conv4[0]= new_layer
            self.layer_names["feature_map.backblock1.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.0"]=new_layer
        elif 'feature_map.backblock1.conv4.1' == layer_name:
            self.feature_map.backblock1.conv4[1]= new_layer
            self.layer_names["feature_map.backblock1.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.1"]=new_layer
        elif 'feature_map.backblock1.conv4.2' == layer_name:
            self.feature_map.backblock1.conv4[2]= new_layer
            self.layer_names["feature_map.backblock1.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.2"]=new_layer
        elif 'feature_map.backblock1.conv5' == layer_name:
            self.feature_map.backblock1.conv5= new_layer
            self.layer_names["feature_map.backblock1.conv5"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5"]=new_layer
        elif 'feature_map.backblock1.conv5.0' == layer_name:
            self.feature_map.backblock1.conv5[0]= new_layer
            self.layer_names["feature_map.backblock1.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.0"]=new_layer
        elif 'feature_map.backblock1.conv5.1' == layer_name:
            self.feature_map.backblock1.conv5[1]= new_layer
            self.layer_names["feature_map.backblock1.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.1"]=new_layer
        elif 'feature_map.backblock1.conv5.2' == layer_name:
            self.feature_map.backblock1.conv5[2]= new_layer
            self.layer_names["feature_map.backblock1.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.2"]=new_layer
        elif 'feature_map.backblock1.conv6' == layer_name:
            self.feature_map.backblock1.conv6= new_layer
            self.layer_names["feature_map.backblock1.conv6"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv6"]=new_layer
        elif 'feature_map.backblock2' == layer_name:
            self.feature_map.backblock2= new_layer
            self.layer_names["feature_map.backblock2"]=new_layer
            self.origin_layer_names["feature_map.backblock2"]=new_layer
        elif 'feature_map.backblock2.conv0' == layer_name:
            self.feature_map.backblock2.conv0= new_layer
            self.layer_names["feature_map.backblock2.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0"]=new_layer
        elif 'feature_map.backblock2.conv0.0' == layer_name:
            self.feature_map.backblock2.conv0[0]= new_layer
            self.layer_names["feature_map.backblock2.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.0"]=new_layer
        elif 'feature_map.backblock2.conv0.1' == layer_name:
            self.feature_map.backblock2.conv0[1]= new_layer
            self.layer_names["feature_map.backblock2.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.1"]=new_layer
        elif 'feature_map.backblock2.conv0.2' == layer_name:
            self.feature_map.backblock2.conv0[2]= new_layer
            self.layer_names["feature_map.backblock2.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.2"]=new_layer
        elif 'feature_map.backblock2.conv1' == layer_name:
            self.feature_map.backblock2.conv1= new_layer
            self.layer_names["feature_map.backblock2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1"]=new_layer
        elif 'feature_map.backblock2.conv1.0' == layer_name:
            self.feature_map.backblock2.conv1[0]= new_layer
            self.layer_names["feature_map.backblock2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.0"]=new_layer
        elif 'feature_map.backblock2.conv1.1' == layer_name:
            self.feature_map.backblock2.conv1[1]= new_layer
            self.layer_names["feature_map.backblock2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.1"]=new_layer
        elif 'feature_map.backblock2.conv1.2' == layer_name:
            self.feature_map.backblock2.conv1[2]= new_layer
            self.layer_names["feature_map.backblock2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.2"]=new_layer
        elif 'feature_map.backblock2.conv2' == layer_name:
            self.feature_map.backblock2.conv2= new_layer
            self.layer_names["feature_map.backblock2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2"]=new_layer
        elif 'feature_map.backblock2.conv2.0' == layer_name:
            self.feature_map.backblock2.conv2[0]= new_layer
            self.layer_names["feature_map.backblock2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.0"]=new_layer
        elif 'feature_map.backblock2.conv2.1' == layer_name:
            self.feature_map.backblock2.conv2[1]= new_layer
            self.layer_names["feature_map.backblock2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.1"]=new_layer
        elif 'feature_map.backblock2.conv2.2' == layer_name:
            self.feature_map.backblock2.conv2[2]= new_layer
            self.layer_names["feature_map.backblock2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.2"]=new_layer
        elif 'feature_map.backblock2.conv3' == layer_name:
            self.feature_map.backblock2.conv3= new_layer
            self.layer_names["feature_map.backblock2.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3"]=new_layer
        elif 'feature_map.backblock2.conv3.0' == layer_name:
            self.feature_map.backblock2.conv3[0]= new_layer
            self.layer_names["feature_map.backblock2.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.0"]=new_layer
        elif 'feature_map.backblock2.conv3.1' == layer_name:
            self.feature_map.backblock2.conv3[1]= new_layer
            self.layer_names["feature_map.backblock2.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.1"]=new_layer
        elif 'feature_map.backblock2.conv3.2' == layer_name:
            self.feature_map.backblock2.conv3[2]= new_layer
            self.layer_names["feature_map.backblock2.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.2"]=new_layer
        elif 'feature_map.backblock2.conv4' == layer_name:
            self.feature_map.backblock2.conv4= new_layer
            self.layer_names["feature_map.backblock2.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4"]=new_layer
        elif 'feature_map.backblock2.conv4.0' == layer_name:
            self.feature_map.backblock2.conv4[0]= new_layer
            self.layer_names["feature_map.backblock2.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.0"]=new_layer
        elif 'feature_map.backblock2.conv4.1' == layer_name:
            self.feature_map.backblock2.conv4[1]= new_layer
            self.layer_names["feature_map.backblock2.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.1"]=new_layer
        elif 'feature_map.backblock2.conv4.2' == layer_name:
            self.feature_map.backblock2.conv4[2]= new_layer
            self.layer_names["feature_map.backblock2.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.2"]=new_layer
        elif 'feature_map.backblock2.conv5' == layer_name:
            self.feature_map.backblock2.conv5= new_layer
            self.layer_names["feature_map.backblock2.conv5"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5"]=new_layer
        elif 'feature_map.backblock2.conv5.0' == layer_name:
            self.feature_map.backblock2.conv5[0]= new_layer
            self.layer_names["feature_map.backblock2.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.0"]=new_layer
        elif 'feature_map.backblock2.conv5.1' == layer_name:
            self.feature_map.backblock2.conv5[1]= new_layer
            self.layer_names["feature_map.backblock2.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.1"]=new_layer
        elif 'feature_map.backblock2.conv5.2' == layer_name:
            self.feature_map.backblock2.conv5[2]= new_layer
            self.layer_names["feature_map.backblock2.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.2"]=new_layer
        elif 'feature_map.backblock2.conv6' == layer_name:
            self.feature_map.backblock2.conv6= new_layer
            self.layer_names["feature_map.backblock2.conv6"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv6"]=new_layer
        elif 'feature_map.backblock3' == layer_name:
            self.feature_map.backblock3= new_layer
            self.layer_names["feature_map.backblock3"]=new_layer
            self.origin_layer_names["feature_map.backblock3"]=new_layer
        elif 'feature_map.backblock3.conv0' == layer_name:
            self.feature_map.backblock3.conv0= new_layer
            self.layer_names["feature_map.backblock3.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv0"]=new_layer
        elif 'feature_map.backblock3.conv0.0' == layer_name:
            self.feature_map.backblock3.conv0[0]= new_layer
            self.layer_names["feature_map.backblock3.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv0.0"]=new_layer
        elif 'feature_map.backblock3.conv0.1' == layer_name:
            self.feature_map.backblock3.conv0[1]= new_layer
            self.layer_names["feature_map.backblock3.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv0.1"]=new_layer
        elif 'feature_map.backblock3.conv0.2' == layer_name:
            self.feature_map.backblock3.conv0[2]= new_layer
            self.layer_names["feature_map.backblock3.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv0.2"]=new_layer
        elif 'feature_map.backblock3.conv1' == layer_name:
            self.feature_map.backblock3.conv1= new_layer
            self.layer_names["feature_map.backblock3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv1"]=new_layer
        elif 'feature_map.backblock3.conv1.0' == layer_name:
            self.feature_map.backblock3.conv1[0]= new_layer
            self.layer_names["feature_map.backblock3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv1.0"]=new_layer
        elif 'feature_map.backblock3.conv1.1' == layer_name:
            self.feature_map.backblock3.conv1[1]= new_layer
            self.layer_names["feature_map.backblock3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv1.1"]=new_layer
        elif 'feature_map.backblock3.conv1.2' == layer_name:
            self.feature_map.backblock3.conv1[2]= new_layer
            self.layer_names["feature_map.backblock3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv1.2"]=new_layer
        elif 'feature_map.backblock3.conv2' == layer_name:
            self.feature_map.backblock3.conv2= new_layer
            self.layer_names["feature_map.backblock3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv2"]=new_layer
        elif 'feature_map.backblock3.conv2.0' == layer_name:
            self.feature_map.backblock3.conv2[0]= new_layer
            self.layer_names["feature_map.backblock3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv2.0"]=new_layer
        elif 'feature_map.backblock3.conv2.1' == layer_name:
            self.feature_map.backblock3.conv2[1]= new_layer
            self.layer_names["feature_map.backblock3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv2.1"]=new_layer
        elif 'feature_map.backblock3.conv2.2' == layer_name:
            self.feature_map.backblock3.conv2[2]= new_layer
            self.layer_names["feature_map.backblock3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv2.2"]=new_layer
        elif 'feature_map.backblock3.conv3' == layer_name:
            self.feature_map.backblock3.conv3= new_layer
            self.layer_names["feature_map.backblock3.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv3"]=new_layer
        elif 'feature_map.backblock3.conv3.0' == layer_name:
            self.feature_map.backblock3.conv3[0]= new_layer
            self.layer_names["feature_map.backblock3.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv3.0"]=new_layer
        elif 'feature_map.backblock3.conv3.1' == layer_name:
            self.feature_map.backblock3.conv3[1]= new_layer
            self.layer_names["feature_map.backblock3.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv3.1"]=new_layer
        elif 'feature_map.backblock3.conv3.2' == layer_name:
            self.feature_map.backblock3.conv3[2]= new_layer
            self.layer_names["feature_map.backblock3.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv3.2"]=new_layer
        elif 'feature_map.backblock3.conv4' == layer_name:
            self.feature_map.backblock3.conv4= new_layer
            self.layer_names["feature_map.backblock3.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv4"]=new_layer
        elif 'feature_map.backblock3.conv4.0' == layer_name:
            self.feature_map.backblock3.conv4[0]= new_layer
            self.layer_names["feature_map.backblock3.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv4.0"]=new_layer
        elif 'feature_map.backblock3.conv4.1' == layer_name:
            self.feature_map.backblock3.conv4[1]= new_layer
            self.layer_names["feature_map.backblock3.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv4.1"]=new_layer
        elif 'feature_map.backblock3.conv4.2' == layer_name:
            self.feature_map.backblock3.conv4[2]= new_layer
            self.layer_names["feature_map.backblock3.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv4.2"]=new_layer
        elif 'feature_map.backblock3.conv5' == layer_name:
            self.feature_map.backblock3.conv5= new_layer
            self.layer_names["feature_map.backblock3.conv5"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv5"]=new_layer
        elif 'feature_map.backblock3.conv5.0' == layer_name:
            self.feature_map.backblock3.conv5[0]= new_layer
            self.layer_names["feature_map.backblock3.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv5.0"]=new_layer
        elif 'feature_map.backblock3.conv5.1' == layer_name:
            self.feature_map.backblock3.conv5[1]= new_layer
            self.layer_names["feature_map.backblock3.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv5.1"]=new_layer
        elif 'feature_map.backblock3.conv5.2' == layer_name:
            self.feature_map.backblock3.conv5[2]= new_layer
            self.layer_names["feature_map.backblock3.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv5.2"]=new_layer
        elif 'feature_map.backblock3.conv6' == layer_name:
            self.feature_map.backblock3.conv6= new_layer
            self.layer_names["feature_map.backblock3.conv6"]=new_layer
            self.origin_layer_names["feature_map.backblock3.conv6"]=new_layer
        elif 'detect_1' == layer_name:
            self.detect_1= new_layer
            self.layer_names["detect_1"]=new_layer
            self.origin_layer_names["detect_1"]=new_layer
        elif 'detect_1.sigmoid' == layer_name:
            self.detect_1.sigmoid= new_layer
            self.layer_names["detect_1.sigmoid"]=new_layer
            self.origin_layer_names["detect_1.sigmoid"]=new_layer
        elif 'detect_2' == layer_name:
            self.detect_2= new_layer
            self.layer_names["detect_2"]=new_layer
            self.origin_layer_names["detect_2"]=new_layer
        elif 'detect_2.sigmoid' == layer_name:
            self.detect_2.sigmoid= new_layer
            self.layer_names["detect_2.sigmoid"]=new_layer
            self.origin_layer_names["detect_2.sigmoid"]=new_layer
        elif 'detect_3' == layer_name:
            self.detect_3= new_layer
            self.layer_names["detect_3"]=new_layer
            self.origin_layer_names["detect_3"]=new_layer
        elif 'detect_3.sigmoid' == layer_name:
            self.detect_3.sigmoid= new_layer
            self.layer_names["detect_3.sigmoid"]=new_layer
            self.origin_layer_names["detect_3.sigmoid"]=new_layer





    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name,order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name]=order

    def get_outshape(self, layer_name):

        if layer_name not in self.out_shapes.keys():
            return False

        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name,out):

        if layer_name not in self.out_shapes.keys():
            return False

        self.out_shapes[layer_name]=out

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name,out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name]=out

    def set_Basic_OPS(self,b):
        self.Basic_OPS=b
    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self,c):
        self.Cascade_OPs=c





    def construct(self, x):
        # print("================================")
        # print(x.shape)
        # print("================================")
        # print("=*===============================")
        # print(input_shape)
        # print("=*===============================")

        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        if not self.keep_detect:
            return big_object_output, medium_object_output, small_object_output
        output_big = self.detect_1(big_object_output, self.input_shape)
        output_me = self.detect_2(medium_object_output, self.input_shape)
        output_small = self.detect_3(small_object_output, self.input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


class Iou(nn.Cell):
    """Calculate the iou of boxes"""

    def __init__(self):
        super(Iou, self).__init__()
        self.min = P.Minimum()
        self.max = P.Maximum()

    def construct(self, box1, box2):
        """
        box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        convert to topLeft and rightDown
        """
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / F.scalar_to_tensor(2.0)  # topLeft
        box1_maxs = box1_xy + box1_wh / F.scalar_to_tensor(2.0)  # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / F.scalar_to_tensor(2.0)
        box2_maxs = box2_xy + box2_wh / F.scalar_to_tensor(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, F.scalar_to_tensor(0.0))
        # P.squeeze: for effiecient slice
        intersect_area = P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = P.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = P.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


class XYLoss(nn.Cell):
    """Loss for x and y."""

    def __init__(self):
        super(XYLoss, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_xy, true_xy):
        xy_loss = object_mask * box_loss_scale * self.cross_entropy(predict_xy, true_xy)
        xy_loss = self.reduce_sum(xy_loss, ())
        return xy_loss


class WHLoss(nn.Cell):
    """Loss for w and h."""

    def __init__(self):
        super(WHLoss, self).__init__()
        self.square = P.Square()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_wh, true_wh):
        wh_loss = object_mask * box_loss_scale * 0.5 * P.Square()(true_wh - predict_wh)
        wh_loss = self.reduce_sum(wh_loss, ())
        return wh_loss


class ConfidenceLoss(nn.Cell):
    """Loss for confidence."""

    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, predict_confidence, ignore_mask):
        confidence_loss = self.cross_entropy(predict_confidence, object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        confidence_loss = self.reduce_sum(confidence_loss, ())
        return confidence_loss


class ClassLoss(nn.Cell):
    """Loss for classification."""

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, predict_class, class_probs):
        class_loss = object_mask * self.cross_entropy(predict_class, class_probs)
        class_loss = self.reduce_sum(class_loss, ())
        return class_loss


class Giou(nn.Cell):
    """Calculating giou"""

    def __init__(self):
        super(Giou, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(self.cast(res_mid0, ms.float32), self.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = C.clip_by_value(giou, -1.0, 1.0)
        return giou


def xywh2x1y1x2y2(box_xywh):
    boxes_x1 = box_xywh[..., 0:1] - box_xywh[..., 2:3] / 2
    boxes_y1 = box_xywh[..., 1:2] - box_xywh[..., 3:4] / 2
    boxes_x2 = box_xywh[..., 0:1] + box_xywh[..., 2:3] / 2
    boxes_y2 = box_xywh[..., 1:2] + box_xywh[..., 3:4] / 2
    boxes_x1y1x2y2 = P.Concat(-1)((boxes_x1, boxes_y1, boxes_x2, boxes_y2))
    return boxes_x1y1x2y2


class YoloLossBlock_ms(nn.Cell):
    """
    Loss block cell of YOLOV4 network.
    """

    def __init__(self, scale, config=default_config):
        super(YoloLossBlock_ms, self).__init__()
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
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = P.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidence_loss = ConfidenceLoss()
        self.class_loss = ClassLoss()

        self.reduce_sum = P.ReduceSum()
        self.giou = Giou()
        self.bbox_class_loss_coff = self.config.bbox_class_loss_coff
        self.ciou_loss_me_coff = int(self.bbox_class_loss_coff[0])
        self.confidence_loss_coff = int(self.bbox_class_loss_coff[1])
        self.class_loss_coff = int(self.bbox_class_loss_coff[2])

    def construct(self, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        """
        prediction : origin output from yolo
        pred_xy: (sigmoid(xy)+grid)/grid_size
        pred_wh: (exp(wh)*anchors)/input_shape
        y_true : after normalize
        gt_box: [batch, maxboxes, xyhw] after normalize
        """
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]
        true_boxes = y_true[:, :, :, :, :4]

        grid_shape = P.Shape()(prediction)[1:3]
        grid_shape = P.Cast()(F.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = P.Select()(P.Equal()(true_wh, 0.0),
                             P.Fill()(P.DType()(true_wh),
                                      P.Shape()(true_wh), 1.0),
                             true_wh)
        true_wh = P.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = P.Shape()(gt_box)
        gt_box = P.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(P.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = P.Cast()(ignore_mask, ms.float32)
        ignore_mask = P.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = F.stop_gradient(ignore_mask)

        confidence_loss = self.confidence_loss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.class_loss(object_mask, prediction[:, :, :, :, 5:], class_probs)

        object_mask_me = P.Reshape()(object_mask, (-1, 1))  # [8, 72, 72, 3, 1]
        box_loss_scale_me = P.Reshape()(box_loss_scale, (-1, 1))
        pred_boxes_me = xywh2x1y1x2y2(pred_boxes)
        pred_boxes_me = P.Reshape()(pred_boxes_me, (-1, 4))
        true_boxes_me = xywh2x1y1x2y2(true_boxes)
        true_boxes_me = P.Reshape()(true_boxes_me, (-1, 4))
        ciou = self.giou(pred_boxes_me, true_boxes_me)
        ciou_loss = object_mask_me * box_loss_scale_me * (1 - ciou)
        ciou_loss_me = self.reduce_sum(ciou_loss, ())
        loss = ciou_loss_me * self.ciou_loss_me_coff + confidence_loss * \
               self.confidence_loss_coff + class_loss * self.class_loss_coff
        batch_size = P.Shape()(prediction)[0]
        return loss / batch_size


def loser(x, yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    input_shape = ops.shape(x)[2:4]
    input_shape = ops.cast(ops.TupleToArray()(input_shape), ms.float32)
    loss_l = YoloLossBlock_ms('l', config)(*yolo_out[0], y_true_0, gt_0, input_shape)
    loss_m = YoloLossBlock_ms('m', config)(*yolo_out[1], y_true_1, gt_1, input_shape)
    loss_s = YoloLossBlock_ms('s', config)(*yolo_out[2], y_true_2, gt_2, input_shape)
    return loss_l + loss_m + loss_s


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args_detection):
        self.eval_ignore_threshold = args_detection.eval_ignore_threshold
        self.labels = config.labels
        self.num_classes = len(self.labels)
        self.results = {}
        self.file_path = ''
        self.save_prefix = args_detection.outputs_dir
        self.ann_file = args_detection.ann_val_file
        self._coco = COCO(self.ann_file)
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.det_boxes = []
        self.nms_thresh = args_detection.nms_thresh
        self.coco_catids = self._coco.getCatIds()
        self.multi_label = config.multi_label
        self.multi_label_thresh = config.multi_label_thresh

    def do_nms_for_results(self):
        """Get result boxes."""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._diou_nms(dets, thresh=0.6)

                keep_box = [{'image_id': int(img_id),
                             'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)}
                            for i in keep_index]
                self.det_boxes.extend(keep_box)
        return self.det_boxes

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _diou_nms(self, dets, thresh=0.5):
        """
        convert xywh -> xmin ymin xmax ymax
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self, result):
        """Save result to file."""
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(result, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def get_eval_result(self):
        """Get eval result."""
        if not self.results:
            logger.warning("[WARNING] result is None.")
            return 0.0, 0.0
        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        res_map = coco_eval.stats[0]
        sys.stdout = stdout
        return rdct.content, float(res_map)

    def detect(self, outputs, batch, image_shape, image_id):
        """Detect boxes."""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                if not self.multi_label:
                    conf = conf.reshape(-1)
                    cls_argmax = cls_argmax.reshape(-1)

                    # create all False
                    flag = np.random.random(cls_emb.shape) > sys.maxsize
                    for i in range(flag.shape[0]):
                        c = cls_argmax[i]
                        flag[i, c] = True
                    confidence = cls_emb[flag] * conf
                    for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence,
                                                                     cls_argmax):
                        if confi < self.eval_ignore_threshold:
                            continue
                        if img_id not in self.results:
                            self.results[img_id] = defaultdict(list)
                        x_lefti = max(0, x_lefti)
                        y_lefti = max(0, y_lefti)
                        wi = min(wi, ori_w)
                        hi = min(hi, ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_catids[clsi]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
                else:
                    conf = conf.reshape(-1, 1)
                    # create all False
                    confidence = cls_emb * conf
                    flag = cls_emb > self.multi_label_thresh
                    flag = flag.nonzero()
                    for index in range(len(flag[0])):
                        i = flag[0][index]
                        j = flag[1][index]
                        confi = confidence[i][j]
                        if confi < self.eval_ignore_threshold:
                            continue
                        if img_id not in self.results:
                            self.results[img_id] = defaultdict(list)
                        x_lefti = max(0, x_top_left[i])
                        y_lefti = max(0, y_top_left[i])
                        wi = min(w[i], ori_w)
                        hi = min(h[i], ori_h)
                        clsi = j
                        # transform catId to match coco
                        coco_clsi = self.coco_catids[clsi]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])


def view_result(args, result, score_threshold=None, recommend_threshold=False):
    from src.coco_visual import CocoVisualUtil
    dataset_coco = COCO(args.ann_val_file)
    coco_visual = CocoVisualUtil()
    eval_types = ["bbox"]
    config.dataset = "coco"
    im_path_dir = config.data_root
    result_files = coco_visual.results2json(dataset_coco, result, "./results.pkl")
    coco_visual.coco_eval(config, result_files, eval_types, dataset_coco, im_path_dir=im_path_dir,
                          score_threshold=score_threshold, recommend_threshold=recommend_threshold)


def apply_eval(eval_param_dict):
    network = eval_param_dict["net"]
    network.set_train(False)
    ds = eval_param_dict["dataset"]
    data_size = eval_param_dict["data_size"]
    args = eval_param_dict["args"]
    detection = DetectionEngine(args)
    for index, data in enumerate(ds.create_dict_iterator(num_epochs=1)):
        image = data["image"]
        image_shape_ = data["image_shape"]
        image_id_ = data["img_id"]
        prediction = network(image)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id_ = image_id_.asnumpy()
        image_shape_ = image_shape_.asnumpy()

        detection.detect([output_small, output_me, output_big], args.per_batch_size, image_shape_, image_id_)
        if index % 100 == 0:
            args.logger.info('Processing... {:.2f}% '.format(index * args.per_batch_size / data_size * 100))

    args.logger.info('Calculating mAP...')
    result = detection.do_nms_for_results()
    result_file_path = detection.write_result(result)
    args.logger.info('result file path: {}'.format(result_file_path))
    eval_result = detection.get_eval_result()
    # view_result(args, result, score_threshold=None, recommend_threshold=config.recommend_threshold)
    print("View eval result completed!", flush=True)
    return eval_result


class COCOYoloDataset:
    """YOLOV4 Dataset for COCO."""

    def __init__(self, root, ann_file, remove_images_without_annotations=True,
                 filter_crowd_anno=True, is_training=True):
        self.coco = COCO(ann_file)
        self.root = root
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.filter_crowd_anno = filter_crowd_anno
        self.is_training = is_training
        self.mosaic = config.mosaic

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

    def _mosaic_preprocess(self, index):
        labels4 = []
        s = 384
        self.mosaic_border = [-s // 2, -s // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        indices = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        for i, img_ids_index in enumerate(indices):
            coco = self.coco
            img_id = self.img_ids[img_ids_index]
            img_path = coco.loadImgs(img_id)[0]["file_name"]
            img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
            img = np.array(img)
            h, w = img.shape[:2]

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 128, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b

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
                out_target.append(tmp)  # 这里out_target是label的实际宽高，对应于图片中的实际度量

            labels = out_target.copy()
            labels = np.array(labels)
            out_target = np.array(out_target)

            labels[:, 0] = out_target[:, 0] + padw
            labels[:, 1] = out_target[:, 1] + padh
            labels[:, 2] = out_target[:, 2] + padw
            labels[:, 3] = out_target[:, 3] + padh
            labels4.append(labels)

        if labels4:
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, :4], 0, 2 * s, out=labels4[:, :4])  # use with random_perspective
        return img4, labels4, [], [], [], [], [], []

    def _convetTopDown(self, bbox):
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]
        return [x_min, y_min, x_min + w, y_min + h]

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

        if self.mosaic and random.random() < 0.5:
            return self._mosaic_preprocess(index)

        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
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
            bbox = self._conve_top_down(bbox)
            tmp.extend(bbox)
            tmp.append(int(label))
            # tmp [x_min y_min x_max y_max, label]
            out_target.append(tmp)
        return img, out_target, [], [], [], [], [], []

    def __len__(self):
        return len(self.img_ids)

    def _conve_top_down(self, bbox):
        x_min = bbox[0]
        y_min = bbox[1]
        w = bbox[2]
        h = bbox[3]
        return [x_min, y_min, x_min + w, y_min + h]


def create_yolo_dataset(image_dir, anno_path, batch_size, max_epoch, device_num, rank,
                        default_config=None, is_training=True, shuffle=True):
    """Create dataset for YOLOV4."""
    cv2.setNumThreads(0)

    if is_training:
        filter_crowd = True
        remove_empty_anno = True
    else:
        filter_crowd = False
        remove_empty_anno = False

    yolo_dataset = COCOYoloDataset(root=image_dir, ann_file=anno_path, filter_crowd_anno=filter_crowd,
                                   remove_images_without_annotations=remove_empty_anno, is_training=is_training)
    distributed_sampler = DistributedSampler(len(yolo_dataset), device_num, rank, shuffle=shuffle)
    hwc_to_chw = CV.HWC2CHW()

    default_config.dataset_size = len(yolo_dataset)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)
    if is_training:
        each_multiscale = default_config.each_multiscale
        multi_scale_trans = MultiScaleTrans(default_config, device_num, each_multiscale)
        dataset_column_names = ["image", "annotation", "bbox1", "bbox2", "bbox3",
                                "gt_box1", "gt_box2", "gt_box3"]
        if device_num != 8:
            ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names,
                                     num_parallel_workers=min(32, num_parallel_workers),
                                     sampler=distributed_sampler)
            ds = ds.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                          num_parallel_workers=min(32, num_parallel_workers), drop_remainder=True)
        else:
            ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names, sampler=distributed_sampler)
            ds = ds.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                          num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    else:
        ds = de.GeneratorDataset(yolo_dataset, column_names=["image", "img_id"],
                                 sampler=distributed_sampler)
        compose_map_func = (lambda image, img_id: reshape_fn(image, img_id, default_config))
        ds = ds.map(operations=compose_map_func, input_columns=["image", "img_id"],
                    output_columns=["image", "image_shape", "img_id"],
                    num_parallel_workers=8)
        ds = ds.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=8)
        ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(max_epoch)

    return ds, len(yolo_dataset)

class yolov4loss_ms(mindspore.nn.Cell):
    def __init__(self):
        super(yolov4loss_ms, self).__init__()
        self.config=default_config
        self.loss_big = YoloLossBlock_ms('l', self.config)
        self.loss_me = YoloLossBlock_ms('m', self.config)
        self.loss_small = YoloLossBlock_ms('s', self.config)

    def construct(self, yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, shape):

            loss_coff = [1, 1, 1]
            loss_l_coff = int(loss_coff[0])
            loss_m_coff = int(loss_coff[1])
            loss_s_coff = int(loss_coff[2])

            loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, shape)
            loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, shape)
            loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, shape)
            return loss_l * loss_l_coff + loss_m * loss_m_coff + loss_s * loss_s_coff


def set_eval(network):
    network.yolo_network.detect_1.conf_training = False
    network.yolo_network.detect_2.conf_training = False
    network.yolo_network.detect_3.conf_training = False
    return network


def settrain(network):
    network.detect_1.conf_training = True
    network.detect_2.conf_training = True
    network.detect_3.conf_training = True
    return network


if __name__ == '__main__':
    degree = 1
    eval_flag = False

    cfg = config
    config.steps_per_epoch = 100
    lr = get_lr(config)
    old_progress = -1
    t_end = time.time()


    images = mindspore.Tensor(numpy.random.randn(1, 3,416,416),dtype=mindspore.float32)
    y_true_0 = mindspore.Tensor(numpy.random.randn(1, 13, 13, 3, 85), dtype=mindspore.float32)
    y_true_1 = mindspore.Tensor(numpy.random.randn(1, 26, 26, 3, 85), dtype=mindspore.float32)
    y_true_2 = mindspore.Tensor(numpy.random.randn(1, 52, 52, 3, 85), dtype=mindspore.float32)
    gt_0 = mindspore.Tensor(numpy.random.randn(1, 90, 4), dtype=mindspore.float32)
    gt_1 = mindspore.Tensor(numpy.random.randn(1, 90, 4), dtype=mindspore.float32)
    gt_2 = mindspore.Tensor(numpy.random.randn(1, 90, 4), dtype=mindspore.float32)
    in_shape = images.shape[2:4]
    in_shape = mindspore.Tensor(tuple(in_shape), dtype=mindspore.float32)

    network = YOLOV4CspDarkNet53_ms()
    network.set_train(True)

    opt = Momentum(params=network.trainable_params(),learning_rate=lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay,loss_scale=cfg.loss_scale)


    yolo_network_out =network(images)

    # loss_fun = yolov4loss_ms
    # loss = loss_fun(yolo_network_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, in_shape)
    #
    # def forward_fn( yolo_network_out, y_true_0,y_true_1,y_true_2,gt_0,gt_1,gt_2,input_shape):
    #
    #     loss =loss_fun(yolo_network_out,y_true_0,y_true_1,y_true_2,gt_0,gt_1,gt_2,input_shape)
    #
    #     return loss
    #
    # grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)
    #
    # def train_step( yolo_network_out, y_true_0,y_true_1,y_true_2,gt_0,gt_1,gt_2,input_shape):
    #     (loss), grads = grad_fn( yolo_network_out,  y_true_0,y_true_1,y_true_2,gt_0,gt_1,gt_2,input_shape)
    #     loss = mindspore.ops.depend(loss, opt(grads))
    #     return loss
    #
    # print("================================================================")
    # yolo_network_out = network(images)
    # loss_ms = train_step( yolo_network_out,y_true_0,y_true_1,y_true_2,gt_0,gt_1,gt_2,in_shape)
    # print("loss_ms", loss_ms)
    # print("================================================================")



    #     if (i + 1) % config.log_interval == 0:
    #         time_used = time.time() - t_end
    #         epoch = int((i + 1) / config.steps_per_epoch)
    #         fps = config.per_batch_size * (i - old_progress) * config.group_size / time_used
    #         if config.rank == 0:
    #             print('epoch[{}], iter[{}], per step time: {:.2f} ms, fps: {:.2f}, lr:{}'.format(
    #                 epoch, i, 1000 * time_used / (i - old_progress), fps, lr))
    #         t_end = time.time()
    #         old_progress = i
    #     if config.need_profiler and profiler is not None:
    #         if i == 10:
    #             profiler.analyse()
    #             break
    #     def inference(network):
    #         print('Start inference....')
    #         start_time = time.time()
    #         network = set_eval(network)
    #         network.set_train(False)
    #         config.outputs_dir = os.path.join(config.log_path)
    #         detection = DetectionEngine(config)
    #         for i, data in enumerate(testdata):
    #             image = data["image"]
    #             image_shape = data["image_shape"]
    #             image_id = data["img_id"]
    #             output_big, output_me, output_small = network(image)
    #             output_big = output_big.asnumpy()
    #             output_me = output_me.asnumpy()
    #             output_small = output_small.asnumpy()
    #             image_id = image_id.asnumpy()
    #             image_shape = image_shape.asnumpy()
    #             detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape, image_id)
    #             if i % 50 == 0:
    #                 print('Processing... {:.2f}% '.format(i / data_size * 100))
    #                 break
    #         print('Calculating mAP...')
    #         result = detection.do_nms_for_results()
    #         result_file_path = detection.write_result(result)
    #         print('result file path: %s', result_file_path)
    #         config.recommend_threshold = False
    #         eval_param_dict = {"net": network, "dataset": ds, "data_size": data_size,
    #                            "anno_json": config.
    #                            ann_val_file, "args": config}
    #         eval_result, _ = apply_eval(eval_param_dict)
    #         cost_time = time.time() - start_time
    #         eval_log_string = '\n=============coco eval reulst=========\n' + eval_result
    #         print(eval_log_string)
    #         print('testing cost time %.2f h', cost_time / 3600.)
    #     if eval_flag:
    #         inference(network)
    # print('==========end training===============')
