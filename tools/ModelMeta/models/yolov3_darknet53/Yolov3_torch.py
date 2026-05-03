import os
import mindspore.dataset as ds
from PIL import Image
from pycocotools.coco import COCO
import cv2
import torch.nn as nn
from torch import optim
import time
import torch
from network.cv.yolov3_darknet53.model_utils.config import config as default_config, get_config
from network.cv.yolov3_darknet53.src.transforms import reshape_fn, MultiScaleTrans
from network.cv.yolov3_darknet53.src.lr_scheduler import get_lr
from network.cv.yolov3_darknet53.src.util import DetectionEngine
import torch.nn.functional as F
from network.cv.yolov3_darknet53.model_utils.config import config

import yaml

if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
    final_device = 'cuda:6'
else:
    final_device = 'cpu'

class YOLOV3DarkNet53(nn.Module):
    # 初始化需要传入is_training（bool）和config
    def __init__(self, is_training, config=default_config):
        super(YOLOV3DarkNet53, self).__init__()
        self.config = config
        self.keep_detect = self.config.keep_detect

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
            "INPUT": [-1, 3, 416, 416],
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



    def forward(self, x):
        input_shape = x.shape[2:4]
        input_shape = torch.tensor(input_shape).float().to(final_device)
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        if not self.keep_detect:
            return big_object_output.to(final_device), medium_object_output.to(final_device), small_object_output.to(final_device)
        output_big = self.detect_1(big_object_output.to(final_device), input_shape)
        output_me = self.detect_2(medium_object_output.to(final_device), input_shape)
        output_small = self.detect_3(small_object_output.to(final_device), input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


class ResidualBlock(nn.Module):
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
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.add(out, identity)

        return out


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1,
               padding=1):
    pad_mode = 'zeros'

    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  padding_mode=pad_mode),
        nn.BatchNorm2d(out_channels, momentum=0.9),
        nn.ReLU()
    )


class DarkNet(nn.Module):
    """
    DarkNet V1 network.

    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        detect: Bool. Whether detect or not. Default:False.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3,f4,f5).

    Examples:
        DarkNet(ResidualBlock,
               [1, 2, 8, 8, 4],
               [32, 64, 128, 256, 512],
               [64, 128, 256, 512, 1024],
               100)
    """

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
        """
        Make Layer for DarkNet.

        :param block: Cell. DarkNet block.
        :param layer_num: Integer. Layer number.
        :param in_channel: Integer. Input channel.
        :param out_channel: Integer. Output channel.

        Examples:
            _make_layer(ConvBlock, 1, 128, 256)
        """
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.Sequential(*layers)

    def forward(self, x):
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


def darknet53():
    """
    Get DarkNet53 neural network.

    Returns:
        Cell, cell instance of DarkNet53 neural network.

    Examples:
        darknet53()
    """
    return DarkNet(ResidualBlock, [1, 2, 8, 8, 4],
                   [32, 64, 128, 256, 512],
                   [64, 128, 256, 512, 1024])


# 更新完了
class DetectionBlock(nn.Module):
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
        self.anchors = torch.Tensor([self.config.anchor_scales[i] for i in idx]).to(final_device)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.conf_training = is_training

    def forward(self, x, input_shape):
        num_batch =x.shape[0]
        grid_size =x.shape[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = torch.reshape(x, (num_batch, self.num_anchors_per_scale, self.num_attrib,
                                       grid_size[0], grid_size[1]))
        prediction = prediction.permute((0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = torch.tensor(range_x).float().to(final_device)

        grid_y = torch.tensor(range_y).float().to(final_device)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = torch.tile(torch.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = torch.tile(torch.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = torch.cat((grid_x, grid_y), dim=-1)

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]

        temp = torch.tensor((grid_size[1], grid_size[0])).to(final_device)
        box_xy = (self.sigmoid(box_xy) + grid.to(final_device)) / temp
        # box_wh is w->h

        box_wh = torch.exp(box_wh) * self.anchors / input_shape

        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)


        if self.conf_training:
            return prediction, box_xy, box_wh#grid,

        return torch.cat((box_xy, box_wh, box_confidence, box_probs), dim=-1)


# 更新完了
class YOLOv3(nn.Module):
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

    def forward(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        # feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        # feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        # img_hight = ops.Shape()(x)[2]
        # img_width = ops.Shape()(x)[3]
        img_hight = x.shape[2]
        img_width = x.shape[3]
        feature_map1, feature_map2, feature_map3 = self.backbone(x)
        con1, big_object_output = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        ups1 = nn.Upsample(size=(img_hight // 16, img_width // 16), mode='nearest')(con1)

        con1 = torch.cat([ups1, feature_map2], dim=1)

        con2, medium_object_output = self.backblock1(con1)

        con2 = self.conv2(con2)

        ups2 = nn.Upsample(size=(img_hight // 8, img_width // 8), mode='nearest')(con2)
        con3 = torch.cat([ups2, feature_map1], dim=1)
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


# 更新完了
class YoloBlock(nn.Module):

    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3, padding=1)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3, padding=1)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3, padding=1)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


# 更新完了
def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.9,
                  eps=1e-5,
                  pad_mode="zeros"):
    return nn.Sequential(nn.Conv2d(in_channel,
                                   out_channel,
                                   kernel_size=ksize,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   padding_mode=pad_mode),
                         # momentum的转换是1-mindspore.momentum的关系
                         nn.BatchNorm2d(out_channel, momentum=1 - momentum, eps=eps),
                         nn.LeakyReLU(alpha)
                         )


# 创建数据集
def create_yolo_dataset(image_dir, anno_path, batch_size, device_num, rank,
                        config=None, is_training=True, shuffle=True):
    """Create dataset for YOLOV3."""
    cv2. \
        setNumThreads(0)

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


def loser(x, yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
    input_shape = x.shape[2:4]
    input_shape = torch.tensor(input_shape)
    loss_l = YoloLossBlock('l', config)(*yolo_out[0], y_true_0, gt_0, input_shape)
    loss_m = YoloLossBlock('m', config)(*yolo_out[1], y_true_1, gt_1, input_shape)
    loss_s = YoloLossBlock('s', config)(*yolo_out[2], y_true_2, gt_2, input_shape)
    return loss_l + loss_m + loss_s


class XYLoss(nn.Module):
    """Loss for x and y."""

    def __init__(self):
        super(XYLoss, self).__init__()

    def forward(self, object_mask, box_loss_scale, predict_xy, true_xy):
        prob = F.sigmoid(predict_xy)
        # weights = torch.tensor([1])
        loss = F.cross_entropy(prob, true_xy)

        xy_loss = object_mask * box_loss_scale * loss
        xy_loss = reduce_sum(xy_loss, ())
        return xy_loss


class ConfidenceLoss(nn.Module):
    """Loss for confidence."""

    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def forward(self, object_mask, predict_confidence, ignore_mask):
        prob = nn.functional.sigmoid(predict_confidence)
        # weights = torch.tensor([1])
        loss = F.cross_entropy(prob, object_mask)
        # confidence_loss = self.cross_entropy(predict_confidence, object_mask)
        confidence_loss = object_mask * loss + (1 - object_mask) * loss * ignore_mask
        confidence_loss = reduce_sum(confidence_loss, ())
        return confidence_loss


class ClassLoss(nn.Module):
    """Loss for classification."""

    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, object_mask, predict_class, class_probs):
        prob = nn.functional.sigmoid(predict_class)
        # weights = torch.tensor([1])
        loss = F.cross_entropy(prob, class_probs)
        class_loss = object_mask * loss
        class_loss = reduce_sum(class_loss, ())
        return class_loss


class WHLoss(nn.Module):
    """Loss for w and h."""

    def __init__(self):
        super(WHLoss, self).__init__()

    def forward(self, object_mask, box_loss_scale, predict_wh, true_wh):
        wh_loss = object_mask * box_loss_scale * 0.5 * torch.square(true_wh - predict_wh)
        wh_loss = reduce_sum(wh_loss, ())
        return wh_loss


def reduce_max(input_x, axis, keepdims):
    value = input_x.cpu().detach().numpy()
    value = np.max(value, axis, keepdims=keepdims)
    value = np.array(value)
    value = torch.tensor(value)
    return value


def reduce_sum(input_x, axis, skip_model=False, keep_dims=False):
    value = None
    if input_x is not None and axis is not None:
        value = input_x.cpu().detach().numpy()
        if isinstance(axis, int):
            pass
        elif axis:
            axis = tuple(set(axis))
        elif axis in ((), []) and skip_model:
            return input_x
        else:
            axis = tuple(range(len(value.shape)))
        value = np.sum(value, axis, keepdims=keep_dims)
        value = np.array(value)
        value = torch.tensor(value)
    return value


# def reduce_min(input_x, axis, keepdims=False):
#     value = input_x.cpu().detach().numpy()
#     value = np.min(value, axis, keepdims=keepdims)
#     value = np.array(value)
#     value = torch.tensor(value)
#     return value


class YoloLossBlock(nn.Module):
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
        self.anchors = torch.tensor([self.config.anchor_scales[i] for i in idx])
        self.ignore_threshold = torch.tensor(self.config.ignore_threshold)
        self.concat = torch.cat
        self.iou = Iou()
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidenceLoss = ConfidenceLoss()
        self.classLoss = ClassLoss()

    def forward(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]

        grid_shape = prediction.shape[1:3]
        grid_shape = torch.tensor((grid_shape[::-1]))
        # print("================================")
        # print(pred_xy.shape)
        # print(pred_wh.shape)
        # print("================================")
        pred_boxes = self.concat((pred_xy, pred_wh), dim=-1)
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        if torch.equal(true_wh, torch.tensor(0.0)):
            true_wh = torch.Tensor.fill_(type(true_wh), true_wh.shape)
        else:
            true_wh = true_wh

        # true_wh = ops.Select()(ops.Equal()(true_wh, 0.0), ops.Fill()(ops.DType()(true_wh),
        #                                                              ops.Shape()(true_wh), 1.0), true_wh)
        true_wh = torch.log(true_wh / self.anchors * input_shape)

        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = gt_box.shape
        gt_box = torch.reshape(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(pred_boxes.unsqueeze(-2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = reduce_max(iou, -1, False)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ignore_mask.float().to(final_device)
        with torch.no_grad():
            ignore_mask = ignore_mask.unsqueeze(-1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        xy_loss = self.xy_loss(object_mask, box_loss_scale, prediction[:, :, :, :, :2], true_xy)
        wh_loss = self.wh_loss(object_mask, box_loss_scale, prediction[:, :, :, :, 2:4], true_wh)
        confidence_loss = self.confidenceLoss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.classLoss(object_mask, prediction[:, :, :, :, 5:], class_probs)
        loss = xy_loss + wh_loss + confidence_loss + class_loss
        batch_size = prediction.shape[0]
        return loss / batch_size


class Iou(nn.Module):
    def __init__(self):
        super(Iou, self).__init__()
        self.min = torch.minimum
        self.max = torch.maximum

    def forward(self, box1, box2):
        # box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        # box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        # convert to topLeft and rightDown
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        # print(box1_xy.shape)
        # print(box1_wh.shape)
        # print(torch.tensor(2.0).shape)
        # print("=============================================")
        box1_mins = box1_xy - box1_wh / torch.tensor(2.0)  # topLeft
        # print(box1_xy.shape)
        # print(box1_wh.shape)
        # print(torch.tensor(2.0).shape)
        # print("=============================================")
        box1_maxs = box1_xy + box1_wh / torch.tensor(2.0)  # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / torch.tensor(2.0)
        box2_maxs = box2_xy + box2_wh / torch.tensor(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, torch.tensor(0.0))
        # ops.squeeze: for effiecient slice
        intersect_area = torch.squeeze(intersect_wh[:, :, :, :, :, 0:1], -1) * \
                         torch.squeeze(intersect_wh[:, :, :, :, :, 1:2], -1)
        box1_area = torch.squeeze(box1_wh[:, :, :, :, :, 0:1], -1) * torch.squeeze((box1_wh[:, :, :, :, :, 1:2]), -1)
        box2_area = torch.squeeze(box2_wh[:, :, :, :, :, 0:1], -1) * torch.squeeze((box2_wh[:, :, :, :, :, 1:2]), -1)
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


def has_valid_annotation(anno):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

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
    if _count_visible_keypoints(anno) >= 10:
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


def get_param_groups(network):
    pg0, pg1, pg2 = [], [], []
    for k, v in network.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    return [pg0, pg1, pg2]


def conver_testing_shape(args):
    """Convert testing shape to list."""
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape


config = get_config()
