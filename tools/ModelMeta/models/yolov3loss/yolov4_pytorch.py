import os
import math
from collections import OrderedDict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.cv.yolov4.model_utils.config import config as default_config

if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
    final_device = 'cuda:6'
else:
    final_device = 'cpu'


class YOLOV4CspDarkNet53_torch(nn.Module):

    def __init__(self):
        super(YOLOV4CspDarkNet53_torch, self).__init__()
        self.config = default_config
        self.keep_detect = self.config.keep_detect
        self.input_shape = torch.tensor(tuple(default_config.test_img_shape), dtype=torch.float32).to(final_device)

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




    def forward(self, x):

        big_object_output, medium_object_output, small_object_output = self.feature_map(x)

        if not self.keep_detect:
            return big_object_output, medium_object_output, small_object_output

        output_big = self.detect_1(big_object_output, self.input_shape)
        output_me = self.detect_2(medium_object_output, self.input_shape)
        output_small = self.detect_3(small_object_output, self.input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


class DetectionBlock(nn.Module):
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
        self.conf_training= True
        self.anchors = torch.tensor([self.config.anchor_scales[i] for i in idx]).to(final_device)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = torch.reshape
        self.tile = torch.tile
        self.concat = torch.cat

    def forward(self, x, input_shape):
        """forward method"""
        num_batch = x.shape[0]
        grid_size = x.shape[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = torch.reshape(x, (num_batch,
                                       self.num_anchors_per_scale,
                                       self.num_attrib,
                                       grid_size[0],
                                       grid_size[1]))
        prediction = prediction.permute(0, 3, 4, 1, 2)

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = torch.tensor(range_x).float().to(final_device)
        grid_y = torch.tensor(range_y).float().to(final_device)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y), dim=-1)

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        # gridsize1 is x
        box_xy = (self.scale_x_y * self.sigmoid(box_xy) - self.offset_x_y + grid) / \
                 torch.tensor((grid_size[1], grid_size[0])).float().to(final_device)

        # box_wh is w->h
        temp=self.anchors / input_shape
        box_wh = torch.exp(box_wh) * temp.to(final_device)
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.conf_training:
            return prediction, box_xy, box_wh
        return self.concat((box_xy, box_wh, box_confidence, box_probs), dim=-1)


class YOLOv4(nn.Module):
    def __init__(self, backbone_shape, backbone, out_channel):
        super(YOLOv4, self).__init__()
        self.out_channel = out_channel
        self.backbone = backbone

        self.conv1 = _conv_bn_leakyrelu(1024, 512, ksize=1)
        self.conv2 = _conv_bn_leakyrelu(512, 1024, ksize=3, padding=1)
        self.conv3 = _conv_bn_leakyrelu(1024, 512, ksize=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv4 = _conv_bn_leakyrelu(2048, 512, ksize=1)

        self.conv5 = _conv_bn_leakyrelu(512, 1024, ksize=3, padding=1)
        self.conv6 = _conv_bn_leakyrelu(1024, 512, ksize=1)
        self.conv7 = _conv_bn_leakyrelu(512, 256, ksize=1)

        self.conv8 = _conv_bn_leakyrelu(512, 256, ksize=1)
        self.backblock0 = YoloBlock(backbone_shape[-2], out_chls=backbone_shape[-3], out_channels=out_channel)

        self.conv9 = _conv_bn_leakyrelu(256, 128, ksize=1)
        self.conv10 = _conv_bn_leakyrelu(256, 128, ksize=1)
        self.conv11 = _conv_bn_leakyrelu(128, 256, ksize=3, stride=2, padding=1)
        self.conv12 = _conv_bn_leakyrelu(256, 512, ksize=3, stride=2, padding=1)

        self.backblock1 = YoloBlock(backbone_shape[-3], out_chls=backbone_shape[-4], out_channels=out_channel)
        self.backblock2 = YoloBlock(backbone_shape[-2], out_chls=backbone_shape[-3], out_channels=out_channel)
        self.backblock3 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.concat = torch.cat

    def forward(self, x):
        """
        input_shape of x is (batch_size, 3, h, w)
        feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        """
        img_hight = x.shape[2]
        img_width = x.shape[3]

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
        spp = torch.cat((m3, m2, m1, con3), dim=1)
        con4 = self.conv4(spp)
        con5 = self.conv5(con4)
        con6 = self.conv6(con5)
        con7 = self.conv7(con6)
        ups1 = nn.Upsample(size=(img_hight // 16, img_width // 16), mode='nearest')(con7)
        con8 = self.conv8(feature_map2)
        con9 = torch.cat((ups1, con8), dim=1)
        con10, _ = self.backblock0(con9)
        con11 = self.conv9(con10)
        ups2 = nn.Upsample(size=(img_hight // 8, img_width // 8), mode='nearest')(con11)
        con12 = self.conv10(feature_map1)
        con13 = torch.cat((ups2, con12), dim=1)
        con14, small_object_output = self.backblock1(con13)
        con15 = self.conv11(con14)
        con16 = torch.cat((con15, con10), dim=1)
        con17, medium_object_output = self.backblock2(con16)
        con18 = self.conv12(con17)
        con19 = torch.cat((con18, con6), dim=1)
        _, big_object_output = self.backblock3(con19)
        return big_object_output, medium_object_output, small_object_output


class Mish(nn.Module):
    """Mish activation method"""

    def __init__(self):
        super(Mish, self).__init__()
        self.mul = torch.mul
        self.tanh = torch.tanh
        self.softplus = F.softplus

    def forward(self, input_x):
        res1 = self.softplus(input_x)
        tanh = self.tanh(res1)
        output = self.mul(input_x, tanh)

        return output


# ������
def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding=0,
               dilation=1):
    pad_mode = 'zeros'

    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  padding_mode=pad_mode),
        nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
        Mish()

    )


# ������
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualBlock, self).__init__()
        out_chls = out_channels
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1, padding=1)
        self.add = torch.add

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


# ������
def _conv_bn_leakyrelu(in_channel,
                       out_channel,
                       ksize,
                       stride=1,
                       padding=0,
                       dilation=1,
                       negative_slope=0.1,
                       momentum=0.9,
                       eps=1e-5,
                       pad_mode="zeros"):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=ksize,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  padding_mode=pad_mode
                  ),
        nn.BatchNorm2d(out_channel, momentum=1 - momentum, eps=eps),
        nn.LeakyReLU(negative_slope)
    )


# ������
class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_leakyrelu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_leakyrelu(out_chls, out_chls_2, ksize=3, padding=1)

        self.conv2 = _conv_bn_leakyrelu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_leakyrelu(out_chls, out_chls_2, ksize=3, padding=1)

        self.conv4 = _conv_bn_leakyrelu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_leakyrelu(out_chls, out_chls_2, ksize=3, padding=1)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        """forward method"""
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


# ������
class CspDarkNet53(nn.Module):
    def __init__(self,
                 block,
                 detect=False):
        super(CspDarkNet53, self).__init__()

        self.outchannel = 1024
        self.detect = detect
        self.concat = torch.cat
        self.add = torch.add

        self.conv0 = conv_block(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv_block(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv3 = conv_block(64, 32, kernel_size=1, stride=1)
        self.conv4 = conv_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv6 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv7 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv8 = conv_block(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv9 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv10 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv11 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv12 = conv_block(128, 128, kernel_size=1, stride=1)
        self.conv13 = conv_block(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv14 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv15 = conv_block(128, 128, kernel_size=1, stride=1)
        self.conv16 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv17 = conv_block(256, 256, kernel_size=1, stride=1)
        self.conv18 = conv_block(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv19 = conv_block(512, 256, kernel_size=1, stride=1)
        self.conv20 = conv_block(256, 256, kernel_size=1, stride=1)
        self.conv21 = conv_block(512, 256, kernel_size=1, stride=1)
        self.conv22 = conv_block(512, 512, kernel_size=1, stride=1)
        self.conv23 = conv_block(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv24 = conv_block(1024, 512, kernel_size=1, stride=1)
        self.conv25 = conv_block(512, 512, kernel_size=1, stride=1)
        self.conv26 = conv_block(1024, 512, kernel_size=1, stride=1)
        self.conv27 = conv_block(1024, 1024, kernel_size=1, stride=1)

        self.layer2 = self._make_layer(block, 2, in_channel=64, out_channel=64)
        self.layer3 = self._make_layer(block, 8, in_channel=128, out_channel=128)
        self.layer4 = self._make_layer(block, 8, in_channel=256, out_channel=256)
        self.layer5 = self._make_layer(block, 4, in_channel=512, out_channel=512)

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(("darkblk", darkblk))

        for i in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(("darblk_{}".format(i), darkblk))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """forward method"""
        c1 = self.conv0(x)
        c2 = self.conv1(c1)  # route
        c3 = self.conv2(c2)
        c4 = self.conv3(c3)
        c5 = self.conv4(c4)
        c6 = self.add(c3, c5)  # �в�ƴ�Ӵ���conv2�����+conv4������Ӹ�conv5������
        c7 = self.conv5(c6)
        c8 = self.conv6(c2)
        c9 = self.concat((c7, c8), dim=1)  # conv5�������conv6���������conv7������
        c10 = self.conv7(c9)
        c11 = self.conv8(c10)  # route
        c12 = self.conv9(c11)
        c13 = self.layer2(c12)
        c14 = self.conv10(c13)
        c15 = self.conv11(c11)
        c16 = self.concat((c14, c15), dim=1)
        c17 = self.conv12(c16)
        c18 = self.conv13(c17)  # route
        c19 = self.conv14(c18)
        c20 = self.layer3(c19)
        c21 = self.conv15(c20)
        c22 = self.conv16(c18)
        c23 = self.concat((c21, c22), dim=1)
        c24 = self.conv17(c23)  # output1
        c25 = self.conv18(c24)  # route
        c26 = self.conv19(c25)
        c27 = self.layer4(c26)
        c28 = self.conv20(c27)
        c29 = self.conv21(c25)
        c30 = self.concat((c28, c29), dim=1)
        c31 = self.conv22(c30)  # output2
        c32 = self.conv23(c31)  # route
        c33 = self.conv24(c32)
        c34 = self.layer5(c33)
        c35 = self.conv25(c34)
        c36 = self.conv26(c32)
        c37 = self.concat((c35, c36), dim=1)
        c38 = self.conv27(c37)  # output3

        if self.detect:
            return c24, c31, c38

        return c38

    def get_out_channels(self):
        return self.outchannel


class Iou(nn.Module):
    """Calculate the iou of boxes"""

    def __init__(self):
        super(Iou, self).__init__()

    def forward(self, box1, box2):
        """
        box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        convert to topLeft and rightDown
        """
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.0  # topLeft
        box1_maxs = box1_xy + box1_wh / 2.0  # rightDown

        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.0
        box2_maxs = box2_xy + box2_wh / 2.0

        intersect_mins = torch.maximum(box1_mins, box2_mins)
        intersect_maxs = torch.minimum(box1_maxs, box2_maxs)
        intersect_wh = torch.maximum(intersect_maxs - intersect_mins, torch.tensor(0.0).to(final_device))
        # torch.squeeze: for efficient slice
        intersect_area = torch.squeeze(intersect_wh[..., 0:1], -1) * \
                         torch.squeeze(intersect_wh[..., 1:2], -1)
        box1_area = torch.squeeze(box1_wh[..., 0:1], -1) * torch.squeeze(box1_wh[..., 1:2], -1)
        box2_area = torch.squeeze(box2_wh[..., 0:1], -1) * torch.squeeze(box2_wh[..., 1:2], -1)
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou





class XYLoss(nn.Module):
    """Loss for x and y."""

    def __init__(self):
        super(XYLoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, object_mask, box_loss_scale, predict_xy, true_xy):
        xy_loss = object_mask * box_loss_scale * self.bce_with_logits(predict_xy, true_xy)
        xy_loss = torch.sum(xy_loss)
        return xy_loss


class WHLoss(nn.Module):
    """Loss for w and h."""

    def __init__(self):
        super(WHLoss, self).__init__()

    def forward(self, object_mask, box_loss_scale, predict_wh, true_wh):
        wh_loss = object_mask * box_loss_scale * 0.5 * torch.square(true_wh - predict_wh)
        wh_loss = torch.sum(wh_loss)
        return wh_loss


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


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


class ConfidenceLoss(nn.Module):
    """Loss for confidence."""

    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, object_mask, predict_confidence, ignore_mask):
        confidence_loss = self.bce_with_logits(predict_confidence, object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        confidence_loss = torch.sum(confidence_loss)
        return confidence_loss


class ClassLoss(nn.Module):
    """Loss for classification."""

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, object_mask, predict_class, class_probs):
        class_loss = object_mask * self.bce_with_logits(predict_class, class_probs)
        class_loss = torch.sum(class_loss)
        return class_loss


class Giou(nn.Module):
    """Calculating giou"""

    def __init__(self):
        super(Giou, self).__init__()
        self.eps = 0.000001

    def forward(self, box_p, box_gt):
        """forward method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = torch.maximum(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = torch.minimum(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = torch.maximum(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = torch.minimum(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = torch.minimum(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = torch.maximum(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = torch.minimum(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = torch.maximum(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = (intersection.float() + self.eps) / (union.float() + self.eps)
        res_mid0 = c_area - union
        res_mid1 = (res_mid0.float() + self.eps) / (c_area.float() + self.eps)
        giou = iou - res_mid1
        giou = torch.clamp(giou, -1.0, 1.0)
        return giou


def xywh2x1y1x2y2(box_xywh):
    boxes_x1 = box_xywh[..., 0:1] - box_xywh[..., 2:3] / 2
    boxes_y1 = box_xywh[..., 1:2] - box_xywh[..., 3:4] / 2
    boxes_x2 = box_xywh[..., 0:1] + box_xywh[..., 2:3] / 2
    boxes_y2 = box_xywh[..., 1:2] + box_xywh[..., 3:4] / 2
    boxes_x1y1x2y2 = torch.cat((boxes_x1, boxes_y1, boxes_x2, boxes_y2), dim=-1)
    return boxes_x1y1x2y2


class YoloLossBlock_torch(nn.Module):
    """
    Loss block module of YOLOV4 network.
    """

    def __init__(self, scale, config=default_config):
        super(YoloLossBlock_torch, self).__init__()
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
        self.anchors = torch.tensor([self.config.anchor_scales[i] for i in idx], dtype=torch.float32).to(final_device)
        self.ignore_threshold = torch.tensor(self.config.ignore_threshold, dtype=torch.float32).to(final_device)
        self.iou = Iou()
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidence_loss = ConfidenceLoss()
        self.class_loss = ClassLoss()
        self.giou = Giou()
        self.bbox_class_loss_coff = self.config.bbox_class_loss_coff
        self.ciou_loss_me_coff = int(self.bbox_class_loss_coff[0])
        self.confidence_loss_coff = int(self.bbox_class_loss_coff[1])
        self.class_loss_coff = int(self.bbox_class_loss_coff[2])

    def forward(self, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        """
        prediction : origin output from yolo
        pred_xy: (sigmoid(xy)+grid)/grid_size
        pred_wh: (exp(wh)*anchors)/input_shape
        y_true : after normalize
        gt_box: [batch, maxboxes, xyhw] after normalize
        """
        object_mask = y_true[..., 4:5]
        class_probs = y_true[..., 5:]
        true_boxes = y_true[..., :4]

        grid_shape = torch.tensor(prediction.shape[1:3], dtype=torch.float32).to(final_device)
        grid_shape = grid_shape.flip(0)

        pred_boxes = torch.cat((pred_xy, pred_wh), -1)

        true_wh = y_true[..., 2:4]
        true_wh = torch.where(true_wh == 0.0, torch.ones_like(true_wh), true_wh).to(final_device)
        input_shape = torch.tensor(input_shape, dtype=torch.float32).to(final_device)
        true_wh = torch.log(true_wh / self.anchors * input_shape)

        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        gt_shape = gt_box.shape
        gt_box = gt_box.view(gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2])

        # add one more dimension for broadcast
        iou = self.iou(pred_boxes.unsqueeze(-2), gt_box)
        best_iou, _ = torch.max(iou, -1)
        # [batch, grid[0], grid[1],        # [batch, grid[0], grid[1], num_anchor]
        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ignore_mask.to(torch.float32)
        ignore_mask = ignore_mask.unsqueeze(-1)
        # ignore_mask backprop will cause a lot of maximum_grad and minimum_grad time consumption.
        # so we turn off its gradient
        ignore_mask = ignore_mask.detach()

        confidence_loss = self.confidence_loss(object_mask, prediction[..., 4:5], ignore_mask)
        class_loss = self.class_loss(object_mask, prediction[..., 5:], class_probs)

        object_mask_me = object_mask.view(-1, 1)
        box_loss_scale_me = box_loss_scale.view(-1, 1)
        pred_boxes_me = xywh2x1y1x2y2(pred_boxes)
        pred_boxes_me = pred_boxes_me.view(-1, 4)
        true_boxes_me = xywh2x1y1x2y2(true_boxes)
        true_boxes_me = true_boxes_me.view(-1, 4)
        ciou = self.giou(pred_boxes_me, true_boxes_me)
        ciou_loss = object_mask_me * box_loss_scale_me * (1 - ciou)
        ciou_loss_me = torch.sum(ciou_loss)
        loss = ciou_loss_me * self.ciou_loss_me_coff + confidence_loss * self.confidence_loss_coff + class_loss * self.class_loss_coff
        batch_size = prediction.shape[0]
        return loss / batch_size


class yolov4loss_torch(torch.nn.Module):
    def __init__(self):
        super(yolov4loss_torch, self).__init__()
        self.config = default_config
        self.loss_big = YoloLossBlock_torch('l', self.config)
        self.loss_me = YoloLossBlock_torch('m', self.config)
        self.loss_small = YoloLossBlock_torch('s', self.config)

    def forward(self, yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):

        loss_coff = [1, 1, 1]
        loss_l_coff = int(loss_coff[0])
        loss_m_coff = int(loss_coff[1])
        loss_s_coff = int(loss_coff[2])

        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        return loss_l * loss_l_coff + loss_m * loss_m_coff + loss_s * loss_s_coff


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
