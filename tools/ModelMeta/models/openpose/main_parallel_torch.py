import json
import os

import cv2
import mindspore
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from models.openpose.eval import detect, draw_person_pose, evaluate_mAP, detect_torch
from models.openpose.src.dataset import create_dataset, valdata
from models.openpose.src.loss_torch import openpose_loss, BuildTrainNetwork
from models.openpose.src.model_utils.config import config
from models.openpose.src.utils import get_lr
# import troubleshooter as ts


class OpenPoseNet(nn.Module):
    insize = 368

    def __init__(self, vggpath='', vgg_with_bn=False):
        super(OpenPoseNet, self).__init__()
        self.base = Base_model(vgg_with_bn=vgg_with_bn)
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None

        self.in_shapes={
            'base.vgg_base.layers.0': [1, 3, 368, 368],
            'base.vgg_base.layers.1': [1, 64, 368, 368],
            'base.vgg_base.layers.2': [1, 64, 368, 368],
            'base.vgg_base.layers.3': [1, 64, 368, 368],
            'base.vgg_base.layers.4': [1, 64, 368, 368],
            'base.vgg_base.layers.5': [1, 64, 184, 184],
            'base.vgg_base.layers.6': [1, 128, 184, 184],
            'base.vgg_base.layers.7': [1, 128, 184, 184],
            'base.vgg_base.layers.8': [1, 128, 184, 184],
            'base.vgg_base.layers.9': [1, 128, 184, 184],
            'base.vgg_base.layers.10': [1, 128, 92, 92],
            'base.vgg_base.layers.11': [1, 256, 92, 92],
            'base.vgg_base.layers.12': [1, 256, 92, 92],
            'base.vgg_base.layers.13': [1, 256, 92, 92],
            'base.vgg_base.layers.14': [1, 256, 92, 92],
            'base.vgg_base.layers.15': [1, 256, 92, 92],
            'base.vgg_base.layers.16': [1, 256, 92, 92],
            'base.vgg_base.layers.17': [1, 256, 92, 92],
            'base.vgg_base.layers.18': [1, 256, 92, 92],
            'base.vgg_base.layers.19': [1, 256, 46, 46],
            'base.vgg_base.layers.20': [1, 512, 46, 46],
            'base.vgg_base.layers.21': [1, 512, 46, 46],
            'base.vgg_base.layers.22': [1, 512, 46, 46],
            'base.conv4_3_CPM': [1, 512, 46, 46],
            'base.conv4_4_CPM': [1, 256, 46, 46],
            'base.relu': [1, 128, 46, 46],
            'stage_1.conv1_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv2_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv3_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv4_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv5_CPM_L1': [1, 512, 46, 46],
            'stage_1.conv1_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv2_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv3_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv4_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv5_CPM_L2': [1, 512, 46, 46],
            'stage_1.relu': [1, 512, 46, 46],
            'stage_2.conv1_L1': [1, 185, 46, 46],
            'stage_2.conv2_L1': [1, 128, 46, 46],
            'stage_2.conv3_L1': [1, 128, 46, 46],
            'stage_2.conv4_L1': [1, 128, 46, 46],
            'stage_2.conv5_L1': [1, 128, 46, 46],
            'stage_2.conv6_L1': [1, 128, 46, 46],
            'stage_2.conv7_L1': [1, 128, 46, 46],
            'stage_2.conv1_L2': [1, 185, 46, 46],
            'stage_2.conv2_L2': [1, 128, 46, 46],
            'stage_2.conv3_L2': [1, 128, 46, 46],
            'stage_2.conv4_L2': [1, 128, 46, 46],
            'stage_2.conv5_L2': [1, 128, 46, 46],
            'stage_2.conv6_L2': [1, 128, 46, 46],
            'stage_2.conv7_L2': [1, 128, 46, 46],
            'stage_2.relu': [1, 128, 46, 46],
            'stage_3.conv1_L1': [1, 185, 46, 46],
            'stage_3.conv2_L1': [1, 128, 46, 46],
            'stage_3.conv3_L1': [1, 128, 46, 46],
            'stage_3.conv4_L1': [1, 128, 46, 46],
            'stage_3.conv5_L1': [1, 128, 46, 46],
            'stage_3.conv6_L1': [1, 128, 46, 46],
            'stage_3.conv7_L1': [1, 128, 46, 46],
            'stage_3.conv1_L2': [1, 185, 46, 46],
            'stage_3.conv2_L2': [1, 128, 46, 46],
            'stage_3.conv3_L2': [1, 128, 46, 46],
            'stage_3.conv4_L2': [1, 128, 46, 46],
            'stage_3.conv5_L2': [1, 128, 46, 46],
            'stage_3.conv6_L2': [1, 128, 46, 46],
            'stage_3.conv7_L2': [1, 128, 46, 46],
            'stage_3.relu': [1, 128, 46, 46],
            'stage_4.conv1_L1': [1, 185, 46, 46],
            'stage_4.conv2_L1': [1, 128, 46, 46],
            'stage_4.conv3_L1': [1, 128, 46, 46],
            'stage_4.conv4_L1': [1, 128, 46, 46],
            'stage_4.conv5_L1': [1, 128, 46, 46],
            'stage_4.conv6_L1': [1, 128, 46, 46],
            'stage_4.conv7_L1': [1, 128, 46, 46],
            'stage_4.conv1_L2': [1, 185, 46, 46],
            'stage_4.conv2_L2': [1, 128, 46, 46],
            'stage_4.conv3_L2': [1, 128, 46, 46],
            'stage_4.conv4_L2': [1, 128, 46, 46],
            'stage_4.conv5_L2': [1, 128, 46, 46],
            'stage_4.conv6_L2': [1, 128, 46, 46],
            'stage_4.conv7_L2': [1, 128, 46, 46],
            'stage_4.relu': [1, 128, 46, 46],
            'stage_5.conv1_L1': [1, 185, 46, 46],
            'stage_5.conv2_L1': [1, 128, 46, 46],
            'stage_5.conv3_L1': [1, 128, 46, 46],
            'stage_5.conv4_L1': [1, 128, 46, 46],
            'stage_5.conv5_L1': [1, 128, 46, 46],
            'stage_5.conv6_L1': [1, 128, 46, 46],
            'stage_5.conv7_L1': [1, 128, 46, 46],
            'stage_5.conv1_L2': [1, 185, 46, 46],
            'stage_5.conv2_L2': [1, 128, 46, 46],
            'stage_5.conv3_L2': [1, 128, 46, 46],
            'stage_5.conv4_L2': [1, 128, 46, 46],
            'stage_5.conv5_L2': [1, 128, 46, 46],
            'stage_5.conv6_L2': [1, 128, 46, 46],
            'stage_5.conv7_L2': [1, 128, 46, 46],
            'stage_5.relu': [1, 128, 46, 46],
            'stage_6.conv1_L1': [1, 185, 46, 46],
            'stage_6.conv2_L1': [1, 128, 46, 46],
            'stage_6.conv3_L1': [1, 128, 46, 46],
            'stage_6.conv4_L1': [1, 128, 46, 46],
            'stage_6.conv5_L1': [1, 128, 46, 46],
            'stage_6.conv6_L1': [1, 128, 46, 46],
            'stage_6.conv7_L1': [1, 128, 46, 46],
            'stage_6.conv1_L2': [1, 185, 46, 46],
            'stage_6.conv2_L2': [1, 128, 46, 46],
            'stage_6.conv3_L2': [1, 128, 46, 46],
            'stage_6.conv4_L2': [1, 128, 46, 46],
            'stage_6.conv5_L2': [1, 128, 46, 46],
            'stage_6.conv6_L2': [1, 128, 46, 46],
            'stage_6.conv7_L2': [1, 128, 46, 46],
            'stage_6.relu': [1, 128, 46, 46],
            'INPUT': [1, 3, 368, 368],
            'OUTPUT1': [1, 38, 46, 46],
            'OUTPUT2': [1, 19, 46, 46],
            'OUTPUT3': [1, 38, 46, 46],
            'OUTPUT4': [1, 19, 46, 46],
            'OUTPUT5': [1, 38, 46, 46],
            'OUTPUT6': [1, 19, 46, 46],
            'OUTPUT7': [1, 38, 46, 46],
            'OUTPUT8': [1, 19, 46, 46],
            'OUTPUT9': [1, 38, 46, 46],
            'OUTPUT10': [1, 19, 46, 46],
            'OUTPUT11': [1, 38, 46, 46],
            'OUTPUT12': [1, 19, 46, 46]
        }

        self.out_shapes = {
            'base.vgg_base.layers.0': [1, 64, 368, 368],
            'base.vgg_base.layers.1': [1, 64, 368, 368],
            'base.vgg_base.layers.2': [1, 64, 368, 368],
            'base.vgg_base.layers.3': [1, 64, 368, 368],
            'base.vgg_base.layers.4': [1, 64, 184, 184],
            'base.vgg_base.layers.5': [1, 128, 184, 184],
            'base.vgg_base.layers.6': [1, 128, 184, 184],
            'base.vgg_base.layers.7': [1, 128, 184, 184],
            'base.vgg_base.layers.8': [1, 128, 184, 184],
            'base.vgg_base.layers.9': [1, 128, 92, 92],
            'base.vgg_base.layers.10': [1, 256, 92, 92],
            'base.vgg_base.layers.11': [1, 256, 92, 92],
            'base.vgg_base.layers.12': [1, 256, 92, 92],
            'base.vgg_base.layers.13': [1, 256, 92, 92],
            'base.vgg_base.layers.14': [1, 256, 92, 92],
            'base.vgg_base.layers.15': [1, 256, 92, 92],
            'base.vgg_base.layers.16': [1, 256, 92, 92],
            'base.vgg_base.layers.17': [1, 256, 92, 92],
            'base.vgg_base.layers.18': [1, 256, 46, 46],
            'base.vgg_base.layers.19': [1, 512, 46, 46],
            'base.vgg_base.layers.20': [1, 512, 46, 46],
            'base.vgg_base.layers.21': [1, 512, 46, 46],
            'base.vgg_base.layers.22': [1, 512, 46, 46],
            'base.conv4_3_CPM': [1, 256, 46, 46],
            'base.conv4_4_CPM': [1, 128, 46, 46],
            'base.relu': [1, 128, 46, 46],
            'stage_1.conv1_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv2_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv3_CPM_L1': [1, 128, 46, 46],
            'stage_1.conv4_CPM_L1': [1, 512, 46, 46],
            'stage_1.conv5_CPM_L1': [1, 38, 46, 46],
            'stage_1.conv1_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv2_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv3_CPM_L2': [1, 128, 46, 46],
            'stage_1.conv4_CPM_L2': [1, 512, 46, 46],
            'stage_1.conv5_CPM_L2': [1, 19, 46, 46],
            'stage_1.relu': [1, 512, 46, 46],
            'stage_2.conv1_L1': [1, 128, 46, 46],
            'stage_2.conv2_L1': [1, 128, 46, 46],
            'stage_2.conv3_L1': [1, 128, 46, 46],
            'stage_2.conv4_L1': [1, 128, 46, 46],
            'stage_2.conv5_L1': [1, 128, 46, 46],
            'stage_2.conv6_L1': [1, 128, 46, 46],
            'stage_2.conv7_L1': [1, 38, 46, 46],
            'stage_2.conv1_L2': [1, 128, 46, 46],
            'stage_2.conv2_L2': [1, 128, 46, 46],
            'stage_2.conv3_L2': [1, 128, 46, 46],
            'stage_2.conv4_L2': [1, 128, 46, 46],
            'stage_2.conv5_L2': [1, 128, 46, 46],
            'stage_2.conv6_L2': [1, 128, 46, 46],
            'stage_2.conv7_L2': [1, 19, 46, 46],
            'stage_2.relu': [1, 128, 46, 46],
            'stage_3.conv1_L1': [1, 128, 46, 46],
            'stage_3.conv2_L1': [1, 128, 46, 46],
            'stage_3.conv3_L1': [1, 128, 46, 46],
            'stage_3.conv4_L1': [1, 128, 46, 46],
            'stage_3.conv5_L1': [1, 128, 46, 46],
            'stage_3.conv6_L1': [1, 128, 46, 46],
            'stage_3.conv7_L1': [1, 38, 46, 46],
            'stage_3.conv1_L2': [1, 128, 46, 46],
            'stage_3.conv2_L2': [1, 128, 46, 46],
            'stage_3.conv3_L2': [1, 128, 46, 46],
            'stage_3.conv4_L2': [1, 128, 46, 46],
            'stage_3.conv5_L2': [1, 128, 46, 46],
            'stage_3.conv6_L2': [1, 128, 46, 46],
            'stage_3.conv7_L2': [1, 19, 46, 46],
            'stage_3.relu': [1, 128, 46, 46],
            'stage_4.conv1_L1': [1, 128, 46, 46],
            'stage_4.conv2_L1': [1, 128, 46, 46],
            'stage_4.conv3_L1': [1, 128, 46, 46],
            'stage_4.conv4_L1': [1, 128, 46, 46],
            'stage_4.conv5_L1': [1, 128, 46, 46],
            'stage_4.conv6_L1': [1, 128, 46, 46],
            'stage_4.conv7_L1': [1, 38, 46, 46],
            'stage_4.conv1_L2': [1, 128, 46, 46],
            'stage_4.conv2_L2': [1, 128, 46, 46],
            'stage_4.conv3_L2': [1, 128, 46, 46],
            'stage_4.conv4_L2': [1, 128, 46, 46],
            'stage_4.conv5_L2': [1, 128, 46, 46],
            'stage_4.conv6_L2': [1, 128, 46, 46],
            'stage_4.conv7_L2': [1, 19, 46, 46],
            'stage_4.relu': [1, 128, 46, 46],
            'stage_5.conv1_L1': [1, 128, 46, 46],
            'stage_5.conv2_L1': [1, 128, 46, 46],
            'stage_5.conv3_L1': [1, 128, 46, 46],
            'stage_5.conv4_L1': [1, 128, 46, 46],
            'stage_5.conv5_L1': [1, 128, 46, 46],
            'stage_5.conv6_L1': [1, 128, 46, 46],
            'stage_5.conv7_L1': [1, 38, 46, 46],
            'stage_5.conv1_L2': [1, 128, 46, 46],
            'stage_5.conv2_L2': [1, 128, 46, 46],
            'stage_5.conv3_L2': [1, 128, 46, 46],
            'stage_5.conv4_L2': [1, 128, 46, 46],
            'stage_5.conv5_L2': [1, 128, 46, 46],
            'stage_5.conv6_L2': [1, 128, 46, 46],
            'stage_5.conv7_L2': [1, 19, 46, 46],
            'stage_5.relu': [1, 128, 46, 46],
            'stage_6.conv1_L1': [1, 128, 46, 46],
            'stage_6.conv2_L1': [1, 128, 46, 46],
            'stage_6.conv3_L1': [1, 128, 46, 46],
            'stage_6.conv4_L1': [1, 128, 46, 46],
            'stage_6.conv5_L1': [1, 128, 46, 46],
            'stage_6.conv6_L1': [1, 128, 46, 46],
            'stage_6.conv7_L1': [1, 38, 46, 46],
            'stage_6.conv1_L2': [1, 128, 46, 46],
            'stage_6.conv2_L2': [1, 128, 46, 46],
            'stage_6.conv3_L2': [1, 128, 46, 46],
            'stage_6.conv4_L2': [1, 128, 46, 46],
            'stage_6.conv5_L2': [1, 128, 46, 46],
            'stage_6.conv6_L2': [1, 128, 46, 46],
            'stage_6.conv7_L2': [1, 19, 46, 46],
            'stage_6.relu': [1, 128, 46, 46],
            'INPUT': [1, 3, 368, 368],
            'OUTPUT1': [1, 38, 46, 46],
            'OUTPUT2': [1, 19, 46, 46],
            'OUTPUT3': [1, 38, 46, 46],
            'OUTPUT4': [1, 19, 46, 46],
            'OUTPUT5': [1, 38, 46, 46],
            'OUTPUT6': [1, 19, 46, 46],
            'OUTPUT7': [1, 38, 46, 46],
            'OUTPUT8': [1, 19, 46, 46],
            'OUTPUT9': [1, 38, 46, 46],
            'OUTPUT10': [1, 19, 46, 46],
            'OUTPUT11': [1, 38, 46, 46],
            'OUTPUT12': [1, 19, 46, 46]
        }

        self.orders={
            'base.vgg_base.layers.0': ['INPUT',
                                    'base.vgg_base.layers.1'],
            'base.vgg_base.layers.1': ['base.vgg_base.layers.0',
                                    'base.vgg_base.layers.2'],
            'base.vgg_base.layers.2': ['base.vgg_base.layers.1',
                                    'base.vgg_base.layers.3'],
            'base.vgg_base.layers.3': ['base.vgg_base.layers.2',
                                    'base.vgg_base.layers.4'],
            'base.vgg_base.layers.4': ['base.vgg_base.layers.3',
                                    'base.vgg_base.layers.5'],
            'base.vgg_base.layers.5': ['base.vgg_base.layers.4',
                                    'base.vgg_base.layers.6'],
            'base.vgg_base.layers.6': ['base.vgg_base.layers.5',
                                    'base.vgg_base.layers.7'],
            'base.vgg_base.layers.7': ['base.vgg_base.layers.6',
                                    'base.vgg_base.layers.8'],
            'base.vgg_base.layers.8': ['base.vgg_base.layers.7',
                                    'base.vgg_base.layers.9'],
            'base.vgg_base.layers.9': ['base.vgg_base.layers.8',
                                    'base.vgg_base.layers.10'],
            'base.vgg_base.layers.10': ['base.vgg_base.layers.9',
                                     'base.vgg_base.layers.11'],
            'base.vgg_base.layers.11': ['base.vgg_base.layers.10',
                                     'base.vgg_base.layers.12'],
            'base.vgg_base.layers.12': ['base.vgg_base.layers.11',
                                     'base.vgg_base.layers.13'],
            'base.vgg_base.layers.13': ['base.vgg_base.layers.12',
                                     'base.vgg_base.layers.14'],
            'base.vgg_base.layers.14': ['base.vgg_base.layers.13',
                                     'base.vgg_base.layers.15'],
            'base.vgg_base.layers.15': ['base.vgg_base.layers.14',
                                     'base.vgg_base.layers.16'],
            'base.vgg_base.layers.16': ['base.vgg_base.layers.15',
                                     'base.vgg_base.layers.17'],
            'base.vgg_base.layers.17': ['base.vgg_base.layers.16',
                                     'base.vgg_base.layers.18'],
            'base.vgg_base.layers.18': ['base.vgg_base.layers.17',
                                     'base.vgg_base.layers.19'],
            'base.vgg_base.layers.19': ['base.vgg_base.layers.18',
                                     'base.vgg_base.layers.20'],
            'base.vgg_base.layers.20': ['base.vgg_base.layers.19',
                                     'base.vgg_base.layers.21'],
            'base.vgg_base.layers.21': ['base.vgg_base.layers.20',
                                     'base.vgg_base.layers.22'],
            'base.vgg_base.layers.22': ['base.vgg_base.layers.21', 'base.conv4_3_CPM'],
            'base.conv4_3_CPM': ['base.vgg_base.layers.22', 'base.conv4_4_CPM'],
            'base.conv4_4_CPM': ['base.conv4_3_CPM', 'base.relu'],
            'base.relu': ['base.conv4_4_CPM', ['stage_1.conv1_CPM_L1','stage_1.conv1_CPM_L2',
                                               'stage_2.conv1_L1','stage_2.conv1_L2',
                                               'stage_3.conv1_L1','stage_3.conv1_L2',
                                               'stage_4.conv1_L1','stage_4.conv1_L2',
                                               'stage_5.conv1_L1','stage_5.conv1_L2',
                                               'stage_6.conv1_L1','stage_6.conv1_L2']],


            'stage_1.conv1_CPM_L1': ['base.relu', 'stage_1.conv2_CPM_L1'],
            'stage_1.conv2_CPM_L1': ['stage_1.conv1_CPM_L1', 'stage_1.conv3_CPM_L1'],
            'stage_1.conv3_CPM_L1': ['stage_1.conv2_CPM_L1', 'stage_1.conv4_CPM_L1'],
            'stage_1.conv4_CPM_L1': ['stage_1.conv3_CPM_L1', 'stage_1.conv5_CPM_L1'],
            # OUTPUT1 h1
            'stage_1.conv5_CPM_L1': ['stage_1.conv4_CPM_L1', ['stage_2.conv1_L1','stage_2.conv1_L2','OUTPUT1']],

            'stage_1.conv1_CPM_L2': ['base.relu', 'stage_1.conv2_CPM_L2'],
            'stage_1.conv2_CPM_L2': ['stage_1.conv1_CPM_L2', 'stage_1.conv3_CPM_L2'],
            'stage_1.conv3_CPM_L2': ['stage_1.conv2_CPM_L2', 'stage_1.conv4_CPM_L2'],
            'stage_1.conv4_CPM_L2': ['stage_1.conv3_CPM_L2', 'stage_1.relu'],
            'stage_1.relu': ['stage_1.conv4_CPM_L2', 'stage_1.conv5_CPM_L2'],
            # OUTPUT2 h2
            'stage_1.conv5_CPM_L2': ['stage_1.relu', ['stage_2.conv1_L1','stage_2.conv1_L2','OUTPUT2']],


            'stage_2.conv1_L1': [['base.relu','stage_1.conv5_CPM_L1','stage_1.conv5_CPM_L2'], 'stage_2.conv2_L1'],
            'stage_2.conv2_L1': ['stage_2.conv1_L1', 'stage_2.conv3_L1'],
            'stage_2.conv3_L1': ['stage_2.conv2_L1', 'stage_2.conv4_L1'],
            'stage_2.conv4_L1': ['stage_2.conv3_L1', 'stage_2.conv5_L1'],
            'stage_2.conv5_L1': ['stage_2.conv4_L1', 'stage_2.conv6_L1'],
            'stage_2.conv6_L1': ['stage_2.conv5_L1', 'stage_2.conv7_L1'],
            # OUTPUT3 h1
            'stage_2.conv7_L1': ['stage_2.conv6_L1', ['stage_3.conv1_L1','stage_3.conv1_L2','OUTPUT3']],

            'stage_2.conv1_L2': [['base.relu','stage_1.conv5_CPM_L1','stage_1.conv5_CPM_L2'], 'stage_2.conv2_L2'],
            'stage_2.conv2_L2': ['stage_2.conv1_L2', 'stage_2.conv3_L2'],
            'stage_2.conv3_L2': ['stage_2.conv2_L2', 'stage_2.conv4_L2'],
            'stage_2.conv4_L2': ['stage_2.conv3_L2', 'stage_2.conv5_L2'],
            'stage_2.conv5_L2': ['stage_2.conv4_L2', 'stage_2.conv6_L2'],
            'stage_2.conv6_L2': ['stage_2.conv5_L2', 'stage_2.relu'],
            'stage_2.relu': ['stage_2.conv6_L2', 'stage_2.conv7_L2'],
            # OUTPUT4 h2
            'stage_2.conv7_L2': ['stage_2.relu', ['stage_3.conv1_L1','stage_3.conv1_L2','OUTPUT4']],


            'stage_3.conv1_L1': [['base.relu','stage_2.conv7_L1','stage_2.conv7_L2'], 'stage_3.conv2_L1'],
            'stage_3.conv2_L1': ['stage_3.conv1_L1', 'stage_3.conv3_L1'],
            'stage_3.conv3_L1': ['stage_3.conv2_L1', 'stage_3.conv4_L1'],
            'stage_3.conv4_L1': ['stage_3.conv3_L1', 'stage_3.conv5_L1'],
            'stage_3.conv5_L1': ['stage_3.conv4_L1', 'stage_3.conv6_L1'],
            'stage_3.conv6_L1': ['stage_3.conv5_L1', 'stage_3.conv7_L1'],
            # OUTPUT5 h1
            'stage_3.conv7_L1': ['stage_3.conv6_L1', ['stage_4.conv1_L1','stage_4.conv1_L2','OUTPUT5']],

            'stage_3.conv1_L2': [['base.relu','stage_2.conv7_L1','stage_2.conv7_L2'], 'stage_3.conv2_L2'],
            'stage_3.conv2_L2': ['stage_3.conv1_L2', 'stage_3.conv3_L2'],
            'stage_3.conv3_L2': ['stage_3.conv2_L2', 'stage_3.conv4_L2'],
            'stage_3.conv4_L2': ['stage_3.conv3_L2', 'stage_3.conv5_L2'],
            'stage_3.conv5_L2': ['stage_3.conv4_L2', 'stage_3.conv6_L2'],
            'stage_3.conv6_L2': ['stage_3.conv5_L2', 'stage_3.relu'],
            'stage_3.relu': ['stage_3.conv6_L2', 'stage_3.conv7_L2'],
            # OUTPUT6 h2
            'stage_3.conv7_L2': ['stage_3.relu', ['stage_4.conv1_L1','stage_4.conv1_L2','OUTPUT6']],


            'stage_4.conv1_L1': [['base.relu','stage_3.conv7_L1','stage_3.conv7_L2'], 'stage_4.conv2_L1'],
            'stage_4.conv2_L1': ['stage_4.conv1_L1', 'stage_4.conv3_L1'],
            'stage_4.conv3_L1': ['stage_4.conv2_L1', 'stage_4.conv4_L1'],
            'stage_4.conv4_L1': ['stage_4.conv3_L1', 'stage_4.conv5_L1'],
            'stage_4.conv5_L1': ['stage_4.conv4_L1', 'stage_4.conv6_L1'],
            'stage_4.conv6_L1': ['stage_4.conv5_L1', 'stage_4.conv7_L1'],
            # OUTPUT7 h1
            'stage_4.conv7_L1': ['stage_4.conv6_L1', ['stage_5.conv1_L1','stage_5.conv1_L2','OUTPUT7']],

            'stage_4.conv1_L2': [['base.relu','stage_3.conv7_L1','stage_3.conv7_L2'], 'stage_4.conv2_L2'],
            'stage_4.conv2_L2': ['stage_4.conv1_L2', 'stage_4.conv3_L2'],
            'stage_4.conv3_L2': ['stage_4.conv2_L2', 'stage_4.conv4_L2'],
            'stage_4.conv4_L2': ['stage_4.conv3_L2', 'stage_4.conv5_L2'],
            'stage_4.conv5_L2': ['stage_4.conv4_L2', 'stage_4.conv6_L2'],
            'stage_4.conv6_L2': ['stage_4.conv5_L2', 'stage_4.relu'],
            'stage_4.relu': ['stage_4.conv6_L2', 'stage_4.conv7_L2'],
            # OUTPUT8 h2
            'stage_4.conv7_L2': ['stage_4.relu', ['stage_5.conv1_L1','stage_5.conv1_L2','OUTPUT8']],


            'stage_5.conv1_L1': [['base.relu','stage_4.conv7_L1','stage_4.conv7_L2'], 'stage_5.conv2_L1'],
            'stage_5.conv2_L1': ['stage_5.conv1_L1', 'stage_5.conv3_L1'],
            'stage_5.conv3_L1': ['stage_5.conv2_L1', 'stage_5.conv4_L1'],
            'stage_5.conv4_L1': ['stage_5.conv3_L1', 'stage_5.conv5_L1'],
            'stage_5.conv5_L1': ['stage_5.conv4_L1', 'stage_5.conv6_L1'],
            'stage_5.conv6_L1': ['stage_5.conv5_L1', 'stage_5.conv7_L1'],
            # PUTPUT9 h1
            'stage_5.conv7_L1': ['stage_5.conv6_L1', ['stage_6.conv1_L1','stage_6.conv1_L2','OUTPUT9']],

            'stage_5.conv1_L2': [['base.relu','stage_4.conv7_L1','stage_4.conv7_L2'], 'stage_5.conv2_L2'],
            'stage_5.conv2_L2': ['stage_5.conv1_L2', 'stage_5.conv3_L2'],
            'stage_5.conv3_L2': ['stage_5.conv2_L2', 'stage_5.conv4_L2'],
            'stage_5.conv4_L2': ['stage_5.conv3_L2', 'stage_5.conv5_L2'],
            'stage_5.conv5_L2': ['stage_5.conv4_L2', 'stage_5.conv6_L2'],
            'stage_5.conv6_L2': ['stage_5.conv5_L2', 'stage_5.relu'],
            'stage_5.relu': ['stage_5.conv6_L2', 'stage_5.conv7_L2'],
            # OUTPUT10 h2
            'stage_5.conv7_L2': ['stage_5.relu', ['stage_6.conv1_L1','stage_6.conv1_L2','OUTPUT10']],


            'stage_6.conv1_L1': [['base.relu','stage_5.conv7_L1','stage_5.conv7_L2'], 'stage_6.conv2_L1'],
            'stage_6.conv2_L1': ['stage_6.conv1_L1', 'stage_6.conv3_L1'],
            'stage_6.conv3_L1': ['stage_6.conv2_L1', 'stage_6.conv4_L1'],
            'stage_6.conv4_L1': ['stage_6.conv3_L1', 'stage_6.conv5_L1'],
            'stage_6.conv5_L1': ['stage_6.conv4_L1', 'stage_6.conv6_L1'],
            'stage_6.conv6_L1': ['stage_6.conv5_L1', 'stage_6.conv7_L1'],
            # OUTPUT11 h1
            'stage_6.conv7_L1': ['stage_6.conv6_L1', 'OUTPUT11'],

            'stage_6.conv1_L2': [['base.relu','stage_5.conv7_L1','stage_5.conv7_L2'], 'stage_6.conv2_L2'],
            'stage_6.conv2_L2': ['stage_6.conv1_L2', 'stage_6.conv3_L2'],
            'stage_6.conv3_L2': ['stage_6.conv2_L2', 'stage_6.conv4_L2'],
            'stage_6.conv4_L2': ['stage_6.conv3_L2', 'stage_6.conv5_L2'],
            'stage_6.conv5_L2': ['stage_6.conv4_L2', 'stage_6.conv6_L2'],
            'stage_6.conv6_L2': ['stage_6.conv5_L2', 'stage_6.relu'],
            'stage_6.relu': ['stage_6.conv6_L2', 'stage_6.conv7_L2'],
            # OUTPUT12 h2
            'stage_6.conv7_L2': ['stage_6.relu', 'OUTPUT12']
        }

        self.layer_input_dtype ={
            'base.vgg_base.layers.0': [torch.float32],
            'base.vgg_base.layers.1': [torch.float32],
            'base.vgg_base.layers.2': [torch.float32],
            'base.vgg_base.layers.3': [torch.float32],
            'base.vgg_base.layers.4': [torch.float32],
            'base.vgg_base.layers.5': [torch.float32],
            'base.vgg_base.layers.6': [torch.float32],
            'base.vgg_base.layers.7': [torch.float32],
            'base.vgg_base.layers.8': [torch.float32],
            'base.vgg_base.layers.9': [torch.float32],
            'base.vgg_base.layers.10': [torch.float32],
            'base.vgg_base.layers.11': [torch.float32],
            'base.vgg_base.layers.12': [torch.float32],
            'base.vgg_base.layers.13': [torch.float32],
            'base.vgg_base.layers.14': [torch.float32],
            'base.vgg_base.layers.15': [torch.float32],
            'base.vgg_base.layers.16': [torch.float32],
            'base.vgg_base.layers.17': [torch.float32],
            'base.vgg_base.layers.18': [torch.float32],
            'base.vgg_base.layers.19': [torch.float32],
            'base.vgg_base.layers.20': [torch.float32],
            'base.vgg_base.layers.21': [torch.float32],
            'base.vgg_base.layers.22': [torch.float32],
            'base.conv4_3_CPM': [torch.float32],
            'base.conv4_4_CPM': [torch.float32],
            'base.relu': [torch.float32],
            'stage_1.conv1_CPM_L1': [torch.float32],
            'stage_1.conv2_CPM_L1': [torch.float32],
            'stage_1.conv3_CPM_L1': [torch.float32],
            'stage_1.conv4_CPM_L1': [torch.float32],
            'stage_1.conv5_CPM_L1': [torch.float32],
            'stage_1.conv1_CPM_L2': [torch.float32],
            'stage_1.conv2_CPM_L2': [torch.float32],
            'stage_1.conv3_CPM_L2': [torch.float32],
            'stage_1.conv4_CPM_L2': [torch.float32],
            'stage_1.conv5_CPM_L2': [torch.float32],
            'stage_1.relu': [torch.float32],
            'stage_2.conv1_L1': [torch.float32],
            'stage_2.conv2_L1': [torch.float32],
            'stage_2.conv3_L1': [torch.float32],
            'stage_2.conv4_L1': [torch.float32],
            'stage_2.conv5_L1': [torch.float32],
            'stage_2.conv6_L1': [torch.float32],
            'stage_2.conv7_L1': [torch.float32],
            'stage_2.conv1_L2': [torch.float32],
            'stage_2.conv2_L2': [torch.float32],
            'stage_2.conv3_L2': [torch.float32],
            'stage_2.conv4_L2': [torch.float32],
            'stage_2.conv5_L2': [torch.float32],
            'stage_2.conv6_L2': [torch.float32],
            'stage_2.conv7_L2': [torch.float32],
            'stage_2.relu': [torch.float32],
            'stage_3.conv1_L1': [torch.float32],
            'stage_3.conv2_L1': [torch.float32],
            'stage_3.conv3_L1': [torch.float32],
            'stage_3.conv4_L1': [torch.float32],
            'stage_3.conv5_L1': [torch.float32],
            'stage_3.conv6_L1': [torch.float32],
            'stage_3.conv7_L1': [torch.float32],
            'stage_3.conv1_L2': [torch.float32],
            'stage_3.conv2_L2': [torch.float32],
            'stage_3.conv3_L2': [torch.float32],
            'stage_3.conv4_L2': [torch.float32],
            'stage_3.conv5_L2': [torch.float32],
            'stage_3.conv6_L2': [torch.float32],
            'stage_3.conv7_L2': [torch.float32],
            'stage_3.relu': [torch.float32],
            'stage_4.conv1_L1': [torch.float32],
            'stage_4.conv2_L1': [torch.float32],
            'stage_4.conv3_L1': [torch.float32],
            'stage_4.conv4_L1': [torch.float32],
            'stage_4.conv5_L1': [torch.float32],
            'stage_4.conv6_L1': [torch.float32],
            'stage_4.conv7_L1': [torch.float32],
            'stage_4.conv1_L2': [torch.float32],
            'stage_4.conv2_L2': [torch.float32],
            'stage_4.conv3_L2': [torch.float32],
            'stage_4.conv4_L2': [torch.float32],
            'stage_4.conv5_L2': [torch.float32],
            'stage_4.conv6_L2': [torch.float32],
            'stage_4.conv7_L2': [torch.float32],
            'stage_4.relu': [torch.float32],
            'stage_5.conv1_L1': [torch.float32],
            'stage_5.conv2_L1': [torch.float32],
            'stage_5.conv3_L1': [torch.float32],
            'stage_5.conv4_L1': [torch.float32],
            'stage_5.conv5_L1': [torch.float32],
            'stage_5.conv6_L1': [torch.float32],
            'stage_5.conv7_L1': [torch.float32],
            'stage_5.conv1_L2': [torch.float32],
            'stage_5.conv2_L2': [torch.float32],
            'stage_5.conv3_L2': [torch.float32],
            'stage_5.conv4_L2': [torch.float32],
            'stage_5.conv5_L2': [torch.float32],
            'stage_5.conv6_L2': [torch.float32],
            'stage_5.conv7_L2': [torch.float32],
            'stage_5.relu': [torch.float32],
            'stage_6.conv1_L1': [torch.float32],
            'stage_6.conv2_L1': [torch.float32],
            'stage_6.conv3_L1': [torch.float32],
            'stage_6.conv4_L1': [torch.float32],
            'stage_6.conv5_L1': [torch.float32],
            'stage_6.conv6_L1': [torch.float32],
            'stage_6.conv7_L1': [torch.float32],
            'stage_6.conv1_L2': [torch.float32],
            'stage_6.conv2_L2': [torch.float32],
            'stage_6.conv3_L2': [torch.float32],
            'stage_6.conv4_L2': [torch.float32],
            'stage_6.conv5_L2': [torch.float32],
            'stage_6.conv6_L2': [torch.float32],
            'stage_6.conv7_L2': [torch.float32],
            'stage_6.relu': [torch.float32]
        }

        self.layer_names = {
            "base": self.base,
            "base.vgg_base": self.base.vgg_base,
            "base.vgg_base.layers": self.base.vgg_base.layers,
            "base.vgg_base.layers.0": self.base.vgg_base.layers[0],
            "base.vgg_base.layers.1": self.base.vgg_base.layers[1],
            "base.vgg_base.layers.2": self.base.vgg_base.layers[2],
            "base.vgg_base.layers.3": self.base.vgg_base.layers[3],
            "base.vgg_base.layers.4": self.base.vgg_base.layers[4],
            "base.vgg_base.layers.5": self.base.vgg_base.layers[5],
            "base.vgg_base.layers.6": self.base.vgg_base.layers[6],
            "base.vgg_base.layers.7": self.base.vgg_base.layers[7],
            "base.vgg_base.layers.8": self.base.vgg_base.layers[8],
            "base.vgg_base.layers.9": self.base.vgg_base.layers[9],
            "base.vgg_base.layers.10": self.base.vgg_base.layers[10],
            "base.vgg_base.layers.11": self.base.vgg_base.layers[11],
            "base.vgg_base.layers.12": self.base.vgg_base.layers[12],
            "base.vgg_base.layers.13": self.base.vgg_base.layers[13],
            "base.vgg_base.layers.14": self.base.vgg_base.layers[14],
            "base.vgg_base.layers.15": self.base.vgg_base.layers[15],
            "base.vgg_base.layers.16": self.base.vgg_base.layers[16],
            "base.vgg_base.layers.17": self.base.vgg_base.layers[17],
            "base.vgg_base.layers.18": self.base.vgg_base.layers[18],
            "base.vgg_base.layers.19": self.base.vgg_base.layers[19],
            "base.vgg_base.layers.20": self.base.vgg_base.layers[20],
            "base.vgg_base.layers.21": self.base.vgg_base.layers[21],
            "base.vgg_base.layers.22": self.base.vgg_base.layers[22],
            "base.conv4_3_CPM": self.base.conv4_3_CPM,
            "base.conv4_4_CPM": self.base.conv4_4_CPM,
            "base.relu": self.base.relu,
            "stage_1": self.stage_1,
            "stage_1.conv1_CPM_L1": self.stage_1.conv1_CPM_L1,
            "stage_1.conv2_CPM_L1": self.stage_1.conv2_CPM_L1,
            "stage_1.conv3_CPM_L1": self.stage_1.conv3_CPM_L1,
            "stage_1.conv4_CPM_L1": self.stage_1.conv4_CPM_L1,
            "stage_1.conv5_CPM_L1": self.stage_1.conv5_CPM_L1,
            "stage_1.conv1_CPM_L2": self.stage_1.conv1_CPM_L2,
            "stage_1.conv2_CPM_L2": self.stage_1.conv2_CPM_L2,
            "stage_1.conv3_CPM_L2": self.stage_1.conv3_CPM_L2,
            "stage_1.conv4_CPM_L2": self.stage_1.conv4_CPM_L2,
            "stage_1.conv5_CPM_L2": self.stage_1.conv5_CPM_L2,
            "stage_1.relu": self.stage_1.relu,
            "stage_2": self.stage_2,
            "stage_2.conv1_L1": self.stage_2.conv1_L1,
            "stage_2.conv2_L1": self.stage_2.conv2_L1,
            "stage_2.conv3_L1": self.stage_2.conv3_L1,
            "stage_2.conv4_L1": self.stage_2.conv4_L1,
            "stage_2.conv5_L1": self.stage_2.conv5_L1,
            "stage_2.conv6_L1": self.stage_2.conv6_L1,
            "stage_2.conv7_L1": self.stage_2.conv7_L1,
            "stage_2.conv1_L2": self.stage_2.conv1_L2,
            "stage_2.conv2_L2": self.stage_2.conv2_L2,
            "stage_2.conv3_L2": self.stage_2.conv3_L2,
            "stage_2.conv4_L2": self.stage_2.conv4_L2,
            "stage_2.conv5_L2": self.stage_2.conv5_L2,
            "stage_2.conv6_L2": self.stage_2.conv6_L2,
            "stage_2.conv7_L2": self.stage_2.conv7_L2,
            "stage_2.relu": self.stage_2.relu,
            "stage_3": self.stage_3,
            "stage_3.conv1_L1": self.stage_3.conv1_L1,
            "stage_3.conv2_L1": self.stage_3.conv2_L1,
            "stage_3.conv3_L1": self.stage_3.conv3_L1,
            "stage_3.conv4_L1": self.stage_3.conv4_L1,
            "stage_3.conv5_L1": self.stage_3.conv5_L1,
            "stage_3.conv6_L1": self.stage_3.conv6_L1,
            "stage_3.conv7_L1": self.stage_3.conv7_L1,
            "stage_3.conv1_L2": self.stage_3.conv1_L2,
            "stage_3.conv2_L2": self.stage_3.conv2_L2,
            "stage_3.conv3_L2": self.stage_3.conv3_L2,
            "stage_3.conv4_L2": self.stage_3.conv4_L2,
            "stage_3.conv5_L2": self.stage_3.conv5_L2,
            "stage_3.conv6_L2": self.stage_3.conv6_L2,
            "stage_3.conv7_L2": self.stage_3.conv7_L2,
            "stage_3.relu": self.stage_3.relu,
            "stage_4": self.stage_4,
            "stage_4.conv1_L1": self.stage_4.conv1_L1,
            "stage_4.conv2_L1": self.stage_4.conv2_L1,
            "stage_4.conv3_L1": self.stage_4.conv3_L1,
            "stage_4.conv4_L1": self.stage_4.conv4_L1,
            "stage_4.conv5_L1": self.stage_4.conv5_L1,
            "stage_4.conv6_L1": self.stage_4.conv6_L1,
            "stage_4.conv7_L1": self.stage_4.conv7_L1,
            "stage_4.conv1_L2": self.stage_4.conv1_L2,
            "stage_4.conv2_L2": self.stage_4.conv2_L2,
            "stage_4.conv3_L2": self.stage_4.conv3_L2,
            "stage_4.conv4_L2": self.stage_4.conv4_L2,
            "stage_4.conv5_L2": self.stage_4.conv5_L2,
            "stage_4.conv6_L2": self.stage_4.conv6_L2,
            "stage_4.conv7_L2": self.stage_4.conv7_L2,
            "stage_4.relu": self.stage_4.relu,
            "stage_5": self.stage_5,
            "stage_5.conv1_L1": self.stage_5.conv1_L1,
            "stage_5.conv2_L1": self.stage_5.conv2_L1,
            "stage_5.conv3_L1": self.stage_5.conv3_L1,
            "stage_5.conv4_L1": self.stage_5.conv4_L1,
            "stage_5.conv5_L1": self.stage_5.conv5_L1,
            "stage_5.conv6_L1": self.stage_5.conv6_L1,
            "stage_5.conv7_L1": self.stage_5.conv7_L1,
            "stage_5.conv1_L2": self.stage_5.conv1_L2,
            "stage_5.conv2_L2": self.stage_5.conv2_L2,
            "stage_5.conv3_L2": self.stage_5.conv3_L2,
            "stage_5.conv4_L2": self.stage_5.conv4_L2,
            "stage_5.conv5_L2": self.stage_5.conv5_L2,
            "stage_5.conv6_L2": self.stage_5.conv6_L2,
            "stage_5.conv7_L2": self.stage_5.conv7_L2,
            "stage_5.relu": self.stage_5.relu,
            "stage_6": self.stage_6,
            "stage_6.conv1_L1": self.stage_6.conv1_L1,
            "stage_6.conv2_L1": self.stage_6.conv2_L1,
            "stage_6.conv3_L1": self.stage_6.conv3_L1,
            "stage_6.conv4_L1": self.stage_6.conv4_L1,
            "stage_6.conv5_L1": self.stage_6.conv5_L1,
            "stage_6.conv6_L1": self.stage_6.conv6_L1,
            "stage_6.conv7_L1": self.stage_6.conv7_L1,
            "stage_6.conv1_L2": self.stage_6.conv1_L2,
            "stage_6.conv2_L2": self.stage_6.conv2_L2,
            "stage_6.conv3_L2": self.stage_6.conv3_L2,
            "stage_6.conv4_L2": self.stage_6.conv4_L2,
            "stage_6.conv5_L2": self.stage_6.conv5_L2,
            "stage_6.conv6_L2": self.stage_6.conv6_L2,
            "stage_6.conv7_L2": self.stage_6.conv7_L2,
            "stage_6.relu": self.stage_6.relu,
        }

    def forward(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(torch.cat((h1, h2, feature_map), dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(torch.cat((h1, h2, feature_map), dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(torch.cat((h1, h2, feature_map), dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(torch.cat((h1, h2, feature_map), dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(torch.cat((h1, h2, feature_map), dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps

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

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self,layer_name,new_layer):
        if 'base' == layer_name:
            self.base= new_layer
            self.layer_names["base"]=new_layer

        elif 'base.vgg_base' == layer_name:
            self.base.vgg_base= new_layer
            self.layer_names["base.vgg_base"]=new_layer

        elif 'base.vgg_base.layers' == layer_name:
            self.base.vgg_base.layers= new_layer
            self.layer_names["base.vgg_base.layers"]=new_layer

        elif 'base.vgg_base.layers.0' == layer_name:
            self.base.vgg_base.layers[0]= new_layer
            self.layer_names["base.vgg_base.layers.0"]=new_layer

        elif 'base.vgg_base.layers.1' == layer_name:
            self.base.vgg_base.layers[1]= new_layer
            self.layer_names["base.vgg_base.layers.1"]=new_layer

        elif 'base.vgg_base.layers.2' == layer_name:
            self.base.vgg_base.layers[2]= new_layer
            self.layer_names["base.vgg_base.layers.2"]=new_layer

        elif 'base.vgg_base.layers.3' == layer_name:
            self.base.vgg_base.layers[3]= new_layer
            self.layer_names["base.vgg_base.layers.3"]=new_layer

        elif 'base.vgg_base.layers.4' == layer_name:
            self.base.vgg_base.layers[4]= new_layer
            self.layer_names["base.vgg_base.layers.4"]=new_layer

        elif 'base.vgg_base.layers.5' == layer_name:
            self.base.vgg_base.layers[5]= new_layer
            self.layer_names["base.vgg_base.layers.5"]=new_layer

        elif 'base.vgg_base.layers.6' == layer_name:
            self.base.vgg_base.layers[6]= new_layer
            self.layer_names["base.vgg_base.layers.6"]=new_layer

        elif 'base.vgg_base.layers.7' == layer_name:
            self.base.vgg_base.layers[7]= new_layer
            self.layer_names["base.vgg_base.layers.7"]=new_layer

        elif 'base.vgg_base.layers.8' == layer_name:
            self.base.vgg_base.layers[8]= new_layer
            self.layer_names["base.vgg_base.layers.8"]=new_layer

        elif 'base.vgg_base.layers.9' == layer_name:
            self.base.vgg_base.layers[9]= new_layer
            self.layer_names["base.vgg_base.layers.9"]=new_layer

        elif 'base.vgg_base.layers.10' == layer_name:
            self.base.vgg_base.layers[10]= new_layer
            self.layer_names["base.vgg_base.layers.10"]=new_layer

        elif 'base.vgg_base.layers.11' == layer_name:
            self.base.vgg_base.layers[11]= new_layer
            self.layer_names["base.vgg_base.layers.11"]=new_layer

        elif 'base.vgg_base.layers.12' == layer_name:
            self.base.vgg_base.layers[12]= new_layer
            self.layer_names["base.vgg_base.layers.12"]=new_layer

        elif 'base.vgg_base.layers.13' == layer_name:
            self.base.vgg_base.layers[13]= new_layer
            self.layer_names["base.vgg_base.layers.13"]=new_layer

        elif 'base.vgg_base.layers.14' == layer_name:
            self.base.vgg_base.layers[14]= new_layer
            self.layer_names["base.vgg_base.layers.14"]=new_layer

        elif 'base.vgg_base.layers.15' == layer_name:
            self.base.vgg_base.layers[15]= new_layer
            self.layer_names["base.vgg_base.layers.15"]=new_layer

        elif 'base.vgg_base.layers.16' == layer_name:
            self.base.vgg_base.layers[16]= new_layer
            self.layer_names["base.vgg_base.layers.16"]=new_layer

        elif 'base.vgg_base.layers.17' == layer_name:
            self.base.vgg_base.layers[17]= new_layer
            self.layer_names["base.vgg_base.layers.17"]=new_layer

        elif 'base.vgg_base.layers.18' == layer_name:
            self.base.vgg_base.layers[18]= new_layer
            self.layer_names["base.vgg_base.layers.18"]=new_layer

        elif 'base.vgg_base.layers.19' == layer_name:
            self.base.vgg_base.layers[19]= new_layer
            self.layer_names["base.vgg_base.layers.19"]=new_layer

        elif 'base.vgg_base.layers.20' == layer_name:
            self.base.vgg_base.layers[20]= new_layer
            self.layer_names["base.vgg_base.layers.20"]=new_layer

        elif 'base.vgg_base.layers.21' == layer_name:
            self.base.vgg_base.layers[21]= new_layer
            self.layer_names["base.vgg_base.layers.21"]=new_layer

        elif 'base.vgg_base.layers.22' == layer_name:
            self.base.vgg_base.layers[22]= new_layer
            self.layer_names["base.vgg_base.layers.22"]=new_layer

        elif 'base.conv4_3_CPM' == layer_name:
            self.base.conv4_3_CPM= new_layer
            self.layer_names["base.conv4_3_CPM"]=new_layer

        elif 'base.conv4_4_CPM' == layer_name:
            self.base.conv4_4_CPM= new_layer
            self.layer_names["base.conv4_4_CPM"]=new_layer

        elif 'base.relu' == layer_name:
            self.base.relu= new_layer
            self.layer_names["base.relu"]=new_layer

        elif 'stage_1' == layer_name:
            self.stage_1= new_layer
            self.layer_names["stage_1"]=new_layer

        elif 'stage_1.conv1_CPM_L1' == layer_name:
            self.stage_1.conv1_CPM_L1= new_layer
            self.layer_names["stage_1.conv1_CPM_L1"]=new_layer

        elif 'stage_1.conv2_CPM_L1' == layer_name:
            self.stage_1.conv2_CPM_L1= new_layer
            self.layer_names["stage_1.conv2_CPM_L1"]=new_layer

        elif 'stage_1.conv3_CPM_L1' == layer_name:
            self.stage_1.conv3_CPM_L1= new_layer
            self.layer_names["stage_1.conv3_CPM_L1"]=new_layer

        elif 'stage_1.conv4_CPM_L1' == layer_name:
            self.stage_1.conv4_CPM_L1= new_layer
            self.layer_names["stage_1.conv4_CPM_L1"]=new_layer

        elif 'stage_1.conv5_CPM_L1' == layer_name:
            self.stage_1.conv5_CPM_L1= new_layer
            self.layer_names["stage_1.conv5_CPM_L1"]=new_layer

        elif 'stage_1.conv1_CPM_L2' == layer_name:
            self.stage_1.conv1_CPM_L2= new_layer
            self.layer_names["stage_1.conv1_CPM_L2"]=new_layer

        elif 'stage_1.conv2_CPM_L2' == layer_name:
            self.stage_1.conv2_CPM_L2= new_layer
            self.layer_names["stage_1.conv2_CPM_L2"]=new_layer

        elif 'stage_1.conv3_CPM_L2' == layer_name:
            self.stage_1.conv3_CPM_L2= new_layer
            self.layer_names["stage_1.conv3_CPM_L2"]=new_layer

        elif 'stage_1.conv4_CPM_L2' == layer_name:
            self.stage_1.conv4_CPM_L2= new_layer
            self.layer_names["stage_1.conv4_CPM_L2"]=new_layer

        elif 'stage_1.conv5_CPM_L2' == layer_name:
            self.stage_1.conv5_CPM_L2= new_layer
            self.layer_names["stage_1.conv5_CPM_L2"]=new_layer

        elif 'stage_1.relu' == layer_name:
            self.stage_1.relu= new_layer
            self.layer_names["stage_1.relu"]=new_layer

        elif 'stage_2' == layer_name:
            self.stage_2= new_layer
            self.layer_names["stage_2"]=new_layer

        elif 'stage_2.conv1_L1' == layer_name:
            self.stage_2.conv1_L1= new_layer
            self.layer_names["stage_2.conv1_L1"]=new_layer

        elif 'stage_2.conv2_L1' == layer_name:
            self.stage_2.conv2_L1= new_layer
            self.layer_names["stage_2.conv2_L1"]=new_layer

        elif 'stage_2.conv3_L1' == layer_name:
            self.stage_2.conv3_L1= new_layer
            self.layer_names["stage_2.conv3_L1"]=new_layer

        elif 'stage_2.conv4_L1' == layer_name:
            self.stage_2.conv4_L1= new_layer
            self.layer_names["stage_2.conv4_L1"]=new_layer

        elif 'stage_2.conv5_L1' == layer_name:
            self.stage_2.conv5_L1= new_layer
            self.layer_names["stage_2.conv5_L1"]=new_layer

        elif 'stage_2.conv6_L1' == layer_name:
            self.stage_2.conv6_L1= new_layer
            self.layer_names["stage_2.conv6_L1"]=new_layer

        elif 'stage_2.conv7_L1' == layer_name:
            self.stage_2.conv7_L1= new_layer
            self.layer_names["stage_2.conv7_L1"]=new_layer

        elif 'stage_2.conv1_L2' == layer_name:
            self.stage_2.conv1_L2= new_layer
            self.layer_names["stage_2.conv1_L2"]=new_layer

        elif 'stage_2.conv2_L2' == layer_name:
            self.stage_2.conv2_L2= new_layer
            self.layer_names["stage_2.conv2_L2"]=new_layer

        elif 'stage_2.conv3_L2' == layer_name:
            self.stage_2.conv3_L2= new_layer
            self.layer_names["stage_2.conv3_L2"]=new_layer

        elif 'stage_2.conv4_L2' == layer_name:
            self.stage_2.conv4_L2= new_layer
            self.layer_names["stage_2.conv4_L2"]=new_layer

        elif 'stage_2.conv5_L2' == layer_name:
            self.stage_2.conv5_L2= new_layer
            self.layer_names["stage_2.conv5_L2"]=new_layer

        elif 'stage_2.conv6_L2' == layer_name:
            self.stage_2.conv6_L2= new_layer
            self.layer_names["stage_2.conv6_L2"]=new_layer

        elif 'stage_2.conv7_L2' == layer_name:
            self.stage_2.conv7_L2= new_layer
            self.layer_names["stage_2.conv7_L2"]=new_layer

        elif 'stage_2.relu' == layer_name:
            self.stage_2.relu= new_layer
            self.layer_names["stage_2.relu"]=new_layer

        elif 'stage_3' == layer_name:
            self.stage_3= new_layer
            self.layer_names["stage_3"]=new_layer

        elif 'stage_3.conv1_L1' == layer_name:
            self.stage_3.conv1_L1= new_layer
            self.layer_names["stage_3.conv1_L1"]=new_layer

        elif 'stage_3.conv2_L1' == layer_name:
            self.stage_3.conv2_L1= new_layer
            self.layer_names["stage_3.conv2_L1"]=new_layer

        elif 'stage_3.conv3_L1' == layer_name:
            self.stage_3.conv3_L1= new_layer
            self.layer_names["stage_3.conv3_L1"]=new_layer

        elif 'stage_3.conv4_L1' == layer_name:
            self.stage_3.conv4_L1= new_layer
            self.layer_names["stage_3.conv4_L1"]=new_layer

        elif 'stage_3.conv5_L1' == layer_name:
            self.stage_3.conv5_L1= new_layer
            self.layer_names["stage_3.conv5_L1"]=new_layer

        elif 'stage_3.conv6_L1' == layer_name:
            self.stage_3.conv6_L1= new_layer
            self.layer_names["stage_3.conv6_L1"]=new_layer

        elif 'stage_3.conv7_L1' == layer_name:
            self.stage_3.conv7_L1= new_layer
            self.layer_names["stage_3.conv7_L1"]=new_layer

        elif 'stage_3.conv1_L2' == layer_name:
            self.stage_3.conv1_L2= new_layer
            self.layer_names["stage_3.conv1_L2"]=new_layer

        elif 'stage_3.conv2_L2' == layer_name:
            self.stage_3.conv2_L2= new_layer
            self.layer_names["stage_3.conv2_L2"]=new_layer

        elif 'stage_3.conv3_L2' == layer_name:
            self.stage_3.conv3_L2= new_layer
            self.layer_names["stage_3.conv3_L2"]=new_layer

        elif 'stage_3.conv4_L2' == layer_name:
            self.stage_3.conv4_L2= new_layer
            self.layer_names["stage_3.conv4_L2"]=new_layer

        elif 'stage_3.conv5_L2' == layer_name:
            self.stage_3.conv5_L2= new_layer
            self.layer_names["stage_3.conv5_L2"]=new_layer

        elif 'stage_3.conv6_L2' == layer_name:
            self.stage_3.conv6_L2= new_layer
            self.layer_names["stage_3.conv6_L2"]=new_layer

        elif 'stage_3.conv7_L2' == layer_name:
            self.stage_3.conv7_L2= new_layer
            self.layer_names["stage_3.conv7_L2"]=new_layer

        elif 'stage_3.relu' == layer_name:
            self.stage_3.relu= new_layer
            self.layer_names["stage_3.relu"]=new_layer

        elif 'stage_4' == layer_name:
            self.stage_4= new_layer
            self.layer_names["stage_4"]=new_layer

        elif 'stage_4.conv1_L1' == layer_name:
            self.stage_4.conv1_L1= new_layer
            self.layer_names["stage_4.conv1_L1"]=new_layer

        elif 'stage_4.conv2_L1' == layer_name:
            self.stage_4.conv2_L1= new_layer
            self.layer_names["stage_4.conv2_L1"]=new_layer

        elif 'stage_4.conv3_L1' == layer_name:
            self.stage_4.conv3_L1= new_layer
            self.layer_names["stage_4.conv3_L1"]=new_layer

        elif 'stage_4.conv4_L1' == layer_name:
            self.stage_4.conv4_L1= new_layer
            self.layer_names["stage_4.conv4_L1"]=new_layer

        elif 'stage_4.conv5_L1' == layer_name:
            self.stage_4.conv5_L1= new_layer
            self.layer_names["stage_4.conv5_L1"]=new_layer

        elif 'stage_4.conv6_L1' == layer_name:
            self.stage_4.conv6_L1= new_layer
            self.layer_names["stage_4.conv6_L1"]=new_layer

        elif 'stage_4.conv7_L1' == layer_name:
            self.stage_4.conv7_L1= new_layer
            self.layer_names["stage_4.conv7_L1"]=new_layer

        elif 'stage_4.conv1_L2' == layer_name:
            self.stage_4.conv1_L2= new_layer
            self.layer_names["stage_4.conv1_L2"]=new_layer

        elif 'stage_4.conv2_L2' == layer_name:
            self.stage_4.conv2_L2= new_layer
            self.layer_names["stage_4.conv2_L2"]=new_layer

        elif 'stage_4.conv3_L2' == layer_name:
            self.stage_4.conv3_L2= new_layer
            self.layer_names["stage_4.conv3_L2"]=new_layer

        elif 'stage_4.conv4_L2' == layer_name:
            self.stage_4.conv4_L2= new_layer
            self.layer_names["stage_4.conv4_L2"]=new_layer

        elif 'stage_4.conv5_L2' == layer_name:
            self.stage_4.conv5_L2= new_layer
            self.layer_names["stage_4.conv5_L2"]=new_layer

        elif 'stage_4.conv6_L2' == layer_name:
            self.stage_4.conv6_L2= new_layer
            self.layer_names["stage_4.conv6_L2"]=new_layer

        elif 'stage_4.conv7_L2' == layer_name:
            self.stage_4.conv7_L2= new_layer
            self.layer_names["stage_4.conv7_L2"]=new_layer

        elif 'stage_4.relu' == layer_name:
            self.stage_4.relu= new_layer
            self.layer_names["stage_4.relu"]=new_layer

        elif 'stage_5' == layer_name:
            self.stage_5= new_layer
            self.layer_names["stage_5"]=new_layer

        elif 'stage_5.conv1_L1' == layer_name:
            self.stage_5.conv1_L1= new_layer
            self.layer_names["stage_5.conv1_L1"]=new_layer

        elif 'stage_5.conv2_L1' == layer_name:
            self.stage_5.conv2_L1= new_layer
            self.layer_names["stage_5.conv2_L1"]=new_layer

        elif 'stage_5.conv3_L1' == layer_name:
            self.stage_5.conv3_L1= new_layer
            self.layer_names["stage_5.conv3_L1"]=new_layer

        elif 'stage_5.conv4_L1' == layer_name:
            self.stage_5.conv4_L1= new_layer
            self.layer_names["stage_5.conv4_L1"]=new_layer

        elif 'stage_5.conv5_L1' == layer_name:
            self.stage_5.conv5_L1= new_layer
            self.layer_names["stage_5.conv5_L1"]=new_layer

        elif 'stage_5.conv6_L1' == layer_name:
            self.stage_5.conv6_L1= new_layer
            self.layer_names["stage_5.conv6_L1"]=new_layer

        elif 'stage_5.conv7_L1' == layer_name:
            self.stage_5.conv7_L1= new_layer
            self.layer_names["stage_5.conv7_L1"]=new_layer

        elif 'stage_5.conv1_L2' == layer_name:
            self.stage_5.conv1_L2= new_layer
            self.layer_names["stage_5.conv1_L2"]=new_layer

        elif 'stage_5.conv2_L2' == layer_name:
            self.stage_5.conv2_L2= new_layer
            self.layer_names["stage_5.conv2_L2"]=new_layer

        elif 'stage_5.conv3_L2' == layer_name:
            self.stage_5.conv3_L2= new_layer
            self.layer_names["stage_5.conv3_L2"]=new_layer

        elif 'stage_5.conv4_L2' == layer_name:
            self.stage_5.conv4_L2= new_layer
            self.layer_names["stage_5.conv4_L2"]=new_layer

        elif 'stage_5.conv5_L2' == layer_name:
            self.stage_5.conv5_L2= new_layer
            self.layer_names["stage_5.conv5_L2"]=new_layer

        elif 'stage_5.conv6_L2' == layer_name:
            self.stage_5.conv6_L2= new_layer
            self.layer_names["stage_5.conv6_L2"]=new_layer

        elif 'stage_5.conv7_L2' == layer_name:
            self.stage_5.conv7_L2= new_layer
            self.layer_names["stage_5.conv7_L2"]=new_layer

        elif 'stage_5.relu' == layer_name:
            self.stage_5.relu= new_layer
            self.layer_names["stage_5.relu"]=new_layer

        elif 'stage_6' == layer_name:
            self.stage_6= new_layer
            self.layer_names["stage_6"]=new_layer

        elif 'stage_6.conv1_L1' == layer_name:
            self.stage_6.conv1_L1= new_layer
            self.layer_names["stage_6.conv1_L1"]=new_layer

        elif 'stage_6.conv2_L1' == layer_name:
            self.stage_6.conv2_L1= new_layer
            self.layer_names["stage_6.conv2_L1"]=new_layer

        elif 'stage_6.conv3_L1' == layer_name:
            self.stage_6.conv3_L1= new_layer
            self.layer_names["stage_6.conv3_L1"]=new_layer

        elif 'stage_6.conv4_L1' == layer_name:
            self.stage_6.conv4_L1= new_layer
            self.layer_names["stage_6.conv4_L1"]=new_layer

        elif 'stage_6.conv5_L1' == layer_name:
            self.stage_6.conv5_L1= new_layer
            self.layer_names["stage_6.conv5_L1"]=new_layer

        elif 'stage_6.conv6_L1' == layer_name:
            self.stage_6.conv6_L1= new_layer
            self.layer_names["stage_6.conv6_L1"]=new_layer

        elif 'stage_6.conv7_L1' == layer_name:
            self.stage_6.conv7_L1= new_layer
            self.layer_names["stage_6.conv7_L1"]=new_layer

        elif 'stage_6.conv1_L2' == layer_name:
            self.stage_6.conv1_L2= new_layer
            self.layer_names["stage_6.conv1_L2"]=new_layer

        elif 'stage_6.conv2_L2' == layer_name:
            self.stage_6.conv2_L2= new_layer
            self.layer_names["stage_6.conv2_L2"]=new_layer

        elif 'stage_6.conv3_L2' == layer_name:
            self.stage_6.conv3_L2= new_layer
            self.layer_names["stage_6.conv3_L2"]=new_layer

        elif 'stage_6.conv4_L2' == layer_name:
            self.stage_6.conv4_L2= new_layer
            self.layer_names["stage_6.conv4_L2"]=new_layer

        elif 'stage_6.conv5_L2' == layer_name:
            self.stage_6.conv5_L2= new_layer
            self.layer_names["stage_6.conv5_L2"]=new_layer

        elif 'stage_6.conv6_L2' == layer_name:
            self.stage_6.conv6_L2= new_layer
            self.layer_names["stage_6.conv6_L2"]=new_layer

        elif 'stage_6.conv7_L2' == layer_name:
            self.stage_6.conv7_L2= new_layer
            self.layer_names["stage_6.conv7_L2"]=new_layer

        elif 'stage_6.relu' == layer_name:
            self.stage_6.relu= new_layer
            self.layer_names["stage_6.relu"]=new_layer



class Base_model(nn.Module):
    def __init__(self, vgg_with_bn=False):
        super(Base_model, self).__init__()
        # Initializing Vgg (actual implementation will be translated later)
        cfgs_zh = {'19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]}
        self.vgg_base = Vgg(cfgs_zh['19'], batch_norm=vgg_with_bn)
        # Initializing other layers
        self.conv4_3_CPM = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_4_CPM = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x


class Vgg(nn.Module):

    def __init__(self, cfg, batch_norm=False):
        super(Vgg, self).__init__()
        self.layers = self._make_layer(cfg, batch_norm=batch_norm)

    def forward(self, x):
        x = self.layers(x)
        return x

    def _make_layer(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=False))
                in_channels = v
        return nn.Sequential(*layers)


class Stage_1(nn.Module):

    def __init__(self):
        super(Stage_1, self).__init__()
        # Initializing convolutional layers
        self.conv1_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = nn.Conv2d(in_channels=512, out_channels=38, kernel_size=1, stride=1, padding=0)
        self.conv1_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L2 = nn.Conv2d(in_channels=512, out_channels=19, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x))
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv1_CPM_L2(x))
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2


class Stage_x(nn.Module):

    def __init__(self):
        super(Stage_x, self).__init__()
        # Initializing convolutional layers for L1
        self.conv1_L1 = nn.Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv2_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv3_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv4_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv5_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv6_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7_L1 = nn.Conv2d(in_channels=128, out_channels=38, kernel_size=1, stride=1, padding=0)

        # Initializing convolutional layers for L2
        self.conv1_L2 = nn.Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv2_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv3_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv4_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv5_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv6_L2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7_L2 = nn.Conv2d(in_channels=128, out_channels=19, kernel_size=1, stride=1, padding=0)

        # ReLU activation layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Processing through the L1 branch
        h1 = self.relu(self.conv1_L1(x))
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)

        # Processing through the L2 branch
        h2 = self.relu(self.conv1_L2(x))
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)

        return h1, h2


def compare_ms_torch():
    device = "CPU"
    network_torch = OpenPoseNet(vggpath=config.vgg_path, vgg_with_bn=config.vgg_with_bn).to(device)
    network_torch.eval()
    weight_dict = torch.load('vgg19-0-97_5004.pth')
    network_torch.load_state_dict(weight_dict)
    from src.openposenet import OpenPoseNet as OpenPoseNet_ms
    network_ms = OpenPoseNet_ms(vggpath=config.vgg_path, vgg_with_bn=config.vgg_with_bn)
    network_ms.set_train(False)
    mindspore.load_checkpoint("vgg19-0-97_5004.ckpt", network_ms)
    aa = np.ones([1, 3, 368, 368])
    a = torch.tensor(aa, dtype=torch.float32).to(device)
    anp = (a,)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=network_torch, ms_net=network_ms, fix_seed=0,
                                                  auto_conv_ckpt=0)  #
    diff_finder.compare(inputs=anp)
    # mindspore.save_checkpoint(network_ms, "vgg19-0-97_5004.ckpt")\


if __name__ == '__main__':
    #compare_ms_torch()
    device = "cuda:6"
    network = OpenPoseNet(vggpath=config.vgg_path, vgg_with_bn=config.vgg_with_bn).to(device)
    # data0 = torch.randn(1, 3, 368, 368)
    # print(network(data0)[0][0].shape)
    criterion = openpose_loss()
    train_net = BuildTrainNetwork(network, criterion).to(device)
    de_dataset_train = create_dataset(config.jsonpath_train, config.imgpath_train, config.maskpath_train,
                                      batch_size=config.batch_size,
                                      rank=0,
                                      group_size=1,
                                      num_worker=1,
                                      multiprocessing=False,
                                      shuffle=True,
                                      repeat_num=1)
    steps_per_epoch = de_dataset_train.get_dataset_size()
    de_dataset_train = de_dataset_train.create_tuple_iterator(output_numpy=True)
    # vgg19_base_params = list(filter(lambda x: 'base.vgg_base' in x.name, train_net.named_parameters()))
    # base_params = list(filter(lambda x: 'base.conv' in x.name, train_net.named_parameters()))
    # stages_params = list(filter(lambda x: 'base' not in x.name, train_net.named_parameters()))
    # lr_stage, lr_base, lr_vgg = get_lr(config.lr * 1,
    #                                    config.lr_gamma,
    #                                    steps_per_epoch,
    #                                    config.max_epoch,
    #                                    config.lr_steps,
    #                                    1,
    #                                    lr_type=config.lr_type,
    #                                    warmup_epoch=config.warmup_epoch)
    # group_params = [{'params': vgg19_base_params, 'lr': lr_vgg},
    #                 {'params': base_params, 'lr': lr_base},
    #                 {'params': stages_params, 'lr': lr_stage}]
    opt = Adam(network.parameters(), lr=1e-4)
    dataset = valdata(config.ann, config.imgpath_val, 0, 1, mode='val')
    dataset_size = dataset.get_dataset_size()
    de_dataset = dataset.create_tuple_iterator(output_numpy=True)
    print("eval dataset size: ", dataset_size)
    epoch_num = 60
    per_batch = 200
    losses_ms_avg1 = []
    for epoch in range(epoch_num):
        nums = 0
        losses_ms = []
        for data in de_dataset_train:
            opt.zero_grad()
            print("data[0].shape: ", data[0].shape)
            nums += data[0].shape[0]
            loss_ms = train_net(torch.tensor(data[0]).to(device), torch.tensor(data[1]).to(device),
                                torch.tensor(data[2]).to(device), torch.tensor(data[3]).to(device))
            loss_ms.backward()
            opt.step()
            if nums % per_batch == 0:
                print("batch:" + str(nums) + " ms_loss1:" + str(
                    loss_ms.asnumpy()))
            losses_ms.append(loss_ms.detach().cpu().numpy())
            break
        losses_ms_avg1.append(np.mean(losses_ms))
        print("epoch {}: ".format(epoch), " ms_loss1: ",
              str(np.mean(losses_ms)))

        kpt_json = []
        with torch.no_grad():
            for _, (img, img_id) in tqdm(enumerate(de_dataset), total=dataset_size):
                # img = img.detach().cpu().numpy()
                img_id = int(img_id[0])
                poses, scores = detect_torch(img, network, device)
                if poses.shape[0] > 0:
                    for index, pose in enumerate(poses):
                        data = dict()
                        pose = pose[[0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10, 1], :].round().astype('i')
                        keypoints = pose.reshape(-1).tolist()
                        keypoints = keypoints[:-3]
                        data['image_id'] = img_id
                        data['score'] = scores[index]
                        data['category_id'] = 1
                        data['keypoints'] = keypoints
                        kpt_json.append(data)
                else:
                    print("Predict poses size is zero.", flush=True)
                img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)

                save_path = os.path.join(config.output_img_path, str(img_id) + ".png")
                cv2.imwrite(save_path, img)

            result_json = 'eval_result.json'
            with open(os.path.join(config.output_img_path, result_json), 'w') as fid:
                json.dump(kpt_json, fid)
            res = evaluate_mAP(os.path.join(config.output_img_path, result_json), ann_file=config.ann)
            print('result: ', res)
