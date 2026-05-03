import copy
import os
import numpy as np
from mindspore_mutation.rules_ms import rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,\
                                        rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18
import torch
import random
import pennylane as qml
import torch.nn as nn
import scipy.io as scio
import torch.fx as fx
import psutil
import mindspore
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager
import mindspore as ms
## 导入模型

from models.deeplabv3.Deeplabv3 import DeepLabV3
from models.UNet.Unet import UNetMedical, create_Unet_dataset
from models.resnet50.resnet50 import resnet50, create_cifar10_dataset, update_params
from models.vgg11.vgg11 import vgg11
from models.vgg16.src.vgg import vgg16
from models.vgg19.original_vgg19 import vgg19, Config
# from models.yolov3.yolov3 import yolov3
from models.PatchCore.src.model import wide_resnet50_2 as wide_resnet50_2_ms
from models.SSD.backbone_mobilenetv1 import SSDWithMobileNetV1 as SSDmobilenetv1_ms
from models.SSD.backbone_resnet50_fpn import ssd_resnet50fpn_ms as SSDresnet50fpn_ms
from models.deeplabv3.main import SegDataset
from models.openpose.src.model_utils.config import config as openpose_config
from models.openpose.src.openposenet import OpenPoseNet as OpenPoseNet_ms

from models.ssimae.src.network import AutoEncoder as AutoEncoder_ms
from models.ssimae.model_utils.config import config as ssimae_cfg

from models.CRNN.src.model_utils.config import config as crnnconfig
from models.CRNN.src.crnn import CRNNV2 as crnn

from models.textcnn.dataset import MovieReview
from models.textcnn.textcnn import TextCNN
from models.mobilenetv2.mobilenetV2 import mobilenet_v2_ms as mobilenet_v2
from models.vit.src.vit import vit_ms as vit_ms
# from models.yolov4.src.yolo import YOLOV4CspDarkNet53 as yolov4
from models.yolov4.main_new import YOLOV4CspDarkNet53_ms as yolov4
from models.yolov3_darknet53.main_new import YOLOV3DarkNet53 as yolov3
from models.SRGAN.src.model.generator import Generator as srgan



mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

dataset_basic_dir = "/home/cvgroup/myz/czx/SemTest_master/test_data"
datasets_path_cargo = {
    "vgg11": dataset_basic_dir+"/vgg11_data0.npy", #1
    "vgg19": dataset_basic_dir + "/Vgg19_data0.npy",  # 3
    "vgg16": dataset_basic_dir+"/cifar10_x.npy", #1
    "resnet": dataset_basic_dir+"/cifar10_x.npy", #4
    "mobilenetv2":dataset_basic_dir+"/cifar10_x.npy",
    "vit":dataset_basic_dir+"/cifar10_x.npy",
    "yolov3": dataset_basic_dir+"/yolov3_x.npy",  # 5
    "yolov4": dataset_basic_dir + "/yolov4_x.npy",
    "TextCNN": dataset_basic_dir+"/textcnn_x.npy",  # 12
    "SSDresnet50fpn":dataset_basic_dir+"/SSDresnet50fpn_x.npy",# 7
    "SSDmobilenetv1":dataset_basic_dir+"/SSDmobilenetv1_x.npy",# 8
    "unet": dataset_basic_dir+"/unet_x.npy", #9
    "openpose": dataset_basic_dir+"/openpose_x.npy", # 6
    "DeepLabV3": dataset_basic_dir+"/deeplabv3_x.npy", #10
    "ssimae": dataset_basic_dir+"/ssimae_x.npy", #15
    "crnn":dataset_basic_dir+"/CRNN_x.npy",
    "patchcore":dataset_basic_dir+"/patchcore_x.npy",
    "srgan":dataset_basic_dir+"/patchcore_x.npy",
}

labels_path_cargo = {
    "vgg16": dataset_basic_dir+"/cifar10_y.npy", #1
    "resnet": dataset_basic_dir+"/cifar10_y.npy", #4
    "mobilenetv2":dataset_basic_dir+"/cifar10_y.npy",
    "vit":dataset_basic_dir+"/cifar10_y.npy",
    "yolov3": [dataset_basic_dir+"/yolov3_y1.npy",
               dataset_basic_dir+"/yolov3_y2.npy",
               dataset_basic_dir+"/yolov3_y3.npy",
               dataset_basic_dir + "/yolov3_y4.npy",
               dataset_basic_dir + "/yolov3_y5.npy",
               dataset_basic_dir + "/yolov3_y6.npy"],
    "yolov4": [dataset_basic_dir+"/yolov4_y1.npy",
               dataset_basic_dir+"/yolov4_y2.npy",
               dataset_basic_dir+"/yolov4_y3.npy",
               dataset_basic_dir + "/yolov4_y4.npy",
               dataset_basic_dir + "/yolov4_y5.npy",
               dataset_basic_dir + "/yolov4_y6.npy"],
    "TextCNN": dataset_basic_dir+"/textcnn_y.npy",  # 12
    "SSDresnet50fpn": [dataset_basic_dir+"/SSDresnet50fpn_y1.npy",dataset_basic_dir+"/SSDresnet50fpn_y2.npy",dataset_basic_dir+"/SSDresnet50fpn_y3.npy"],
    "SSDmobilenetv1": [dataset_basic_dir+"/SSDmobilenetv1_y1.npy",dataset_basic_dir+"/SSDmobilenetv1_y2.npy",dataset_basic_dir+"/SSDmobilenetv1_y3.npy"],# 8
    "unet": dataset_basic_dir+"/unet_y.npy", #9
    "openpose": [dataset_basic_dir+"/openpose_y1.npy",dataset_basic_dir+"/openpose_y2.npy",dataset_basic_dir+"/openpose_y3.npy"],
    "crnn":dataset_basic_dir+"/CRNN_y.npy",
    "DeepLabV3": dataset_basic_dir+"/deeplabv3_y.npy", #10
}




net_cargo = {
        "vgg19": vgg19,
        "vgg11": vgg11,
        "vgg16": vgg16,
        "resnet": resnet50,
        "vit":vit_ms,
        "mobilenetv2":mobilenet_v2,
        "yolov4":yolov4,
        "unet": UNetMedical,
        "DeepLabV3": DeepLabV3,
        "yolov3": yolov3,
        "TextCNN": TextCNN,
        "openpose": OpenPoseNet_ms,
        "patchcore":wide_resnet50_2_ms,
        "SSDresnet50fpn":SSDresnet50fpn_ms,
        "SSDmobilenetv1":SSDmobilenetv1_ms,
        "crnn":crnn,
        "ssimae":AutoEncoder_ms,
        "gpt2": None,
        "srgan":srgan,

    }


def get_model(model_name):
    if model_name == "vgg11":
        model = net_cargo[model_name]()
    if model_name == "vgg16":
        model = net_cargo[model_name]()
    elif model_name == "vgg19":
        model = net_cargo[model_name](10, args=Config({}))
    elif model_name == "resnet":
        model = net_cargo[model_name]()
    elif model_name == "unet":
        model = net_cargo[model_name](n_channels=1, n_classes=2)
    elif model_name == "DeepLabV3":
        model = net_cargo[model_name]('eval', 21, 8, False)
    elif model_name == "yolov3":
        model = net_cargo[model_name](is_training=True)
    elif model_name == "TextCNN":
        model = net_cargo[model_name](vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
    elif model_name == "openpose":
        model = net_cargo[model_name](vggpath=openpose_config.vgg_path, vgg_with_bn=openpose_config.vgg_with_bn)
    elif model_name == "patchcore":
        model = net_cargo[model_name]()
    elif model_name == "SSDmobilenetv1":
        model = net_cargo[model_name]()
    elif model_name == "SSDresnet50fpn":
        model = net_cargo[model_name]()
    elif model_name == "mobilenetv2":
        model = net_cargo[model_name]()
    elif model_name == "crnn":
        crnnconfig.batch_size = 1  # batch_size
        model = net_cargo[model_name](crnnconfig)
    elif model_name == "ssimae":
        model = net_cargo[model_name](ssimae_cfg)
    elif model_name == "yolov4":
        model = net_cargo[model_name]()
    elif model_name == "vit":
        model = net_cargo[model_name]()
    elif model_name == "albert":
        model = net_cargo[model_name]()
    elif model_name == "gpt2":
        model = net_cargo[model_name]()
    elif model_name == "srgan":
        model = net_cargo[model_name](4)
    return model

def loss_unet_ms():
    from models.UNet.Unet import CrossEntropyWithLogits
    return CrossEntropyWithLogits()


def loss_unet_torch():
    from models.UNet.main_torch import CrossEntropyWithLogits
    return CrossEntropyWithLogits()


def loss_textcnn_ms():
    from models.textcnn.run_textcnn import loss_com_ms
    return loss_com_ms


def loss_textcnn_torch():
    from models.textcnn.run_textcnn_torch import loss_com
    return loss_com()


def loss_ssimae_ms():
    from models.ssimae.src.network import SSIMLoss
    return SSIMLoss()


def loss_ssimae_torch():
    from models.ssimae.src.network_torch import SSIMLoss as SSIMLoss_torch
    return SSIMLoss_torch()

def loss_deepv3plus_ms():
    from models.deeplabv3plus.main import deeplabv3_mindspore as loss_ms
    return loss_ms

def loss_deepv3plus_torch():
    from models.deeplabv3plus.main_torch import deeplabv3_torch
    return deeplabv3_torch()

def get_optimizer(optimize_name):
    from mindspore.nn.optim import AdamWeightDecay
    optimizer = {}
    optimizer['SGD'] = [mindspore.nn.SGD, torch.optim.SGD]
    optimizer['adam'] = [mindspore.nn.Adam, torch.optim.Adam]
    optimizer['adamweightdecay'] = [AdamWeightDecay, torch.optim.AdamW]
    return optimizer[optimize_name]


def loss_SSDmultibox_ms():
    return SSDmultibox_ms_cal


def SSDmultibox_ms_cal(pred_loc, pred_label, gt_loc, gt_label, num_matched_boxes):
    mask = mindspore.ops.less(0, gt_label).astype(mindspore.float32)
    num_matched_boxes = mindspore.numpy.sum(num_matched_boxes.astype(mindspore.float32))

    # Positioning loss
    mask_loc = mindspore.ops.tile(mindspore.ops.expand_dims(mask, -1), (1, 1, 4))
    smooth_l1 = mindspore.nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
    loss_loc = mindspore.numpy.sum(mindspore.numpy.sum(smooth_l1, -1), -1)

    # Category loss
    from models.SSD.ssd_utils import class_loss
    loss_cls = class_loss(pred_label, gt_label)
    loss_cls = mindspore.numpy.sum(loss_cls, (1, 2))

    return mindspore.numpy.sum((loss_cls + loss_loc) / num_matched_boxes)


class loss_SSDmultibox_torch(torch.nn.Module):
    def __init__(self):
        super(loss_SSDmultibox_torch, self).__init__()

    def forward(self, pred_loc, pred_label, gt_loc, gt_label, num_matched_boxes):
        mask = (gt_label > 0).float()
        num_matched_boxes = num_matched_boxes.float().sum()

        # Positioning loss
        mask_loc = mask.unsqueeze(-1).repeat(1, 1, 4)
        smooth_l1 = torch.nn.SmoothL1Loss(reduction='none')(pred_loc, gt_loc) * mask_loc
        loss_loc = smooth_l1.sum(dim=-1).sum(dim=-1)

        # Category loss
        from models.SSD.ssd_utils_torch import class_loss
        loss_cls = class_loss(pred_label, gt_label)
        loss_cls = loss_cls.sum(dim=(1, 2))

        return ((loss_cls + loss_loc) / num_matched_boxes).sum()


def loss_yolo_ms():
    from models.yolov4.main_new import yolov4loss_ms
    return yolov4loss_ms()


def loss_yolo_torch():
    from models.yolov4.yolov4_pytorch import yolov4loss_torch
    return yolov4loss_torch()

def loss_yolov4_ms():
    from models.yolov4.main_new import yolov4loss_ms
    return yolov4loss_ms()


def loss_yolov4_torch():
    from models.yolov4.yolov4_pytorch import yolov4loss_torch
    return yolov4loss_torch()

def get_loss(loss_name):
    loss = {}
    loss['CrossEntropy'] = [mindspore.nn.CrossEntropyLoss, torch.nn.CrossEntropyLoss]
    loss['ssdmultix'] = [loss_SSDmultibox_ms, loss_SSDmultibox_torch]
    loss['unetloss'] = [loss_unet_ms, loss_unet_torch]
    loss['textcnnloss'] = [loss_textcnn_ms, loss_textcnn_torch]
    loss['ssimaeloss'] = [loss_ssimae_ms, loss_ssimae_torch]
    loss['deepv3plusloss'] = [loss_deepv3plus_ms, loss_deepv3plus_torch]
    loss['yololoss'] = [loss_yolo_ms, loss_yolo_torch]
    loss['yolov4loss'] = [loss_yolov4_ms, loss_yolov4_torch]
    return loss[loss_name]


def max_seed_model_api_times(model_name):
    if model_name == "vgg11": #1
        return 28
    elif model_name == "vgg16": #2
        return 39
    elif model_name == "vgg19": #3
        return 61
    elif model_name == "resnet": #4
        return 158

    elif model_name == "yolov3": #5
        return 222
    elif model_name == "openpose": #6
        return 175
    elif model_name == "SSDresnet50fpn":#7
        return 271
    elif model_name == "SSDmobilenetv1":#8
        return 141

    elif model_name == "unet":#9
        return 49
    elif model_name == "DeepLabV3":#10
        return 329

    elif model_name == "LSTM": #11
        return 3
    elif model_name == "TextCNN": #12
        return 12
    elif model_name == "FastText": #13
        return 2

    elif model_name == "patchcore": #14
        return 127
    elif model_name == "ssimae":#15
        return 39
    elif model_name == "srgan":#15
        return 39
    # 新
    elif model_name=='mobilenetv2':
        return 140
    elif model_name=='vit':
        return 160
    elif model_name=='yolov4': # to do
        return 30# yolov4_torch()
    elif model_name=='crnn':
        crnnconfig.batch_size = 1  # batch_size
        return 29


def reflect_name(option_name,option_rule):
    if option_rule is rule1:
        new_name = option_name + "_mutated_rule1"
    elif option_rule is rule2:
        new_name = option_name + "_mutated_rule2"
    elif option_rule is rule3:
        new_name = option_name + "_mutated_rule3"
    elif option_rule is rule4:
        new_name = option_name + "_mutated_rule4"
    elif option_rule is rule5:
        new_name = option_name + "_mutated_rule5"
    elif option_rule is rule6:
        new_name = option_name + "_mutated_rule6"
    elif option_rule is rule7:
        new_name = option_name + "_mutated_rule7"
    elif option_rule is rule8:
        new_name = option_name + "_mutated_rule8"
    elif option_rule is rule9:
        new_name = option_name + "_mutated_rule9"
    elif option_rule is rule10:
        new_name = option_name + "_mutated_rule10"
    elif option_rule is rule11:
        new_name = option_name + "_mutated_rule11"
    elif option_rule is rule12:
        new_name = option_name + "_mutated_rule12"
    elif option_rule is rule13:
        new_name = option_name + "_mutated_rule13"
    elif option_rule is rule14:
        new_name = option_name + "_mutated_rule14"
    elif option_rule is rule15:
        new_name = option_name + "_mutated_rule15"
    elif option_rule is rule16:
        new_name = option_name + "_mutated_rule16"
    elif option_rule is rule17:
        new_name = option_name + "_mutated_rule17"
    elif option_rule is rule18:
        new_name = option_name + "_mutated_rule18"
    return new_name

def match_rule(option_rule_name): # 使用字典将字符串名称映射到相应的规则对象
    match_rule_dict = {
        'rule1': rule1,
        'rule2': rule2,
        'rule3': rule3,
        'rule4': rule4,
        'rule5': rule5,
        'rule6': rule6,
        'rule7': rule7,
        'rule8': rule8,
        'rule9': rule9,
        'rule10': rule10,
        'rule11': rule11,
        'rule12': rule12,
        'rule13': rule13,
        'rule14': rule14,
        'rule15': rule15,
        'rule16': rule16,
        'rule17': rule17,
        'rule18': rule18
    }
    return match_rule_dict.get(option_rule_name, None)


def rule_reflect_class(option_rule, option_instance):
    if option_rule is rule1:
        if isinstance(option_instance, ms.nn.Conv2d):
            return rule1.TransLayerRule1Conv2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule1.TransLayerRule1AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule1.TransLayerRule1MaxPool2d
        elif isinstance(option_instance, ms.nn.ReLU):
            return rule1.TransLayerRule1ReLU
        elif isinstance(option_instance, ms.nn.ReLU6):
            return rule1.TransLayerRule1ReLU6
        elif isinstance(option_instance, ms.nn.BatchNorm2d):
            return rule1.TransLayerRule1BatchNorm2d
        # elif isinstance(option_instance, ms.nn.Dense):
        #     return rule1.TransLayer_rule1_Dense
        elif isinstance(option_instance, ms.nn.Flatten):
            return rule1.TransLayerRule1Flatten
        elif isinstance(option_instance, ms.nn.HSigmoid):
            return rule1.TransLayerRule1Hardsigmoid
        elif isinstance(option_instance, ms.nn.Sigmoid):
            return rule1.TransLayerRule1Sigmoid
        elif isinstance(option_instance, ms.nn.Softmax):
            return rule1.TransLayerRule1Softmax
        elif isinstance(option_instance, ms.nn.Tanh):
            return rule1.TransLayerRule1Tanh
        elif isinstance(option_instance, ms.nn.Conv2dTranspose):
            return rule1.TransLayerRule1ConvTranspose2d
        elif isinstance(option_instance, ms.nn.LeakyReLU):
            return rule1.TransLayerRule1LeakyReLU
        elif isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule1.TransLayerRule1AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.Dropout):
            return rule1.TransLayerRule1Dropout
        elif isinstance(option_instance, ms.nn.Embedding):
            return rule1.TransLayerRule1Embedding
        elif isinstance(option_instance, ms.nn.LSTM):
            return rule1.TransLayerRule1LSTM
    elif option_rule is rule2:
        return rule2.TransLayerRule2
    elif option_rule is rule3:
        if isinstance(option_instance, ms.nn.Conv2d):
            return rule3.Conv2dToConv3d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule3.MaxPool2dToMaxPool3d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule3.AvgPool2dToAvgPool3d
    elif option_rule is rule4:
        return rule4.TransLayerRule4
    elif option_rule is rule5:
        return rule5.TransLayerRule5
    elif option_rule is rule6:
        return rule6.TransLayerRule6
    elif option_rule is rule7:
        return rule7.TransLayerRule7
    elif option_rule is rule8:
        return rule8.TransLayerRule8
    elif option_rule is rule9:
        return rule9.TransLayerRule9
    elif option_rule is rule10:
        return rule10.TransLayerRule10
    elif option_rule is rule11:
        return rule11.TransLayer_rule11
    elif option_rule is rule12:
        if isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule12.TransLayerRule12AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule12.TransLayerRule12AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule12.TransLayerRule12MaxPool2d
    elif option_rule is rule13:
        if isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule13.TransLayer_rule13_AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule13.TransLayer_rule13_AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule13.TransLayer_rule13_MaxPool2d
    elif option_rule is rule14:
        if isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule14.TransLayer_rule14_AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule14.TransLayer_rule14_AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule14.TransLayer_rule14_MaxPool2d
    elif option_rule is rule15:
        if isinstance(option_instance, ms.nn.ReLU):
            return rule15.TransLayerRule15ReLU
        elif isinstance(option_instance, ms.nn.LeakyReLU):
            return rule15.TransLayerRule15LeakyReLU
    elif option_rule is rule16:
        return rule16.TransLayer_rule16


def select_places(sequence, k):
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            # print("sequence is", sequence)
            # print("k is", k)
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            # print("subs_place is", subs_place)
            # print("chosen is", chosen)
            return subs_place, chosen
    # print("Cannot find suitable places")
    return None, None

# def select_places(sequence, k): # (range(0, len(nodelist)), 5)
#     for i in range(5):
#         try:
#             chosen = random.choices(sequence, k=k)
#         except Exception as e:
#             print("sequence is", sequence)
#             return None, None
#         subs_place = max(chosen)
#         chosen.remove(subs_place)
#         if max(chosen) != subs_place:
#             return subs_place, chosen
#     print("Cannot find suitable places")
#     return None, None

# def select_places(sequence, k): # (range(0, len(nodelist)), 5)
#     try:
#         chosen = random.choices(sequence, k=k)
#         return chosen
#     except Exception as e:
#         print("Cannot find suitable places")
#         print("sequence is", sequence)
#         return None

# MCMC
np.random.seed(20200501)
class MCMC:
    class Mutator:
        def __init__(self, name, total=0, delta_bigger_than_zero=0, epsilon=1e-7):
            self.name = name
            self.total = total
            self.delta_bigger_than_zero = delta_bigger_than_zero
            self.epsilon = epsilon

        @property
        def score(self, epsilon=1e-7):
            rate = self.delta_bigger_than_zero / (self.total + epsilon)
            return rate

    def __init__(self, mutate_ops=['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']):
        self.p = 1 / len(mutate_ops)
        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self):#用变异算子名称返回这个算子对象
        mus = {}
        for mu in self._mutators:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None: # which means it's the first mutation
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        else:
            self.sort_mutators() #根据每个算子的得分降序排序
            k1 = self.index(mu1) #当前算子排名
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self._mutators[k2]
            return mu2.name

    def sort_mutators(self):
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self._mutators):
            if mu.name == mutator_name:
                return i
        return -1

## QRDQN相关网络
# 量子计算设置
n_qubits = 4  # 量子比特数量
n_layers = 2  # 量子线路层数
dev = qml.device("default.qubit", wires=n_qubits)

# 定义量子电路
def quantum_circuit(params, inputs):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 将量子电路作为一个Pytorch层
weight_shapes = {"params": (n_layers, n_qubits)}
quantum_layer = qml.qnn.TorchLayer(qml.QNode(quantum_circuit, dev), weight_shapes)

# 定义QRDQN的神经网络部分
class QRDQN(nn.Module):
    def __init__(self, state_dim, n_actions, num_quantiles):
        super(QRDQN, self).__init__()
        self.num_quantiles = num_quantiles
        self.n_actions = n_actions
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, n_qubits)  # 确保与量子比特数匹配
        self.quantum_layer = quantum_layer
        self.fc3 = nn.Linear(n_qubits, num_quantiles*n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state.view(1, -1)))
        x = torch.relu(self.fc2(x))
        x = self.quantum_layer(x)
        q_values = self.fc3(x)
        return q_values.view(-1, self.n_actions, self.num_quantiles)

# 定义DQN的神经网络部分
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, num_quantiles):
        super(DQN, self).__init__()
        self.num_quantiles = num_quantiles
        self.n_actions = n_actions
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, n_qubits)
        self.fc3 = nn.Linear(n_qubits, num_quantiles*n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values.view(-1, self.n_actions, self.num_quantiles)
# class QRDQN(nn.Module):
#     def __init__(self, state_dim, n_actions):
#         super(QRDQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, n_qubits)  # 确保与量子比特数匹配
#         self.quantum_layer = quantum_layer
#         self.fc3 = nn.Linear(n_qubits, n_actions)

#     def forward(self, state):
#         x = torch.relu(self.fc1(state.view(1, -1)))
#         x = torch.relu(self.fc2(x))
#         x = self.quantum_layer(x)
#         q_values = self.fc3(x)
#         return q_values
    

import pynvml
pynvml.nvmlInit()
def compute_gpu_cpu(): #GPU CPU使用情况1
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  # # GPU显存使用情况  MB
    mem = psutil.virtual_memory()  # 当前内存使用情况
    gpu_memory = float(memory_info.used) / (1024 ** 2)
    cpu_memory = float(mem.available) / 1024 / 1024
    return gpu_memory,cpu_memory

### ms新加
banned_ops = [mindspore.ops.operations.array_ops.Shape,
              mindspore.ops.operations.array_ops.Concat,
              type(None)
              ]
banned_cell = [mindspore.nn.layer.CentralCrop, ]
banned_trees = [mindspore.ops.ResizeBilinearV2, #版本不一致 to do对应的模型也要替换
                mindspore.ops.operations.Shape,
                type(None)
                ]
# hash_table：一个用于记录节点访问次数的字典，防止重复处理节点。 变化
# nodedict：用于存储符合特定条件的节点（即 CallCell、CallPrimitive 等）节点的标识符作为键，节点所属的符号树作为值
# depth：记录当前递归的深度
def scan_node(stree, hash_table, nodedict=None, depth=0):
    # global hash_table
    # for node in stree.nodes(all_nodes=False):
    if type(stree) == mindspore.rewrite.api.symbol_tree.SymbolTree:
        stree = stree._symbol_tree
    for node in stree.all_nodes():
        if isinstance(node, NodeManager):
            for sub_node in node.get_tree_nodes():
                subtree = sub_node.symbol_tree
                scan_node(subtree, hash_table, nodedict=nodedict, depth=depth + 1)
        if (node.get_node_type() == NodeType.CallCell and node.get_instance_type() not in banned_cell) or (
                node.get_node_type() == NodeType.CallPrimitive and node.get_instance_type() not in banned_ops) \
                or (node.get_node_type() == NodeType.Tree and node.get_instance_type() not in banned_trees) \
                or node.get_node_type() == NodeType.CellContainer:
            if hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] == 1:
                continue
            hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] += 1
            if node.get_node_type() not in [NodeType.CellContainer, NodeType.Tree]:
                nodedict[mindspore.rewrite.api.node.Node(node).get_handler()] = node.get_belong_symbol_tree()
    return True,nodedict

def check_node(node):
    if len(node.get_users()) == 0 or len(node.get_targets()) != 1:
        # print('len(node.get_users():',len(node.get_users()), ',len(node.get_targets()):',len(node.get_targets()))
        return False
    return True



class qrdqnloss(nn.Module):
    def __init__(self,quantiles):
        super(qrdqnloss, self).__init__()
        self.quantiles = quantiles


    def forward(self, target_quantiles, quantile_values):
        td_error = target_quantiles.unsqueeze(1) - quantile_values.unsqueeze(2)
        huber_loss = torch.where(td_error.abs() <= 1.0, 0.5 * td_error ** 2,td_error.abs() - 0.5)  # 形状: (batch_size, num_quantiles, num_quantiles)

        quantile_loss = (self.quantiles - (td_error.detach() < 0).float()).abs() * huber_loss  # 形状: (batch_size, num_quantiles, num_quantiles)
        loss = quantile_loss.mean()  # 标量
        return loss


# if __name__ == '__main__':
#     Quantum_Q = QRDQN(10, 4, 20).to("cuda:6")
#     Target_Q = QRDQN(10, 4, 20).to("cuda:6")
#     optimizer = torch.optim.Adam(Quantum_Q.parameters(), lr=0.001)
#     criterion = torch.nn.MSELoss().to("cuda:6")
#
#     target_update = 1
#     reward = 0.354612587
#     gamma = 0.04
#     done = 0
#     num_quantiles = 20
#     formatted_data_torch = torch.tensor(np.random.randn(1, 10), dtype=torch.float32).to("cuda:6")
#
#     next_quantile_values = Quantum_Q(formatted_data_torch)
#     next_q_values = next_quantile_values.mean(dim=2)
#     next_actions = torch.argmax(next_q_values, dim=1)
#     next_quantiles = next_quantile_values[range(formatted_data_torch.shape[0]), next_actions]
#     target_quantiles = reward+gamma*(1-done)*next_quantiles
#
#
#     # selected_MR_structure_idx = q_value.argmax().item()
#     quantiles = (torch.linspace(0,1,num_quantiles)+0.5/num_quantiles).to("cuda:6")
#     #loss_fun = qrdqnloss(quantiles)
#
#     quantile_values = Target_Q(formatted_data_torch)
#     quantile_values = quantile_values[range(formatted_data_torch.shape[0]), next_actions]
#     td_error = target_quantiles.unsqueeze(1) - quantile_values.unsqueeze(2)
#     huber_loss = torch.where(td_error.abs() <= 1.0, 0.5 * td_error ** 2,td_error.abs() - 0.5)  # 形状: (batch_size, num_quantiles, num_quantiles)
#
#     quantile_loss = (quantiles - (td_error.detach() < 0).float()).abs() * huber_loss  # 形状: (batch_size, num_quantiles, num_quantiles)
#     loss = quantile_loss.mean()  # 标量
#
#     # 优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()





