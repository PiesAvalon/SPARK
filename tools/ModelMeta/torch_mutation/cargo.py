import os

import numpy as np
from torch_mutation.rules_torch import rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,\
                                        rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18
import torch
import random
import pennylane as qml
import torch.nn as nn
import scipy.io as scio
import torch.fx as fx
import psutil
import mindspore
## 导入模型
from models.vgg11.vgg11_torch import vgg11 #1
from models.vgg16.vgg16_torch import VGG as vgg16 # 2
from models.vgg19.vgg19_torch import vgg19 #3
from models.resnet50.resnet50_torch import resnet50 #4
from models.yolov3_darknet53.main_new import YOLOV3DarkNet53 as yolov3
# from models.yolov4.src.yolo import YOLOV4CspDarkNet53 as yolov4
from models.yolov4.main_new import YOLOV4CspDarkNet53_ms as yolov4
# from models.yolov3.main_torch import YOLOV3DarkNet53 #5
from models.openpose.main_parallel_torch import OpenPoseNet as OpenPoseNet_torch #6
from models.openpose.src.model_utils.config import config as openpose_config #6
from models.SSD.backbone_resnet50_fpn_pytorch import ssd_resnet50fpn_torch #7
from models.SSD.backbone_mobilenetv1_pytorch import SSDWithMobileNetV1 #8

from models.UNet.main_torch import UNetMedical_torch #9
from models.deeplabv3.main_torch import DeepLabV3_torch #10

from models.textcnn.textcnn_torch import TextCNN #12
# from models.FastText.fasttext_torch import FastText_torch #13

from models.PatchCore.model_torch import wide_resnet50_2 as PatchCore #14
from models.ssimae.src.network_torch import AutoEncoder as AutoEncoder_torch #15 #{'enable_modelarts': 'Whether training on modelarts defau
from models.ssimae.model_utils.config import config as ssimae_cfg  #15

# 新
from models.mobilenetv2.model_v2_withms import mobilenet_v2_torch as mobilenet_v2_torch
from models.vit.vit_torch import get_vit_torch as vit_torch
from models.yolov4.yolov4_pytorch import YOLOV4CspDarkNet53_torch as yolov4_torch
from models.CRNN.src.model_utils.config import config as crnnconfig
from models.CRNN.crnn_torch import CRNNV2 as crnn_torch # crnn_torch.py中的device记得改 to do
from models.SRGAN.model_torch import Generator as srgan
dataset_basic_dir = "/home/cvgroup/myz/czx/SemTest_master/test_data"
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
    "srgan":dataset_basic_dir+"/patchcore_x.npy",
}


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

def get_model(model_name, device):
    net_cargo = {
        # 图像分类,结果都是Size([10])
        "vgg11": vgg11, #1
        "vgg16": vgg16, #2
        "vgg19": vgg19, #3
        "resnet": resnet50, #4
        # 目标检测
        "yolov3": yolov3,  # 5
        "yolov4":yolov4,
        "openpose": OpenPoseNet_torch,  # 6
        "SSDresnet50fpn": ssd_resnet50fpn_torch,  # 7
        "SSDmobilenetv1": SSDWithMobileNetV1,  # 8
        # 语义分割
        "unet": UNetMedical_torch,#9
        "DeepLabV3": DeepLabV3_torch, #10
        # 文本分类
        "TextCNN": TextCNN,  # 12
        # "FastText": FastText_torch, #13
        # 异常检测
        "patchcore": PatchCore,  # 14
        "crnn":crnn_torch,
        "ssimae": AutoEncoder_torch,  # 15
        "srgan":srgan,
    }

    if model_name == "vgg11": #1
        model = net_cargo[model_name]()
        return model
    elif model_name == "vgg16": #2
        def make_features(cfg: list, batch_norm=True, ):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, bias=False)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                    else:
                        layers += [conv2d, nn.ReLU()]
                    in_channels = v
            return nn.Sequential(*layers)

        features = make_features([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        model = net_cargo[model_name](features, num_classes=10, has_dropout=False, init_weights=False, phase="train", include_top=True)
        return model
    elif model_name == "vgg19": #3
        model = net_cargo[model_name](10)
        return model
    elif model_name == "resnet": #4
        model = net_cargo[model_name]()
        return model

    elif model_name == "yolov3": #5
        model = net_cargo[model_name](True)
        return model
    elif model_name == "openpose": #6
        model = net_cargo[model_name](vggpath=openpose_config.vgg_path, vgg_with_bn=openpose_config.vgg_with_bn)
        return model
    elif model_name == "SSDresnet50fpn":#7
        model = net_cargo[model_name]()
        return model
    elif model_name == "SSDmobilenetv1":#8
        model = net_cargo[model_name]()
        return model

    elif model_name == "unet":#9
        model = net_cargo[model_name](1, 2)
        return model
    elif model_name == "DeepLabV3":#10
        model = net_cargo[model_name](21)
        return model

    elif model_name == "TextCNN": #12
        model = net_cargo[model_name](vocab_len=20305, word_len=51, num_classes=52, vec_length=40)
        return model

    elif model_name == "FastText": #13
        model = net_cargo[model_name]()
        return model

    elif model_name == "patchcore": #14
        model = net_cargo[model_name]()
        return model
    elif model_name == "ssimae":#15
        model = net_cargo[model_name](ssimae_cfg)
        return model

    # 新
    elif model_name=='mobilenetv2':
        return mobilenet_v2_torch()
    elif model_name=='vit':
        return vit_torch()
    elif model_name=='yolov4':
        return yolov4_torch()
    elif model_name=='crnn':
        crnnconfig.batch_size = 1  # batch_size
        return crnn_torch(crnnconfig)
    elif model_name == "srgan":
        model = net_cargo[model_name](4)
    return model

# if __name__ == '__main__':
#     #  1."vgg11" 2."vgg16" 3."vgg19" 4."resnet" 5."yolov3" 6."openpose" 7."SSDresnet50fpn" 8."SSDmobilenetv1" 测试fx
#     # 9."UNetMedical" 10."DeepLabV3" 11."LSTM" 12."textcnn" 13."FastText" 14."patchcore" 15."ssimae"
#     # 新
#     # 'MobileNetV2'  'ViT'没有成功  'Yolov4'   'CRNN'

#     model_name = "CRNN"
#     model=get_model(model_name,"CPU")
#     print(model)
#     d=fx.symbolic_trace(model)
#     print("-*"*10)
#     print(d)

#     option_layers = list()  # 可被选择的算子列表（形式：(node,instance,name)）
#     nn_types=set()
#     for node in d.graph.nodes:
#         if node.op == 'call_module' and '_mutate' not in node.name:  # 如果这个算子是module类型的且不是被变异过的算子，就加入列表
#             module_name = node.target
#             module_instance = d.get_submodule(module_name)  # 获取模块实例
#             option_layers.append((node, module_instance, node.name))
#             nn_types.add(type(module_instance))
#     print(option_layers)
#     print(len(option_layers))
#     print(nn_types)



# def get_loss(loss_name):
#     loss = {}
#     loss['CrossEntropy'] = [mindspore.nn.CrossEntropyLoss, torch.nn.CrossEntropyLoss]
#     loss['ssdmultix'] = [loss_SSDmultibox_ms, loss_SSDmultibox_torch]
#     loss['retinafacemultix'] = [loss_retinaface_ms, loss_retinaface_torch]
#     loss['yololoss'] = [loss_yolo_ms, loss_yolo_torch]
#     loss['unetloss'] = [loss_unet_ms, loss_unet_torch]
#     loss['unet3dloss'] = [loss_unet3d_ms, loss_unet3d_torch]
#     loss['fasttextloss'] = [loss_fasttext_ms, loss_fasttext_torch]
#     loss['textcnnloss'] = [loss_textcnn_ms, loss_textcnn_torch]
#     loss['deepv3plusloss'] = [loss_deepv3plus_ms, loss_deepv3plus_torch]
#     loss['transformerloss'] = [loss_transformer_ms, loss_transformer_torch]
#     loss['bertloss'] = [loss_bert_ms, loss_bert_torch]
#     loss['rcnnloss'] = [loss_maskrcnn_ms, loss_maskrcnn_torch]
#     loss['panguloss'] = [loss_pangu_ms, loss_pangu_torch]
#
#     return loss[loss_name]

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
    return loss_com


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
    loss['CrossEntropy'] = [torch.nn.CrossEntropyLoss, torch.nn.CrossEntropyLoss]
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

    elif model_name == "UNetMedical":#9
        return 49
    elif model_name == "DeepLabV3":#10
        return 329
    elif model_name == "srgan":#15
        return 39
    elif model_name == "LSTM": #11
        return 3
    elif model_name == "textcnn": #12
        return 12
    elif model_name == "FastText": #13
        return 2

    elif model_name == "patchcore": #14
        return 127
    elif model_name == "ssimae":#15
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


def rule_reflect_class(option_rule,option_instance):
    if option_rule is rule1:
        if isinstance(option_instance, torch.nn.Conv2d):
            return rule1.TransLayer_rule1_Conv2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule1.TransLayer_rule1_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule1.TransLayer_rule1_MaxPool2d
        elif isinstance(option_instance, torch.nn.ReLU):
            return rule1.TransLayer_rule1_ReLU
        elif isinstance(option_instance, torch.nn.ReLU6):
            return rule1.TransLayer_rule1_ReLU6
        elif isinstance(option_instance, torch.nn.BatchNorm2d):
            return rule1.TransLayer_rule1_BatchNorm2d
        elif isinstance(option_instance, torch.nn.Linear):
            return rule1.TransLayer_rule1_Linear
        elif isinstance(option_instance, torch.nn.Flatten):
            return rule1.TransLayer_rule1_Flatten
        elif isinstance(option_instance, torch.nn.Hardsigmoid):
            return rule1.TransLayer_rule1_Hardsigmoid
        elif isinstance(option_instance, torch.nn.Sigmoid):
            return rule1.TransLayer_rule1_Sigmoid
        elif isinstance(option_instance, torch.nn.Softmax):
            return rule1.TransLayer_rule1_Softmax
        elif isinstance(option_instance, torch.nn.Tanh):
            return rule1.TransLayer_rule1_Tanh
        elif isinstance(option_instance, torch.nn.ConvTranspose2d):
            return rule1.TransLayer_rule1_ConvTranspose2d
        elif isinstance(option_instance, torch.nn.LeakyReLU):
            return rule1.TransLayer_rule1_LeakyReLU
        elif isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule1.TransLayer_rule1_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.Dropout):
            return rule1.TransLayer_rule1_Dropout
        elif isinstance(option_instance, torch.nn.Embedding):
            return rule1.TransLayer_rule1_Embedding
        elif isinstance(option_instance, torch.nn.LSTM):
            return rule1.TransLayer_rule1_LSTM
    elif option_rule is rule2:
        return rule2.TransLayer_rule2
    elif option_rule is rule3:
        if isinstance(option_instance, torch.nn.Conv2d):
            return rule3.TransLayer_rule3_Conv2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule3.TransLayer_rule3_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule3.TransLayer_rule3_MaxPool2d
    elif option_rule is rule4:
        return rule4.TransLayer_rule4
    elif option_rule is rule5:
        return rule5.TransLayer_rule5
    elif option_rule is rule6:
        return rule6.TransLayer_rule6
    elif option_rule is rule7:
        return rule7.TransLayer_rule7
    elif option_rule is rule8:
        return rule8.TransLayer_rule8
    elif option_rule is rule9:
        return rule9.TransLayer_rule9
    elif option_rule is rule10:
        return rule10.TransLayer_rule10
    elif option_rule is rule11:
        return rule11.TransLayer_rule11
    elif option_rule is rule12:
        if isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule12.TransLayer_rule12_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule12.TransLayer_rule12_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule12.TransLayer_rule12_MaxPool2d
    elif option_rule is rule13:
        if isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule13.TransLayer_rule13_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule13.TransLayer_rule13_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule13.TransLayer_rule13_MaxPool2d
    elif option_rule is rule14:
        if isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule14.TransLayer_rule14_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule14.TransLayer_rule14_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule14.TransLayer_rule14_MaxPool2d
    elif option_rule is rule15:
        if isinstance(option_instance, torch.nn.ReLU):
            return rule15.TransLayer_rule15_ReLU
        elif isinstance(option_instance, torch.nn.LeakyReLU):
            return rule15.TransLayer_rule15_LeakyReLU
    elif option_rule is rule16:
        return rule16.TransLayer_rule16


def select_places(sequence, k): # (range(0, len(nodelist)), 5)
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            print("sequence is", sequence)
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            return subs_place, chosen
    print("Cannot find suitable places")
    return None, None

def select_places(sequence, k): # (range(0, len(nodelist)), 5)
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            print("sequence is", sequence)
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            return subs_place, chosen
    print("Cannot find suitable places")
    return None, None

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
    def __init__(self, state_dim, n_actions):
        super(QRDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, n_qubits)  # 确保与量子比特数匹配
        self.quantum_layer = quantum_layer
        self.fc3 = nn.Linear(n_qubits, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state.view(1, -1)))
        x = torch.relu(self.fc2(x))
        x = self.quantum_layer(x)
        q_values = self.fc3(x)
        return q_values

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
    
def compute_gpu_cpu(): #GPU CPU使用情况1
    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  # # GPU显存使用情况  MB
    mem = psutil.virtual_memory()  # 当前内存使用情况
    cpu_memory = float(mem.available) / 1024 / 1024
    return gpu_memory,cpu_memory

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