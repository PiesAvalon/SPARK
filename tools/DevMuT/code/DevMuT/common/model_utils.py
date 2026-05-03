import os
import numpy as np
import json
from copy import deepcopy
import torch
import mindspore as ms
import troubleshooter as ts
import scipy.io as scio

if os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
    device = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[0]
    device = "cuda:" + device
else:
    device = "cpu"


def find_Cascade_OP(layer_names):
    Cascade_ops = []
    for i in range(len(layer_names)):
        if "_del" in layer_names[i] or "empty" in layer_names[i]:
            continue
        c1 = layer_names[i].split(".")
        for j in range(i + 1, len(layer_names)):
            if "_del" in layer_names[j] or "empty" in layer_names[i]:
                continue
            c2 = layer_names[j].split(".")
            if layer_names[i] in layer_names[j] and len(c1) == len(c2) - 1:
                Cascade_ops.append(layer_names[i])
                break
    return Cascade_ops


def model_prepare(model, input_size):
    layer_names = deepcopy(list(model.layer_names.keys()))
    Cascade_OPs = find_Cascade_OP(layer_names)
    Basic_OPS = list(set(layer_names) - set(Cascade_OPs))

    if "transformer" in str(model.__class__.__name__).lower():
        Cascade_OPs_new = []
        Basic_OPS_new = []
        for val in Basic_OPS:
            if "decoder" in val or "create_attention_mask_from_input_mask" in val or "tfm_embedding_lookup" in val:
                continue
            Basic_OPS_new.append(val)

        for val in Cascade_OPs:
            if "decoder" in val or "create_attention_mask_from_input_mask" in val or "tfm_embedding_lookup" in val:
                continue
            Cascade_OPs_new.append(val)

        Cascade_OPs = Cascade_OPs_new
        Basic_OPS = Basic_OPS_new

    elif "pangu" in str(model.__class__.__name__).lower():
        not_pair_layers = []
        f = open(os.getcwd() + "/network/nlp/S_Pangu_alpha/notpairwithtorch", "r")
        lines = f.readlines()
        for line in lines:
            not_pair_layers.append(line[:-1])
        f.close()

        layer_names = deepcopy(list(model.layer_names.keys()))
        layer_names = list(set(layer_names) - set(not_pair_layers))

        Cascade_OPs = find_Cascade_OP(layer_names)
        Basic_OPS = list(set(layer_names) - set(Cascade_OPs))

        Cascade_OPs_new = []
        Basic_OPS_new = []

        self_ops = list(model.orders.keys())

        for val in Basic_OPS:
            if not val in self_ops:
                continue
            Basic_OPS_new.append(val)

        for Cascade_OP in Cascade_OPs:
            flag = False
            for val in Basic_OPS_new:
                if Cascade_OP in val:
                    flag = True
                    break
            if flag and (not "backbone" == Cascade_OP):
                Cascade_OPs_new.append(Cascade_OP)

        Cascade_OPs = Cascade_OPs_new
        Basic_OPS = Basic_OPS_new

    model.set_Basic_OPS(deepcopy(Basic_OPS))
    model.set_Cascade_OPS(deepcopy(Cascade_OPs))

    Cascade_OPs_inshapes, Cascade_OPs_outshapes = {}, {}
    remove_Cascade = []

    for Cascade_OP in Cascade_OPs:
        yezi_ops = find_Child_leaf_OP(model.layer_names, Cascade_OP, model.Basic_OPS, model.add_Cascade_OPs)

        bsize = model.in_shapes[list(model.in_shapes.keys())[0]][0]
        first_childs, final_childs, last_ops, next_ops, in_shape, out_shape = find_Cascade_OP_shape(model, bsize,
                                                                                                    Cascade_OP,
                                                                                                    yezi_ops)

        if len(last_ops) > 1 or len(final_childs) > 1:
            remove_Cascade.append(Cascade_OP)
            continue

        if len(last_ops) == 0 or len(next_ops) == 0:
            remove_Cascade.append(Cascade_OP)
            continue

        if not model.out_shapes[last_ops[0]] == in_shape:
            remove_Cascade.append(Cascade_OP)
            continue

        if not model.in_shapes[next_ops[0]] == out_shape:
            remove_Cascade.append(Cascade_OP)
            continue

        assert len(in_shape) == 1
        assert len(final_childs) == 1

        in_shape, out_shape = list(in_shape[0].split(",")), list(out_shape[0].split(","))
        in_shape, out_shape = [int(val) for val in in_shape], [int(val) for val in out_shape]

        Cascade_OPs_inshapes[Cascade_OP] = in_shape
        Cascade_OPs_outshapes[Cascade_OP] = out_shape

    Cascade_OPs = deepcopy(model.Cascade_OPs)
    Cascade_OPs_after_del = []
    for Cascade_OP in Cascade_OPs:
        if Cascade_OP in remove_Cascade:
            continue

        Cascade_OPs_after_del.append(Cascade_OP)

    model.set_Cascade_OPS(deepcopy(Cascade_OPs_after_del))
    model.Cascade_OPs_inshapes = Cascade_OPs_inshapes
    model.Cascade_OPs_outshapes = Cascade_OPs_outshapes

    shape_keys = list(model.out_shapes.keys())

    for shape_key in shape_keys:
        if "INPUT" in shape_key:
            scan_batchsize = model.in_shapes[shape_key][0]
            break

    bsize_mul = abs(input_size[0]/scan_batchsize)

    for shape_key in shape_keys:
        if "rnn" in shape_key.lower() or (len(model.in_shapes[shape_key])==3 and model.in_shapes[shape_key][1]==scan_batchsize):
            if abs(model.in_shapes[shape_key][1]) == 1 or input_size[0] == model.in_shapes[shape_key][1]:
                model.in_shapes[shape_key][1] = input_size[0]
            else:
                model.in_shapes[shape_key][1] = int(model.in_shapes[shape_key][1] * bsize_mul)
            if abs(model.out_shapes[shape_key][1]) == 1 or input_size[0] == model.out_shapes[shape_key][1]:
                model.out_shapes[shape_key][1] = input_size[0]
            else:
                model.out_shapes[shape_key][1] = int(model.out_shapes[shape_key][1] * bsize_mul)
            continue

        if abs(model.in_shapes[shape_key][0]) == 1 or input_size[0] == model.in_shapes[shape_key][0]:
            model.in_shapes[shape_key][0] = input_size[0]
        else:
            model.in_shapes[shape_key][0] = int(model.in_shapes[shape_key][0] * bsize_mul)
        if abs(model.out_shapes[shape_key][0]) == 1 or input_size[0] == model.out_shapes[shape_key][0]:
            model.out_shapes[shape_key][0] = input_size[0]
        else:
            model.out_shapes[shape_key][0] = int(model.out_shapes[shape_key][0] * bsize_mul)

    check_orderinfo_selfcorrect(model)
    return model


def check_orderinfo_selfcorrect(model):
    orders = model.orders
    layer_names = list(orders.keys())
    for layer_name in layer_names:
        qianqu, houji = orders[layer_name]
        if isinstance(qianqu, list):
            for qianqu_single in qianqu:
                if "INPUT" in qianqu_single:
                    continue
                assert (orders[qianqu_single][1] == layer_name or layer_name in orders[qianqu_single][1])
        else:
            if "INPUT" not in qianqu:
                assert (orders[qianqu][1] == layer_name or layer_name in orders[qianqu][1])

        if isinstance(houji, list):
            for houji_single in houji:
                if "OUTPUT" in houji_single:
                    continue
                assert (orders[houji_single][0] == layer_name or layer_name in orders[houji_single][0])
        else:
            if "OUTPUT" not in houji:
                assert (orders[houji][0] == layer_name or layer_name in orders[houji][0])


def find_Cascade_OP_shape(model, b_size, del_layer_name, yezi_ops):
    first_childs, final_childs = [], []
    last_ops, next_ops = [], []
    input_shapes, out_shapes = [], []
    for yezi_op in yezi_ops:
        qianqu_info = model.get_order(yezi_op)[0]
        houji_info = model.get_order(yezi_op)[1]

        # check qianqu info
        flag_firstchild = True
        if isinstance(qianqu_info, list):
            for qianqu_info_single in qianqu_info:
                flag_lastop = True
                if not (del_layer_name in qianqu_info_single):
                    flag_firstchild = False
                    flag_lastop = False
                    break

            if not flag_lastop:
                last_ops.append(qianqu_info_single)

        else:
            if del_layer_name not in qianqu_info:
                flag_firstchild = False
                last_ops.append(qianqu_info)

        if not flag_firstchild:
            first_childs.append(yezi_op)
            in_shape = deepcopy(model.in_shapes[yezi_op])
            if abs(in_shape[0]) == 1 or in_shape[0] == b_size:
                in_shape[0] = b_size
            else:
                in_shape[0] = abs(in_shape[0]) * b_size

            input_shapes.append(in_shape)

        # check houji info
        flag_finalchild = True
        if isinstance(houji_info, list):
            for houji_info_single in houji_info:
                flag_nextop = True
                if not (del_layer_name in houji_info_single):
                    flag_finalchild = False
                    flag_nextop = False

            if not flag_nextop:
                next_ops.append(houji_info_single)

        else:
            if not (del_layer_name in houji_info):
                flag_finalchild = False
                next_ops.append(houji_info)

        if not flag_finalchild:
            final_childs.append(yezi_op)
            out_shape = deepcopy(model.out_shapes[yezi_op])
            if abs(out_shape[0]) == 1 or out_shape[0] == b_size:
                out_shape[0] = b_size
            else:
                out_shape[0] = abs(out_shape[0]) * b_size
            out_shapes.append(out_shape)

    last_ops, next_ops = list(set(last_ops)), list(set(next_ops))

    input_shapes_str, out_shapes_str = [], []
    for val in input_shapes:
        input_shapes_str.append(str(val)[1:-1])
    for val in out_shapes:
        out_shapes_str.append(str(val)[1:-1])

    input_shapes, out_shapes = list(set(input_shapes_str)), list(set(out_shapes_str))

    return first_childs, final_childs, last_ops, next_ops, input_shapes, out_shapes


def find_Child_leaf_OP(layer_names, del_layer_name, Basic_op_names, add_Cascade_OP_names):
    yezi_ops = []
    for layer_name in layer_names:
        if "_del" in layer_name or "empty" in layer_name:
            continue

        flag = (del_layer_name + "." in layer_name) and not (del_layer_name == layer_name) and \
               (layer_name in Basic_op_names or layer_name in add_Cascade_OP_names)

        if flag:
            yezi_ops.append(layer_name)

    return yezi_ops


def get_vgg16():
    from network.cv.vgg16.src.vgg import vgg16
    from network.cv.vgg16.vgg16_torch import vgg
    model1 = vgg16()
    model2 = vgg()
    return model1, model2


def get_textcnn():
    from network.nlp.textcnn.src.textcnn import TextCNN as textcnn_ms
    from network.nlp.textcnn.src.textcnn_torch import TextCNN as textcnn_torch
    model1 = textcnn_ms(vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
    model2 = textcnn_torch(vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
    return model1, model2


def get_yolov4():
    from network.cv.yolov4.main_new import YOLOV4CspDarkNet53_ms as yolov4_ms
    from network.cv.yolov4.yolov4_pytorch import YOLOV4CspDarkNet53_torch as yolov4_torch

    model1 = yolov4_ms()
    model2 = yolov4_torch()
    return model1, model2

def get_unet():
    from network.cv.unet.main import UNetMedical as unet_ms
    from network.cv.unet.main_torch import UNetMedical_torch as unet_torch
    model1 = unet_ms(n_channels=1, n_classes=2)
    model2 = unet_torch(n_channels=1, n_classes=2)

    return model1, model2


def get_unetplus():
    from network.cv.unetplus.mainplus import NestedUNet as unetplus_ms
    from network.cv.unetplus.mainplus_torch import NestedUNet as unetplus_torch
    model1 = unetplus_ms(in_channel=1, n_class=2, use_deconv=True, use_bn=True, use_ds=False)
    model2 = unetplus_torch(in_channels=1, n_classes=2, feature_scale=2, use_deconv=True, use_bn=True, use_ds=False)

    return model1, model2



def get_resnet50():
    from network.cv.resnet.src.model_utils.config import config
    from network.cv.resnet.src.resnet import resnet50 as resnet

    config.class_num = 10
    from network.cv.resnet.resnet50_torch import resnet50 as resnet50_torch

    model1 = resnet(class_num=config.class_num)
    model2 = resnet50_torch(config.class_num)
    return model1, model2



def get_model(model_name, input_size=1, only_ms=False, scaned=True):
    models_dict = {
        'vgg16': get_vgg16,
        'resnet50': get_resnet50,
        "textcnn": get_textcnn,
        'unet': get_unet,
        'unetplus': get_unetplus,
        'yolov4': get_yolov4,
    }
    if "transformer" == model_name or "maskrcnn" == model_name or "fasterrcnn" == model_name or "pangu" == model_name or "crnn"==model_name:
        model_ms, model_pt = models_dict[model_name](input_size[0])
    else:
        model_ms, model_pt = models_dict[model_name]()



    model_ms.set_train(True)
    model_pt.train()

    if scaned:
        model_ms = model_prepare(model_ms, input_size)
        model_pt = model_prepare(model_pt, input_size)

    if only_ms:
       return model_ms

    if os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
        device = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[0]
        model_pt.to('cuda:'+device)
    else:
        model_pt.to('cpu')

    ts.migrator.get_weight_map(pt_net=model_pt,
                               weight_map_save_path='./torch_net_map.json',
                               print_map=False)
    torch.save(model_pt.state_dict(), './torch_net.path')
    ts.migrator.convert_weight(weight_map_path='./torch_net_map.json',
                               pt_file_path='./torch_net.path',
                               ms_file_save_path='./convert_ms.ckpt',
                               print_conv_info=False, print_save_path=False)
    param_dict = ms.load_checkpoint('./convert_ms.ckpt')
    ms.load_param_into_net(model_ms, param_dict)

    return model_ms, model_pt
