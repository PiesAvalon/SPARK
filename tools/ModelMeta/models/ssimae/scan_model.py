import json
import os
from copy import deepcopy

import torch
import mindspore
# import util as util
from util import find_Cascade_OP,check_orderinfo_selfcorrect
import util as util
from infoplus.TorchInfoPlus import torchinfoplus
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
import numpy as np
import troubleshooter as ts
from models.ssimae.model_utils.config import config
from models.ssimae.src.network_torch import AutoEncoder as AutoEncoder_torch
from models.ssimae.src.network import AutoEncoder as AutoEncoder_ms


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
            if not del_layer_name in qianqu_info:
                flag_firstchild = False
                last_ops.append(qianqu_info)

        if not flag_firstchild:
            first_childs.append(yezi_op)
            in_shape = model.in_shapes[yezi_op]
            if abs(in_shape[0]) > 1:
                in_shape[0] = abs(in_shape[0]) * b_size
            else:
                in_shape[0] = b_size

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
            out_shape = model.out_shapes[yezi_op]
            if abs(out_shape[0]) > 1:
                out_shape[0] = abs(out_shape[0]) * b_size
            else:
                out_shape[0] = b_size
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

        flag = (del_layer_name + "." in layer_name) and not (del_layer_name == layer_name) \
               and (layer_name in Basic_op_names or layer_name in add_Cascade_OP_names)

        if flag:
            yezi_ops.append(layer_name)

    return yezi_ops

def remove_empty_Cascade_ops(model, Cascade_ops, Basic_ops):
    del_idxs = []
    for i in range(len(Cascade_ops)):
        c1 = Cascade_ops[i]
        flag = False
        for j in range(len(Basic_ops)):
            c2 = Basic_ops[j]
            if c1 in c2:
                flag = True
                break
        if not flag:
            del_idxs.append(i)
    del_flag = 0
    for del_idx in del_idxs:
        model.layer_names.pop(Cascade_ops[del_idx - del_flag])
        del Cascade_ops[del_idx - del_flag]
        del_flag += 1
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
        if abs(model.in_shapes[shape_key][0]) == 1:
            model.in_shapes[shape_key][0] = input_size[0]
        else:
            model.in_shapes[shape_key][0] = model.in_shapes[shape_key][0] * input_size[0]

        if abs(model.out_shapes[shape_key][0]) == 1:
            model.out_shapes[shape_key][0] = input_size[0]
        else:
            model.out_shapes[shape_key][0] = model.out_shapes[shape_key][0] * input_size[0]

    check_orderinfo_selfcorrect(model)
    return model



def check_ms_torch_modelinfo(model_ms, model_torch):
    # check layer_names
    layer_names_ms, layer_names_torch = list(model_ms.layer_names.keys()), list(model_torch.layer_names.keys())
    layer_names_ms.sort()
    layer_names_torch.sort()
    assert layer_names_ms == layer_names_torch

    # check input_shapes
    in_shapes_ms, in_shapes_torch = model_ms.in_shapes, model_torch.in_shapes,
    inshape_keys_ms, inshape_keys_torch = list(in_shapes_ms.keys()), list(in_shapes_torch.keys())
    inshape_keys_ms.sort()
    inshape_keys_torch.sort()
    assert inshape_keys_ms == inshape_keys_torch
    for inshape_key in in_shapes_torch:
        # print("inshape_key: " + str(inshape_key))
        in_shape_ms, in_shape_torch = tuple(in_shapes_ms[inshape_key]), tuple(in_shapes_torch[inshape_key])
        assert in_shape_ms[1:] == in_shape_torch[1:]
        assert abs(in_shape_ms[0]) == abs(in_shape_torch[0])

    # check output_shapes
    out_shapes_ms, out_shapes_torch = model_ms.out_shapes, model_torch.out_shapes,
    outshape_keys_ms, outshape_keys_torch = list(out_shapes_ms.keys()), list(out_shapes_torch.keys())
    outshape_keys_ms.sort()
    outshape_keys_torch.sort()
    assert outshape_keys_ms == outshape_keys_torch
    for outshape_key in out_shapes_torch:
        # print("outshape_key: "+str(outshape_key))
        out_shape_ms, out_shape_torch = tuple(out_shapes_ms[outshape_key]), tuple(out_shapes_torch[outshape_key])
        assert out_shape_ms[1:] == out_shape_torch[1:]
        assert abs(out_shape_ms[0]) == abs(out_shape_torch[0])

    assert inshape_keys_ms == outshape_keys_torch
    assert outshape_keys_ms == inshape_keys_torch
    assert inshape_keys_torch == outshape_keys_torch
    assert inshape_keys_ms == outshape_keys_ms

    layer_names_ms_left = list(set(layer_names_ms) - set(model_ms.Cascade_OPs))

    for layer_name_ms_left in layer_names_ms_left:
        if "INPUT" in layer_name_ms_left or "OUTPUT" in layer_name_ms_left or (
        not layer_name_ms_left in model_ms.Basic_OPS):
            continue

        # print("inshape_key_ms check layer_names: "+str(layer_name_ms_left))
        assert layer_name_ms_left in in_shapes_torch
        assert layer_name_ms_left in out_shapes_torch

        assert layer_name_ms_left in in_shapes_ms
        assert layer_name_ms_left in out_shapes_ms

    # check orders
    orders_ms, orders_torch = list(model_ms.orders.keys()), list(model_torch.orders.keys())
    orders_ms.sort()
    orders_torch.sort()
    assert orders_ms == orders_torch
    for order_ms in orders_ms:
        # print("order_key: "+str(order_ms))
        tuopu_info_ms, tuopu_info_torch = model_ms.orders[order_ms], model_torch.orders[order_ms]
        assert tuopu_info_torch[0] == tuopu_info_ms[0]
        assert tuopu_info_torch[1] == tuopu_info_ms[1]

    shape_keys_set = set(inshape_keys_torch)
    order_keys_set = set(orders_ms)

    left_names = shape_keys_set - order_keys_set
    print("left names: " + str(left_names))
    for name in left_names:
        print("name: ", name)
        assert ("INPUT" in name or "OUTPUT" in name)


if __name__ == '__main__':
    # model_name = "resnet50"
    # device = "CPU"
    # model_ms, model_t = get_model(model_name, device, device_id=0, input_size=(1, 3, 224, 224))

    # from network.recommend.wide_and_deep.Wide_Deep_config import config
    # config.batch_size = 1
    # from network.recommend.wide_and_deep.wide_and_deep_torch import WideDeepModel
    # model_t = WideDeepModel(config)

    model_name = "auto_encoder"
    cfg=config

    # model1 = model_ms()
    # model2 = model_torch()
    device = "CPU"
    auto_encoder = AutoEncoder_torch(cfg).to(device)
    auto_encoder = model_prepare(auto_encoder, (1, 3, 256, 256))
    auto_encoder.eval()
    # weight_dict = torch.load('vgg19-0-97_5004.pth')
    # auto_encoder.load_state_dict(weight_dict)


    auto_encoder_ms = AutoEncoder_ms(cfg)
    auto_encoder_ms = model_prepare(auto_encoder_ms, (1, 3, 256, 256))
    # auto_encoder_ms.set_train(False)
    # mindspore.load_checkpoint("vgg19-0-97_5004.ckpt", auto_encoder_ms)

    input_shape = (1, 3, 256, 256)
    model_input_size = [input_shape]
    model_dtypes_t = [torch.float32]
    model_dtypes_ms = [mindspore.float32]

    result, global_layer_info = torchinfoplus.summary(model=auto_encoder, input_size=model_input_size,
                                                      dtypes=model_dtypes_t,
                                                      col_names=['input_size', 'output_size', 'name'], depth=10,
                                                      verbose=1)
    in_shapes2, out_shapes2 = torchinfoplus.get_input_size(global_layer_info), torchinfoplus.get_output_size(global_layer_info)
    dtypes_dict2 = torchinfoplus.get_dtypes(global_layer_info)
    input_dict2 = torchinfoplus.get_input_datas(global_layer_info)
    out_dict2 = torchinfoplus.get_output_datas(global_layer_info)
    exits2 = torchinfoplus.get_primitive_orders(global_layer_info)
    output_data2 = torchinfoplus.get_output_datas(global_layer_info)

    in_shapes2['INPUT'] = list(input_shape)
    in_shapes2['OUTPUT'] = [1, 10]
    out_shapes2['INPUT'] = list(input_shape)
    out_shapes2['OUTPUT'] = [1, 10]

########################################################

    result, global_layer_info = mindsporeinfoplus.summary_plus(model = auto_encoder_ms, input_size=model_input_size,
                                                      dtypes=model_dtypes_ms,
                                                      col_names=['input_size', 'output_size', 'name'], depth=10,
                                                      verbose=1)

    in_shapes1, out_shapes1 = mindsporeinfoplus.get_input_size(global_layer_info), torchinfoplus.get_output_size(global_layer_info)
    dtypes_dict1 = mindsporeinfoplus.get_dtypes(global_layer_info)
    input_dict1 = mindsporeinfoplus.get_input_datas(global_layer_info)
    out_dict1 = mindsporeinfoplus.get_output_datas(global_layer_info)
    orders1 = mindsporeinfoplus.get_primitive_orders(global_layer_info)
    output_data1 = mindsporeinfoplus.get_output_datas(global_layer_info)

    in_shapes1['INPUT'] = list(input_shape)
    in_shapes1['OUTPUT'] = [1, 10]
    out_shapes1['INPUT'] = list(input_shape)
    out_shapes1['OUTPUT'] = [1, 10]


    util.write_layernames(auto_encoder, name=model_name)
    util.write_setmethod(auto_encoder, name=model_name)


    util.check_orderinfo_selfcorrect(auto_encoder_ms)
    util.check_orderinfo_selfcorrect(auto_encoder)
    util.check_layers_and_shapes(auto_encoder_ms)
    util.check_layers_and_shapes(auto_encoder)
    # compare the models constructed by two frameworks
    check_ms_torch_modelinfo(auto_encoder, auto_encoder_ms)

