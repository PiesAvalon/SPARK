import random
from copy import deepcopy
import numpy as np
import yaml
import torch
from common.mutation_torch.Layer_utils import *
from common.mutation_ms.OP_parameter_mutate_utils import get_new_basicop, get_new_cascadeop

if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET']=='GPU':
    final_device = f'cuda:0'
else:
    final_device = 'cpu'


def update_params(old_op, new_op):
    attrs_list = list(old_op.__dict__.items())
    edit_flag = False
    for i in range(len(attrs_list)):
        if "grad_ops_label" in attrs_list[i][0]:
            edit_flag = True
            continue
        if edit_flag and "grad_ops_label" not in attrs_list[i][0]:
            if "Prim" in str(attrs_list[i][1]) and "<" in str(attrs_list[i][1]):
                edit_flag = False
                continue
            if hasattr(new_op, attrs_list[i][0]):
                setattr(new_op, str(attrs_list[i][0]), attrs_list[i][1])
    return new_op


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


def find_Cascade_OP_shape(model, input_size, del_layer_name, yezi_ops):
    first_childs, final_childs = [], []
    last_ops, next_ops = [], []
    input_shapes, out_shapes = [], []
    for yezi_op in yezi_ops:
        qianqu_info = model.get_order(yezi_op)[0]
        houji_info = model.get_order(yezi_op)[1]

        flag = True
        if isinstance(qianqu_info, list):
            for qianqu_info_single in qianqu_info:
                flag1 = True
                if not (del_layer_name in qianqu_info_single):
                    flag = False
                    flag1 = False
                if not flag1:
                    last_ops.append(qianqu_info_single)
        else:
            if del_layer_name not in qianqu_info:
                flag = False
                last_ops.append(qianqu_info)

        if not flag:
            first_childs.append(yezi_op)
            in_shape = model.in_shapes[yezi_op]
            in_shape[0] = input_size[0]
            input_shapes.append(in_shape)

        flag = True
        if isinstance(houji_info, list):
            for houji_info_single in houji_info:
                flag2 = True
                if not (del_layer_name in houji_info_single):
                    flag = False
                    flag2 = False

                if not flag2:
                    next_ops.append(houji_info_single)

        else:
            if not (del_layer_name in houji_info):
                flag = False
                next_ops.append(houji_info)

        if not flag:
            final_childs.append(yezi_op)
            out_shape = model.out_shapes[yezi_op]
            out_shape[0] = input_size[0]
            out_shapes.append(out_shape)

    last_ops, next_ops = list(set(last_ops)), list(set(next_ops))

    input_shapes_str, out_shapes_str = [], []
    for val in input_shapes:
        input_shapes_str.append(str(val)[1:-1])
    for val in out_shapes:
        out_shapes_str.append(str(val)[1:-1])

    input_shapes, out_shapes = list(set(input_shapes_str)), list(set(out_shapes_str))

    return last_ops, next_ops, input_shapes, out_shapes


def find_Child_leaf_OP(layer_names, del_layer_name, Basic_op_names, add_Cascade_OP_names):
    yezi_ops = []
    for layer_name in layer_names:
        if "_del" in layer_name or "empty" in layer_name:
            continue

        flag = (del_layer_name in layer_name) and not (del_layer_name == layer_name) and \
               (layer_name in Basic_op_names or layer_name in add_Cascade_OP_names)
        if flag:
            yezi_ops.append(layer_name)

    return yezi_ops


def judge_legenacy(model, input_size, train_configs):
    batch_size = input_size[0]
    input_sizes = train_configs["input_size"]
    model_name = train_configs["model_name"]
    test_inputs = []
    dtype_list = train_configs['dtypes']

    assert len(input_sizes) == len(dtype_list)

    for i in range(len(input_sizes)):
        input_size_each = input_sizes[i]
        dtype_str = dtype_list[i]

        input_size_str = input_size_each[1:-1].split(",")
        input_size_list = []
        for val in input_size_str:
            if val == "":
                continue
            input_size_list.append(int(val))
        input_size_list = tuple(input_size_list)
        input_size_tuple = [batch_size] + list(input_size_list[1:])
        test_data_numpy = np.ones(input_size_tuple)

        if dtype_str == "float":
            dtype = torch.float32
            test_input = torch.tensor(test_data_numpy, dtype=dtype).to(final_device)
        elif dtype_str == "int":
            dtype = torch.int64
            test_input = torch.tensor(test_data_numpy, dtype=dtype).to(final_device)
        elif dtype_str == "bool":
            dtype = torch.bool
            test_input = torch.tensor(test_data_numpy, dtype=dtype).to(final_device)

        test_inputs.append(test_input)

    try:
        if model_name== "pinns":
            model(*test_inputs)
        else:
            with torch.no_grad():
                model(*test_inputs)
        return True
    except Exception as e:
        return False


def LD_update_Cascade_lastandnext_info(model, last_ops, next_ops, del_layer_name):
    if not isinstance(last_ops, list):
        if not ("INPUT" in last_ops):
            lastop_houjiinfo = model.get_order(last_ops)[1]

            if isinstance(lastop_houjiinfo, list):
                # delete all of the childrens of del_layer_name
                lastop_houjiinfo = del_Cascade_op_info(lastop_houjiinfo, del_layer_name)

                if isinstance(next_ops, list):
                    lastop_houjiinfo_new = next_ops + lastop_houjiinfo
                else:
                    lastop_houjiinfo_new = [next_ops] + lastop_houjiinfo
                model.orders[last_ops] = [model.orders[last_ops][0], lastop_houjiinfo_new]
            else:
                model.orders[last_ops] = [model.orders[last_ops][0], next_ops]

    else:
        for last_op_single in last_ops:
            if "INPUT" in last_op_single:
                continue
            lastop_houjiinfo = model.get_order(last_op_single)[1]
            if isinstance(lastop_houjiinfo, list):
                lastop_houjiinfo = del_Cascade_op_info(lastop_houjiinfo, del_layer_name)
                if isinstance(next_ops, list):
                    lastop_houjiinfo_new = next_ops + lastop_houjiinfo
                else:
                    lastop_houjiinfo_new = [next_ops] + lastop_houjiinfo
                model.orders[last_op_single] = [model.orders[last_op_single][0], lastop_houjiinfo_new]
            else:
                model.orders[last_op_single] = [model.orders[last_op_single][0], next_ops]

    # update the information of next_ops
    if not isinstance(next_ops, list):
        if not ("OUTPUT" in next_ops):
            nextop_qianquinfo = model.get_order(next_ops)[0]
            if isinstance(nextop_qianquinfo, list):
                nextop_qianquinfo = del_Cascade_op_info(nextop_qianquinfo, del_layer_name)
                if isinstance(last_ops, list):
                    nextop_qianquinfo_new = last_ops + nextop_qianquinfo
                else:
                    nextop_qianquinfo_new = [last_ops] + nextop_qianquinfo
                model.orders[next_ops] = [nextop_qianquinfo_new, model.orders[next_ops][1]]
            else:
                model.orders[next_ops] = [last_ops, model.orders[next_ops][1]]

    else:
        for next_op_single in next_ops:
            if "OUTPUT" in next_op_single:
                continue
            nextop_qianquinfo = model.get_order(next_op_single)[0]
            if isinstance(nextop_qianquinfo, list):
                nextop_qianquinfo = del_Cascade_op_info(nextop_qianquinfo, del_layer_name)
                if isinstance(last_ops, list):
                    nextop_qianquinfo_new = last_ops + nextop_qianquinfo
                else:
                    nextop_qianquinfo_new = [last_ops] + nextop_qianquinfo
                model.orders[next_op_single] = [nextop_qianquinfo_new, model.orders[next_op_single][1]]
            else:
                model.orders[next_op_single] = [last_ops, model.orders[next_op_single][1]]


def RA_update_Cascade_lastandnext_info(model, last_ops, next_ops, del_layer_name):
    if not isinstance(last_ops, list):
        if not ("INPUT" in last_ops):
            lastop_houjiinfo = model.get_order(last_ops)[1]
            if isinstance(lastop_houjiinfo, list):
                lastop_houjiinfo = del_Cascade_op_info(lastop_houjiinfo, del_layer_name)
                lastop_houjiinfo.append(del_layer_name)
                model.orders[last_ops] = [model.orders[last_ops][0], lastop_houjiinfo]
            else:
                model.orders[last_ops] = [model.orders[last_ops][0], del_layer_name]

    else:
        for last_op_single in last_ops:
            if "INPUT" in last_op_single:
                continue
            lastop_houjiinfo = model.get_order(last_op_single)[1]
            if isinstance(lastop_houjiinfo, list):
                lastop_houjiinfo = del_Cascade_op_info(lastop_houjiinfo, del_layer_name)
                lastop_houjiinfo.append(del_layer_name)
                model.orders[last_op_single] = [model.orders[last_op_single][0], lastop_houjiinfo]
            else:
                model.orders[last_op_single] = [model.orders[last_op_single][0], del_layer_name]

    if not isinstance(next_ops, list):
        if not ("OUTPUT" in next_ops):
            nextop_qianquinfo = model.get_order(next_ops)[0]
            if isinstance(nextop_qianquinfo, list):
                nextop_qianquinfo = del_Cascade_op_info(nextop_qianquinfo, del_layer_name)
                nextop_qianquinfo.append(del_layer_name)
                model.orders[next_ops] = [nextop_qianquinfo, model.orders[next_ops][1]]
            else:
                model.orders[next_ops] = [del_layer_name, model.orders[next_ops][1]]

    else:
        for next_op_single in next_ops:
            if "OUTPUT" in next_op_single:
                continue
            nextop_qianquinfo = model.get_order(next_op_single)[0]
            if isinstance(nextop_qianquinfo, list):
                nextop_qianquinfo = del_Cascade_op_info(nextop_qianquinfo, del_layer_name)
                nextop_qianquinfo.append(del_layer_name)
                model.orders[next_op_single] = [nextop_qianquinfo, model.orders[next_op_single][1]]
            else:
                model.orders[next_op_single] = [del_layer_name, model.orders[next_op_single][1]]


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


def del_Cascade_op_info(qianqu_houji_list, del_layer_name):
    """
    delete del_layer_name itself and all of its childs
    """
    del_idxs = []
    for idx in range(len(qianqu_houji_list)):
        qianqu_houji_list_single = qianqu_houji_list[idx]
        if del_layer_name in qianqu_houji_list_single:
            del_idxs.append(idx)

    del_flag = 0
    for idx in del_idxs:
        del qianqu_houji_list[idx - del_flag]
        del_flag += 1
    return qianqu_houji_list

def get_alternative_Basicops(in_shape, out_shape, mut_type):

    if mut_type == "CM" or mut_type == "RA":
        insert_layer_inshape = deepcopy(in_shape)
        insert_inchannels = in_shape[1]
    elif mut_type == "LA" or mut_type == "LC":
        insert_layer_inshape = deepcopy(out_shape)
        insert_inchannels = out_shape[1]
    insert_outchannels = out_shape[1]

    kernel_size = 1
    stride = 1

    dimension = "2D"
    if len(insert_layer_inshape) == 5:
        dimension = "3D"

    alternative_insert_layers = BasicOPUtils().extension_ops
    activation_names = BasicOPUtils().activation_names
    alternative_insert_layers['conv'] = BasicOPUtils.available_convs
    alternative_insert_layers['batchnorm'] = BasicOPUtils.available_BN
    alternative_insert_layers['pool'] = BasicOPUtils.available_pool

    del alternative_insert_layers['embedding']
    if not dimension == "2D":
        del alternative_insert_layers['pool']
        del alternative_insert_layers['linear']
        #have not add layernorm


    alternative_insert_layers_instances = get_new_basicop(alternative_insert_layers,  activation_names, insert_inchannels, insert_outchannels, kernel_size, stride, dimension)
    return alternative_insert_layers_instances




def get_alternative_Cascadeops(in_shape, out_shape, mut_type, activation_name):
    if mut_type == "CM" or mut_type == "RA":
        insert_layer_inshape = deepcopy(in_shape)
        insert_inchannels = in_shape[1]
    elif mut_type == "LA" or mut_type == "LC":
        insert_layer_inshape = deepcopy(out_shape)
        insert_inchannels = out_shape[1]
    insert_outchannels = out_shape[1]

    kernel_size = 1

    alternative_insert_layers = CascadeOPUtils().extension_ops
    alternative_insert_layers = get_new_cascadeop(alternative_insert_layers, insert_inchannels, insert_outchannels, kernel_size, stride=1, activation=activation_name.lower())
    return alternative_insert_layers



def get_lubrication_op(inshape, layer, input_size):
    layer_type = str(layer.__class__.__name__).lower()

    pro = 1
    for val in inshape:
        pro *= val
    product = pro

    newshape = tuple()

    if layer_type in CascadeOPUtils.get_cascadeops_names() and layer_type not in ["se", 'inception_a']:
        size = int(pow(int(product / input_size[0] / layer.in_channels), 0.5))
        newshape = (input_size[0], layer.in_channels, size, size)

    elif layer_type == "se" or layer_type == "inception_a":
        size = int(pow(int(product / input_size[0] / input_size[1]), 0.5))
        newshape = (input_size[0], input_size[1], size, size)

    # Basic ops
    elif "conv" in layer_type:
        size = int(pow(int(product / input_size[0] / layer.in_channels), 0.5))
        if "3d" in layer_type and (not len(inshape) == 5):
            newshape = (input_size[0], layer.in_channels, 1, size, size)
        elif "2d" in layer_type and (not len(inshape) == 4):
            newshape = (input_size[0], layer.in_channels, size, size)

    elif "batchnorm" in layer_type:
        size = int(pow(int(product / input_size[0] / inshape[1]), 0.5))
        if "3d" in layer_type and (not len(inshape) == 5):
            newshape = (input_size[0], inshape[1], 1, size, size)
        elif "2d" in layer_type and (not len(inshape) == 4):
            newshape = (input_size[0], layer.num_features, size, size)

    elif "pool" in layer_type:
        size = int(pow(int(product / input_size[0] / input_size[1]), 0.5))
        if "3d" in layer_type and (not len(inshape) == 5):
            newshape = (input_size[0], input_size[1], 1, size, size)

        elif "2d" in layer_type and (not len(inshape) == 4):
            newshape = (input_size[0], input_size[1], size, size)

    if len(newshape) == 0:
        return
    else:
        return Replace_torch(inshape, newshape)


def set_layer(model, layer, mutlayer_name, mut_type):
    try:
        model.set_layers(mutlayer_name, layer)
        return True
    except Exception as e:
        return "{} set layers failure!".format(mut_type)


def create_replacecell(input_shape, output_shape):
    replace_cell = Replace_torch(input_shape, output_shape)
    return replace_cell
