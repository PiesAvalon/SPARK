import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from collections import OrderedDict
import numpy as np
import _ctypes
import json

from deeplab_v3plus import DeepLabV3Plus


def ms_prod(x):
    pro = 1
    for val in x:
        pro *= val
    print("Execute Prod!")
    return pro


def cell_2_id(model):
    names = {}
    is_leaf = {}

    def make_dict(cell, cells=None, name_prefix=''):
        cell_id = str(id(cell))
        t_cells = cells if cells else set()
        if cell in t_cells:
            return

        t_cells.add(cell)
        names[cell_id] = name_prefix
        is_leaf[cell_id] = True
        yield name_prefix, cell

        for name, cell_ in cell._cells.items():
            if cell_:
                cells_name_prefix = name
                is_leaf[cell_id] = False
                if name_prefix:
                    cells_name_prefix = name_prefix + '.' + cells_name_prefix
                for ele in make_dict(cell_, t_cells, cells_name_prefix):
                    yield ele

    name_gen = make_dict(model)
    for layer in name_gen:
        print(f'layer:{layer}')
    # print(f'names:{names}')
    return names, is_leaf


def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(cell):
        def hook(cell_id, input, output):
            # print(type(cell))
            # print('hook!')
            class_name = str(cell_id)
            class_name = class_name[:cell_id.index("(")]
            # class_id = str(cell_id)[cell_id.index("(") + 1: cell_id.index(")")]
            # clazz = _ctypes.PyObj_FromPtr(int(class_id))
            # print(f'class:{clazz}')
            #
            # cells_ = clazz._cells
            # print(f'cells_:{cells_}')
            # name_cells = clazz.name_cells().values()
            # print(f'name_cells:{name_cells}')
            # print(f'cells_and_names')
            # cells_and_names = clazz.cells_and_names()
            # for cell_ in cells_and_names:
            #     print(cell_[0])

            # print(f'cell op name')
            # print(cell_id)
            # cell_and_names = cell.cells_and_names()
            # print(f'get name:{cell_and_names[0][0]}')
            # print('model._cells')
            # print(model._cells.keys())
            # print('!' * 20)
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].shape)
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.shape)[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            # print(hasattr(cell, "weight"))
            # print(hasattr(cell.weight, "size"))
            if hasattr(cell_id, "weight") and hasattr(cell_id.weight, "size"):
                print("temp1_input:" + str(cell_id.weight.shape))
                temp1 = ms_prod(cell_id.weight.shape)
                print("temp1_output:" + str(temp1))
                params += temp1
                summary[m_key]["trainable"] = cell_id.weight.requires_grad
            if hasattr(cell_id, "bias") and hasattr(cell_id.bias, "size"):
                print("temp2_input:" + str(cell_id.bias.shape))
                temp2 = ms_prod(cell_id.bias.shape)
                print("temp2_output:" + str(temp2))
                params += temp2
            summary[m_key]["nb_params"] = params

        # if (
        #         not isinstance(cell, nn.SequentialCell)
        #         and not isinstance(cell, nn.CellList)
        #         # and not (cell == model)
        # ):
        #     hooks.append(cell.register_forward_hook(hook))
        hooks.append(cell.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # if device == "cuda" and torch.cuda.is_available():
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    stdnormal = ops.StandardNormal(seed=2)
    shape = (2,) + input_size[0]
    x1 = stdnormal(shape)
    x = [x1]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    print("x type:" + str(type(x)))
    print("x len:" + str(len(x)))
    print("x[0].shape:" + str(x[0].shape))
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20} {:>25} {:>25} {:>15}".format("Layer (type)", "Input_shape", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]) + ',',
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


def summary2(model, input_size, batch_size=-1, device="cuda", names=None, is_leaf=None):
    def register_hook(cell):
        def hook(cell_id, input, output):
            # print(type(cell))
            class_name = str(cell_id)
            class_name = class_name[:cell_id.index("(")]
            class_id = str(cell_id)[cell_id.index("(") + 1: cell_id.index(")")]
            clazz = _ctypes.PyObj_FromPtr(int(class_id))
            cell_name = names.get(class_id)

            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            # cell_and_names = cell.cells_and_names()
            # print(f'get name:{cell_and_names[0][0]}')
            # print('model._cells')
            # print(model._cells.keys())
            summary[m_key] = OrderedDict()
            summary[m_key]['cell_name'] = cell_name
            summary[m_key]['is_leaf'] = is_leaf.get(class_id, False)
            summary[m_key]["input_shape"] = list(input[0].shape)
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.shape)[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            # print(hasattr(cell, "weight"))
            # print(hasattr(cell.weight, "size"))
            if hasattr(cell_id, "weight") and hasattr(cell_id.weight, "size"):
                print("temp1_input:" + str(cell_id.weight.shape))
                temp1 = ms_prod(cell_id.weight.shape)
                print("temp1_output:" + str(temp1))
                params += temp1
                summary[m_key]["trainable"] = cell_id.weight.requires_grad
            if hasattr(cell_id, "bias") and hasattr(cell_id.bias, "size"):
                print("temp2_input:" + str(cell_id.bias.shape))
                temp2 = ms_prod(cell_id.bias.shape)
                print("temp2_output:" + str(temp2))
                params += temp2
            summary[m_key]["nb_params"] = params

        # if (
        #         not isinstance(cell, nn.SequentialCell)
        #         and not isinstance(cell, nn.CellList)
        #         # and not (cell == model)
        # ):
        #     hooks.append(cell.register_forward_hook(hook))
        hooks.append(cell.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # if device == "cuda" and torch.cuda.is_available():
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    stdnormal = ops.StandardNormal(seed=2)
    shape = (2,) + input_size[0]
    x1 = stdnormal(shape)
    x = [x1]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    print("x type:" + str(type(x)))
    print("x len:" + str(len(x)))
    print("x[0].shape:" + str(x[0].shape))
    output = model(*x)
    output_shape = output.shape
    if isinstance(output_shape, tuple):
        output_shape = list(output_shape)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>30} {:>40} {:>25} {:>25} {:>15}".format("Layer (type)", 'cell_name', "Input_shape", "Output Shape",
                                                           "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>30} {:>40} {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]['cell_name']),
            str(summary[layer]["input_shape"]) + ',',
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if summary[layer]['is_leaf']:
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return summary, output_shape


def get_out_shapes(summary, input_size, output_size):
    print('self.out_shapes=', end='')
    out_shapes = {'INPUT': input_size}
    for layer in summary:
        if not summary[layer]['is_leaf']:
            continue
        out_shapes[summary[layer]['cell_name']] = summary[layer]['output_shape']
    out_shapes['OUTPUT'] = output_size
    print(out_shapes)
    return out_shapes


def get_in_shapes(summary, input_size, output_size):
    print('self.in_shapes=', end='')

    in_shapes = {'INPUT': input_size}
    for layer in summary:
        if not summary[layer]['is_leaf']:
            continue
        in_shapes[summary[layer]['cell_name']] = summary[layer]['input_shape']
    in_shapes['OUTPUT'] = output_size
    print(in_shapes)
    return in_shapes


def get_parent(cell_name: str):
    lst_point = cell_name.rfind('.')
    if lst_point == -1:
        return ''
    return cell_name[:lst_point]


def create_tree(summary):
    name_tree = {}
    for layer in summary:
        cell_name = summary[layer]['cell_name']
        if cell_name not in name_tree:
            name_tree[cell_name] = []
        parent = get_parent(cell_name)
        if not parent == '':
            if parent not in name_tree:
                name_tree[parent] = []
            name_tree[parent].append(cell_name)

    return name_tree


def add_edge(orders, pre, cur):
    if pre not in orders:
        orders[pre] = ['', '']
    if cur not in orders:
        orders[cur] = ['', '']
    orders[cur][0] = pre
    orders[pre][1] = cur


def get_orders(summary):
    r"""
    summary: dict, {m_key: Layer(type), cell_name, is_leaf}
    """
    orders = {}

    name_tree = create_tree(summary)
    # print(name_tree)
    for layer in summary:
        cell_name = summary[layer]['cell_name']
        children_list = name_tree[cell_name]
        if len(children_list) == 0:
            continue
        all_leaves = True
        for child in children_list:
            grandchild_list = name_tree[child]
            if len(grandchild_list) != 0:
                all_leaves = False
                break
        if not all_leaves:
            continue
        children_list.sort()

        for i in range(len(children_list) - 1):
            add_edge(orders, children_list[i], children_list[i + 1])

    print('self.orders=', end='')
    print(orders)
    return orders


if __name__ == '__main__':
    batch_size = 1
    input_size = [-1, 3, 513, 513]
    std_normal = ops.StandardNormal(seed=2)
    model = DeepLabV3Plus()

    print('+' * 10)
    layers = model.cells_and_names()
    for layer in layers:
        # print('=' * 10)
        # print(layer)
        # print('+' * 10)
        print(layer[0])
        # print('-' * 10)

    summary(model, input_size=tuple(input_size[1:]))

    print('-' * 50)
    names, is_leaf = cell_2_id(model)
    ms_summary, output_size = summary2(model, input_size=tuple(input_size[1:]), names=names, is_leaf=is_leaf)
    get_out_shapes(ms_summary, input_size, output_size)
    get_in_shapes(ms_summary, input_size, output_size)
    get_orders(ms_summary)
