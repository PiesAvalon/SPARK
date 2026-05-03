import mindspore
import mindspore.nn as nn
import numpy as np
from utils.infoplus.MindSporeInfoPlus import mindsporeinfoplus
from common.model_utils import get_model
from mindspore import Tensor, SymbolTree, Node


def update_params(old_op, ans_dict):
    type_list = [bool, str, int, tuple, list, float, np.ndarray, Tensor]
    attrs_list = list(old_op.__dict__.items())
    edit_flag = False
    ans = {}
    for i in range(len(attrs_list)):
        if "grad_ops_label" in attrs_list[i][0]:
            edit_flag = True
            continue
        if edit_flag and "grad_ops_label" not in attrs_list[i][0]:
            if "Prim" in str(attrs_list[i][1]) and "<" in str(attrs_list[i][1]):
                edit_flag = False
                continue
            # print(type(getattr(old_op, attrs_list[i][0])))
            ans[attrs_list[i][0]] = getattr(old_op, attrs_list[i][0]) \
                if type(getattr(old_op, attrs_list[i][0])) in type_list else None
    if old_op.__class__.__name__ not in ans_dict.keys():
        ans_dict[old_op.__class__.__name__] = set()
    ans_dict[old_op.__class__.__name__].add(str(ans))


def calculate_layer_shape(model: nn.Cell, np_data: list, model_dtypes_ms: list):
    input_data = mindsporeinfoplus.np_2_tensor(np_data, model_dtypes_ms)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=model,
        input_data=input_data,
        dtypes=model_dtypes_ms,
        col_names=['input_size', 'output_size', 'name'],
        mode="train",
        verbose=0,
        depth=10
    )
    current_layer_shape_dict = mindsporeinfoplus.get_input_size(global_layer_info)
    shape_fenzi = 0
    for key in current_layer_shape_dict.keys():
        shape_fenzi += len(current_layer_shape_dict[key])
    return shape_fenzi


def calculate_layer_dtype(model: nn.Cell, np_data: list, model_dtypes_ms: list):
    input_data = mindsporeinfoplus.np_2_tensor(np_data, model_dtypes_ms)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=model,
        input_data=input_data,
        dtypes=model_dtypes_ms,
        col_names=['input_size', 'output_size', 'name'],
        mode="train",
        verbose=0,
        depth=10
    )
    current_layer_dtype_dict = mindsporeinfoplus.get_dtypes(global_layer_info)
    dtype_fenzi = 0
    for key in current_layer_dtype_dict.keys():
        dtype_fenzi += len(current_layer_dtype_dict[key])
    return dtype_fenzi


def calculate_layer_sequence(model: nn.Cell):
    current_layer_sequence_set = set()
    stree = SymbolTree.create(model)
    if mindspore.__version__ == "2.2.0":
        head_node = stree._symbol_tree.get_head()
    else:
        head_node = stree._symbol_tree.get_head_node()
    if head_node is None:
        print("head_node None, return")
        return 0
    node: Node = head_node.get_next()
    prev_layer = None
    while node is not None:
        if node.get_instance() is not None:
            if prev_layer is not None:
                current_layer_sequence_set.add((prev_layer, node.get_instance().__class__.__name__))
            prev_layer = node.get_instance().__class__.__name__
        node = node.get_next()
    return len(current_layer_sequence_set)


def calculate_op_num(model: nn.Cell):
    current_op_list = []
    for _, cell in model.cells_and_names():
        current_op_list.append(type(cell))
    return len(current_op_list)


def calculate_op_type(model: nn.Cell):
    current_op_set = set()
    stree = SymbolTree.create(model)
    if mindspore.__version__ == "2.2.0":
        head_node = stree._symbol_tree.get_head()
    else:
        head_node = stree._symbol_tree.get_head_node()
    if head_node is None:
        print("head_node None, return")
        return 0
    node: Node = head_node.get_next()
    while node is not None:
        if node.get_instance() is not None:
            current_op_set.add(node.get_instance().__class__.__name__)
        node = node.get_next()
    # print(current_op_set)
    return len(current_op_set)


def calculate_edge_num(model: nn.Cell):
    current_edge_list = []
    stree = SymbolTree.create(model)
    if mindspore.__version__ == "2.2.0":
        for in_node in stree.nodes(all_nodes=True):
            for out_node in in_node.get_users():
                current_edge_list.append(out_node.get_name())
    else:
        for in_node in stree.nodes():
            for out_node in in_node.get_users():
                current_edge_list.append(out_node.get_instance().__class__.__name__)
    return len(current_edge_list)


def calculate_all_coverage(model: nn.Cell, np_data: list, model_dtypes_ms: list):
    return calculate_layer_shape(model, np_data, model_dtypes_ms), calculate_layer_dtype \
        (model, np_data, model_dtypes_ms), calculate_layer_sequence(model), \
        calculate_op_num(model), calculate_op_type(model), calculate_edge_num(model)


if __name__ == '__main__':
    net_ms,_ = get_model(model_name="resnet50",input_size=(2,3,224,224))
    inpu_np = np.ones([2, 3, 224, 224])
    np_data = [inpu_np]
    model_dtypes_ms = [mindspore.float32]
    print(calculate_all_coverage(net_ms, np_data, model_dtypes_ms))
