import os

import mindspore
import numpy as np

from mindspore import Tensor
import json
from Coverage import CoverageCalculatornew
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
import networkx as nx


mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")

def traverse_network(new_net, layer_config):
    layer_type_list = []
    current_layer_num = 0
    layers = new_net.cells_and_names()
    type_list = [bool, str, int, tuple, list, float, np.ndarray, Tensor]
    edit_flag = False
    ans = {}
    layer_config_new = {}
    for i, j in layers:

        if i != '':
            if not has_child_node(new_net, i):
                current_layer_num += 1
                attrs_list = list(j.__dict__.items())
                for i in range(len(attrs_list)):
                    if "grad_ops_label" in attrs_list[i][0]:
                        edit_flag = True
                        continue
                    if edit_flag and "grad_ops_label" not in attrs_list[i][0]:
                        if "Prim" in str(attrs_list[i][1]) and "<" in str(attrs_list[i][1]):
                            edit_flag = False
                            continue
                        ans[attrs_list[i][0]] = getattr(j, attrs_list[i][0]) \
                            if type(getattr(j, attrs_list[i][0])) in type_list else None
                if j.__class__.__name__ not in layer_config.keys():
                    layer_config[j.__class__.__name__] = []
                layer_config[j.__class__.__name__].append(ans)
                if j.__class__.__name__.split('Opt')[0] not in layer_type_list:
                    layer_type_list.append(j.__class__.__name__.split('Opt')[0])
    for key, value in layer_config.items():
        # try:
        layer_config_new[key.split('Opt')[0]] = [
            dict(t) for t in {
                tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items()) for d in layer_config[key]
            }
        ]

        # except Exception as e:
        #     print(e)
        #     {tuple(d.items()) for d in layer_config[key]}
        #     print("success")
    return layer_config_new, layer_type_list, current_layer_num


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def ms_model2json(new_net, input_tensor, dtypes):
    res, global_layer_info, summary_list = mindsporeinfoplus.summary_plus(
        model=new_net,
        input_data=input_tensor,
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        mode="train",
        verbose=0,
        depth=10
    )
    output_datas = mindsporeinfoplus.get_output_datas(global_layer_info)
    model_json = {}
    edge_list_list = []
    cur_edge_num = 0
    index = 0
    index_next = 1
    while 1:
        if int(summary_list[index].depth) >= 1:
            edge_list = []
            for index_next in range(1, len(summary_list)):
                if summary_list[index + index_next].children == []:
                    break

            input_type = summary_list[index].class_name
            if index_next == 1:
                index += 1
                if index == len(summary_list) - 1:
                    break
                else:
                    output_type = summary_list[index].class_name
            else:
                index = index + index_next
                if index == len(summary_list) - 1:
                    break
                output_type = summary_list[index].class_name
            edge_list.append(input_type.split('Opt')[0])
            edge_list.append(output_type.split('Opt')[0])
            cur_edge_num += 1
            if edge_list not in edge_list_list:
                edge_list_list.append(edge_list)
        else:
            index += 1
            if index == len(summary_list) - 1:
                break

    model_json["edges"] = edge_list_list
    layer_config = {}
    layer_config, layer_type_list, current_layer_num = traverse_network(new_net, layer_config)
    model_json["layer_config"] = layer_config
    layer_input_info = {}
    for layer_info in summary_list:
        layer_input_info_dist = {}
        if int(layer_info.depth) >= 1:
            input_name = layer_info.class_name.split('Opt')[0]
            if len(layer_info.input_size) == 0:
                continue
            input_size = layer_info.input_size[0]
            len_input_size = len(input_size)
            if input_name not in layer_input_info.keys():
                layer_input_info_dist["input_dims"] = [len_input_size]
                layer_input_info_dist["dtype"] = [str(dtypes)]
                layer_input_info_dist["shape"] = [str(input_size)]
                layer_input_info[input_name] = layer_input_info_dist
            else:
                if len_input_size not in layer_input_info[input_name]["input_dims"]:
                    layer_input_info[input_name]["input_dims"].append(len_input_size)
                if str(dtypes) not in layer_input_info[input_name]["dtype"]:
                    layer_input_info[input_name]["dtype"].append(str(dtypes))
                if str(input_size) not in layer_input_info[input_name]["shape"]:
                    layer_input_info[input_name]["shape"].append(str(input_size))

    model_json["layer_input_info"] = layer_input_info
    model_json["cur_edge_num"] = cur_edge_num
    model_json["layer_num"] = current_layer_num
    model_json["layer_type"] = layer_type_list

    layer_dims = {}
    for layer_info in summary_list:
        if int(layer_info.depth) >= 1:
            input_name = layer_info.class_name.split('Opt')[0]
            if len(layer_info.input_size) == 0:
                continue
            input_size = layer_info.input_size[0]
            output_size = layer_info.output_size
            len_input_size = len(input_size)
            len_output_size = len(output_size)
            if input_name not in layer_dims:
                layer_dims[input_name] = {
                    "input_dims": [len_input_size],
                    "output_dims": [len_output_size]
                }
            else:
                if len_input_size not in layer_dims[input_name]["input_dims"]:
                    layer_dims[input_name]["input_dims"].append(len_input_size)
                if len_output_size not in layer_dims[input_name]["output_dims"]:
                    layer_dims[input_name]["output_dims"].append(len_output_size)

    model_json["layer_dims"] = layer_dims
    inside = model_json["layer_num"] / model_json["cur_edge_num"]
    return model_json, inside, output_datas


def has_child_node(net, node_name):
    layers = net.cells_and_names()
    parent_node = None
    for name, _ in layers:
        if name == node_name:
            parent_node = name
            continue
        if parent_node is not None and name.startswith(parent_node + '.'):
            return True
    return False


def model2cov(model, input, dtype, file_path_1, all_json_path, api_config_pool_path, folder_path):
    model_json_1, inside, output_datas = ms_model2json(model, input, dtype)
    with open(file_path_1, 'w', encoding='utf-8') as file:
        json.dump(model_json_1, file, ensure_ascii=False, indent=4)
    file.close()

    all_json_path = "/root/MR/mutation_mindspore/all_layer_info.json"

    cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
    cal_cov.load_json(file_path_1)
    input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov = cal_cov.cal_coverage()
    return input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas


def calc_inner_div(model):
    graph = nx.DiGraph()
    for name, node in model.cells_and_names:
        graph.add_node(name)
        graph.add_edge(node, name)
    longest_path = nx.dag_longest_path(graph)
    return len(longest_path) / len(graph)


if __name__ == "__main__":
    mindspore.set_context(pynative_synchronize=True)
    model_1 = vgg11()
    model_2 = resnet50(10)
    tar_set = set()
    input_dtypes = [mindspore.float32]

    input_1 = np.random.randn(1, 5, 3, 32, 32)
    input_2 = np.random.randn(1, 5, 3, 224, 224)

    input_dtypes_1 = [mindspore.float32 for _ in input_1.shape]
    input_dtypes_2 = [mindspore.float32 for _ in input_1.shape]

    data1 = mindsporeinfoplus.np_2_tensor(input_1, input_dtypes_1)
    data2 = mindsporeinfoplus.np_2_tensor(input_2, input_dtypes_2)

    model_json_1 = ms_model2json(model_1, data1, input_dtypes)
    model_json_2 = ms_model2json(model_2, data2, input_dtypes)
    inside_1 = model_json_1["layer_num"] / model_json_1["cur_edge_num"]
    inside_2 = model_json_2["layer_num"] / model_json_2["cur_edge_num"]

    tar_set_2 = set()
    if len(set(model_json_2['layer_type'])) > len(tar_set):
        tar_set_2 = set(model_json_2['layer_type'])

    outer_div = len(tar_set_2 - set(model_json_1['layer_type']))
    # exit(6666)

    with open("./json/ms_vgg11.json", 'w', encoding='utf-8') as file:
        json.dump(model_json_1, file, ensure_ascii=False, indent=4)
    file.close()
    with open("./json/ms_resnet50.json", 'w', encoding='utf-8') as file:
        json.dump(model_json_2, file, ensure_ascii=False, indent=4)
    file.close()
    folder_path = "./json/"
    api_config_pool_path = './mindspore_api_config_pool.json'
    all_json_path = "all_layer_info.json"

    # vgg11
    print('vgg11-----------------------------------')
    file_1 = "./json/ms_vgg11.json"
    cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
    cal_cov.load_json(file_1)
    cal_cov.cal_coverage()

    # resnet50
    print('resnet50-----------------------------------')
    file_2 = "./json/ms_resnet50.json"
    cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
    cal_cov.load_json(file_2)
    cal_cov.cal_coverage()
