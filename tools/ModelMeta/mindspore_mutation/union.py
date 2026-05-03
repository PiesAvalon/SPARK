import json
import os

def make_hashable(input_list):
    # 将列表中的每个元素转换为元组，如果元素本身是列表
    return [tuple(item) if isinstance(item, list) else item for item in input_list]

def union_json(single_json_path, all_json_path):
    """
    single_json_path:单个json文件的路径
    all_json_path:分母json文件的路径
    """
    with open(single_json_path, 'r') as json_file:
        model_info = json.load(json_file)
    if not os.path.exists(all_json_path):
        all_layer_info = {}
    else:
        with open(all_json_path, 'r') as all_json_file:
            all_layer_info = json.load(all_json_file)

    if 'edges' not in all_layer_info.keys():
        all_layer_info['edges'] = []
        edge_list = []
    else:
        edge_list = all_layer_info['edges']
    for i in model_info["edges"]:
        if i not in edge_list:
            edge_list.append(i)
    all_layer_info['edges'] = edge_list

    if 'layer_config' not in all_layer_info.keys():
        all_layer_info['layer_config'] = {}

    for layer_class, layer_configs in model_info['layer_config'].items():
        if layer_class not in all_layer_info['layer_config'].keys():
            all_layer_info['layer_config'][layer_class] = layer_configs
        else:
            for layer_config in layer_configs:
                if layer_config not in all_layer_info['layer_config'][layer_class]:
                    all_layer_info['layer_config'][layer_class].append(layer_config)

    if 'layer_input_info' not in all_layer_info.keys():
        all_layer_info['layer_input_info'] = {}
    for layer_class, layer_input_info in model_info['layer_input_info'].items():
        if layer_class not in all_layer_info['layer_input_info'].keys():
            all_layer_info['layer_input_info'][layer_class] = layer_input_info
        else:
            for attr in ["input_dims", "dtype", "shape"]:
                if attr not in all_layer_info['layer_input_info'][layer_class].keys():
                    all_layer_info['layer_input_info'][layer_class][attr] = layer_input_info[attr]
                else:
                    hashable_layer_input_info = make_hashable(layer_input_info[attr])
                    hashable_all_layer_info = make_hashable(all_layer_info['layer_input_info'][layer_class][attr])
                    
                    all_layer_info['layer_input_info'][layer_class][attr] = list(set(hashable_layer_input_info).union(set(hashable_all_layer_info)))

                    # all_layer_info['layer_input_info'][layer_class][attr] = list(
                    #     set(layer_input_info[attr]).union(set(all_layer_info['layer_input_info'][layer_class][attr])))

    if 'layer_dims' not in all_layer_info.keys():
        all_layer_info['layer_dims'] = {}
    for layer_class, layer_dims in model_info['layer_dims'].items():
        if layer_class not in all_layer_info['layer_dims'].keys():
            all_layer_info['layer_dims'][layer_class] = layer_dims
        else:
            for attr in ["input_dims", "output_dims"]:
                if attr not in all_layer_info['layer_dims'][layer_class].keys():
                    all_layer_info['layer_dims'][layer_class][attr] = layer_dims[attr]
                else:
                    all_layer_info['layer_dims'][layer_class][attr] = list(
                        set(layer_dims[attr]).union(set(all_layer_info['layer_dims'][layer_class][attr])))

    if 'layer_type' not in all_layer_info.keys():
        for index,i in enumerate(model_info['layer_type']):

            model_info['layer_type'][index] = i.split("Opt")[0]
        all_layer_info['layer_type'] = model_info['layer_type']
    else:
        for index,i in enumerate(model_info['layer_type']):

            model_info['layer_type'][index] = i.split("Opt")[0]
        all_layer_info['layer_type'] = list(set(model_info['layer_type']).union(set(all_layer_info['layer_type'])))

    # if 'max_edge_num' not in all_layer_info.keys():
    #     all_layer_info['max_edge_num'] = model_info['cur_edge_num']
    # else:
    #     all_layer_info['max_edge_num'] = max(all_layer_info['max_edge_num'], model_info['cur_edge_num'])

    # if 'max_layer_num' not in all_layer_info.keys():
    #     all_layer_info['max_layer_num'] = model_info['layer_num']
    # else:
    #     all_layer_info['max_layer_num'] = max(all_layer_info['max_layer_num'], model_info['layer_num'])

    if 'cur_edge_num' not in all_layer_info.keys():
        all_layer_info['cur_edge_num'] = model_info['cur_edge_num']
    else:
        all_layer_info['cur_edge_num'] = max(all_layer_info['cur_edge_num'], model_info['cur_edge_num'])

    if 'layer_num' not in all_layer_info.keys():
        all_layer_info['layer_num'] = model_info['layer_num']
    else:
        all_layer_info['layer_num'] = max(all_layer_info['layer_num'], model_info['layer_num'])

    with open(all_json_path, 'w') as json_file:
        json.dump(all_layer_info, json_file, indent=4)

if __name__ == '__main__':
    # folder_path = "./data_all_new_ms/"
    all_json_path = "./ms_data/log/random/data_all_random.json"
    # for file in os.listdir(folder_path):
    #     if file != 'all_layer_info.json':
    #         file_path = os.path.join(folder_path, file)
    #         union_json(file_path, all_json_path)

    folder_path = './ms_data/log/random'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    union_json(file_path, all_json_path)
                except Exception as e:
                    print(file_path)
    # with open("./data6_all_layer_info.json", 'r') as json_file:
    #     model_info = json.load(json_file)
    # key1 = model_info["layer_type"]

    # with open("./tensorflow_api_config_pool.json", 'r') as json_file:
    #     model_info = json.load(json_file)
    # key2 = model_info.keys()
    # for i in key2:
    #     if i not in key1:
    #         print(i) 