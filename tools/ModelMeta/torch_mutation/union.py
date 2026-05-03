import json
import os
def union_json(single_json_path, all_json_path):
    """
    single_json_path:
    all_json_path:
    """
    with open(single_json_path, 'r') as json_file:
        model_info = json.load(json_file)
    if not os.path.exists(all_json_path):
        all_layer_info = {}
    else:
        with open(all_json_path, 'r') as all_json_file:
            all_layer_info = json.load(all_json_file)

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
                    all_layer_info['layer_input_info'][layer_class][attr] = list(
                        set(layer_input_info[attr]).union(set(all_layer_info['layer_input_info'][layer_class][attr])))

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
        all_layer_info['layer_type'] = model_info['layer_type']
    else:
        all_layer_info['layer_type'] = list(set(model_info['layer_type']).union(set(all_layer_info['layer_type'])))

    if 'max_edge_num' not in all_layer_info.keys():
        all_layer_info['max_edge_num'] = model_info['cur_edge_num']
    else:
        all_layer_info['max_edge_num'] = max(all_layer_info['max_edge_num'], model_info['cur_edge_num'])

    if 'max_layer_num' not in all_layer_info.keys():
        all_layer_info['max_layer_num'] = model_info['layer_num']
    else:
        all_layer_info['max_layer_num'] = max(all_layer_info['max_layer_num'], model_info['layer_num'])

    with open(all_json_path, 'w') as json_file:
        json.dump(all_layer_info, json_file, indent=4)

if __name__ == '__main__':
    single_json_path = r"F:\NEW\比赛\项目\MR2023-master\torch_mutated_net\YOLOV3DarkNet53\2023_12_13_21_52_05\model_json\model_0.json"
    all_json_path = r"F:\NEW\比赛\项目\MR2023-master\torch_mutated_net\YOLOV3DarkNet53\2023_12_13_21_52_05\model_json\model_8.json"
    union_json(single_json_path,all_json_path)