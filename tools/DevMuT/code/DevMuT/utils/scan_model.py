import json
import torch
from common.model_utils import get_model
import numpy as np
import mindspore
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
from infoplus.TorchInfoPlus import torchinfoplus
import os
import yaml
import utils.util as util

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

def check_helpinfo(model_ms, model_t):
    # compare the models constructed by two frameworks
    util.check_orderinfo_selfcorrect(model_ms)
    util.check_orderinfo_selfcorrect(model_t)
    util.check_layers_and_shapes(model_ms)
    util.check_layers_and_shapes(model_t)
    check_ms_torch_modelinfo(model_ms, model_t)


if __name__ == '__main__':
    model_name = "SSDmobilenetv1"
    device = "cpu"

    save_path = "./help_data/"+model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_path = os.getcwd() + '/config/' + model_name + '.yaml'
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    input_size_lists = config['train_config']['input_size']
    model_input_size = []
    for input_size_each in input_size_lists:
        input_size_str = input_size_each[1:-1].split(",")
        input_size_list = []
        for val in input_size_str:
            if val == "":
                continue
            input_size_list.append(int(val))
        input_size_list = tuple(input_size_list)
        model_input_size.append(input_size_list)


    model1, model2 = get_model(model_name, input_size=model_input_size[0], scaned=True)

    util.check_orderinfo_selfcorrect(model1)
    util.check_orderinfo_selfcorrect(model2)
    util.check_layers_and_shapes(model1)
    util.check_layers_and_shapes(model2)
    # compare the models constructed by two frameworks
    check_ms_torch_modelinfo(model1, model2)

    # dtypes = config['train_config']['dtypes']
    # model_dtypes_ms = []
    # model_dtypes_torch = []
    # for dtype in dtypes:
    #     if dtypes[0] == "float":
    #         model_dtypes_ms.append(mindspore.float32)
    #         model_dtypes_torch.append(torch.float32)
    #     elif dtypes[0] == "int":
    #         model_dtypes_ms.append(mindspore.int32)
    #         model_dtypes_torch.append(torch.int32)
    #
    # load_external_data = False
    # if not load_external_data:
    #     np_data = [np.ones(model_input_size[i]) for i in range(len(model_input_size))] #[np.ones(model_input_size[i]) for i in range(len(model_input_size))]
    #
    #
    #
    # input_data = torchinfoplus.np_2_tensor(np_data, model_dtypes_torch, device="cuda:0")
    # res, global_layer_info = torchinfoplus.summary(
    #     model=model_t,
    #     input_data=input_data,
    #     dtypes=model_dtypes_torch,
    #     col_names=['input_size', 'output_size', 'name'],
    #     verbose=1,
    #     depth=10
    # )
    # in_shapes1, out_shapes1 = torchinfoplus.get_input_size(global_layer_info), torchinfoplus.get_output_size(global_layer_info)
    # dtypes_dict1 = torchinfoplus.get_dtypes(global_layer_info)
    # input_dict1 = torchinfoplus.get_input_datas(global_layer_info)
    # out_dict1 = torchinfoplus.get_output_datas(global_layer_info)
    # orders1 = torchinfoplus.get_primitive_orders(global_layer_info)
    # output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    #
    #
    # input_data = mindsporeinfoplus.np_2_tensor(np_data, model_dtypes_ms)
    # res, global_layer_info = mindsporeinfoplus.summary_plus(
    #         model=model_ms,
    #         input_data=input_data,
    #         dtypes=model_dtypes_ms,
    #         col_names=['input_size', 'output_size', 'name'],
    #         mode="train",
    #         verbose=1,
    #         depth=10
    # )
    #
    # in_shapes2, out_shapes2 = mindsporeinfoplus.get_input_size(global_layer_info), mindsporeinfoplus.get_output_size(global_layer_info)
    # dtypes_dict2 = mindsporeinfoplus.get_dtypes(global_layer_info)
    # input_dict2 = mindsporeinfoplus.get_input_datas(global_layer_info)
    # out_dict2 = mindsporeinfoplus.get_output_datas(global_layer_info)
    # orders2 = mindsporeinfoplus.get_primitive_orders(global_layer_info)
    # output_data2 = mindsporeinfoplus.get_output_datas(global_layer_info)
    #
    #
    # in_shapes1['INPUT'] = [1, 3, 224, 224]
    # out_shapes1['OUTPUT1'] = [1, 512, 28, 28]
    # out_shapes1['OUTPUT2'] = [1, 1024, 14, 14]
    #
    # in_shapes2['INPUT'] = [1, 3, 224, 224]
    # out_shapes2['OUTPUT1'] = [1, 512, 28, 28]
    # out_shapes2['OUTPUT2'] = [1, 1024, 14, 14]
    #
    #
    # util.write_layernames(model_ms, name=model_name, save_path=save_path)
    # util.write_setmethod(model_ms, name=model_name, save_path=save_path)
    # util.write_layernames(model_t, name=model_name, save_path=save_path)
    # util.write_setmethod(model_t, name=model_name, save_path=save_path)
    #
    #
    # json_str = json.dumps(model_ms.in_shapes)
    # with open(save_path + f'/{model_name}_ms_inshape.json', 'w') as f:
    #     f.write(json_str)
    #
    # json_str = json.dumps(model_ms.out_shapes)
    # with open(save_path + f'/{model_name}_ms_outshape.json', 'w') as f:
    #     f.write(json_str)
    #
    # json_str = json.dumps(model_ms.orders)
    # with open(save_path + f'/{model_name}_ms_order.json', 'w') as f:
    #     f.write(json_str)
    #
    # json_str = json.dumps(model_t.in_shapes)
    # with open(save_path + f'/{model_name}_torch_inshape.json', 'w') as f:
    #     f.write(json_str)
    #
    # json_str = json.dumps(model_t.out_shapes)
    # with open(save_path + f'/{model_name}_torch_outshape.json', 'w') as f:
    #     f.write(json_str)
    #
    # json_str = json.dumps(model_t.orders)
    # with open(save_path + f'/{model_name}_torch_order.json', 'w') as f:
    #     f.write(json_str)

