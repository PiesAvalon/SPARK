import mindspore
import numpy as np
import torch.nn
from utils.infoplus.MindSporeInfoPlus import mindsporeinfoplus
from utils.infoplus.TorchInfoPlus import torchinfoplus


def ChebyshevDistance(x, y):
    if isinstance(x, mindspore.Tensor):
        x = x.asnumpy()
    elif isinstance(x, torch.Tensor):
        if torch.get_device(x) != "cpu":
            x = x.cpu()
        x = x.detach().numpy()
    if isinstance(y, mindspore.Tensor):
        y = y.asnumpy()
    elif isinstance(y, torch.Tensor):
        if torch.get_device(y) != "cpu":
            y = y.cpu()
        y = y.detach().numpy()

    out = np.mean(np.abs(x - y))

    return out


def compare_layer(input_data_dict_new, input_data_dict_old):

    distance_dict={}

    for layer in input_data_dict_new.keys():
        if layer == "[]" or layer == "":
            continue

        if input_data_dict_new[layer] is not None and input_data_dict_old[layer] is not None:
            layer_np_new = input_data_dict_new[layer][0]
            layer_up_old = input_data_dict_old[layer][0]

            dis = ChebyshevDistance(layer_np_new, layer_up_old)

            distance_dict[layer] = dis



    return distance_dict


def compare_models(network: mindspore.nn.Cell, net: torch.nn.Module, np_data=[np.ones([2, 3, 513, 513])],
                   ms_dtypes=[mindspore.float32], torch_dtypes=[torch.float32], device="cpu"):



    input_data = mindsporeinfoplus.np_2_tensor(np_data, ms_dtypes)

    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network,
        input_data=input_data,
        dtypes=ms_dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    net.eval()

    torch_data = torchinfoplus.np_2_tensor(np_data, torch_dtypes, device=device)

    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        dtypes=torch_dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)

    output_data1 = torchinfoplus.get_output_datas(global_layer_info)

    result = compare_layer(output_data, output_data1)

    del_keys = list(set(net.Basic_OPS) - set(result.keys()))
    for del_key in del_keys:
        result.pop(del_key)

    return result






def get_torch_outputs(net: torch.nn.Module, np_data=[np.ones([2, 3, 513, 513])], torch_dtypes=[torch.float32], device="cpu"):

    torch_data = torchinfoplus.np_2_tensor(np_data, torch_dtypes, device=device)

    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        dtypes=torch_dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)

    output_data = torchinfoplus.get_output_datas(global_layer_info)

    return output_data



def get_ms_outputs(network: mindspore.nn.Cell, np_data=[np.ones([2, 3, 513, 513])], ms_dtypes=[mindspore.float32]):



    input_data = mindsporeinfoplus.np_2_tensor(np_data, ms_dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network,
        input_data=input_data,
        dtypes=ms_dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)

    return output_data





if __name__ == '__main__':
    from network.cv.PatchCore.model_torch import wide_resnet50_2 as patchcore_t
    from network.cv.PatchCore.model_ms import wide_resnet50_2 as patchcore_ms

    data = [np.ones([1, 3, 224, 224])]
    ms_dtypes = [mindspore.float32]
    model_dtypes_t = [torch.float32]
    model_input_size = [(1, 3, 224, 224)]

    model_ms = patchcore_ms()
    model_t = patchcore_t()
    model_t.eval()
    model_ms.set_train(False)

    result, global_layer_info = mindsporeinfoplus.summary_plus(model=model_ms, input_size=model_input_size,
                                                      dtypes = ms_dtypes,
                                                      col_names=['input_size', 'output_size', 'name'], depth=10,
                                                      verbose=1)

    in_shapes2, out_shapes2 = mindsporeinfoplus.get_input_size(global_layer_info), torchinfoplus.get_output_size(global_layer_info)
    dtypes_dict2 = mindsporeinfoplus.get_dtypes(global_layer_info)
    input_dict2 = mindsporeinfoplus.get_input_datas(global_layer_info)
    out_dict2 = mindsporeinfoplus.get_output_datas(global_layer_info)
    orders2 = mindsporeinfoplus.get_primitive_orders(global_layer_info)
    output_data2 = mindsporeinfoplus.get_output_datas(global_layer_info)



    result, global_layer_info = torchinfoplus.summary(model=model_t, input_size=model_input_size,
                                                      dtypes=model_dtypes_t,
                                                      col_names=['input_size', 'output_size', 'name'], depth=10,
                                                      verbose=1)

    in_shapes1, out_shapes1 = torchinfoplus.get_input_size(global_layer_info), torchinfoplus.get_output_size(global_layer_info)
    dtypes_dict = torchinfoplus.get_dtypes(global_layer_info)
    input_dict = torchinfoplus.get_input_datas(global_layer_info)
    out_dict = torchinfoplus.get_output_datas(global_layer_info)
    orders = torchinfoplus.get_primitive_orders(global_layer_info)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)


    # origin_distance = compare_models(model_ms, model_t, np_data=data, ms_dtypes=ms_dtypes,
    #                                  torch_dtypes=model_dtypes_t)
    # print(origin_distance)




