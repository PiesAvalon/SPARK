import mindspore
import numpy as np
import torch.nn
from bug_localization.infoplus.MindSporeInfoPlus import mindsporeinfoplus
from bug_localization.infoplus.TorchInfoPlus import torchinfoplus
# from models.deeplabv3.Deeplabv3 import DeepLabV3
# from models.deeplabv3.main_torch import DeepLabV3_torch


def ChebyshevDistance(x, y):
    if isinstance(x, mindspore.Tensor):
        x = x.asnumpy()
    elif isinstance(x, torch.Tensor):
        if torch.get_device(x) != "CPU":
            x = x.cpu()
        x = x.detach().numpy()
    if isinstance(y, mindspore.Tensor):
        y = y.asnumpy()
    elif isinstance(y, torch.Tensor):
        if torch.get_device(y) != "CPU":
            y = y.cpu()
        y = y.detach().numpy()
    # x = x.asnumpy()
    # y = y.asnumpy()
    # try:
    out = np.max(np.abs(x - y))
    # except ValueError as e:
    #     print(e)
    #     out = e
    return out


def compare_layer(input_data_dict_new, input_data_dict_old):
    # pprint(input_data_dict_new)
    maximum = 0
    for layer in input_data_dict_new.keys():
        if layer == "[]" or layer == "":
            continue
        if input_data_dict_new[layer] is not None and input_data_dict_old[layer] is not None:
            layer_np_new = input_data_dict_new[layer][0]
            layer_up_old = input_data_dict_old[layer][0]
            print("layer: ", layer, "distance: ", )
            print(ChebyshevDistance(layer_np_new, layer_up_old))
            maximum = max(maximum, ChebyshevDistance(layer_np_new, layer_up_old))
    return maximum


def compare_models(network: mindspore.nn.Cell, net: torch.nn.Module, np_data=[np.ones([2, 3, 513, 513])],
                   ms_dtypes=[mindspore.float32], torch_dtypes=[torch.float32], device="CPU"):
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
    # print("res: ", res)
    torch_data = torchinfoplus.np_2_tensor(np_data, torch_dtypes, device=device)

    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=torch_dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    # print("result: ", result)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    print("===========================================")
    print("maximum", compare_layer(output_data, output_data1))


# if __name__ == '__main__':
#     network = DeepLabV3('train', 21, 8, False)
#     net = DeepLabV3_torch()
#     compare_models(network, net)
