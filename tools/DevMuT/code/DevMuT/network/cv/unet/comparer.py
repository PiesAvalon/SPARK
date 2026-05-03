from pprint import pprint

import mindspore
import numpy as np
import torch
import troubleshooter as ts
from mindspore import load_param_into_net, load_checkpoint

from infoplus.MindSporeInfoPlus import mindsporeinfoplus
from infoplus.TorchInfoPlus import torchinfoplus
from main import UNetMedical, create_dataset, dice_coeff, UnetEval
from main_torch import UNetMedical_torch, UnetEval_torch

bn_ms2pt = {"gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var",
            "embedding_table": "weight",
            }


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
    # x = x.asnumpy()
    # y = y.asnumpy()
    # try:
    out = np.max(np.abs(x - y))
    # except ValueError as e:
    # print(e)
    # out = e
    return out


def distance(x1, x2):
    distance_real = ChebyshevDistance
    dis = distance_real(x1, x2)
    return dis


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        for bn_name in bn_ms2pt:
            if bn_name in name:
                name = name.replace(bn_name, bn_ms2pt[bn_name])
        value = param.data.asnumpy()
        value = torch.tensor(value, dtype=torch.float32)
        # print(name)
        ms_params[name] = value
    return ms_params


def compare_layer(input_data_dict_new, input_data_dict_old):
    # pprint(input_data_dict_new)
    maximum = 0
    for layer in input_data_dict_new.keys():
        if input_data_dict_new[layer] is not None and input_data_dict_old[layer] is not None:
            layer_np_new = input_data_dict_new[layer][0]
            layer_up_old = input_data_dict_old[layer][0]
            print("layer: ", layer, "distance chess: ", distance(layer_np_new, layer_up_old)
                  # , "distance_euclidean: ",
                  # EuclideanDistance(layer_np_new, layer_up_old)
                  )
            # try:
            maximum = max(maximum, distance(layer_np_new, layer_up_old))
            # except TypeError as e:
            #     print(e)
            #     return 0
    return maximum


def eval_Unet(model_input, data_dir, cross_valid_ind=1):
    from Unetconfig import config
    config.use_deconv = True
    config.use_ds = False
    config.use_bn = False
    config.batch_size = 1
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                      do_crop=[388, 388], img_size=[572, 572])

    valid_ds = valid_dataset.create_tuple_iterator(output_numpy=True)
    metric = dice_coeff()

    for data in valid_ds:
        metric.clear()
        # inputs, labels = data[0].numpy(), data[1].numpy()
        inputs, labels = data[0], data[1]
        inputs, labels = mindspore.Tensor(inputs, mindspore.float32), mindspore.Tensor(labels,
                                                                                       mindspore.int32)  # Send tensors to the appropriate device (CPU or GPU)
        inputs_torch, labels_torch = torch.tensor(data[0], dtype=torch.float32).to(device), \
            torch.tensor(data[1], dtype=torch.int64).to(device)
        if str(model_input.__class__) == "<class 'main.UnetEval'>":
            logits = model_input(inputs)
        else:
            # print("str(model_input.__class__):", str(model_input.__class__))
            logits = model_input(inputs_torch)
            logits = mindspore.Tensor(logits.detach().cpu().numpy(), mindspore.float32)
        metric.update(logits, labels)

    # print("logits shape:", logit.shape, "labels shape:", label.shape)
    dice = metric.eval()
    if str(model_input.__class__) == "<class 'main.UnetEval'>":
        print("mindspore accuracy", dice)
    else:
        print("torch accuracy", dice)
    return dice


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 20230818
    ts.widget.fix_random(seed)
    input_size = (1, 1, 572, 572)
    network_ms = UNetMedical(1, 2)
    network_ms.set_train(False)
    ckpt_path = "/data/CKPTS/Unet/unet2d_ascend_v190_isbichallenge_official_cv_iou90.00.ckpt"
    load_param_into_net(network_ms, load_checkpoint(ckpt_path))
    ms_param = mindspore_params(network_ms)
    inpu_np = np.ones([1, 1, 572, 572])
    np_data = [inpu_np for i in range(1)]
    dtypes = [mindspore.float32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network_ms,
        input_data=input_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    # print(res)
    # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    torch.save(ms_param, 'Unet_medical.pth')
    weights_dict = torch.load('Unet_medical.pth', map_location=device)
    net_torch = UNetMedical_torch(n_channels=1, n_classes=2).to(device)
    net_torch.eval()
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net_torch.state_dict()}
    pprint(weights_dict.keys())
    pprint(load_weights_dict.keys())
    net_torch.load_state_dict(load_weights_dict, strict=False)
    dtypes = [torch.float32]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, "cpu")

    result, global_layer_info = torchinfoplus.summary(
        model=net_torch,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    # print(result)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    # print("maximum", compare_layer(output_data, output_data1))
    # diff_finder = ts.migrator.NetDifferenceFinder(pt_net=net_torch, ms_net=network_ms, fix_seed=seed,
    #                                               auto_conv_ckpt=0)  #
    # diff_finder.compare(auto_inputs=((input_size, np.float32),))
    eval_ms_net = UnetEval(network_ms, eval_activate="Softmax".lower())
    eval_torch_net = UnetEval_torch(net_torch, eval_activate="Softmax".lower())
    print(eval_Unet(eval_ms_net, "/data/MR/datasets/archive", cross_valid_ind=1))
    print(eval_Unet(eval_torch_net, "/data/MR/datasets/archive", cross_valid_ind=1))
