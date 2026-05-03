from pprint import pprint

import mindspore
import numpy as np
import troubleshooter as ts
from tqdm import tqdm

from src.dataset import MovieReview

seed = 20230818
ts.widget.fix_random(seed)
import torch
from mindspore import load_param_into_net, load_checkpoint, nn

from infoplus.MindSporeInfoPlus import mindsporeinfoplus
from infoplus.TorchInfoPlus import torchinfoplus
from src.textcnn import TextCNN

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


def eval_TextCNN(model_torch: torch.nn.Module, model_ms: nn.Cell):
    model_torch.eval()
    model_ms.set_train(False)

    test_data_size = 0
    correct_torch = 0
    correct_ms = 0
    instance = MovieReview(root_dir="/data/MR/datasets/rt-polaritydata", maxlen=51, split=0.9)
    test_dataset = instance.create_train_dataset(batch_size=1)
    dataset_size = test_dataset.get_dataset_size()
    test_iter = test_dataset.create_dict_iterator(output_numpy=False)

    for item in tqdm(test_iter, total=dataset_size):
        text, targets = item['data'], item['label']
        test_data_size += text.shape[0]
        text_array, targets_array = text.asnumpy(), targets.asnumpy()
        with torch.no_grad():
            text_tensor, targets_tensor = torch.LongTensor(text_array).to(device), torch.LongTensor(
                targets_array).to(device)

            output_torch = model_torch(text_tensor)
        output_ms = model_ms(text)
        indices_ms = np.argmax(output_ms.asnumpy(), axis=1)
        result_ms = (np.equal(indices_ms, targets.asnumpy()) * 1).reshape(-1)
        accuracy_ms = result_ms.sum()
        correct_ms = correct_ms + accuracy_ms
        with torch.no_grad():
            indices = torch.argmax(output_torch.to(device), dim=1)
            result = (np.equal(indices.detach().cpu().numpy(), targets_tensor.detach().cpu().numpy()) * 1).reshape(-1)
            accuracy = result.sum()
            correct_torch = correct_torch + accuracy

    print("Pytorch Test Accuracy: {}%".format(
        100 * correct_torch / test_data_size) + " " + "Mindpsore Test Accuacy: {}%".format(
        100 * correct_ms / test_data_size))


if __name__ == '__main__':
    device = "cpu"
    network = TextCNN(vocab_len=20288, word_len=51, num_classes=2, vec_length=40)  # TextCNN mindspore
    ckpt_path = "/data1/CKPTS/textcnn/textcnn_ascend_v190_moviereview_official_nlp_acc77.44.ckpt"
    load_param_into_net(network, load_checkpoint(ckpt_path))
    print("=" * 20)
    ms_param = mindspore_params(network)
    torch.save(ms_param, 'textcnn.pth')
    inpu_np1 = np.ones([1, 51])
    network.set_train(False)
    np_data = [inpu_np1]
    dtypes = [mindspore.int32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info = mindsporeinfoplus.summary_plus(
        model=network,
        input_data=input_data,
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        verbose=0,
        depth=8)
    # print("net_ms: ", [i.name for i in net_ms.get_parameters()])
    input_data = mindsporeinfoplus.get_input_datas(global_layer_info)
    output_data = mindsporeinfoplus.get_output_datas(global_layer_info)
    from src.textcnn_torch import TextCNN as TextCNN_torch

    net = TextCNN_torch(vocab_len=20288, word_len=51, num_classes=2, vec_length=40).to(device)  # TextCNN Pytorch
    # for i in net.state_dict():
    #     print(i)
    print("device", device)
    weights_dict = torch.load("textcnn.pth", map_location="cpu")
    # param_convert(weights_dict, net.state_dict())
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in net.state_dict()}
    pprint(weights_dict.keys())
    pprint(net.state_dict().keys())
    pprint(load_weights_dict.keys())
    net.load_state_dict(load_weights_dict, strict=False)
    dtypes = [torch.int64, torch.int64]
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, "cpu")
    result, global_layer_info = torchinfoplus.summary(
        model=net,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=0)
    input_data1 = torchinfoplus.get_input_datas(global_layer_info)
    output_data1 = torchinfoplus.get_output_datas(global_layer_info)
    # print(input_datas)
    # pprint(input_datas)
    print("===========================================")
    # pprint(input_datas2)
    print("maximum", compare_layer(input_data, input_data1))
    print("maximum", compare_layer(output_data, output_data1))
    input_size = (1, 51)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=net, ms_net=network, fix_seed=seed, auto_conv_ckpt=0)  #
    diff_finder.compare(auto_inputs=((input_size, np.int32),))
    eval_TextCNN(net, network)
