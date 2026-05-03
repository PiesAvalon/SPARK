import numpy as np

import mindspore
from mindspore import Tensor
import torch
from torch import tensor

def conv2d_test():
    weight = np.random.randn(240, 120, 4, 4)
    bias = np.random.randn(240)

    # weight = np.ones((240, 120, 4, 4))
    # bias = np.ones((240))

    # PyTorch
    x_ = np.random.randn(1, 120, 1024, 640)
    x = tensor(x_, dtype=torch.float32)
    net1 = torch.nn.Conv2d(120, 240, 4)
    net1.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
    net1.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    output1 = net1(x).detach().numpy()


    # MindSpore
    x = Tensor(x_, mindspore.float32)
    net2 = mindspore.nn.Conv2d(120, 240, 4, pad_mode='valid', has_bias=True,
                     weight_init=mindspore.Tensor(weight, mindspore.float32),
                     bias_init=mindspore.Tensor(bias, mindspore.float32))
    output2 = net2(x).asnumpy()


    x = output1
    y = output2
    print(np.mean(np.abs(x - y)))




def dense_test():
    # PyTorch

    # weight = np.ones((4,3))
    # bias = np.ones((4))

    weight = np.random.randn(10, 2048)
    bias = np.random.randn(10)

    net1 = torch.nn.Linear(2048, 10)
    net1.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
    net1.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    x = torch.tensor(np.ones((64, 2048)), dtype=torch.float)
    output1 = net1(x).detach().numpy()
    print(output1)

    print("-*-*" * 50)
    # MindSpore

    x = Tensor(np.ones((64, 2048)), mindspore.float32)
    net2 = mindspore.nn.Dense(2048, 10, has_bias=True, weight_init=mindspore.Tensor(weight, mindspore.float32),
                              bias_init=mindspore.Tensor(bias, mindspore.float32))
    output2 = net2(x).asnumpy()
    print(output2)




def batchnorm2d_test():
    data = np.random.randn(100, 3, 224, 224).astype(np.float32)
    # PyTorch

    m1 = torch.nn.BatchNorm2d(num_features=3, momentum=0.1)
    input_py = tensor(data, dtype=torch.float32)
    output1 = m1(input_py).detach().numpy()
    print(output1)


    print("-*-*" * 50)
    # MindSpore

    m2 = mindspore.nn.BatchNorm2d(num_features=3, momentum=0.9)
    m2.set_train()

    input_ms = Tensor(data, mindspore.float32)
    output2 = m2(input_ms).asnumpy()
    print(output2)

def avgpool2d_test():
    import numpy as np

    import mindspore
    from mindspore import Tensor
    import torch
    from torch import tensor
    from mindspore import context
    context.set_context(device_target="GPU")
    data = np.random.randn(6, 6, 8, 8).astype(np.float32)
    pool = mindspore.nn.AvgPool2d(4, stride=2, ceil_mode=True, pad_mode='pad', padding=2)
    x1 = Tensor(data, mindspore.float32)
    output1 = pool(x1).asnumpy()

    pool = torch.nn.AvgPool2d(4, stride=2, ceil_mode=True, padding=2).cuda()
    x1 = tensor(data, dtype=torch.float32).cuda()
    output2 = pool(x1).detach().cpu().numpy()

    print(np.max(np.mean(output1 - output2)))

def maxpool2d_test():
    import numpy as np

    import mindspore
    from mindspore import Tensor
    import torch
    from torch import tensor
    from mindspore import context
    context.set_context(device_target="GPU")
    data = np.random.randn(6, 6, 8, 8).astype(np.float32)

    max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, return_indices=False).cuda()
    x = torch.tensor(data, dtype=torch.float32).cuda()
    output1 = max_pool(x).detach().cpu().numpy()

    max_pool = mindspore.nn.MaxPool2d(kernel_size=2, stride=1, pad_mode='pad', padding=1, dilation=1,
                                      return_indices=False)
    x1 = mindspore.Tensor(data, mindspore.float32)
    output2 = max_pool(x1).asnumpy()

    print(np.max(np.mean(output1 - output2)))


if __name__ == '__main__':
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="GPU",device_id=1)
    data1_np = np.load("./ms_input.npy")

    data1, data2 = mindspore.Tensor(data1_np, mindspore.float32), torch.tensor(data1_np,dtype=torch.float32).cuda()

    a = mindspore.nn.GraphCell(mindspore.load('/data1/myz/empirical_expcopy/log/ms_layer.mindir'))
    b = torch.nn.Linear(in_features=288, out_features=2, bias=True).cuda()
    weights_dict = torch.load('/data1/myz/empirical_expcopy/log/torch_layer.pth', map_location="cuda:0")
    load_weights_dict = {k: v for k, v in weights_dict.items() if k in b.state_dict()}
    b.load_state_dict(load_weights_dict, strict=False)

    import troubleshooter as ts
    ts.migrator.get_weight_map(pt_net=b, weight_map_save_path='./torch_net_map.json', print_map=False)

    torch.save(b.state_dict(), './torch_net.path')

    ts.migrator.convert_weight(weight_map_path='./torch_net_map.json', pt_file_path='./torch_net.path',
                               ms_file_save_path='./convert_ms.ckpt', print_conv_info=False)

    param_dict = mindspore.load_checkpoint('./convert_ms.ckpt')
    mindspore.load_param_into_net(a, param_dict)

    x = mindspore.nn.Flatten()(a(data1)).asnumpy()
    y = torch.nn.Flatten()(b(data2)).detach().cpu().numpy()

    in_dis = np.mean(np.abs(data1.asnumpy() - data2.detach().cpu().numpy()))
    out = np.mean(np.abs(x - y))
    print(out)








