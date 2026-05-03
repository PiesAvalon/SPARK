from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def module_2_id(model):
    names = {}
    is_leaf = {}

    def make_dict(module, memo=None, name_prefix=''):
        cell_id = str(id(module))
        if memo is None:
            memo = set()
        if module in memo:
            return

        memo.add(module)
        names[cell_id] = name_prefix
        is_leaf[cell_id] = True
        yield name_prefix, module

        for name, module_ in module._modules.items():
            if module_:
                cells_name_prefix = name
                is_leaf[cell_id] = False
                if name_prefix:
                    cells_name_prefix = name_prefix + '.' + cells_name_prefix
                for ele in make_dict(module_, memo, cells_name_prefix):
                    yield ele

    name_gen = make_dict(model)
    for layer in name_gen:
        continue
        # print(f'torch layer:{layer}')
    return names, is_leaf


def torch_summary_plus(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):

            cell_id = str(id(module))

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['module_name'] = names.get(cell_id)
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    names, is_leaf = module_2_id(model)

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20} {:>30} {:>25}  {:>25} {:>15}".format("Layer (type)", "Module name", "Input shape",
                                                            "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>30} {:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]['module_name']),
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return summary


def name_2_layer(summary):
    res = {}
    for s in summary:
        name = str(summary[s]['module_name'])
        res[name] = s
    return res


if __name__ == '__main__':
    print('torch summary plus start')
    # resnet = Resnet(Bottleneck, [3, 4, 23, 3], 16)
    # model = DeepLabV3(resnet).cuda()
    # # print(model._modules)
    # torch_summary_plus(model, (3, 513, 513))
    #
    # named_modules = model.named_modules()
    # for name in named_modules:
    #     print(name[0])
    #
    # layers = model.named_modules()
    # for layer in layers:
    #     print(layer[0])
