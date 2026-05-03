from pprint import pprint

import mindspore
import torch
from mindspore import load_param_into_net, load_checkpoint
# from backbone_mobilenetv1 import SSDWithMobileNetV1 as SSD_ms
# from backbone_mobilenetv1_pytorch import SSDWithMobileNetV1 as SSD_torch
from backbone_resnet50_fpn import ssd_resnet50fpn_ms as SSD_ms
from backbone_resnet50_fpn_pytorch import ssd_resnet50fpn_torch as SSD_torch
bn_ms2pt = {"gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var",
            "embedding_table": "weight",
            }


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


network = SSD_ms()
net_torch = SSD_torch()
ckpt_path = "ssdresnet50fpn_ascend_v190_coco2017_official_cv_acc37.56.ckpt"
weight_dict = load_checkpoint(ckpt_path)
# for k, v in weight_dict.items():
#     print(k, v.shape)
param_dict = mindspore.load_checkpoint(ckpt_path)
loaded_params = {}
for x in list(param_dict.keys()):
    param_dict[x.replace(".feature_extractor.resnet", "")] = param_dict[x]
    loaded_params[x.replace(".feature_extractor.resnet", "")] = param_dict[x]
    del param_dict[x]
param_not_load, ckpt_not_load = mindspore.load_param_into_net(network.network, param_dict)
print("param_not_load: ")
pprint(param_not_load)
print("ckpt_not_load: ")
pprint(ckpt_not_load)
print("loaded_params: ")
pprint(loaded_params)
load_param_into_net(network, weight_dict)
print("=" * 20)
ms_param = mindspore_params(network)
weights_dict = ms_param
load_weights_dict = {k: v for k, v in weights_dict.items()
                     if k in net_torch.state_dict()}
net_torch.load_state_dict(load_weights_dict, strict=False)
torch.save(net_torch.state_dict(), 'ssdresnet50fpn_ascend_v190_coco2017_official_cv_acc37.56.pth')

