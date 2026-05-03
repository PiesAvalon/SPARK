import torch
from mindspore import load_param_into_net, load_checkpoint

from model_utils.config import config
from src.network import AutoEncoder

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


network = AutoEncoder(config)
from main_parallel_torch import AutoEncoder as AutoEncoder_torch
net_torch = AutoEncoder_torch(config)
ckpt_path = "ssimae_ascend_v190_mvtecadbottle_official_cv_ok96.8_nok96.8_avg95.2.ckpt"
load_param_into_net(network, load_checkpoint(ckpt_path))
print("=" * 20)
ms_param = mindspore_params(network)
weights_dict = ms_param
load_weights_dict = {k: v for k, v in weights_dict.items()
                     if k in net_torch.state_dict()}
net_torch.load_state_dict(load_weights_dict, strict=False)
torch.save(net_torch.state_dict(), 'ssimae_ascend_v190_mvtecadbottle_official_cv_ok96.8_nok96.8_avg95.2.pth')

