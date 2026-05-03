import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infoplus.MindSporeInfoPlus import mindsporeinfoplus
from infoplus.TorchInfoPlus import torchinfoplus
from models.vgg11.vgg11 import vgg11 as vgg11_ms
from models.vgg11.vgg11_torch import vgg11 as vgg11_torch
import mindspore
import torch
import numpy as np
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target='GPU')


device = 'cpu'
# model_ms = vgg11_ms()
model_torch = vgg11_torch()


data = np.load("/home/cvgroup/myz/czx/semtest/data/vgg11_data0.npy")
samples = np.random.choice(data.shape[0], 2, replace=False)
samples_data = data[samples] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
data_ms = mindspore.Tensor(samples_data, dtype=mindspore.float32)
data_torch = torch.tensor(samples_data, dtype=torch.float32).to(device)

dtypes_ms = [mindspore.float32]
# model_ms(data_ms)
model_torch(data_torch)

# res, global_layer_info, summary_list = mindsporeinfoplus.summary_plus(
#     model=model_ms,
#     input_data=data_ms,
#     dtypes=dtypes_ms,
#     col_names=['input_size', 'output_size', 'name'],
#     mode="train",
#     verbose=0,
#     depth=10
# )

dtypes_torch = [torch.float32]
res, global_layer_info, summary_list = torchinfoplus.summary(
    model=model_torch,
    input_data=data_torch,
    dtypes=dtypes_torch,
    col_names=['input_size', 'output_size', 'name'],
    mode="train",
    verbose=0,
    depth=10
)

output_datas_torch = torchinfoplus.get_output_datas(global_layer_info)

print(output_datas_torch)

