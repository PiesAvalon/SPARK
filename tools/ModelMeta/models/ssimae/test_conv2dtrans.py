import numpy as np
import torch
import mindspore

from model_utils.config import config

cfg = config
a = np.random.randn(128, 32, 8, 8)
ta = torch.tensor(a, dtype=torch.float32)
ma = mindspore.Tensor(a, dtype=mindspore.float32)
net_torch = torch.nn.Conv2d(
                    cfg.flc, cfg.z_dim, 8, stride=1, bias=False
                )
net_torch.eval()
out_torch = net_torch(ta)
net = mindspore.nn.Conv2d(
                    cfg.flc, cfg.z_dim, 8, stride=1, pad_mode="valid",  has_bias=False
                )
net.set_train(False)
out_ms = net(ma)
print(np.max(np.abs(out_torch.detach().numpy() - out_ms.asnumpy())))
