import numpy as np
import torch
import mindspore
import mindspore.nn

a = np.random.randn(1, 3, 300, 300)
ta = torch.tensor(a, dtype=torch.float32)
ma = mindspore.Tensor(a, dtype=mindspore.float32)
net_torch = torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1,  # for 'same' padding
                            dilation=1,
                            groups=1,
                            bias=False)
net_torch.eval()
out_torch = net_torch(ta)
net = mindspore.nn.Conv2d(3, 32, (3, 3), (2, 2), pad_mode='same', padding=0, group=1, dilation=(1, 1))
net.set_train(False)
out_ms = net(ma)
print(out_torch.shape, out_ms.shape)
print(np.max(np.abs(out_torch.detach().numpy() - out_ms.asnumpy())))
