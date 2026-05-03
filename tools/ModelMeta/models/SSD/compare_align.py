import mindspore
mindspore.set_context(pynative_synchronize=True, device_target="GPU")
import numpy as np
import torch
import troubleshooter as ts
# from backbone_mobilenetv1 import SSDWithMobileNetV1 as SSD_ms
# from backbone_mobilenetv1_pytorch import SSDWithMobileNetV1 as SSD_torch
# from backbone_mobilenetv2 import SSDWithMobileNetV2 as SSD_ms
# from backbone_mobilenetv2_pytorch import SSDWithMobileNetV2 as SSD_torch
from backbone_resnet50_fpn import ssd_resnet50fpn_ms as SSD_ms
from backbone_resnet50_fpn_pytorch import ssd_resnet50fpn_torch as SSD_torch
from comparer import compare_models
model1 = SSD_ms()
model1.set_train(False)
# mindspore.load_checkpoint("ssdmobilenetv1.ckpt", model1)
mindspore.load_checkpoint("/root/SSD_2023/ssdresnet50.ckpt", model1)
model2 = SSD_torch()
model2.eval()


# a = torch.randn(2, 3, 640, 640)
# ma = mindspore.Tensor(a.detach().numpy(), mindspore.float32)
# b = model1(ma)
# print(b[0].shape, b[1].shape)
# tb = model2(a)
# print(tb[0].shape, tb[1].shape)
weight_dict = torch.load("/root/SSD_2023/ssdresnet50.pth")
#weight_dict = torch.load("ssdmobilenetv1.pth")
model2.load_state_dict(weight_dict, strict=False)
input_size = (2, 3, 640, 640)
diff_finder = ts.migrator.NetDifferenceFinder(pt_net=model2, ms_net=model1, fix_seed=0, auto_conv_ckpt=1)
diff_finder.compare(auto_inputs=((input_size, np.float32),))
aa = np.ones([2, 3, 640, 640])
compare_models(model1, model2, np_data=[aa])
# mindspore.save_checkpoint(model1, "ssdmobilenetv1.ckpt")
# mindspore.save_checkpoint(model1, "ssdresnet50.ckpt")
