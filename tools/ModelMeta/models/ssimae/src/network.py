import mindspore
import numpy as np
from mindspore import nn
from mindspore import ops


# class AutoEncoder(nn.Cell):
#     def __init__(self, cfg):
#         super(AutoEncoder, self).__init__()
#
#         encoded_layers = []
#         encoded_layers.extend(
#             [
#                 nn.Conv2d(
#                     cfg.input_channel,
#                     cfg.flc,
#                     4,
#                     stride=2,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc,
#                     cfg.flc,
#                     4,
#                     stride=2,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#             ]
#         )
#         encoded_layers.extend(
#             [
#                 nn.Conv2d(
#                     cfg.flc,
#                     cfg.flc,
#                     4,
#                     stride=2,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#             ]
#         )
#         encoded_layers.extend(
#             [
#                 nn.Conv2d(
#                     cfg.flc,
#                     cfg.flc,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc,
#                     cfg.flc * 2,
#                     4,
#                     stride=2,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc * 2,
#                     cfg.flc * 2,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc * 2,
#                     cfg.flc * 4,
#                     4,
#                     stride=2,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc * 4,
#                     cfg.flc * 2,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc * 2,
#                     cfg.flc,
#                     3,
#                     stride=1,
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc, cfg.z_dim, 8, stride=1, pad_mode="valid",  has_bias=False
#                 ),
#             ]
#         )
#         self.encoded = nn.SequentialCell(encoded_layers)
#         decoded_layers = []
#         decoded_layers.extend(
#             [
#                 nn.Conv2dTranspose(
#                     cfg.z_dim, cfg.flc, 8, stride=1, pad_mode="valid",  has_bias=False
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc,
#                     cfg.flc * 2,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc * 2,
#                     cfg.flc * 4,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2dTranspose(cfg.flc * 4, cfg.flc * 2, 4, stride=2, pad_mode='valid', has_bias=False),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc * 2,
#                     cfg.flc * 2,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2dTranspose(cfg.flc * 2, cfg.flc, 4, stride=2, pad_mode='valid',  has_bias=False),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2d(
#                     cfg.flc,
#                     cfg.flc,
#                     3,
#                     stride=1,
#
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.LeakyReLU(alpha=0.2),
#                 nn.Conv2dTranspose(cfg.flc, cfg.flc, 4, stride=2, pad_mode='valid',  has_bias=False),
#                 nn.LeakyReLU(alpha=0.2),
#             ]
#         )
#         decoded_layers.extend(
#             [
#                 nn.Conv2dTranspose(cfg.flc, cfg.flc, 4, stride=2, pad_mode='valid',  has_bias=False),
#                 nn.LeakyReLU(alpha=0.2),
#             ]
#         )
#         decoded_layers.extend(
#             [
#                 nn.Conv2dTranspose(
#                     cfg.flc,
#                     cfg.input_channel,
#                     4,
#                     stride=2,
#                     has_bias=False,
#                     pad_mode="pad",
#                     padding=1,
#                 ),
#                 nn.Sigmoid(),
#             ]
#         )
#         self.decoded = nn.SequentialCell(decoded_layers)
#         self.resize = ops.interpolate
#
#     def construct(self, input_batch):
#         temp = self.encoded(input_batch)
#         output_batch = self.decoded(temp)
#         output_batch = self.resize(output_batch, input_batch.shape[2:])
#         return output_batch


class AutoEncoder(nn.Cell):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()

        # Encoding layers
        self.enc_conv1 = nn.Conv2d(cfg.input_channel, cfg.flc, 4, stride=2, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu1 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv2 = nn.Conv2d(cfg.flc, cfg.flc, 4, stride=2, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu2 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv3 = nn.Conv2d(cfg.flc, cfg.flc, 4, stride=2, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu3 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv4 = nn.Conv2d(cfg.flc, cfg.flc, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu4 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv5 = nn.Conv2d(cfg.flc, cfg.flc * 2, 4, stride=2, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu5 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv6 = nn.Conv2d(cfg.flc * 2, cfg.flc * 2, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu6 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv7 = nn.Conv2d(cfg.flc * 2, cfg.flc * 4, 4, stride=2, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu7 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv8 = nn.Conv2d(cfg.flc * 4, cfg.flc * 2, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu8 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv9 = nn.Conv2d(cfg.flc * 2, cfg.flc, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.enc_relu9 = nn.LeakyReLU(alpha=0.2)
        self.enc_conv10 = nn.Conv2d(cfg.flc, cfg.z_dim, 8, stride=1, pad_mode="valid", has_bias=False)

        # Decoding layers
        self.dec_convT1 = nn.Conv2dTranspose(cfg.z_dim, cfg.flc, 8, stride=1, pad_mode="valid", has_bias=False)
        self.dec_relu1 = nn.LeakyReLU(alpha=0.2)
        self.dec_conv1 = nn.Conv2d(cfg.flc, cfg.flc * 2, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.dec_relu2 = nn.LeakyReLU(alpha=0.2)
        self.dec_conv2 = nn.Conv2d(cfg.flc * 2, cfg.flc * 4, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.dec_relu3 = nn.LeakyReLU(alpha=0.2)
        self.dec_convT2 = nn.Conv2dTranspose(cfg.flc * 4, cfg.flc * 2, 4, stride=2, pad_mode='valid', has_bias=False)
        self.dec_relu4 = nn.LeakyReLU(alpha=0.2)
        self.dec_conv3 = nn.Conv2d(cfg.flc * 2, cfg.flc * 2, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.dec_relu5 = nn.LeakyReLU(alpha=0.2)
        self.dec_convT3 = nn.Conv2dTranspose(cfg.flc * 2, cfg.flc, 4, stride=2, pad_mode='valid', has_bias=False)
        self.dec_relu6 = nn.LeakyReLU(alpha=0.2)
        self.dec_conv4 = nn.Conv2d(cfg.flc, cfg.flc, 3, stride=1, has_bias=False, pad_mode="pad", padding=1)
        self.dec_relu7 = nn.LeakyReLU(alpha=0.2)
        self.dec_convT4 = nn.Conv2dTranspose(cfg.flc, cfg.flc, 4, stride=2, pad_mode='valid', has_bias=False)
        self.dec_relu8 = nn.LeakyReLU(alpha=0.2)
        self.dec_convT5 = nn.Conv2dTranspose(cfg.flc, cfg.flc, 4, stride=2, pad_mode='valid', has_bias=False)
        self.dec_relu9 = nn.LeakyReLU(alpha=0.2)
        self.dec_convT6 = nn.Conv2dTranspose(cfg.flc, cfg.input_channel, 4, stride=2, has_bias=False, pad_mode="pad",
                                             padding=1)
        self.dec_final = nn.Sigmoid()
        self.resize = ops.interpolate

    def construct(self, input_batch):
        # Encoding
        x = self.enc_relu1(self.enc_conv1(input_batch))
        x = self.enc_relu2(self.enc_conv2(x))
        x = self.enc_relu3(self.enc_conv3(x))
        x = self.enc_relu4(self.enc_conv4(x))
        x = self.enc_relu5(self.enc_conv5(x))
        x = self.enc_relu6(self.enc_conv6(x))
        x = self.enc_relu7(self.enc_conv7(x))
        x = self.enc_relu8(self.enc_conv8(x))
        x = self.enc_relu9(self.enc_conv9(x))
        x = self.enc_conv10(x)  # Last layer doesn't have a ReLU

        # Decoding
        x = self.dec_relu1(self.dec_convT1(x))
        x = self.dec_relu2(self.dec_conv1(x))
        x = self.dec_relu3(self.dec_conv2(x))
        x = self.dec_relu4(self.dec_convT2(x))
        x = self.dec_relu5(self.dec_conv3(x))
        x = self.dec_relu6(self.dec_convT3(x))
        x = self.dec_relu7(self.dec_conv4(x))
        x = self.dec_relu8(self.dec_convT4(x))
        x = self.dec_relu9(self.dec_convT5(x))
        x = self.dec_final(self.dec_convT6(x))

        # Resize to match input batch dimensions
        output_batch = self.resize(x, input_batch.shape[2:])
        return output_batch


class SSIMLoss(nn.Cell):
    def __init__(self, max_val=1.0):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val
        self.loss_fn = nn.SSIM(max_val=self.max_val)
        self.reduce_mean = ops.ReduceMean()

    def construct(self, input_batch, target):
        output = self.loss_fn(input_batch, target)
        loss = 1 - self.reduce_mean(output)
        return loss


class NetWithLoss(nn.Cell):
    def __init__(self, net, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, input_batch):
        output = self._net(input_batch)
        return self._loss_fn(output, input_batch)


if __name__ == '__main__':
    from models.ssimae.model_utils.config import config as cfg
    old_net = AutoEncoder(cfg)
    # new_net = AutoEncoder_new(cfg)
    from infoplus.MindSporeInfoPlus import mindsporeinfoplus
    np_data = [np.random.randn(1, 3, 256, 256)]
    dtypes = [mindspore.float32]
    input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    res, global_layer_info, summary_list = mindsporeinfoplus.summary_plus(
        model=old_net,
        input_data=input_data,
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'],
        mode="train",
        verbose=1,
        depth=10
    )
    print("Old model summary: ", res)