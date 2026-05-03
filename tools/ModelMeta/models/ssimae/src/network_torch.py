import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ssimae import pytorch_ssim


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        encoded_layers = []
        encoded_layers.extend(
            [
                nn.Conv2d(
                    cfg.input_channel,
                    cfg.flc,
                    4,
                    stride=2,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc,
                    4,
                    stride=2,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
            ]
        )
        if cfg.crop_size == 256:
            encoded_layers.extend(
                [
                    nn.Conv2d(
                        cfg.flc,
                        cfg.flc,
                        4,
                        stride=2,
                        bias=False,
                        padding=1,
                    ),
                    nn.LeakyReLU(negative_slope=0.2),
                ]
            )
        encoded_layers.extend(
            [
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc * 2,
                    4,
                    stride=2,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 4,
                    4,
                    stride=2,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc * 4,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc, cfg.z_dim, 8, stride=1, bias=False
                ),
            ]
        )
        self.encoded = nn.Sequential(*encoded_layers)
        decoded_layers = []
        decoded_layers.extend(
            [
                nn.ConvTranspose2d(
                    cfg.z_dim, cfg.flc, 8, stride=1, bias=False
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 4,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.ConvTranspose2d(cfg.flc * 4, cfg.flc * 2, 4, stride=2, bias=False),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc * 2,
                    cfg.flc * 2,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.ConvTranspose2d(cfg.flc * 2, cfg.flc, 4, stride=2, bias=False),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(
                    cfg.flc,
                    cfg.flc,
                    3,
                    stride=1,
                    bias=False,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                nn.ConvTranspose2d(cfg.flc, cfg.flc, 4, stride=2, bias=False),
                nn.LeakyReLU(negative_slope=0.2),
            ]
        )
        if cfg.crop_size == 256:
            decoded_layers.extend(
                [
                    nn.ConvTranspose2d(cfg.flc, cfg.flc, 4, stride=2, bias=False),
                    nn.LeakyReLU(negative_slope=0.2),
                ]
            )
        decoded_layers.extend(
            [
                nn.ConvTranspose2d(
                    cfg.flc,
                    cfg.input_channel,
                    4,
                    stride=2,
                    bias=False,
                    padding=1,
                ),
                nn.Sigmoid(),
            ]
        )
        self.decoded = nn.Sequential(*decoded_layers)
        self.resize = F.interpolate

        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None
        self.in_shapes={
            'encoded.0': [1, 3, 256, 256],
            'encoded.1': [1, 32, 128, 128],
            'encoded.2': [1, 32, 128, 128],
            'encoded.3': [1, 32, 64, 64],
            'encoded.4': [1, 32, 64, 64],
            'encoded.5': [1, 32, 32, 32],
            'encoded.6': [1, 32, 32, 32],
            'encoded.7': [1, 32, 32, 32],
            'encoded.8': [1, 32, 32, 32],
            'encoded.9': [1, 64, 16, 16],
            'encoded.10': [1, 64, 16, 16],
            'encoded.11': [1, 64, 16, 16],
            'encoded.12': [1, 64, 16, 16],
            'encoded.13': [1, 128, 8, 8],
            'encoded.14': [1, 128, 8, 8],
            'encoded.15': [1, 64, 8, 8],
            'encoded.16': [1, 64, 8, 8],
            'encoded.17': [1, 32, 8, 8],
            'encoded.18': [1, 32, 8, 8],
            'decoded.0': [1, 500, 1, 1],
            'decoded.1': [1, 32, 8, 8],
            'decoded.2': [1, 32, 8, 8],
            'decoded.3': [1, 64, 8, 8],
            'decoded.4': [1, 64, 8, 8],
            'decoded.5': [1, 128, 8, 8],
            'decoded.6': [1, 128, 8, 8],
            'decoded.7': [1, 64, 18, 18],
            'decoded.8': [1, 64, 18, 18],
            'decoded.9': [1, 64, 18, 18],
            'decoded.10': [1, 64, 18, 18],
            'decoded.11': [1, 32, 38, 38],
            'decoded.12': [1, 32, 38, 38],
            'decoded.13': [1, 32, 38, 38],
            'decoded.14': [1, 32, 38, 38],
            'decoded.15': [1, 32, 78, 78],
            'decoded.16': [1, 32, 78, 78],
            'decoded.17': [1, 32, 158, 158],
            'decoded.18': [1, 32, 158, 158],
            'decoded.19': [1, 3, 316, 316],
            'INPUT': [1, 3, 256, 256],
            'OUTPUT': [1, 3, 316, 316]
        }
        self.out_shapes = {
            'encoded.0': [1, 32, 128, 128],
            'encoded.1': [1, 32, 128, 128],
            'encoded.2': [1, 32, 64, 64],
            'encoded.3': [1, 32, 64, 64],
            'encoded.4': [1, 32, 32, 32],
            'encoded.5': [1, 32, 32, 32],
            'encoded.6': [1, 32, 32, 32],
            'encoded.7': [1, 32, 32, 32],
            'encoded.8': [1, 64, 16, 16],
            'encoded.9': [1, 64, 16, 16],
            'encoded.10': [1, 64, 16, 16],
            'encoded.11': [1, 64, 16, 16],
            'encoded.12': [1, 128, 8, 8],
            'encoded.13': [1, 128, 8, 8],
            'encoded.14': [1, 64, 8, 8],
            'encoded.15': [1, 64, 8, 8],
            'encoded.16': [1, 32, 8, 8],
            'encoded.17': [1, 32, 8, 8],
            'encoded.18': [1, 500, 1, 1],
            'decoded.0': [1, 32, 8, 8],
            'decoded.1': [1, 32, 8, 8],
            'decoded.2': [1, 64, 8, 8],
            'decoded.3': [1, 64, 8, 8],
            'decoded.4': [1, 128, 8, 8],
            'decoded.5': [1, 128, 8, 8],
            'decoded.6': [1, 64, 18, 18],
            'decoded.7': [1, 64, 18, 18],
            'decoded.8': [1, 64, 18, 18],
            'decoded.9': [1, 64, 18, 18],
            'decoded.10': [1, 32, 38, 38],
            'decoded.11': [1, 32, 38, 38],
            'decoded.12': [1, 32, 38, 38],
            'decoded.13': [1, 32, 38, 38],
            'decoded.14': [1, 32, 78, 78],
            'decoded.15': [1, 32, 78, 78],
            'decoded.16': [1, 32, 158, 158],
            'decoded.17': [1, 32, 158, 158],
            'decoded.18': [1, 3, 316, 316],
            'decoded.19': [1, 3, 316, 316],
            'INPUT': [1, 3, 256, 256],
            'OUTPUT': [1, 3, 316, 316]
        }
        self.orders = {
            'encoded.0': ['INPUT', 'encoded.1'],
            'encoded.1': ['encoded.0', 'encoded.2'],
            'encoded.2': ['encoded.1', 'encoded.3'],
            'encoded.3': ['encoded.2', 'encoded.4'],
            'encoded.4': ['encoded.3', 'encoded.5'],
            'encoded.5': ['encoded.4', 'encoded.6'],
            'encoded.6': ['encoded.5', 'encoded.7'],
            'encoded.7': ['encoded.6', 'encoded.8'],
            'encoded.8': ['encoded.7', 'encoded.9'],
            'encoded.9': ['encoded.8', 'encoded.10'],
            'encoded.10': ['encoded.9', 'encoded.11'],
            'encoded.11': ['encoded.10', 'encoded.12'],
            'encoded.12': ['encoded.11', 'encoded.13'],
            'encoded.13': ['encoded.12', 'encoded.14'],
            'encoded.14': ['encoded.13', 'encoded.15'],
            'encoded.15': ['encoded.14', 'encoded.16'],
            'encoded.16': ['encoded.15', 'encoded.17'],
            'encoded.17': ['encoded.16', 'encoded.18'],
            'encoded.18': ['encoded.17', 'decoded.0'],
            'decoded.0': ['encoded.18', 'decoded.1'],
            'decoded.1': ['decoded.0', 'decoded.2'],
            'decoded.2': ['decoded.1', 'decoded.3'],
            'decoded.3': ['decoded.2', 'decoded.4'],
            'decoded.4': ['decoded.3', 'decoded.5'],
            'decoded.5': ['decoded.4', 'decoded.6'],
            'decoded.6': ['decoded.5', 'decoded.7'],
            'decoded.7': ['decoded.6', 'decoded.8'],
            'decoded.8': ['decoded.7', 'decoded.9'],
            'decoded.9': ['decoded.8', 'decoded.10'],
            'decoded.10': ['decoded.9', 'decoded.11'],
            'decoded.11': ['decoded.10', 'decoded.12'],
            'decoded.12': ['decoded.11', 'decoded.13'],
            'decoded.13': ['decoded.12', 'decoded.14'],
            'decoded.14': ['decoded.13', 'decoded.15'],
            'decoded.15': ['decoded.14', 'decoded.16'],
            'decoded.16': ['decoded.15', 'decoded.17'],
            'decoded.17': ['decoded.16', 'decoded.18'],
            'decoded.18': ['decoded.17', 'decoded.19'],
            'decoded.19': ['decoded.18', 'OUTPUT']
        }
        self.layer_input_dtype = {
            'encoded.0': [torch.float32],
            'encoded.1': [torch.float32],
            'encoded.2': [torch.float32],
            'encoded.3': [torch.float32],
            'encoded.4': [torch.float32],
            'encoded.5': [torch.float32],
            'encoded.6': [torch.float32],
            'encoded.7': [torch.float32],
            'encoded.8': [torch.float32],
            'encoded.9': [torch.float32],
            'encoded.10': [torch.float32],
            'encoded.11': [torch.float32],
            'encoded.12': [torch.float32],
            'encoded.13': [torch.float32],
            'encoded.14': [torch.float32],
            'encoded.15': [torch.float32],
            'encoded.16': [torch.float32],
            'encoded.17': [torch.float32],
            'encoded.18': [torch.float32],
            'decoded.0': [torch.float32],
            'decoded.1': [torch.float32],
            'decoded.2': [torch.float32],
            'decoded.3': [torch.float32],
            'decoded.4': [torch.float32],
            'decoded.5': [torch.float32],
            'decoded.6': [torch.float32],
            'decoded.7': [torch.float32],
            'decoded.8': [torch.float32],
            'decoded.9': [torch.float32],
            'decoded.10': [torch.float32],
            'decoded.11': [torch.float32],
            'decoded.12': [torch.float32],
            'decoded.13': [torch.float32],
            'decoded.14': [torch.float32],
            'decoded.15': [torch.float32],
            'decoded.16': [torch.float32],
            'decoded.17': [torch.float32],
            'decoded.18': [torch.float32],
            'decoded.19': [torch.float32]
        }
        self.layer_names = {
            "encoded": self.encoded,
            "encoded.0": self.encoded[0],
            "encoded.1": self.encoded[1],
            "encoded.2": self.encoded[2],
            "encoded.3": self.encoded[3],
            "encoded.4": self.encoded[4],
            "encoded.5": self.encoded[5],
            "encoded.6": self.encoded[6],
            "encoded.7": self.encoded[7],
            "encoded.8": self.encoded[8],
            "encoded.9": self.encoded[9],
            "encoded.10": self.encoded[10],
            "encoded.11": self.encoded[11],
            "encoded.12": self.encoded[12],
            "encoded.13": self.encoded[13],
            "encoded.14": self.encoded[14],
            "encoded.15": self.encoded[15],
            "encoded.16": self.encoded[16],
            "encoded.17": self.encoded[17],
            "encoded.18": self.encoded[18],
            "decoded": self.decoded,
            "decoded.0": self.decoded[0],
            "decoded.1": self.decoded[1],
            "decoded.2": self.decoded[2],
            "decoded.3": self.decoded[3],
            "decoded.4": self.decoded[4],
            "decoded.5": self.decoded[5],
            "decoded.6": self.decoded[6],
            "decoded.7": self.decoded[7],
            "decoded.8": self.decoded[8],
            "decoded.9": self.decoded[9],
            "decoded.10": self.decoded[10],
            "decoded.11": self.decoded[11],
            "decoded.12": self.decoded[12],
            "decoded.13": self.decoded[13],
            "decoded.14": self.decoded[14],
            "decoded.15": self.decoded[15],
            "decoded.16": self.decoded[16],
            "decoded.17": self.decoded[17],
            "decoded.18": self.decoded[18],
            "decoded.19": self.decoded[19],
        }


    def forward(self, input_batch):
        temp = self.encoded(input_batch)
        output_batch = self.decoded(temp)
        # print("output_batch.shape: ", output_batch.shape)
        # if (input_batch.shape[2] != output_batch.shape[2]) or (input_batch.shape[3] != output_batch.shape[3]):
        output_batch = self.resize(output_batch, input_batch.shape[2:])
        return output_batch

    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name, order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name] = order

    def get_outshape(self, layer_name):

        if layer_name not in self.out_shapes.keys():
            return False

        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name, out):

        if layer_name not in self.out_shapes.keys():
            return False

        self.out_shapes[layer_name] = out

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):
        if 'encoded' == layer_name:
            self.encoded = new_layer
            self.layer_names["encoded"] = new_layer

        elif 'encoded.0' == layer_name:
            self.encoded[0] = new_layer
            self.layer_names["encoded.0"] = new_layer

        elif 'encoded.1' == layer_name:
            self.encoded[1] = new_layer
            self.layer_names["encoded.1"] = new_layer

        elif 'encoded.2' == layer_name:
            self.encoded[2] = new_layer
            self.layer_names["encoded.2"] = new_layer

        elif 'encoded.3' == layer_name:
            self.encoded[3] = new_layer
            self.layer_names["encoded.3"] = new_layer

        elif 'encoded.4' == layer_name:
            self.encoded[4] = new_layer
            self.layer_names["encoded.4"] = new_layer

        elif 'encoded.5' == layer_name:
            self.encoded[5] = new_layer
            self.layer_names["encoded.5"] = new_layer

        elif 'encoded.6' == layer_name:
            self.encoded[6] = new_layer
            self.layer_names["encoded.6"] = new_layer

        elif 'encoded.7' == layer_name:
            self.encoded[7] = new_layer
            self.layer_names["encoded.7"] = new_layer

        elif 'encoded.8' == layer_name:
            self.encoded[8] = new_layer
            self.layer_names["encoded.8"] = new_layer

        elif 'encoded.9' == layer_name:
            self.encoded[9] = new_layer
            self.layer_names["encoded.9"] = new_layer

        elif 'encoded.10' == layer_name:
            self.encoded[10] = new_layer
            self.layer_names["encoded.10"] = new_layer

        elif 'encoded.11' == layer_name:
            self.encoded[11] = new_layer
            self.layer_names["encoded.11"] = new_layer

        elif 'encoded.12' == layer_name:
            self.encoded[12] = new_layer
            self.layer_names["encoded.12"] = new_layer

        elif 'encoded.13' == layer_name:
            self.encoded[13] = new_layer
            self.layer_names["encoded.13"] = new_layer

        elif 'encoded.14' == layer_name:
            self.encoded[14] = new_layer
            self.layer_names["encoded.14"] = new_layer

        elif 'encoded.15' == layer_name:
            self.encoded[15] = new_layer
            self.layer_names["encoded.15"] = new_layer

        elif 'encoded.16' == layer_name:
            self.encoded[16] = new_layer
            self.layer_names["encoded.16"] = new_layer

        elif 'encoded.17' == layer_name:
            self.encoded[17] = new_layer
            self.layer_names["encoded.17"] = new_layer

        elif 'encoded.18' == layer_name:
            self.encoded[18] = new_layer
            self.layer_names["encoded.18"] = new_layer

        elif 'decoded' == layer_name:
            self.decoded = new_layer
            self.layer_names["decoded"] = new_layer

        elif 'decoded.0' == layer_name:
            self.decoded[0] = new_layer
            self.layer_names["decoded.0"] = new_layer

        elif 'decoded.1' == layer_name:
            self.decoded[1] = new_layer
            self.layer_names["decoded.1"] = new_layer

        elif 'decoded.2' == layer_name:
            self.decoded[2] = new_layer
            self.layer_names["decoded.2"] = new_layer

        elif 'decoded.3' == layer_name:
            self.decoded[3] = new_layer
            self.layer_names["decoded.3"] = new_layer

        elif 'decoded.4' == layer_name:
            self.decoded[4] = new_layer
            self.layer_names["decoded.4"] = new_layer

        elif 'decoded.5' == layer_name:
            self.decoded[5] = new_layer
            self.layer_names["decoded.5"] = new_layer

        elif 'decoded.6' == layer_name:
            self.decoded[6] = new_layer
            self.layer_names["decoded.6"] = new_layer

        elif 'decoded.7' == layer_name:
            self.decoded[7] = new_layer
            self.layer_names["decoded.7"] = new_layer

        elif 'decoded.8' == layer_name:
            self.decoded[8] = new_layer
            self.layer_names["decoded.8"] = new_layer

        elif 'decoded.9' == layer_name:
            self.decoded[9] = new_layer
            self.layer_names["decoded.9"] = new_layer

        elif 'decoded.10' == layer_name:
            self.decoded[10] = new_layer
            self.layer_names["decoded.10"] = new_layer

        elif 'decoded.11' == layer_name:
            self.decoded[11] = new_layer
            self.layer_names["decoded.11"] = new_layer

        elif 'decoded.12' == layer_name:
            self.decoded[12] = new_layer
            self.layer_names["decoded.12"] = new_layer

        elif 'decoded.13' == layer_name:
            self.decoded[13] = new_layer
            self.layer_names["decoded.13"] = new_layer

        elif 'decoded.14' == layer_name:
            self.decoded[14] = new_layer
            self.layer_names["decoded.14"] = new_layer

        elif 'decoded.15' == layer_name:
            self.decoded[15] = new_layer
            self.layer_names["decoded.15"] = new_layer

        elif 'decoded.16' == layer_name:
            self.decoded[16] = new_layer
            self.layer_names["decoded.16"] = new_layer

        elif 'decoded.17' == layer_name:
            self.decoded[17] = new_layer
            self.layer_names["decoded.17"] = new_layer

        elif 'decoded.18' == layer_name:
            self.decoded[18] = new_layer
            self.layer_names["decoded.18"] = new_layer

        elif 'decoded.19' == layer_name:
            self.decoded[19] = new_layer
            self.layer_names["decoded.19"] = new_layer


class SSIMLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val
        self.loss_fn = pytorch_ssim.SSIM(window_size=11)
        self.reduce_mean = torch.mean

    def forward(self, input_batch, target):
        output = self.loss_fn(input_batch, target)
        loss = 1 - self.reduce_mean(output)
        return loss


class NetWithLoss(nn.Module):
    def __init__(self, net, loss_fn):
        super(NetWithLoss, self).__init__()
        self._net = net
        self._loss_fn = loss_fn

    def forward(self, input_batch):
        output = self._net(input_batch)
        return self._loss_fn(output, input_batch)