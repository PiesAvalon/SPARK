# import mindspore
import numpy as np
import torch
from torch import nn
import os
import torch_mutation.config as config

# if os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
#     device = "cuda:6"
# else:
#     device = "CPU"
device=config.device

torch.fx.wrap("reshape")

def reshape(x, shape):
    return torch.reshape(x, shape)

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, batch_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(in_features=nHidden * 2, out_features=nOut)
        self.h0 = torch.tensor(np.zeros([1 * 2, batch_size, nHidden]).astype(np.float32)).to(device)
        self.c0 = torch.tensor(np.zeros([1 * 2, batch_size, nHidden]).astype(np.float32)).to(device)

    def forward(self, x):
        recurrent, _ = self.rnn(x, (self.h0, self.c0))
        T, b, h = recurrent.shape
        t_rec = reshape(recurrent, (T * b, h,))
        out = self.embedding(t_rec)  # [T * b, nOut]
        out = reshape(out, (T, b, -1,))

        return out


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9)


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, use_bn=False, pad_mode='same'):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=0)
        self.bn = _bn(out_channel)
        self.Relu = nn.ReLU()
        self.use_bn = use_bn

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.Relu(out)
        return out


class VGG(nn.Module):
    """VGG Network structure"""

    def __init__(self, is_training=True):
        super(VGG, self).__init__()
        self.conv1 = Conv(3, 64, use_bn=True)
        self.conv2 = Conv(64, 128, use_bn=True)
        self.conv3 = Conv(128, 256, use_bn=True)
        self.conv4 = Conv(256, 256, use_bn=True)
        self.conv5 = Conv(256, 512, use_bn=True)
        self.conv6 = Conv(512, 512, use_bn=True)
        self.conv7 = Conv(512, 512, kernel_size=2, pad_mode='valid', use_bn=True)
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool2d1(x)
        x = self.conv2(x)
        x = self.maxpool2d1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2d2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2d2(x)
        x = self.conv7(x)
        return x


class CRNNV2(nn.Module):
    """
     Define a CRNN network which contains Bidirectional LSTM layers and vgg layer.

     Args:
        input_size(int): Size of time sequence. Usually, the input_size is equal to three times of image height for
        text images.
        batch_size(int): batch size of input data, default is 64
        hidden_size(int): the hidden size in LSTM layers, default is 512
     """

    def __init__(self, config):
        super(CRNNV2, self).__init__()
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.class_num
        self.reshape = torch.reshape
        self.transpose = torch.permute
        self.vgg = VGG().to(device)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.input_size, self.hidden_size, self.hidden_size, self.batch_size),
            BidirectionalLSTM(self.hidden_size, self.hidden_size, self.num_classes, self.batch_size))

        self.in_shapes = {
            'INPUT': [1, 3, 32, 100],
            'vgg.conv1.conv': [1, 3, 32, 100],
            'vgg.conv1.bn': [1, 64, 32, 100],
            'vgg.conv1.Relu': [1, 64, 32, 100],
            'vgg.conv2.conv': [1, 64, 16, 50],
            'vgg.conv2.bn': [1, 128, 16, 50],
            'vgg.conv2.Relu': [1, 128, 16, 50],
            'vgg.conv3.conv': [1, 128, 8, 25],
            'vgg.conv3.bn': [1, 256, 8, 25],
            'vgg.conv3.Relu': [1, 256, 8, 25],
            'vgg.conv4.conv': [1, 256, 8, 25],
            'vgg.conv4.bn': [1, 256, 8, 25],
            'vgg.conv4.Relu': [1, 256, 8, 25],
            'vgg.conv5.conv': [1, 256, 4, 25],
            'vgg.conv5.bn': [1, 512, 4, 25],
            'vgg.conv5.Relu': [1, 512, 4, 25],
            'vgg.conv6.conv': [1, 512, 4, 25],
            'vgg.conv6.bn': [1, 512, 4, 25],
            'vgg.conv6.Relu': [1, 512, 4, 25],
            'vgg.conv7.conv': [1, 512, 2, 25],
            'vgg.conv7.bn': [1, 512, 1, 24],
            'vgg.conv7.Relu': [1, 512, 1, 24],
            'vgg.maxpool2d1': [1, 128, 16, 50],
            'vgg.maxpool2d2': [1, 512, 4, 25],
            'rnn.0.embedding': [24, 512],
            'rnn.1.embedding': [24, 512],
            'OUTPUT': [24, 1, 37]
        }
        self.out_shapes = {
            'INPUT': [1, 3, 32, 100],
            'vgg.conv1.conv': [1, 64, 32, 100],
            'vgg.conv1.bn': [1, 64, 32, 100],
            'vgg.conv1.Relu': [1, 64, 32, 100],
            'vgg.conv2.conv': [1, 128, 16, 50],
            'vgg.conv2.bn': [1, 128, 16, 50],
            'vgg.conv2.Relu': [1, 128, 16, 50],
            'vgg.conv3.conv': [1, 256, 8, 25],
            'vgg.conv3.bn': [1, 256, 8, 25],
            'vgg.conv3.Relu': [1, 256, 8, 25],
            'vgg.conv4.conv': [1, 256, 8, 25],
            'vgg.conv4.bn': [1, 256, 8, 25],
            'vgg.conv4.Relu': [1, 256, 8, 25],
            'vgg.conv5.conv': [1, 512, 4, 25],
            'vgg.conv5.bn': [1, 512, 4, 25],
            'vgg.conv5.Relu': [1, 512, 4, 25],
            'vgg.conv6.conv': [1, 512, 4, 25],
            'vgg.conv6.bn': [1, 512, 4, 25],
            'vgg.conv6.Relu': [1, 512, 4, 25],
            'vgg.conv7.conv': [1, 512, 1, 24],
            'vgg.conv7.bn': [1, 512, 1, 24],
            'vgg.conv7.Relu': [1, 512, 1, 24],
            'vgg.maxpool2d1': [1, 128, 8, 25],
            'vgg.maxpool2d2': [1, 512, 2, 25],
            'rnn.0.embedding': [24, 256],
            'rnn.1.embedding': [24, 37],
            'OUTPUT': [24, 1, 37]
        }
        self.orders = {
            'vgg.conv1.conv': ['INPUT', 'vgg.conv1.bn'],
            'vgg.conv1.bn': ['vgg.conv1.conv', 'vgg.conv1.Relu'],
            'vgg.conv1.Relu': ['vgg.conv1.bn', 'vgg.conv2.conv'],
            'vgg.conv2.conv': ['vgg.conv1.Relu', 'vgg.conv2.bn'],
            'vgg.conv2.bn': ['vgg.conv2.conv', 'vgg.conv2.Relu'],
            'vgg.conv2.Relu': ['vgg.conv2.bn', 'vgg.maxpool2d1'],
            'vgg.maxpool2d1': ['vgg.conv2.Relu', 'vgg.conv3.conv'],
            'vgg.conv3.conv': ['vgg.maxpool2d1', 'vgg.conv3.bn'],
            'vgg.conv3.bn': ['vgg.conv3.conv', 'vgg.conv3.Relu'],
            'vgg.conv3.Relu': ['vgg.conv3.bn', 'vgg.conv4.conv'],
            'vgg.conv4.conv': ['vgg.conv3.Relu', 'vgg.conv4.bn'],
            'vgg.conv4.bn': ['vgg.conv4.conv', 'vgg.conv4.Relu'],
            'vgg.conv4.Relu': ['vgg.conv4.bn', 'vgg.conv5.conv'],
            'vgg.conv5.conv': ['vgg.conv4.Relu', 'vgg.conv5.bn'],
            'vgg.conv5.bn': ['vgg.conv5.conv', 'vgg.conv5.Relu'],
            'vgg.conv5.Relu': ['vgg.conv5.bn', 'vgg.conv6.conv'],
            'vgg.conv6.conv': ['vgg.conv5.Relu', 'vgg.conv6.bn'],
            'vgg.conv6.bn': ['vgg.conv6.conv', 'vgg.conv6.Relu'],
            'vgg.conv6.Relu': ['vgg.conv6.bn', 'vgg.maxpool2d2'],
            'vgg.maxpool2d2': ['vgg.conv6.Relu', 'vgg.conv7.conv'],
            'vgg.conv7.conv': ['vgg.maxpool2d2', 'vgg.conv7.bn'],
            'vgg.conv7.bn': ['vgg.conv7.conv', 'vgg.conv7.Relu'],
            'vgg.conv7.Relu': ['vgg.conv7.bn', 'rnn.0.embedding'],
            'rnn.0.embedding': ['vgg.conv7.Relu', 'rnn.1.embedding'],
            'rnn.1.embedding': ['rnn.0.embedding', 'OUTPUT']
        }

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

        self.layer_names = {
            "vgg.conv1": self.vgg.conv1,
            "vgg.conv1.conv": self.vgg.conv1.conv,
            "vgg.conv1.bn": self.vgg.conv1.bn,
            "vgg.conv1.Relu": self.vgg.conv1.Relu,
            "vgg.conv2": self.vgg.conv2,
            "vgg.conv2.conv": self.vgg.conv2.conv,
            "vgg.conv2.bn": self.vgg.conv2.bn,
            "vgg.conv2.Relu": self.vgg.conv2.Relu,
            "vgg.conv3": self.vgg.conv3,
            "vgg.conv3.conv": self.vgg.conv3.conv,
            "vgg.conv3.bn": self.vgg.conv3.bn,
            "vgg.conv3.Relu": self.vgg.conv3.Relu,
            "vgg.conv4": self.vgg.conv4,
            "vgg.conv4.conv": self.vgg.conv4.conv,
            "vgg.conv4.bn": self.vgg.conv4.bn,
            "vgg.conv4.Relu": self.vgg.conv4.Relu,
            "vgg.conv5": self.vgg.conv5,
            "vgg.conv5.conv": self.vgg.conv5.conv,
            "vgg.conv5.bn": self.vgg.conv5.bn,
            "vgg.conv5.Relu": self.vgg.conv5.Relu,
            "vgg.conv6": self.vgg.conv6,
            "vgg.conv6.conv": self.vgg.conv6.conv,
            "vgg.conv6.bn": self.vgg.conv6.bn,
            "vgg.conv6.Relu": self.vgg.conv6.Relu,
            "vgg.conv7": self.vgg.conv7,
            "vgg.conv7.conv": self.vgg.conv7.conv,
            "vgg.conv7.bn": self.vgg.conv7.bn,
            "vgg.conv7.Relu": self.vgg.conv7.Relu,
            "vgg.maxpool2d1": self.vgg.maxpool2d1,
            "vgg.maxpool2d2": self.vgg.maxpool2d2,
            "rnn.0.embedding": self.rnn[0].embedding,
            "rnn.1.embedding": self.rnn[1].embedding,
        }




    def forward(self, x):
        x = self.vgg(x)

        x = self.reshape(x, (self.batch_size, self.input_size, -1))
        x = self.transpose(x, (2, 0, 1))

        x = self.rnn(x)

        return x

    def set_layers(self,layer_name,new_layer):
        if 'vgg' == layer_name:
            self.vgg= new_layer
            self.layer_names["vgg"]=new_layer

        elif 'vgg.conv1' == layer_name:
            self.vgg.conv1= new_layer
            self.layer_names["vgg.conv1"]=new_layer

        elif 'vgg.conv1.conv' == layer_name:
            self.vgg.conv1.conv= new_layer
            self.layer_names["vgg.conv1.conv"]=new_layer

        elif 'vgg.conv1.bn' == layer_name:
            self.vgg.conv1.bn= new_layer
            self.layer_names["vgg.conv1.bn"]=new_layer

        elif 'vgg.conv1.Relu' == layer_name:
            self.vgg.conv1.Relu= new_layer
            self.layer_names["vgg.conv1.Relu"]=new_layer

        elif 'vgg.conv2' == layer_name:
            self.vgg.conv2= new_layer
            self.layer_names["vgg.conv2"]=new_layer

        elif 'vgg.conv2.conv' == layer_name:
            self.vgg.conv2.conv= new_layer
            self.layer_names["vgg.conv2.conv"]=new_layer

        elif 'vgg.conv2.bn' == layer_name:
            self.vgg.conv2.bn= new_layer
            self.layer_names["vgg.conv2.bn"]=new_layer

        elif 'vgg.conv2.Relu' == layer_name:
            self.vgg.conv2.Relu= new_layer
            self.layer_names["vgg.conv2.Relu"]=new_layer

        elif 'vgg.conv3' == layer_name:
            self.vgg.conv3= new_layer
            self.layer_names["vgg.conv3"]=new_layer

        elif 'vgg.conv3.conv' == layer_name:
            self.vgg.conv3.conv= new_layer
            self.layer_names["vgg.conv3.conv"]=new_layer

        elif 'vgg.conv3.bn' == layer_name:
            self.vgg.conv3.bn= new_layer
            self.layer_names["vgg.conv3.bn"]=new_layer

        elif 'vgg.conv3.Relu' == layer_name:
            self.vgg.conv3.Relu= new_layer
            self.layer_names["vgg.conv3.Relu"]=new_layer

        elif 'vgg.conv4' == layer_name:
            self.vgg.conv4= new_layer
            self.layer_names["vgg.conv4"]=new_layer

        elif 'vgg.conv4.conv' == layer_name:
            self.vgg.conv4.conv= new_layer
            self.layer_names["vgg.conv4.conv"]=new_layer

        elif 'vgg.conv4.bn' == layer_name:
            self.vgg.conv4.bn= new_layer
            self.layer_names["vgg.conv4.bn"]=new_layer

        elif 'vgg.conv4.Relu' == layer_name:
            self.vgg.conv4.Relu= new_layer
            self.layer_names["vgg.conv4.Relu"]=new_layer

        elif 'vgg.conv5' == layer_name:
            self.vgg.conv5= new_layer
            self.layer_names["vgg.conv5"]=new_layer

        elif 'vgg.conv5.conv' == layer_name:
            self.vgg.conv5.conv= new_layer
            self.layer_names["vgg.conv5.conv"]=new_layer

        elif 'vgg.conv5.bn' == layer_name:
            self.vgg.conv5.bn= new_layer
            self.layer_names["vgg.conv5.bn"]=new_layer

        elif 'vgg.conv5.Relu' == layer_name:
            self.vgg.conv5.Relu= new_layer
            self.layer_names["vgg.conv5.Relu"]=new_layer

        elif 'vgg.conv6' == layer_name:
            self.vgg.conv6= new_layer
            self.layer_names["vgg.conv6"]=new_layer

        elif 'vgg.conv6.conv' == layer_name:
            self.vgg.conv6.conv= new_layer
            self.layer_names["vgg.conv6.conv"]=new_layer

        elif 'vgg.conv6.bn' == layer_name:
            self.vgg.conv6.bn= new_layer
            self.layer_names["vgg.conv6.bn"]=new_layer

        elif 'vgg.conv6.Relu' == layer_name:
            self.vgg.conv6.Relu= new_layer
            self.layer_names["vgg.conv6.Relu"]=new_layer

        elif 'vgg.conv7' == layer_name:
            self.vgg.conv7= new_layer
            self.layer_names["vgg.conv7"]=new_layer

        elif 'vgg.conv7.conv' == layer_name:
            self.vgg.conv7.conv= new_layer
            self.layer_names["vgg.conv7.conv"]=new_layer

        elif 'vgg.conv7.bn' == layer_name:
            self.vgg.conv7.bn= new_layer
            self.layer_names["vgg.conv7.bn"]=new_layer

        elif 'vgg.conv7.Relu' == layer_name:
            self.vgg.conv7.Relu= new_layer
            self.layer_names["vgg.conv7.Relu"]=new_layer

        elif 'vgg.maxpool2d1' == layer_name:
            self.vgg.maxpool2d1= new_layer
            self.layer_names["vgg.maxpool2d1"]=new_layer

        elif 'vgg.maxpool2d2' == layer_name:
            self.vgg.maxpool2d2= new_layer
            self.layer_names["vgg.maxpool2d2"]=new_layer

        elif 'rnn' == layer_name:
            self.rnn= new_layer
            self.layer_names["rnn"]=new_layer

        elif 'rnn.0' == layer_name:
            self.rnn[0]= new_layer
            self.layer_names["rnn.0"]=new_layer

        elif 'rnn.0.embedding' == layer_name:
            self.rnn[0].embedding= new_layer
            self.layer_names["rnn.0.embedding"]=new_layer

        elif 'rnn.1' == layer_name:
            self.rnn[1]= new_layer
            self.layer_names["rnn.1"]=new_layer

        elif 'rnn.1.embedding' == layer_name:
            self.rnn[1].embedding= new_layer
            self.layer_names["rnn.1.embedding"]=new_layer

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


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




from torch.nn import functional as F
class CTCLoss_torch(torch.nn.Module):
    """
     CTCLoss definition

     Args:
        max_sequence_length(int): max number of sequence length. For text images, the value is equal to image
        width
        max_label_length(int): max number of label length for each input.
        batch_size(int): batch size of input logits
     """

    def __init__(self, max_sequence_length, max_label_length, batch_size):
        super(CTCLoss_torch, self).__init__()
        self.sequence_length = torch.tensor(np.array([max_sequence_length] * batch_size), dtype=torch.int32)
        labels_indices = []
        for i in range(batch_size):
            for j in range(max_label_length):
                labels_indices.append([i, j])
        self.labels_indices = torch.tensor(np.array(labels_indices), dtype=torch.int64)
        self.reshape = torch.reshape
        # self.ctc_loss = torch.nn.CTCLoss()

    def forward(self, logit, label):
        batch_size = logit.size(1)
        targets = torch.randint(1, 20, (batch_size, 2), dtype=torch.long)
        input_lengths = torch.ones(logit.size(1)).long() * logit.size(0)
        target_lengths = torch.ones(logit.size(1)).long() * 2
        loss = F.ctc_loss(logit, targets, input_lengths, target_lengths)
        return loss

