import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.nn.functional as F


# class SoftmaxCrossEntropyLoss(nn.Module):
#     def __init__(self, num_cls=21, ignore_label=255):
#         super(SoftmaxCrossEntropyLoss, self).__init__()
#         self.num_cls = num_cls
#         self.ignore_label = ignore_label
#         self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
#
#     def forward(self, logits, labels):
#         labels = labels.type(torch.long)
#         loss = self.criterion(logits, labels)
#         return loss


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def forward(self, logits, labels):
        labels = labels.view(-1).long()
        logits = logits.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
        logits = logits.view(-1, self.num_classes)

        weights = (labels != self.ignore_label).float()
        labels = labels.clamp(0, self.num_classes - 1)  # Clamp the ignore_label to be within the valid range

        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = loss * weights
        loss = loss.sum() / weights.sum()
        return loss


class SegDataset(Dataset):
    def __init__(self,
                 image_mean,
                 image_std,
                 data_file='',
                 crop_size=512,
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21):

        self.data_file = data_file
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        assert max_scale > min_scale

    def __len__(self):
        return len(self.data_file)

    def preprocess_(self, image, label):
        # bgr image
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]

        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def __getitem__(self, idx):
        image, label = self.data_file[idx]
        image_out, label_out = self.preprocess_(image, label)
        image_out = torch.from_numpy(image_out).type(torch.FloatTensor)
        label_out = torch.from_numpy(label_out).type(torch.LongTensor)
        return image_out, label_out


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    # print("in_planes:", in_planes, "out_planes:", out_planes, "stride:", stride, "dilation:", dilation, "padding:",
    #       padding)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                     bias=False)


class Resnet(nn.Module):
    def __init__(self, block, block_num, output_stride):
        super(Resnet, self).__init__()
        self.inplanes = 64
        # weight_shape = (self.inplanes, 3, 7, 7)
        # weight = torch.tensor(np.zeros(weight_shape), dtype=torch.float32)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1.weight = torch.nn.Parameter(weight)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=2, grids=[1, 2, 4])
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=4, grids=[1, 2, 4])

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        if grids is None:
            grids = [1] * blocks
        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0])
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=base_dilation * grids[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)
        return out


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate=1):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate,
                             bias=False)

        self.aspp_conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.aspp_conv(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.size()
        out = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        out = self.conv(out)
        out = torch.nn.functional.interpolate(out, size[2:], mode='bilinear', align_corners=True)
        return out


class ASPP(nn.Module):
    def __init__(self, atrous_rates, in_channels=2048, num_classes=21):
        super(ASPP, self).__init__()
        out_channels = 256
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0])
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1])
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2])
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3])
        self.aspp_pooling = ASPPPooling(in_channels, out_channels)

        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=True)
        self.drop = nn.Dropout(0.7)

    def forward(self, x):
        # print("ASPP input shape: ", x.shape)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)

        x = torch.cat((x1, x2), dim=1)
        x = torch.cat((x, x3), dim=1)
        x = torch.cat((x, x4), dim=1)
        x = torch.cat((x, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.training:
            x = self.drop(x)
        x = self.conv2(x)
        return x


#
# # Instantiate your custom ResNet class here
# model = Resnet(Bottleneck, [3, 4, 23, 3], 16)
# resnet = model


class DeepLabV3_torch(nn.Module):
    def __init__(self, num_classes=21):
        super(DeepLabV3_torch, self).__init__()
        self.resnet = Resnet(Bottleneck, [3, 4, 23, 3], 8)
        self.aspp = ASPP([
            1,
            6,
            12,
            18,
        ], num_classes=num_classes)

    def forward(self, x):
        size = x.shape
        # print("================================================================")
        # print(size)
        # print("================================================================")
        out = self.resnet(x)
        out = self.aspp(out)
        out = torch.nn.functional.interpolate(out, size[2:], mode='bilinear', align_corners=True)
        return out


def train_eval_deeplabv3_torch():
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = DeepLabV3_torch(num_classes=21)
    criterion = SoftmaxCrossEntropyLoss(num_classes=21, ignore_label=255)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0001)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    a = np.random.randn(4, 3, 513, 513)
    # batch_size must be larger than 1
    a = torch.tensor(a, dtype=torch.float32).to(device)
    b = np.random.randn(4, 513, 513)
    b = torch.tensor(b, dtype=torch.float32).to(device)
    print(model(a))

    # Define the train step
    def train_step(data, label):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Training loop

    print("================================")
    loss_value = train_step(a, b)
    print(loss_value)
    print("================================")
    return


if __name__ == '__main__':
    net = DeepLabV3_torch(num_classes=21)
    # batch_size must be larger than 1
    a = torch.tensor(np.random.randn(2, 3, 513, 513), dtype=torch.float32)
    print(net(a).shape)
    torch.onnx.export(net, a, "deeplabv3_torch.onnx", verbose=True)
