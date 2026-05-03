import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv_layer(kernel_size):
    return nn.Conv2d(in_channels=1, out_channels=96, kernel_size=kernel_size, padding=1,
                     padding_mode="zeros", bias=True)


class TextCNN(nn.Module):
    def __init__(self, vocab_len, word_len, num_classes, vec_length, embedding_table='uniform'):
        super(TextCNN, self).__init__()
        self.vec_length = vec_length
        self.word_len = word_len
        self.num_classes = num_classes
        self.vocab_len = vocab_len

        self.unsqueeze = torch.unsqueeze
        self.embedding = nn.Embedding(vocab_len, self.vec_length)  # embedding_table=embedding_table

        # self.slice =
        self.layer1 = self.make_layer(kernel_height=3)
        self.layer2 = self.make_layer(kernel_height=4)
        self.layer3 = self.make_layer(kernel_height=5)
        self.reducemax = torch.max
        self.concat = torch.cat  # dim = 1

        self.fc = nn.Linear(96 * 3, self.num_classes)
        self.drop = nn.Dropout(p=0.5)

        self.Cascade_OPs = []
        self.Basic_OPS = []
        self.add_Cascade_OPs = []

        self.out_shapes = {
            "INPUT": [-1, 51],
            "embedding": [-1, 1, 51, 40],
            "layer1.0": [-1, 96, 51, 3],
            "layer1.1": [-1, 96, 51, 3],
            "layer1.2": [-1, 96, 1, 3],
            "layer2.0": [-1, 96, 50, 3],
            "layer2.1": [-1, 96, 50, 3],
            "layer2.2": [-1, 96, 1, 3],
            "layer3.0": [-1, 96, 49, 3],
            "layer3.1": [-1, 96, 49, 3],
            "layer3.2": [-1, 96, 1, 3],
            "drop": [-1, 288],
            "fc": [-1, 2],
            "OUTPUT": [-1, 2]
        }
        self.in_shapes = {
            "INPUT": [-1, 51],
            "embedding": [-1, 1, 51],
            "layer1.0": [-1, 1, 51, 40],
            "layer1.1": [-1, 96, 51, 3],
            "layer1.2": [-1, 96, 51, 3],
            "layer2.0": [-1, 1, 51, 40],
            "layer2.1": [-1, 96, 50, 3],
            "layer2.2": [-1, 96, 50, 3],
            "layer3.0": [-1, 1, 51, 40],
            "layer3.1": [-1, 96, 49, 3],
            "layer3.2": [-1, 96, 49, 3],
            "drop": [-1, 288],
            "fc": [-1, 288],
            "OUTPUT": [-1, 2]
        }

        self.orders = {
            "embedding": ["INPUT", ["layer1.0", "layer2.0", "layer3.0"]],
            "layer1.0": ["embedding", "layer1.1"],
            "layer1.1": ["layer1.0", "layer1.2"],
            "layer1.2": ["layer1.1", "drop"],
            "layer2.0": ["embedding", "layer2.1"],
            "layer2.1": ["layer2.0", "layer2.2"],
            "layer2.2": ["layer2.1", "drop"],
            "layer3.0": ["embedding", "layer3.1"],
            "layer3.1": ["layer3.0", "layer3.2"],
            "layer3.2": ["layer3.1", "drop"],
            "drop": [["layer1.2", "layer2.2", "layer3.2"], "fc"],
            "fc": ["drop", "OUTPUT"]
        }
        self.layer_names = {
            "embedding": self.embedding,
            "layer1": self.layer1,
            "layer1.0": self.layer1[0],
            "layer1.1": self.layer1[1],
            "layer1.2": self.layer1[2],
            "layer2": self.layer2,
            "layer2.0": self.layer2[0],
            "layer2.1": self.layer2[1],
            "layer2.2": self.layer2[2],
            "layer3": self.layer3,
            "layer3.0": self.layer3[0],
            "layer3.1": self.layer3[1],
            "layer3.2": self.layer3[2],
            "drop": self.drop,
            "fc": self.fc,
        }

    def make_layer(self, kernel_height):
        return nn.Sequential(
            make_conv_layer((kernel_height, self.vec_length)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.word_len - kernel_height + 1, 1)),
        )

    def forward(self, x):

        x = self.unsqueeze(x, 1)
        # print("unsqueeze.dtype: ", x.dtype)
        x = x.to(dtype=torch.int64)
        x = torch.clamp(x, 0, self.vocab_len - 1)
        x = self.embedding(x)
        # print("embedding.dtype: ", x.dtype)

        x = x.to(dtype=torch.float32)
        x1 = self.layer1(x)
        # print("layer1.dtype: ", x.dtype)

        x2 = self.layer2(x)
        # print("layer2.dtype: ", x.dtype)

        x3 = self.layer3(x)
        # print("layer3.dtype: ", x.dtype)

        x1 = x1.to(dtype=torch.float32)
        x2 = x2.to(dtype=torch.float32)
        x3 = x3.to(dtype=torch.float32)

        x1, _ = self.reducemax(x1, 2, keepdim=False)
        x1, _ = self.reducemax(x1, 2, keepdim=False)
        x2, _ = self.reducemax(x2, 2, keepdim=False)
        x2, _ = self.reducemax(x2, 2, keepdim=False)
        x3, _ = self.reducemax(x3, 2, keepdim=False)
        x3, _ = self.reducemax(x3, 2, keepdim=False)

        # print("x1.dtype", x1.dtype)
        # print("x2.dtype", x2.dtype)
        # print("x3.dtype", x3.dtype)

        x = self.concat((x1, x2, x3), dim=1)
        x = self.drop(x)
        # x = x.to(dtype=torch.float32)
        # print("drop.dtype", x.dtype)

        x = self.fc(x)

        # print("fc.dtype", x.dtype)

        return x

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self, layer_name, new_layer):

        if "embedding" == layer_name:
            self.embedding = new_layer
            self.layer_names['layers'] = new_layer
        elif "layer1" == layer_name:
            self.layer1 = new_layer = new_layer
            self.layer_names["layer1"] = new_layer
        elif "layer1.0" == layer_name:
            self.layer1[0] = new_layer
            self.layer_names["layer1.0"] = new_layer
        elif "layer1.1" == layer_name:
            self.layer1[1] = new_layer
            self.layer_names["layer1.1"] = new_layer
        elif "layer1.2" == layer_name:
            self.layer1[2] = new_layer
            self.layer_names["layer1.2"] = new_layer
        elif "layer2" == layer_name:
            self.layer2 = new_layer
            self.layer_names["layer2"] = new_layer
        elif "layer2.0" == layer_name:
            self.layer2[0] = new_layer
            self.layer_names["layer2.0"] = new_layer
        elif "layer2.1" == layer_name:
            self.layer2[1] = new_layer
            self.layer_names["layer2.1"] = new_layer
        elif "layer2.2" == layer_name:
            self.layer2[2] = new_layer
            self.layer_names["layer2.2"] = new_layer
        elif "layer3" == layer_name:
            self.layer3 = new_layer
            self.layer_names["layer3"] = new_layer
        elif "layer3.0" == layer_name:
            self.layer3[0] = new_layer
            self.layer_names["layer3.0"] = new_layer
        elif "layer3.1" == layer_name:
            self.layer3[1] = new_layer
            self.layer_names["layer3.1"] = new_layer
        elif "layer3.2" == layer_name:
            self.layer3[2] = new_layer
            self.layer_names["layer3.2"] = new_layer
        elif "drop" == layer_name:
            self.drop = new_layer
            self.layer_names["drop"] = new_layer
        elif "fc" == layer_name:
            self.fc = new_layer
            self.layer_names["fc"] = new_layer

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

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out


