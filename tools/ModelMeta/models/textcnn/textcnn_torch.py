import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace


def make_conv_layer(kernel_size):
    return nn.Conv2d(in_channels=1, out_channels=96, kernel_size=kernel_size, padding=1, bias=True)


class TextCNN(nn.Module):
    """
    TextCNN architecture
    """

    def __init__(self, vocab_len, word_len, num_classes, vec_length):
        super(TextCNN, self).__init__()
        self.vec_length = vec_length
        self.word_len = word_len
        self.num_classes = num_classes
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, self.vec_length)

        self.layer1 = self.make_layer(kernel_height=3)
        self.layer2 = self.make_layer(kernel_height=4)
        self.layer3 = self.make_layer(kernel_height=5)

        self.fc = nn.Linear(96 * 3, self.num_classes)
        self.drop = nn.Dropout(p=0.5)

    def make_layer(self, kernel_height):
        return nn.Sequential(
            make_conv_layer((kernel_height, self.vec_length)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.word_len - kernel_height + 1, 1)),
        )

    def forward(self, x):
        """
        forward
        """
        x = torch.unsqueeze(x, 1)
        x = torch.clamp(x, 0, self.vocab_len - 1)
        # print("Embedding weight shape:", self.embedding.weight.shape)
        # print("Max index in text tensor:", x.max())
        # print("Min index in text tensor:", x.min())
        x = self.embedding(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        # print("x1.shape", x1.shape)
        # print("x2.shape", x2.shape)
        # print("x3.shape", x3.shape)
        x1 = F.adaptive_max_pool2d(x1, (1, 1)).view(x.size(0), -1)
        x2 = F.adaptive_max_pool2d(x2, (1, 1)).view(x.size(0), -1)
        x3 = F.adaptive_max_pool2d(x3, (1, 1)).view(x.size(0), -1)
        # print("x1.shape", x1.shape)
        # print("x2.shape", x2.shape)
        # print("x3.shape", x3.shape)
        x = torch.cat((x1, x2, x3), 1)

        x = self.drop(x)
        x = self.fc(x)

        return x

class MyCustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        # Check if the current module is '_make_layer'
        # print("module_qualified_name", module_qualified_name)
        if module_qualified_name.endswith('_make_layer'):
            return True
        if module_qualified_name.startswith('layers'):
            return True
        return super().is_leaf_module(m, module_qualified_name)


if __name__ == '__main__':
    model = TextCNN(vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
    a = torch.tensor(np.random.randn(8, 51).astype(np.int32), dtype=torch.int64)
    # print("type", a.dtype)
    # print(model(a).shape)
    # symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
    # # print(symbolic_traced.graph)
    # # # print(net)
    # graph: torch.fx.Graph = torch.fx.Tracer().trace(symbolic_traced)
    # for node in graph.nodes:
    #     print("name:", node.name)
    #     print("op:", node.op)
    # torch.onnx.export(model, a, "textcnn_torch.onnx", verbose=True)
    symbolic_traced = MyCustomTracer().trace(model)
    traced = torch.fx.GraphModule(model, symbolic_traced)
    for node in symbolic_traced.nodes:
        print("mutable_op_name", node.name, "op_type", node.op, "name", node.name)