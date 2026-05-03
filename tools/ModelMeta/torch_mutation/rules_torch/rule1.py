## rule1 利用torch.jit.script(layer)进行优化,适用于所有层
import torch
import torch.nn as nn
import torch.jit

class TransLayer_rule1_Conv2d(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Conv2d, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_AvgPool2d(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_AvgPool2d, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_MaxPool2d(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_MaxPool2d, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_ReLU(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_ReLU, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_ReLU6(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_ReLU6, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_BatchNorm2d(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_BatchNorm2d, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Linear(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Linear, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Flatten(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Flatten, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Hardsigmoid(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Hardsigmoid, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Sigmoid(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Sigmoid, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Softmax(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Softmax, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Tanh(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Tanh, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_ConvTranspose2d(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_ConvTranspose2d, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_LeakyReLU(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_LeakyReLU, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_AdaptiveAvgPool2d(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_AdaptiveAvgPool2d, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Dropout(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Dropout, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_Embedding(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_Embedding, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)

class TransLayer_rule1_LSTM(nn.Module):
    def __init__(self, layer):
        super(TransLayer_rule1_LSTM, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError("This wrapper only supports nn.Module layers")
        self.optimized_layer = torch.jit.script(layer)

    def forward(self, x):
        return self.optimized_layer(x)






















# fn=nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# optimized_fn = TransLayer(fn)
#
# # 测试输入
# input_tensor = torch.randn((5,64,32,32))
#
# # 变异前的结果
# output_before = fn(input_tensor)
# # 变异后的结果
# output_after = optimized_fn(input_tensor)
#
# # 验证输出是否接近 完全一样！！
# are_outputs_close = torch.allclose(output_before, output_after, atol=1e-6)
# print(f"Are outputs close? {are_outputs_close}")
# print(output_before==output_after)
#
# print(fn)
# print(optimized_fn)