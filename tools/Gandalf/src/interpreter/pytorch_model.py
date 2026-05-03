"""
    pytorch的模型及模型生成器类
"""
import torch
from src.configuration.support.torch_expansion.bottleneck import Bottleneck
from src.configuration.support.torch_expansion.add import Add


class PyTorchNetwork(torch.nn.Module):
    def __init__(self, network_json, interpreter, input_shape):
        super(PyTorchNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.fake_layers = []
        self.branch_index = set()
        # 创建network
        # branch_to 字典
        branch_to = {}
        # 记录当前维度
        current_shape = input_shape
        self.layer_output_shape = [current_shape]
        # 剩余层
        for layer in network_json:
            layer_name = layer['name']
            params = layer['params'] if layer.__contains__('params') else {}
            # index 字段
            if layer.__contains__('index'):
                # 首先判断是否需要上/下采样或升/降维
                if current_shape != branch_to[layer['index']][1]:
                    bottleneck = Bottleneck(current_shape, branch_to[layer['index']][1])
                    # 添加的是算子和第二个输入的索引
                    self.layers.append(bottleneck)
                    self.fake_layers.append(branch_to[layer['index']][0])
                else:
                    self.layers.append(Add())
                    self.fake_layers.append(branch_to[layer['index']][0])
            # index处理完毕
            op, current_shape = interpreter.get_op_and_shape(layer_name, params, current_shape)
            self.layer_output_shape.append(current_shape)
            if isinstance(op, list):
                for o in op:
                    self.layers.append(o)
                    self.fake_layers.append(None)
            else:
                self.layers.append(op)
                self.fake_layers.append(None)
            # branch_to 字段
            if layer.__contains__('branch_to'):
                # 前者为当前层在self.layers中的索引 后者为其shape
                branch_to[layer['branch_to']] = (len(self.layers) - 1, current_shape)
                self.branch_index.add(len(self.layers) - 1)

    def forward(self, x):
        dic = {}
        for i in range(len(self.layers)):
            f = self.layers[i]
            if self.fake_layers[i] is not None:
                # bottleneck的情况
                x = f([x, dic[self.fake_layers[i]]])
            else:
                x = f(x)
            if i in self.branch_index:
                r = x
                dic[i] = r
        return x
