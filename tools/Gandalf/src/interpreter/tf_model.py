"""
    tf的模型及模型生成器类
"""
import tensorflow as tf
from src.configuration.support.tf_expansion.bottleneck import Bottleneck


class TensorFlowNetwork(tf.keras.Model):
    def __init__(self, network_json, interpreter, input_shape):
        super(TensorFlowNetwork, self).__init__()
        self.my_layers = []
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
                    self.my_layers.append((bottleneck, branch_to[layer['index']][0]))
                else:
                    self.my_layers.append((tf.keras.layers.Add(), branch_to[layer['index']][0]))
            # index处理完毕
            op, current_shape = interpreter.get_op_and_shape(layer_name, params, current_shape)
            self.layer_output_shape.append(current_shape)
            # if isinstance(op, list):
            #     for o in op:
            #         self.layers.append(o)
            # else:
            self.my_layers.append(op)
            # branch_to 字段
            if layer.__contains__('branch_to'):
                # 前者为当前层在self.layers中的索引 后者为其shape
                branch_to[layer['branch_to']] = (len(self.my_layers) - 1, current_shape)
                self.branch_index.add(len(self.my_layers) - 1)

    def call(self, x, training=False):
        dic = {}
        for i in range(len(self.my_layers)):
            f = self.my_layers[i]
            if isinstance(f, tuple):
                # bottleneck的情况
                real_f = f[0]
                x = real_f([x, dic[f[1]]], training=training)
            else:
                x = f(x, training=training)
            if i in self.branch_index:
                r = x
                dic[i] = r
        return x
