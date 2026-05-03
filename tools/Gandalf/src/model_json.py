"""
    json模型的类
"""

import json
import logging

import numpy as np

import tensorflow as tf
import torch
import jittor
from torchsummary import summary as t_summary
from jittorsummary import summary as j_summary

from src.interpreter.interpreter.tf_interpreter import TFInterpreter
from src.interpreter.interpreter.torch_interpreter import TorchInterpreter
from src.interpreter.interpreter.jittor_interpreter import JittorInterpreter

from src.interpreter.pytorch_model import PyTorchNetwork
from src.interpreter.jittor_model import JittorNetwork

from src.grammar.top_level_checker import TopLevelChecker

from src.configuration.support.tf_expansion.bottleneck import Bottleneck

from src.train_process.tf.normal import normal_train as tf_normal_train
from src.train_process.torch.normal import normal_train as torch_normal_train
from src.train_process.jittor.normal import normal_train as jittor_normal_train

from src.test_process.tf.normal import normal_test as tf_normal_test
from src.test_process.torch.normal import normal_test as torch_normal_test
from src.test_process.jittor.normal import normal_test as jittor_normal_test


class ModelJSON:
    model_num = 0
    interpreter_table = {
        'TensorFlow': TFInterpreter(),
        'PyTorch': TorchInterpreter(),
        'Jittor': JittorInterpreter()
    }
    dim_convert = [(0, 2, 1), (0, 3, 1, 2), (0, 4, 1, 2, 3)]
    top_level_checker = TopLevelChecker()

    # 初始化方法区
    def __init__(self, file_path):
        if isinstance(file_path, str):
            self.file_path = file_path
            with open(file_path, 'r') as f:
                total_json = json.load(f)
        elif isinstance(file_path, dict):
            total_json = file_path
        # 初步检查语法
        ModelJSON.top_level_checker.grammar_checker(total_json)
        # name字段
        if total_json.__contains__('name'):
            self.name = total_json['name']
        else:
            self.name = 'jsonModel_{0}'.format(ModelJSON.model_num)
        ModelJSON.model_num += 1
        # framework字段
        self.framework = total_json['framework']
        # input_shape字段
        self.input_shape = total_json['input_shape']
        # network字段
        network_in_json = total_json['network']
        self.__init_network(network_in_json)
        # 输出提示
        logging.info('[JsonDL] {0} has been defined.'.format(self.name))

    # # 初始化方法区
    # def __init__(self, model1, model2, name=None):
    #     # name字段
    #     if name is not None:
    #         self.name = name
    #     else:
    #         self.name = 'jsonModel_{0}'.format(ModelJSON.model_num)
    #     ModelJSON.model_num += 1
    #     # framework字段
    #     if model1.framework == model2.framework:
    #         self.framework = model1.framework
    #     else:
    #         raise Exception('Models of different framework implement cannot concatenate.')
    #     # input_shape字段
    #     self.input_shape = model1.input_shape
    #     # network字段
    #     model1_network = model1.network
    #     model2_network = model2.network
    #     if self.framework == 'TensorFlow':
    #         self.network = model2_network(model1_network)

        # 输出提示
        logging.info('[JsonDL] {0} has been defined.'.format(self.name))

    def __init_network(self, network_in_json):
        interpreter = ModelJSON.interpreter_table[self.framework]
        if self.framework == 'TensorFlow':
            # 创建network
            # 首层输入层
            inp = tf.keras.layers.Input(self.input_shape)
            network = inp
            # branch_to 字典
            branch_to = {}
            # 记录当前维度
            current_shape = self.input_shape
            self.layer_output_shape = [current_shape]
            # 剩余层
            for layer in network_in_json:
                layer_name = layer['name']
                params = layer['params'] if layer.__contains__('params') else {}
                # index 字段
                if layer.__contains__('index'):
                    # 首先判断是否需要上/下采样或升/降维
                    if current_shape != branch_to[layer['index']][1]:
                        bottleneck = Bottleneck(current_shape, branch_to[layer['index']][1])
                        network = bottleneck([network, branch_to[layer['index']][0]])
                    else:
                        bottleneck = tf.keras.layers.Add()
                        network = bottleneck([network, branch_to[layer['index']][0]])
                # index处理完毕
                op, current_shape = interpreter.get_op_and_shape(layer_name, params, current_shape)
                network = op(network)
                self.layer_output_shape.append(current_shape)
                # branch_to 字段
                if layer.__contains__('branch_to'):
                    branch_to[layer['branch_to']] = (network, current_shape)
            model = tf.keras.Model(inp, network)
            self.network = model
            # self.network = TensorFlowNetwork(network_in_json, interpreter, self.input_shape)
            # trial_inp = tuple([1] + self.input_shape)
            # self.network.predict(np.ones(trial_inp, dtype=np.float32))
            # self.layer_output_shape = self.network.layer_output_shape
        elif self.framework == 'PyTorch':
            self.network = PyTorchNetwork(network_in_json, interpreter, self.input_shape)
            # 确保使用了gpu
            gpu = torch.device("cuda:0")
            self.network.to(gpu)
            self.layer_output_shape = self.network.layer_output_shape
        elif self.framework == 'Jittor':
            self.network = JittorNetwork(network_in_json, interpreter, self.input_shape)
            self.layer_output_shape = self.network.layer_output_shape
        else:
            raise Exception('No support DL framework')

    # 关于权重及模型的保存与加载
    def save_weights_across_framework(self, path):
        model = self.network
        if self.framework == 'TensorFlow':
            model.save_weights(path)
        elif self.framework == 'PyTorch':
            if path[-4:] == '.pkl':
                torch.save(model.state_dict(), path)
            else:
                torch.save(model.state_dict(), path + '.pkl')
        elif self.framework == 'Jittor':
            if path[-4:] == '.pkl':
                jittor.save(model.state_dict(), path)
            else:
                jittor.save(model.state_dict(), path + '.pkl')
        else:
            raise Exception('No support DL framework')
        logging.info('[JsonDL] weights of {0} has been saved.'.format(self.name))

    def save_model_across_framework(self, path):
        model = self.network
        if self.framework == 'TensorFlow':
            if path[-3:] == '.h5':
                model.save(path)
            else:
                model.save(path + '.h5')
        elif self.framework == 'PyTorch':
            if path[-4:] == '.pkl':
                torch.save(model, path)
            else:
                torch.save(model, path + '.pkl')
        elif self.framework == 'Jittor':
            if path[-4:] == '.pkl':
                jittor.save(model, path)
            else:
                jittor.save(model, path + '.pkl')
        else:
            raise Exception('No support DL framework')
        logging.info('[JsonDL] {0} has been saved.'.format(self.name))

    def load_weights_across_framework(self, path):
        if self.framework == 'TensorFlow':
            self.network.load_weights(path)
        elif self.framework == 'PyTorch':
            if path[-4:] == '.pkl':
                self.network.load_state_dict(torch.load(path))
            else:
                self.network.load_state_dict(torch.load(path + '.pkl'))
        elif self.framework == 'Jittor':
            if path[-4:] == '.pkl':
                self.network.load_state_dict(jittor.load(path))
            else:
                self.network.load_state_dict(jittor.load(path + '.pkl'))
        else:
            raise Exception('No support DL framework')
        logging.info('[JsonDL] weights of {0} has been loaded.'.format(self.name))

    def load_model_across_framework(self, path):
        if self.framework == 'TensorFlow':
            if path[-3:] == '.h5':
                self.network = tf.keras.models.load_model(path)
            else:
                self.network = tf.keras.models.load_model(path + '.h5')
        elif self.framework == 'PyTorch':
            if path[-4:] == '.pkl':
                self.network = torch.load(path)
            else:
                self.network = torch.load(path + '.pkl')
        elif self.framework == 'Jittor':
            if path[-4:] == '.pkl':
                self.network = jittor.load(path)
            else:
                self.network = jittor.load(path + '.pkl')
        else:
            raise Exception('No support DL framework')
        logging.info('[JsonDL] {0} has been loaded.'.format(self.name))

    # 功能方法区
    def predict(self, x, format="normal", convert=False, **kwargs):
        if format not in ["normal", "NCHW", "NHWC"]:
            raise Exception('Parameter \"format\" should be chosen between \"normal\", \"NCHW\" and \"NHWC\".')
        if not isinstance(convert, bool):
            raise Exception('Parameter \"convert\" should be bool')
        if self.framework == 'TensorFlow':
            res = self.network(x, **kwargs)
            shape = list(res.shape)
            if format == 'NCHW' and len(shape) == 4:
                res = tf.transpose(res, (0, 3, 1, 2))
            elif format == 'NCHW' and len(shape) == 3:
                res = tf.transpose(res, (0, 2, 1))
            elif format == 'NCHW' and len(shape) == 5:
                res = tf.transpose(res, (0, 4, 1, 2, 3))
        elif self.framework == 'PyTorch':
            x = np.transpose(x, ModelJSON.dim_convert[len(self.input_shape) - 2])
            t = torch.from_numpy(x)
            gpu = torch.device("cuda:0")
            t = t.to(gpu)
            res = self.network(t, **kwargs)
            shape = list(res.shape)
            if format == 'NHWC' and len(shape) == 4:
                res = res.permute(0, 2, 3, 1)
            elif format == 'NHWC' and len(shape) == 3:
                res = res.permute(0, 2, 1)
            elif format == 'NHWC' and len(shape) == 5:
                res = res.permute(res, (0, 2, 3, 4, 1))
        elif self.framework == 'Jittor':
            x = np.transpose(x, ModelJSON.dim_convert[len(self.input_shape) - 2])
            t = jittor.Var(x)
            res = self.network(t, **kwargs)
            shape = list(res.shape)
            if format == 'NHWC' and len(shape) == 4:
                res = res.permute(0, 2, 3, 1)
            elif format == 'NHWC' and len(shape) == 3:
                res = res.permute(0, 2, 1)
            elif format == 'NHWC' and len(shape) == 5:
                res = res.permute(res, (0, 2, 3, 4, 1))
        else:
            raise Exception('No support DL framework')
        # 最后统一处理输出tensor还是numpy数组
        if convert:
            return res.numpy()
        else:
            return res

    def summary(self):
        if self.framework == 'TensorFlow':
            self.network.summary()
        elif self.framework == 'PyTorch':
            t_summary(self.network, input_size=(self.input_shape[-1], self.input_shape[-3], self.input_shape[-2]),
                      device="cuda")
        elif self.framework == 'Jittor':
            j_summary(self.network, input_size=(self.input_shape[-1], self.input_shape[-3], self.input_shape[-2]))
        else:
            raise Exception('No support DL framework')

    def train(self, optimizer, loss, epochs, batch_size, ds_train, ds_valid=None, metrics=['OneHotAccuracy'],
              localized=False, shuffle=True):
        if self.framework == 'TensorFlow':
            r1, r2 = tf_normal_train(self.network, optimizer, loss, epochs, batch_size, ds_train, ds_valid, metrics,
                            localized, shuffle)
        elif self.framework == 'PyTorch':
            r1, r2 = torch_normal_train(self.network, optimizer, loss, epochs, batch_size, ds_train, ds_valid, metrics,
                            localized, shuffle)
        elif self.framework == 'Jittor':
            r1, r2 = jittor_normal_train(self.network, optimizer, loss, epochs, batch_size, ds_train, ds_valid, metrics,
                               localized, shuffle)
        else:
            raise Exception('No support DL framework.')

        return r1, r2

    def test(self, ds_test, metrics=['OneHotAccuracy']):
        if self.framework == 'TensorFlow':
            res = tf_normal_test(self.network, ds_test, metrics)
        elif self.framework == 'PyTorch':
            res = torch_normal_test(self.network, ds_test, metrics)
        elif self.framework == 'Jittor':
            res = jittor_normal_test(self.network, ds_test, metrics)
        else:
            raise Exception('No support DL framework.')

        return res

# if __name__ == '__main__':
#     x = np.ones((4, 224, 224, 3))
#     model = ModelJSON('example_jittor.json')
#     print(model.predict(x))

    # x = np.ones((1, 28, 28, 1))
    # inp = tf.Tensor(x)
    # print(list(inp.shape))
    # conv2d = torch.nn.Conv2d(1, 32, 3)
    # summary(conv2d, input_size=(1, 28, 28), device='cpu')
    # conv2d = conv2d(inp)
    # conv2d_2 = torch.nn.Conv2d(32, 64, 3)(conv2d)
    # print(conv2d_2)

    # x = np.ones((1, 28, 28, 1))
    # inp = tf.keras.layers.Input((28, 28, 1))
    # conv2d = tf.keras.layers.Conv2D(32, 3)
    # print(list(conv2d(x).shape))
    # conv2d_2 = tf.keras.layers.Conv2D(64, 3)
    # flatten = tf.keras.layers.Flatten()
    # inp2 = tf.keras.layers.Input((1, 36846))
    # dense = tf.keras.layers.Dense(1024)
    #
    #
    # network = inp
    # network = conv2d(network)
    # network = conv2d_2(network)
    # network = flatten(network)
    # # network = inp2(network)
    # network = dense(network)
    #
    # model = tf.keras.Model(inp, network)
    # print(model(x))
