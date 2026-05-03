import os
import sys
import random

import cv2
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import torch
import traceback
import json
import numpy as np

from src.model_json import ModelJSON
from torch.utils.data.dataset import TensorDataset
import src.interpreter.interpreter.ops as ops
from preliminary_trials.env.weights_unite import WeightsUnite

from src.train_process.tf.phased_normal import phased_normal_train as tf_train
from src.train_process.torch.phased_normal import phased_normal_train as torch_train

DURATION_BY_SECONDS = 4 * 60 * 60
weight_unite = WeightsUnite()


class LeNetEnv:
    def __load_dataset(self):
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        # lst = []
        # for i in range(50000):
        #     lst.append(cv2.resize(x_train[i, :, :], (224, 224)))
        # x_train = np.asarray(lst)
        x_train = x_train.reshape((-1, 32, 32, 3))
        y_train = y_train.reshape((-1))
        self.predict_label = y_train[:self.batch_size]
        y = self.__generate_one_hot(y_train)
        y = y.astype(np.float32)
        x = x_train / 255.0
        x = x.astype(np.float32)

        data = tf.data.Dataset.from_tensor_slices((x, y))
        self.tf_dataset = data
        self.predict_data = x[:self.batch_size]
        self.predict_data_mean = tf.reduce_mean(self.predict_data)

        x = x.transpose((0, 3, 1, 2))
        data = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        self.torch_dataset = data

    def __generate_one_hot(self, y):
        materials = np.eye(10)
        return materials[y]

    def __init__(self):
        self.epochs = 5
        self.batch_size = 2
        self.lr = 1e-4
        self.__load_dataset()
        self.epsilon = 1e-5
        self.scale_factor = np.asarray(5., np.float32)

        tf_model = ModelJSON('./tf_vgg-16.json')
        torch_model = ModelJSON('./torch_vgg-16.json')

        weights_unify(tf_model.network, torch_model.network, weight_unite)

        p1 = tf_model.predict(self.predict_data)
        p2 = torch_model.predict(self.predict_data)
        self.base_inconsistency = self.cal_inconsistency(p1, p2)
        print(self.base_inconsistency)
        with open('./tf_vgg-16.json', 'r') as f:
            self.total_json = json.load(f)

    def cal_inconsistency(self, predict1, predict2):
        if len(predict2.shape) == 4:
            predict2 = predict2.permute(0, 2, 3, 1)
        if predict2.device.type == 'cuda':
            predict2 = predict2.cpu()
        predict2 = predict2.detach().numpy()
        diff = tf.maximum(predict1 - predict2, predict2 - predict1)
        return tf.reduce_sum(diff) / tf.cast(tf.size(diff), tf.float32)

    def copy_from_tf_model(self):
        res = {}
        res["name"] = "VGG-16"
        res["framework"] = "PyTorch"
        res["input_shape"] = [32, 32, 3]
        res["network"] = []
        for layer in self.total_json['network']:
            copy_layer = {}
            copy_layer['name'] = layer['name']
            if layer.__contains__('params'):
                copy_layer['params'] = {}
                for k, v in layer['params'].items():
                    copy_layer['params'][k] = v
            res["network"].append(copy_layer)
        return res

def weights_unify(tf_model, torch_model, weight_unite):
    def find_next_p(tf_layers, p):
        # 搜索下一个需要的层
        while p < len(tf_layers):
            cur_layer = tf_layers[p]
            cur_layer_name = cur_layer.name
            flag = None
            for name in sensitive_table:
                if cur_layer_name.find(name) != -1:
                    flag = name
                    break
            if flag:
                break
            else:
                p += 1
        return p, flag

    sensitive_table = ['conv', 'embedding', 'dense']
    func_table = {
        'conv': 'Conv',
        'embedding': 'Embedding',
        'dense': 'Linear'
    }
    # 前者为list形式 后者为iterator形式
    tf_layers = tf_model.layers
    torch_module_lst = torch_model.children()
    torch_layers = None
    for module_lst in torch_module_lst:
        torch_layers = module_lst.children()
        break
    assert torch_layers

    # 寻找第一个p
    # p表示当前遍历位置，超出范围则为无下一个需要权重的层
    # flag表示当前层的名字，None则无下一个需要权重的层
    p, flag = find_next_p(tf_layers, 0)
    # 遍历torch model
    for layer in torch_layers:
        # 没有权重需要统一了 直接跳出
        if flag is None:
            break
        torch_class = layer.__class__.__name__
        # 当前层不是我们需要的层
        if torch_class.find(func_table[flag]) == -1:
            continue
        else:
            # 交给专门的权重统一器处理
            weight_unite.weight_unify_for_layer(tf_layers[p], layer, torch_class)
            p, flag = find_next_p(tf_layers, p + 1)
    assert p >= len(tf_layers)
    assert flag is None


def save_meta_with_weights(tf_json, reward, err_type, c, err_info=None):
    tf_meta = {
        'reward': reward,
        'type': err_type,
        'json': tf_json
    }
    with open('./pool/{0}.json'.format(c), 'w') as f:
        json.dump(tf_meta, f)

    # 保存weights或错误日志
    if err_type == 'CRASH':
        # crash样本 保存错误信息
        with open('./crash/{0}'.format(c), 'w+') as f:
            f.write(err_info)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.config.experimental_run_functions_eagerly(True)

    start_time = time.time()
    counter = 0
    ln = LeNetEnv()
    pool_sizes = [2, 4, 6, 8, 10, 12, 14]

    iter = 0
    while True:
        print("Iter: {0}".format(iter))
        iter += 1
        cur_time = time.time()
        if cur_time - start_time >= DURATION_BY_SECONDS:
            with open('./num0', 'w+') as f:
                f.write(str((cur_time - start_time) / 3600))
            sys.exit()
        network = ln.total_json['network']
        # layer 0
        conv2d_1 = network[1]['params']
        rand1 = random.randint(32, 64)
        conv2d_1['out_channels'] = rand1
        conv2d_1['kernel_size'] = random.randint(1, 15)
        # layer 1
        conv2d_2 = network[2]['params']
        conv2d_2['in_channels'] = rand1
        conv2d_2['out_channels'] = rand1
        conv2d_2['kernel_size'] = random.randint(1, 15)
        # layer 2
        avg_pool_1 = network[3]['params']
        avg_pool_1['pool_size'] = pool_sizes[random.randint(0, 6)]
        # layer 3
        conv2d_3 = network[4]['params']
        conv2d_3['in_channels'] = rand1
        rand2 = random.randint(64, 128)
        conv2d_3['out_channels'] = rand2
        conv2d_3['kernel_size'] = random.randint(1, 15)
        # layer 4
        conv2d_4 = network[5]['params']
        conv2d_4['in_channels'] = rand2
        conv2d_4['out_channels'] = rand2
        conv2d_4['kernel_size'] = random.randint(1, 15)
        # layer 5
        avg_pool_2 = network[6]['params']
        avg_pool_2['pool_size'] = pool_sizes[random.randint(0, 6)]
        # layer 6
        conv2d_5 = network[7]['params']
        rand3 = random.randint(128, 256)
        conv2d_5['in_channels'] = rand2
        conv2d_5['out_channels'] = rand3
        conv2d_5['kernel_size'] = random.randint(1, 15)
        # layer 7
        conv2d_6 = network[8]['params']
        conv2d_6['in_channels'] = rand3
        conv2d_6['out_channels'] = rand3
        conv2d_6['kernel_size'] = random.randint(1, 15)
        # layer 8
        avg_pool_3 = network[9]['params']
        avg_pool_3['pool_size'] = pool_sizes[random.randint(0, 6)]
        # layer 9
        conv2d_7 = network[10]['params']
        rand4 = random.randint(256, 512)
        conv2d_7['in_channels'] = rand3
        conv2d_7['out_channels'] = rand4
        conv2d_7['kernel_size'] = random.randint(1, 15)
        # layer 10
        conv2d_8 = network[11]['params']
        conv2d_8['in_channels'] = rand4
        conv2d_8['out_channels'] = rand4
        conv2d_8['kernel_size'] = random.randint(1, 15)
        # layer 11
        avg_pool_4 = network[12]['params']
        avg_pool_4['pool_size'] = pool_sizes[random.randint(0, 6)]
        # layer 12
        conv2d_9 = network[13]['params']
        conv2d_9['in_channels'] = rand4
        conv2d_9['out_channels'] = rand4
        conv2d_9['kernel_size'] = random.randint(1, 15)
        # layer 13
        conv2d_10 = network[14]['params']
        conv2d_10['in_channels'] = rand4
        conv2d_10['out_channels'] = rand4
        conv2d_10['kernel_size'] = random.randint(1, 15)
        # layer 14
        avg_pool_5 = network[15]['params']
        avg_pool_5['pool_size'] = pool_sizes[random.randint(0, 6)]
        # layer 16
        dense_1 = network[17]['params']
        dense_1['in_features'] = 49 * rand4

        tf_dict = ln.total_json
        torch_dict = ln.copy_from_tf_model()
        print(tf_dict)
        print(torch_dict)

        try:
            torch_model = ModelJSON(torch_dict)
            tf_model = ModelJSON(tf_dict)

            weights_unify(tf_model.network, torch_model.network, weight_unite)

            # 优化器
            optimizer = {
                'name': 'Adam',
                'params': {'lr': 1e-4}
            }
            # 阶段性训练
            tf_phased_train = tf_train(tf_model.network, optimizer, "CrossEntropy", ln.epochs,
                                       ln.batch_size, ln.tf_dataset, shuffle=False, show_iter=True)
            torch_phased_train = torch_train(torch_model.network, optimizer, "CrossEntropy", ln.epochs,
                                             ln.batch_size, ln.torch_dataset, shuffle=False, show_iter=True)

            while True:
                tf_info, _, tf_done = next(tf_phased_train)
                torch_info, _, torch_done = next(torch_phased_train)

                predict_tf = tf_model.predict(ln.predict_data)
                predict_torch = torch_model.predict(ln.predict_data)

                # 首先，检查输出是否存在NAN/INF
                if tf.reduce_any(tf.math.is_nan(predict_tf)) or tf.reduce_any(tf.math.is_inf(predict_tf)):
                    save_meta_with_weights(tf_dict, 1.,'OUTPUT-NAN/INF', counter)
                    counter += 1
                    print('output nan')
                    break
                if torch.any(torch.isnan(predict_torch)) or torch.any(torch.isinf(predict_torch)):
                    save_meta_with_weights(tf_dict, 1., 'OUTPUT-NAN/INF', counter)
                    counter += 1
                    print('output nan')
                    break
                # 检查loss项是否为nan或inf
                # 提取loss
                tf_loss = tf_info[0][1]
                torch_loss = torch_info[0][1]
                # 检查
                if tf.reduce_any(tf.math.is_nan(tf_loss)) or tf.reduce_any(tf.math.is_inf(tf_loss)):
                    save_meta_with_weights(tf_dict, 1., 'LOSS-NAN/INF', counter)
                    counter += 1
                    print('loss nan')
                    break
                if torch.any(torch.isnan(torch_loss)) or torch.any(torch.isinf(torch_loss)):
                    save_meta_with_weights(tf_dict, 1., 'LOSS-NAN/INF', counter)
                    counter += 1
                    print('loss nan')
                    break
                # 无nan或inf 检查inconsistency
                reward = ln.cal_inconsistency(predict_tf, predict_torch) - ln.base_inconsistency
                print(reward)
                if tf.reduce_all(reward >= ln.epsilon):
                    save_meta_with_weights(tf_dict, reward.numpy(), 'INCONSISTENCY', counter)
                    counter += 1
                    break
                if tf_done or torch_done:
                    break
        except Exception as e:
            err_info = traceback.format_exc()
            print('crash occur')
            save_meta_with_weights(tf_dict, 1., 'CRASH', counter, err_info)
            counter += 1
