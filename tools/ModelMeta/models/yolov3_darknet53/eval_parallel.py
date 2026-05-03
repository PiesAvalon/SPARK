# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YoloV3 eval."""
import os
import datetime
import time

import mindspore as ms
import torch
from tqdm import tqdm

from network.cv.yolov3_darknet53.main_new import YOLOV3DarkNet53
# from src.yolo import YOLOV3DarkNet53
from network.cv.yolov3_darknet53.src.logger import get_logger
from network.cv.yolov3_darknet53.src.yolo_dataset import create_yolo_dataset

from network.cv.yolov3_darknet53.model_utils.config import config
from network.cv.yolov3_darknet53.model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process
from network.cv.yolov3_darknet53.util import DetectionEngine
import yaml

f = open("./config.txt", "r")
path = f.readline().splitlines()[0]
f.close()

with open(path, "r", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
train_configs = data['train_config']

device = train_configs['device']
device_id = train_configs['device_id']
if device == "GPU":
    final_device = "cuda:" + str(device_id)
else:
    final_device = "cpu"

# from util import DetectionEngine

def set_eval(model1, model2):
    model1.detect_1.conf_training = False
    model1.detect_2.conf_training = False
    model1.detect_3.conf_training = False
    model2.detect_1.conf_training = False
    model2.detect_2.conf_training = False
    model2.detect_3.conf_training = False
    return model1, model2

def conver_testing_shape(args):
    """Convert testing shape to list."""
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape


def load_parameters(network, file_name):
    config.logger.info("yolov3 pretrained network model: %s", file_name)
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)
    config.logger.info('load_model %s success', file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_test():
    """The function of eval."""
    start_time = time.time()
    config.data_root = os.path.join(config.data_dir, 'val2014')
    config.annFile = os.path.join(config.data_dir, 'annotations/new_instances_val2014_5%.json')

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=devid)

    # logger
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    config.logger = get_logger(config.outputs_dir, rank_id)

    ms.reset_auto_parallel_context()
    parallel_mode = ms.ParallelMode.STAND_ALONE
    ms.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    config.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)

    if os.path.isfile(config.pretrained):
        load_parameters(network, config.pretrained)
        print("evaling from pretrained parameters", config.pretrained)
    else:
        print("evaling from random parameters")
        # raise FileNotFoundError(f"{config.pretrained} not exists or not a pre-trained file.")

    if config.testing_shape:
        config.test_img_shape = conver_testing_shape(config)

    ds = create_yolo_dataset(config.data_root, config.annFile, is_training=False,
                             batch_size=config.per_batch_size, device_num=1,
                             rank=rank_id, shuffle=False, config=config)
    size = ds.get_dataset_size()
    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('totol %d images to eval', ds.get_dataset_size() * config.per_batch_size)
    ds_iter = ds.create_dict_iterator(num_epochs=1)
    network.set_train(False)

    # init detection engine
    detection = DetectionEngine(config)

    config.logger.info('Start inference....')
    for i, data in tqdm(enumerate(ds_iter), total=size):
        image = data["image"]
        image_shape = data["image_shape"]
        image_id = data["img_id"]
        output_big, output_me, output_small = network(image)
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id = image_id.asnumpy()
        image_shape = image_shape.asnumpy()
        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape, image_id)
        if i % 50 == 0:
            config.logger.info('Processing... {:.2f}% '.format(i / ds.get_dataset_size() * 100))
            # break

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    config.logger.info('result file path: %s', result_file_path)
    eval_result, map = detection.get_eval_result()

    cost_time = time.time() - start_time
    eval_print_str = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_print_str)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)
    print("map:", map)

def eval_yolov3(model_ms, model_torch):
    network, network_torch = set_eval(model_ms, model_torch)
    network_torch.train(False)
    network.set_train(False)


    start_time = time.time()
    config.data_root = os.path.join(config.data_dir, 'val2014')
    config.annFile = os.path.join(config.data_dir, 'annotations/new_instances_val2014_5%.json')

    # logger
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    config.logger = get_logger(config.outputs_dir, rank_id)

    ms.reset_auto_parallel_context()
    parallel_mode = ms.ParallelMode.STAND_ALONE
    ms.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    if os.path.isfile(config.pretrained):
        load_parameters(network, config.pretrained)
        print("evaling from pretrained parameters", config.pretrained)
    else:
        print("evaling from random parameters")
        # raise FileNotFoundError(f"{config.pretrained} not exists or not a pre-trained file.")

    if config.testing_shape:
        config.test_img_shape = conver_testing_shape(config)

    ds = create_yolo_dataset(config.data_root, config.annFile, is_training=False,
                             batch_size=config.per_batch_size, device_num=1,
                             rank=rank_id, shuffle=False, config=config)
    size = ds.get_dataset_size()
    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('totol %d images to eval', ds.get_dataset_size() * config.per_batch_size)
    ds_iter = ds.create_dict_iterator(num_epochs=1)
    network.set_train(False)
    network_torch.eval()

    # init detection engine
    detection = DetectionEngine(config)
    detection_torch = DetectionEngine(config)

    config.logger.info('Start inference....')
    for i, data in tqdm(enumerate(ds_iter), total=size):
        image = data["image"]
        image_shape = data["image_shape"]
        image_id = data["img_id"]
        image_t = torch.tensor(image.asnumpy(), dtype=torch.float32).to(final_device)
        output_big, output_me, output_small = network(image)
        output_big_t, output_me_t, output_small_t = network_torch(image_t)
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        output_big_t = output_big_t.cpu().detach().numpy()
        output_me_t = output_me_t.cpu().detach().numpy()
        output_small_t = output_small_t.cpu().detach().numpy()
        image_id = image_id.asnumpy()
        image_shape = image_shape.asnumpy()
        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape, image_id)
        detection_torch.detect([output_small_t, output_me_t, output_big_t], config.per_batch_size, image_shape,
                               image_id)
        if i % 50 == 0:
            config.logger.info('Processing... {:.2f}% '.format(i / ds.get_dataset_size() * 100))
            # break

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results_ms()
    detection_torch.do_nms_for_results()
    result_file_path = detection.write_result()
    result_file_path_torch = detection_torch.write_result_torch()
    print("result_file_path:", result_file_path)
    print("result_file_path_torch:", result_file_path_torch)
    config.logger.info('result file path: %s', result_file_path)
    eval_result, map = detection.get_eval_result()
    eval_result_torch, map_torch = detection_torch.get_eval_result_torch()
    cost_time = time.time() - start_time
    eval_print_str = '\n=============coco eval result_mindspore=========\n' + eval_result
    eval_print_str_torch = '\n=============coco eval result_torch=========\n' + eval_result_torch
    config.logger.info(eval_print_str)
    config.logger.info(eval_print_str_torch)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)
    print("map_ms:", map)
    print("map_torch:", map_torch)
    return map,map_torch


if __name__ == "__main__":
    device = "cuda:2"
    start_time = time.time()
    config.data_root = os.path.join(config.data_dir, 'val2014')
    config.annFile = os.path.join(config.data_dir, 'annotations/new_instances_val2014_10%.json')

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=devid)

    # logger
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    config.logger = get_logger(config.outputs_dir, rank_id)

    ms.reset_auto_parallel_context()
    parallel_mode = ms.ParallelMode.STAND_ALONE
    ms.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    config.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)
    from main_torch import YOLOV3DarkNet53 as YOLOV3DarkNet53_torch
    network_torch = YOLOV3DarkNet53_torch(is_training=False).to(final_device)

    if os.path.isfile(config.pretrained):
        load_parameters(network, config.pretrained)
        print("evaling from pretrained parameters", config.pretrained)
    else:
        print("evaling from random parameters")
        # raise FileNotFoundError(f"{config.pretrained} not exists or not a pre-trained file.")

    if config.testing_shape:
        config.test_img_shape = conver_testing_shape(config)

    ds = create_yolo_dataset(config.data_root, config.annFile, is_training=False,
                             batch_size=config.per_batch_size, device_num=1,
                             rank=rank_id, shuffle=False, config=config)
    size = ds.get_dataset_size()
    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('totol %d images to eval', ds.get_dataset_size() * config.per_batch_size)
    ds_iter = ds.create_dict_iterator(num_epochs=1)
    network.set_train(False)
    network_torch.eval()

    # init detection engine
    detection = DetectionEngine(config)
    detection_torch = DetectionEngine(config)

    config.logger.info('Start inference....')
    for i, data in tqdm(enumerate(ds_iter), total=size):
        image = data["image"]
        image_shape = data["image_shape"]
        image_id = data["img_id"]
        image_t = torch.tensor(image.asnumpy(), dtype=torch.float32).to(final_device)
        output_big, output_me, output_small = network(image)
        output_big_t, output_me_t, output_small_t = network_torch(image_t)
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        output_big_t = output_big_t.cpu().detach().numpy()
        output_me_t = output_me_t.cpu().detach().numpy()
        output_small_t = output_small_t.cpu().detach().numpy()
        image_id = image_id.asnumpy()
        image_shape = image_shape.asnumpy()
        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape, image_id)
        detection_torch.detect([output_small_t, output_me_t, output_big_t], config.per_batch_size, image_shape, image_id)
        if i % 50 == 0:
            config.logger.info('Processing... {:.2f}% '.format(i / ds.get_dataset_size() * 100))
            # break

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results_ms()
    detection_torch.do_nms_for_results()
    result_file_path = detection.write_result()
    result_file_path_torch = detection_torch.write_result_torch()
    print("result_file_path:", result_file_path)
    print("result_file_path_torch:", result_file_path_torch)
    config.logger.info('result file path: %s', result_file_path)
    eval_result, map = detection.get_eval_result()
    eval_result_torch, map_torch = detection_torch.get_eval_result_torch()
    cost_time = time.time() - start_time
    eval_print_str = '\n=============coco eval result_mindspore=========\n' + eval_result
    eval_print_str_torch = '\n=============coco eval result_torch=========\n' + eval_result_torch
    config.logger.info(eval_print_str)
    config.logger.info(eval_print_str_torch)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)
    print("map_ms:", map)
    print("map_torch:", map_torch)
