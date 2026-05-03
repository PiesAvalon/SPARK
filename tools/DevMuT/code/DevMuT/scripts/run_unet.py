import sys

sys.path.append(".")
import os
import numpy as np
import argparse
from common.log_recoder import Logger
from copy import deepcopy
from pprint import pformat
import time
import mindspore
import torch
from common.model_utils import get_model
from common.loss_utils import get_loss
from common.opt_utils import get_optimizer
from common.dataset_utils import get_dataset
import psutil
from common.analyzelog_util import train_result_analyze
from network.cv.unet.main import UnetEval, dice_coeff
from network.cv.unet.main_torch import UnetEval_torch
import jax
import jax.numpy as jnp
import optax


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


config_plus = Config({
    # Url for modelarts
    'data_url': "",
    'train_url': "",
    'checkpoint_url': "",
    # Path for local
    'data_path': "ischanllge",
    'output_path': "/cache/train",
    'load_path': "/cache/checkpoint_path/",
    'device_target': "GPU",
    'enable_profiling': False,

    # ==============================================================================
    # Training options
    'model_name': "unet_nested",
    'include_background': True,
    'run_eval': True,
    'run_distribute': False,
    'dataset': "ISBI",
    'crop': "None",
    'image_size': [96, 96],
    'train_augment': False,
    'lr': 0.0003,
    'epochs': 200,
    'repeat': 10,
    'distribute_epochs': 1600,
    'batch_size': 16,
    'distribute_batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 3,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,
    'use_ds': False,
    'use_bn': False,
    'use_deconv': True,
    'resume': False,
    'resume_ckpt': "./",
    'transfer_training': False,
    'filter_weight': ["final1.weight", "final2.weight", "final3.weight", "final4.weight"],
    'show_eval': False,
    'color': [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 255]],

    # Eval options
    'eval_metrics': "dice_coeff",
    'eval_start_epoch': 0,
    'eval_interval': 1,
    'keep_checkpoint_max': 10,
    'eval_activate': "Softmax",
    'eval_resize': False,
    'checkpoint_path': "./checkpoint/",
    'checkpoint_file_path': "ckpt_unet_nested_adam-4-75.ckpt",
    'rst_path': "./result_Files/",
    'result_path': "./preprocess_Result",
    # Export options
    'width': 96,
    'height': 96,
    'file_name': "unetplusplus",
    'file_format': "MINDIR",
})

config_noplus = Config({
    "enable_modelarts": False,
    "data_url": "",
    "train_url": "",
    "checkpoint_url": "",
    "data_path": "/data/pzy/Unet/archive",
    "output_path": "/cache/train",
    "load_path": "/cache/checkpoint_path/",
    "device_target": "GPU",
    "enable_profiling": False,

    "model_name": 'unet_medical',
    "include_background": True,
    "run_eval": False,
    "run_distribute": False,
    "crop": [388, 388],
    "image_size": [572, 572],
    "train_augment": True,
    "lr": 0.0001,
    "epochs": 400,
    "repeat": 400,
    "distribute_epochs": 1600,
    "batch_size": 1,
    "cross_valid_ind": 1,
    "num_classes": 2,
    "num_channels": 1,
    "weight_decay": 0.0005,
    "loss_scale": 1024.0,
    "FixedLossScaleManager": 1024.0,
    "resume": False,
    "resume_ckpt": "./",
    "transfer_training": False,
    "filter_weight": ["outc.weight", "outc.bias"],
    "show_eval": False,
    "color": [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 255]],

    # Eval options
    "eval_metrics": "dice_coeff",
    "eval_start_epoch": 0,
    "eval_interval": 1,
    "keep_checkpoint_max": 10,
    "eval_activate": "Softmax",
    "eval_resize": False,
    "checkpoint_path": "./checkpoint/",
    "checkpoint_file_path": "ckpt_unet_medical_adam-4_75.ckpt",
    "rst_path": "./result_Files/",
    "result_path": "",
    "width": 572,
    "height": 572,
    "file_name": "unet",
    "file_format": "MINDIR",

})


def chebyshev_distance(dict1, dict2):
    distance = np.max(np.abs(dict1 - dict2))
    return distance


def start_unet_train(model_ms, model_torch, yml_train_configs, train_logger, ture_log_path):
    loss_name = yml_train_configs['loss_name']
    learning_rate = yml_train_configs['learning_rate']
    batch_size = yml_train_configs['batch_size']
    per_batch = batch_size * 100
    dataset_name = yml_train_configs['dataset_name']
    optimizer = yml_train_configs['optimizer']
    max_epoch = yml_train_configs['epoch']
    model_name = yml_train_configs['model_name']
    loss_truth, acc_truth, memory_truth = yml_train_configs['loss_ground_truth'], yml_train_configs[
        'eval_ground_truth'], \
        yml_train_configs['memory_threshold']
    process = psutil.Process()
    isplus = model_name == 'unetplus'
    if isplus:
        config_unet = config_plus
    else:
        config_unet = config_noplus
    epoch_num = max_epoch

    config_unet.batch_size = batch_size
    config_unet.data_path = yml_train_configs['dataset_path']

    if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        device = devices[-2]
        final_device = "cuda:" + device
    else:
        final_device = 'cpu'

    loss_ms, loss_torch, loss_jax_fun = get_loss(loss_name)
    loss_ms, loss_torch = loss_ms(), loss_torch()
    loss_torch = loss_torch.to(final_device)

    dataset = get_dataset(dataset_name)
    train_dataset = dataset(data_dir=config_unet.data_path, batch_size=config_unet.batch_size, is_train=True)
    valid_dataset = dataset(data_dir=config_unet.data_path, batch_size=config_unet.batch_size,
                            is_train=False)  # Batch size should be 1 when in evaluation. Otherwise will raise exception

    train_ds = train_dataset.create_dict_iterator(output_numpy=True)
    valid_ds = valid_dataset.create_dict_iterator(output_numpy=True)

    modelms_trainable_params = model_ms.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in modelms_trainable_params:
        modelms_trainable_param.name = model_ms.__class__.__name__ + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1

    optimizer_ms, optimizer_torch, optimizer_jax = get_optimizer(optimizer)
    optimizer_ms = optimizer_ms(params=model_ms.trainable_params(), learning_rate=learning_rate,
                                weight_decay=config_unet.weight_decay)
    optimizer_torch = optimizer_torch(model_torch.parameters(), lr=learning_rate, weight_decay=config_unet.weight_decay)
    optimizer_jax = optimizer_jax(learning_rate)

    params_torch = {key: value.detach().cpu().numpy() for key, value in model_torch.state_dict().items()}
    params_jax = {name: jnp.array(value, dtype=jnp.float32) for name, value in params_torch.items()}
    opt_state = optimizer_jax.init(params_jax)

    from network.cv.unet import mainplus
    from network.cv.unet import mainplus_torch

    testnet1 = mainplus_torch.UnetEval(model_torch, eval_activate=config_unet.eval_activate.lower())
    testnet2 = mainplus.UnetEval(model_ms, eval_activate=config_unet.eval_activate.lower())
    metric1 = mainplus_torch.DiceCoeff(show_eval=False)
    metric2 = mainplus.dice_coeff(show_eval=True)

    def forward_fn(data, label):
        logits = model_ms(data)
        loss = loss_ms(logits, label)
        return loss, logits

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer_ms(grads))
        return loss, grads

    losses_ms_avg, losses_torch_avg, losses_jax_avg = [], [], []
    ms_memorys_avg, torch_memorys_avg, jax_memorys_avg = [], [], []
    ms_times_avg, torch_times_avg, jax_times_avg = [], [], []
    eval_ms, eval_torch = [], []

    for epoch in range(epoch_num):
        train_logger.info('----------------------------')
        train_logger.info(f"epoch: {epoch}/{epoch_num}")
        f = open(ture_log_path, 'a+')
        f.write('----------------------------')
        f.close()
        f = open(ture_log_path, 'a+')
        f.write(f"epoch: {epoch}/{epoch_num}")
        f.close()

        nums = 0
        model_torch.train()
        model_ms.set_train(True)
        model_torch.to(final_device)
        losses_torch, losses_ms, losses_jax = [], [], []
        ms_memorys, torch_memorys, jax_memorys = [], [], []
        ms_times, torch_times, jax_times = [], [], []
        torch_mindsore_distance, ms_jax_distance, jax_troch_distance = [], [], []
        for data in train_ds:
            nums += data['image'].shape[0]
            imgs_array, targets_array = deepcopy(data['image']), deepcopy(data['mask'])
            imgs_torch, targets_torch = torch.tensor(imgs_array, dtype=torch.float32).to(final_device), torch.tensor(
                targets_array, dtype=torch.float32).to(final_device)
            imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mindspore.float32), mindspore.Tensor(targets_array,
                                                                                                    mindspore.float32)

            memory_info = process.memory_info()
            torch_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            torch_time_start = time.time()

            logits_torch = model_torch(imgs_torch)

            loss_torch_result = loss_torch(logits_torch, targets_torch)
            loss_torch_result.backward()
            optimizer_torch.step()

            torch_time_end = time.time()
            torch_time_train = torch_time_end - torch_time_start
            memory_info = process.memory_info()
            torch_memory_train_end = memory_info.rss / 1024 / 1024
            torch_memory_train = torch_memory_train_end - torch_memory_train_start
            optimizer_torch.zero_grad()
            old_torch_state_dict = model_torch.state_dict()
            torch.save(old_torch_state_dict, './model_weights1.pth')

            memory_info = process.memory_info()
            ms_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            ms_time_start = time.time()
            loss_ms_result, grads_ms_result = train_step(imgs_ms, targets_ms)

            ms_times_end = time.time()
            ms_time_train = ms_times_end - ms_time_start
            memory_info = process.memory_info()
            ms_memory_train = memory_info.rss / 1024 / 1024 - ms_memory_train_start

            memory_info = process.memory_info()
            jax_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            jax_time_start = time.time()

            # jaxparams 2 torchparams
            params_jax_numpy = {name: np.array(value) for name, value in params_jax.items()}
            params_torch_updated = {name: torch.from_numpy(value) for name, value in params_jax_numpy.items()}
            model_torch.load_state_dict(params_torch_updated)
            logits_torch = model_torch(imgs_torch)

            jax_out_put = logits_torch.detach().cpu().numpy()
            jax_out_put_targets = targets_torch.detach().cpu().numpy()
            loss_jax, jax_grads = jax.value_and_grad(loss_jax_fun)(params_jax, jax_out_put, jax_out_put_targets)

            updates, opt_state = optimizer_jax.update(jax_grads, opt_state, params_jax)
            params_jax = optax.apply_updates(params_jax, updates)
            jax_time_end = time.time()
            jax_time_train = jax_time_end - jax_time_start
            memory_info = process.memory_info()
            jax_memory_train = memory_info.rss / 1024 / 1024 - jax_memory_train_start

            torch_grads = {key: value.detach().cpu().numpy() for key, value in model_torch.state_dict().items()}
            mindspore_grads = {param.name: grad.asnumpy() for param, grad in zip(model_ms.trainable_params(), grads_ms_result)}

            torch_grads_distance = chebyshev_distance(list(torch_grads.values())[-1],
                                                      list(mindspore_grads.values())[-1])
            mindspore_grads_distance = chebyshev_distance(list(mindspore_grads.values())[-1],
                                                          list(params_jax[list(torch_grads.keys())[-1]])[-1])
            jax_grads_distance = chebyshev_distance(list(torch_grads.values())[-1],
                                                    list(params_jax[list(torch_grads.keys())[-1]])[-1])
            loaded_state_dict = torch.load('./model_weights1.pth')
            model_torch.load_state_dict(loaded_state_dict)

            if nums % per_batch == 0:
                f = open(ture_log_path, 'a+')
                f.write(
                    f"batch: {nums}, \n \
                    torch_loss: {loss_torch_result.item()}, ms_loss: {loss_ms_result.asnumpy()}, jax_loss: {np.array(loss_jax)}, \n \
                    torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, jax_memory:  {jax_memory_train}MB , \n \
                    torch_time: {torch_time_train}, ms_time:  {ms_time_train},jax_time:  {jax_time_train} , \n \
                    torch_mindsore_distance: {torch_grads_distance},  \n  ms_jax_distance:  {mindspore_grads_distance},  \n jax_troch_distance:  {jax_grads_distance} , \n \
                    ")
                f.close()
                train_logger.info(
                    f"batch: {nums}, \n \
                    torch_loss: {loss_torch_result.item()}, ms_loss: {loss_ms_result.asnumpy()}, jax_loss: {np.array(loss_jax)}, \n \
                    torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, jax_memory:  {jax_memory_train}MB , \n \
                    torch_time: {torch_time_train}, ms_time:  {ms_time_train},jax_time:  {jax_time_train} , \n \
                    torch_mindsore_distance: {torch_grads_distance},  \n  ms_jax_distance:  {mindspore_grads_distance},  \n jax_troch_distance:  {jax_grads_distance} , \n \
                    ")
                if nums == 2000:
                    break

            losses_torch.append(loss_torch_result.item())
            losses_ms.append(loss_ms_result.asnumpy())
            ms_memorys.append(ms_memory_train)
            torch_memorys.append(torch_memory_train)
            ms_times.append(ms_time_train)
            torch_times.append(torch_time_train)
            losses_jax.append(np.array(loss_jax))
            jax_memorys.append(jax_memory_train)
            jax_times.append(jax_time_train)
            torch_mindsore_distance.append(torch_grads_distance)
            ms_jax_distance.append(mindspore_grads_distance)
            jax_troch_distance.append(jax_grads_distance)
        losses_ms_avg.append(np.mean(losses_ms))
        losses_torch_avg.append(np.mean(losses_torch))
        losses_jax_avg.append(np.mean(loss_jax))
        ms_memorys_avg.append(np.mean(ms_memorys))
        torch_memorys_avg.append(np.mean(torch_memorys))
        jax_memorys_avg.append(np.mean(jax_memorys))
        ms_times_avg.append(np.mean(ms_times))
        torch_times_avg.append(np.mean(torch_times))
        jax_times_avg.append(np.mean(jax_times))
        f = open(ture_log_path, 'a+')
        f.write(
            f"epoch: {epoch}, \n,\
                  torch_loss_avg: {np.mean(losses_torch)}, ms_loss_avg: {np.mean(losses_ms)}, jax_loss_avg: {np.mean(losses_jax)}, \n,\
                      torch_memory_avg: {np.mean(torch_memorys)}MB, ms_memory_avg:  {np.mean(ms_memorys)}MB, jax_memory_avg:  {np.mean(jax_memorys)}MB, \n,\
                          torch_time_avg: {np.mean(torch_times)}, ms_time_avg:  {np.mean(ms_times)}, jax_time_avg:  {np.mean(jax_times)}, \n,\
                            torch_mindsore_distance_avg: {np.mean(torch_mindsore_distance)},  \n  ms_jax_distance_avg:  {np.mean(ms_jax_distance)},  \n jax_troch_distance_avg:  {np.mean(jax_troch_distance)} , \n \
                            ")
        f.close()
        train_logger.info(
            f"epoch: {epoch}, \n, torch_loss_avg: {np.mean(losses_torch)}, ms_loss_avg: {np.mean(losses_ms)}, jax_loss_avg: {np.mean(losses_jax)}, \n, torch_memory_avg: {np.mean(torch_memory_train)}MB, ms_memory_avg:  {np.mean(ms_memory_train)}MB, jax_memory_avg:  {np.mean(jax_memory_train)}MB, \n, torch_time_avg: {np.mean(torch_time_train)}, ms_time_avg:  {np.mean(ms_time_train)}, jax_time_avg:  {np.mean(jax_time_train)}")

        # start eval stage
    #     metric1 = dice_coeff()
    #     metric2 = dice_coeff()

    #     testnet1 = UnetEval(model_ms, eval_activate="Softmax".lower())
    #     testnet2 = UnetEval_torch(model_torch, eval_activate="Softmax".lower())

    #     metric1.clear()
    #     metric2.clear()
    #     for tdata in valid_ds:
    #         inputs, labels = tdata['image'], tdata['mask']
    #         inputs_ms, labels_ms = mindspore.Tensor(inputs, mindspore.float32), mindspore.Tensor(labels,
    #                                                                                              mindspore.int32)  # Send tensors to the appropriate device (CPU or GPU)
    #         inputs_torch, labels_torch = torch.tensor(inputs, dtype=torch.float32).to(final_device), torch.tensor(
    #             labels, dtype=torch.float32).to(final_device)

    #         logits1 = testnet1(inputs_ms)
    #         logits2 = testnet2(inputs_torch)
    #         logits2 = mindspore.Tensor(logits2.detach().cpu().numpy(), mindspore.float32)

    #         metric1.update(logits1, labels)
    #         metric2.update(logits2, labels)

    #     dice1 = metric1.eval()
    #     dice2 = metric2.eval()
    #     train_logger.info(f"MindSpore_dice: {dice1}" + " PyTorch_dice: {}".format(dice2))
    #     eval_torch.append(dice1[0])
    #     eval_ms.append(dice2[0])

    # train_logger.generation = yml_train_configs['generation']
    # analyze_util = train_result_analyze(model_name=model_name, epochs=max_epoch, loss_ms=losses_ms_avg,
    #                                     loss_torch=losses_torch_avg, eval_ms=eval_ms, eval_torch=eval_torch,
    #                                     memories_ms=ms_memorys, memories_torch=torch_memorys, loss_truth=loss_truth,
    #                                     acc_truth=acc_truth, memory_truth=memory_truth, train_logger=train_logger)
    # analyze_util.analyze_main()


if __name__ == '__main__':
    """
    export CONTEXT_DEVICE_TARGET=GPU
    export CUDA_VISIBLE_DEVICES=2,3
    """

    train_configs = {
        "model_name": "unet",
        'dataset_name': "ischanllge",
        'batch_size': 1,
        'input_size': (1, 1, 572, 572),
        'test_size': 1,
        'dtypes': ['float'],
        'epoch': 5,
        'loss_name': "unetloss",
        'optimizer': "SGD",
        'learning_rate': 0.02,
        'loss_ground_truth': 2.950969386100769,
        'eval_ground_truth': 0.998740881321355,
        'memory_threshold': 0.01,
        'device_target': 'GPU',
        'device_id': 0,
        "dataset_path": "/data1/pzy/raw/ischanllge"

    }

    data = [np.ones(train_configs["input_size"])]
    ms_dtypes = [mindspore.float32]
    torch_dtypes = [torch.float32]
    model_ms_origin, model_torch_origin = get_model(train_configs["model_name"], train_configs["input_size"],
                                                    only_ms=False, scaned=True)

    log_path = '/data1/myz/empirical_exp/common/log/E3/rq2result/patchcore-2023.9.20.8.43.32/mutation.txt'

    model_name = train_configs['model_name']
    logger = Logger(log_file='./log/debug.log')
    args_opt = argparse.Namespace(
        model=model_name,
        dataset_path=r'/data1/pzy/mindb/ssd/datamind/ssd.mindrecord0',
        batch_size=5,
        epoch=200,
        mutation_iterations=5,
        selected_model_num=1,
        mutation_type=["WS", "NS", "NAI", "NEB", "GF"],
        # "LA","LD","WS","NS","NAI","NEB","GF","LC","LS","RA","WS", "NS", "NAI", "NEB", "GF"
        mutation_log='/data1/myz/netsv/common/log',
        selected_generation=None,
        mutation_strategy="random"
    )
    model_ms, model_torch, train_configs = model_ms_origin, model_torch_origin, train_configs

    start_unet_train(model_ms, model_torch, train_configs, logger.logger)
