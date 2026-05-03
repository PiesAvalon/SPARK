import sys

# sys.path.append("./")
sys.path.append("../")
import argparse
import os
from copy import deepcopy
import numpy as np
import psutil
import torch
import mindspore
from mindspore.common import dtype as mstype
from common.dataset_utils import get_dataset
from common.loss_utils import get_loss
from common.opt_utils import get_optimizer
from common.analyzelog_util import train_result_analyze
import time
from common.log_recoder import Logger
from common.model_utils import get_model
import jax
import jax.numpy as jnp
import optax
import copy
# os.environ['CONTEXT_DEVICE_TARGET'] = 'GPU'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def chebyshev_distance(dict1, dict2):
    # max_distance = []
    # if isinstance(dict1, tuple):
    #     dict1 = dict1[0]
    # if isinstance(dict2, tuple):
    #     dict2 = dict2[0]
    # # print(dict1)
    # for key in dict1.keys():
    #     value1, value2 = dict1[key], dict2[key]
    distance = np.max(np.abs(dict1 - dict2))
    #     max_distance.append(distance)
    # str_list = [str(num) for num in max_distance]
    # result_string = ','.join(str_list)
    return distance


def start_imageclassification_train(model_ms, model_torch, train_configs, train_logger, ture_log_path):
    loss_name = train_configs['loss_name']
    learning_rate = train_configs['learning_rate']
    batch_size = train_configs['batch_size']
    per_batch = batch_size * 100
    dataset_name = train_configs['dataset_name']
    optimizer = train_configs['optimizer']
    epochs = train_configs['epoch']
    model_name = train_configs['model_name']
    loss_truth, acc_truth, memory_truth = train_configs['loss_ground_truth'], train_configs['eval_ground_truth'], \
        train_configs['memory_threshold']
    
    process = psutil.Process()

    if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        device = devices[-2]
        final_device = "cuda:" + device
    else:
        final_device = 'cpu'

    loss_fun_ms, loss_fun_torch = get_loss(loss_name)
    loss_fun_ms, loss_fun_torch = loss_fun_ms(), loss_fun_torch()
    loss_fun_torch = loss_fun_torch.to(final_device)

    optimizer_ms, optimizer_torch= get_optimizer(optimizer)
    optimizer_torch = optimizer_torch(model_torch.parameters(), lr=learning_rate)
    optimizer_jax = optax.sgd(learning_rate)


    # params = model_torch.parameters()
    # params_torch = {name: param.detach().cpu().numpy() for name, param in model_torch.named_parameters()}
    # params_jax = {name: jnp.array(value) for name, value in params_torch.items()}
    params_torch = {key: value.detach().cpu().numpy() for key, value in model_torch.state_dict().items()}
    params_jax = {name: jnp.array(value, dtype=jnp.float32) for name, value in params_torch.items()}
    opt_state = optimizer_jax.init(params_jax)

    # params 2 jax

    modelms_trainable_params = model_ms.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in modelms_trainable_params:
        modelms_trainable_param.name = model_ms.__class__.__name__ + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    optimizer_ms = optimizer_ms(params=new_trainable_params, learning_rate=learning_rate, momentum=0.9,
                                weight_decay=0.0001)

    # old_params
    # old_torch_grads = {key: value.detach().cpu().numpy() for key, value in model_torch.state_dict().items()}

    # old_jax_grads = params_jax

    dataset = get_dataset(dataset_name)
    train_set = dataset(data_dir=train_configs['dataset_path'], batch_size=batch_size, is_train=True)
    test_set = dataset(data_dir=train_configs['dataset_path'], batch_size=batch_size, is_train=False)

    train_iter = train_set.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    test_iter = test_set.create_dict_iterator(output_numpy=True, num_epochs=epochs)

    def forward_fn(data, label):
        logits = model_ms(data)
        loss = loss_fun_ms(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer_ms(grads))
        return loss, grads
    
    def loss_fn(params_, output, label):
        loss = optax.softmax_cross_entropy(output, jax.nn.one_hot(label, 10)).mean()
        return loss

    losses_ms_avg, losses_torch_avg, losses_jax_avg = [], [], []
    ms_memorys_avg, torch_memorys_avg, jax_memorys_avg = [], [], []
    ms_times_avg, torch_times_avg, jax_times_avg = [], [], []
    eval_ms, eval_torch = [], []

    for epoch in range(epochs):
        train_logger.info('----------------------------')
        train_logger.info(f"epoch: {epoch}/{epochs}")
        f = open(ture_log_path, 'a+')
        f.write('----------------------------')
        f.close()
        f = open(ture_log_path, 'a+')
        f.write(f"epoch: {epoch}/{epochs}")
        f.close()
        model_torch.train()
        model_ms.set_train(True)

        losses_torch, losses_ms, losses_jax = [], [], []
        ms_memorys, torch_memorys, jax_memorys = [], [], []
        ms_times, torch_times, jax_times = [], [], []
        torch_mindsore_distance, ms_jax_distance, jax_troch_distance = [], [], []
        batch = 0
        nums = 0
        index = 1
        index1 = 0
        for item in train_iter:
            nums += item['image'].shape[0]
            imgs_array, targets_array = deepcopy(item['image']), deepcopy(item['label'])
            imgs_torch, targets_torch = torch.tensor(imgs_array, dtype=torch.float32).to(final_device), torch.tensor(
                targets_array, dtype=torch.long).to(final_device)
            imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mstype.float32), mindspore.Tensor(targets_array,
                                                                                                 mstype.int32)
            # if index1 == 0:

            #     loss_ms, grads = train_step(imgs_ms, targets_ms)
            #     old_mindspore_grads = {param.name: grad.asnumpy() for param, grad in zip(model_ms.trainable_params(), grads)}

            #     outputs_torch_tensor = model_torch(imgs_torch)
            #     jax_out_put  = outputs_torch_tensor.detach().cpu().numpy()
            #     jax_out_put_targets  = targets_torch.detach().cpu().numpy()
            #     loss_jax, old_jax_grads = jax.value_and_grad(loss_fn)(params_jax, jax_out_put, jax_out_put_targets)

            #     index1 += 1
            # torch
            memory_info = process.memory_info()
            torch_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            torch_time_start = time.time()

            outputs_torch_tensor = model_torch(imgs_torch)
            loss_torch = loss_fun_torch(outputs_torch_tensor, targets_torch)
            loss_torch.backward()
            optimizer_torch.step()

            torch_time_end = time.time()
            torch_time_train = torch_time_end - torch_time_start

            memory_info = process.memory_info()
            torch_memory_train_end = memory_info.rss / 1024 / 1024
            torch_memory_train = torch_memory_train_end - torch_memory_train_start
            optimizer_torch.zero_grad()
            old_torch_state_dict = model_torch.state_dict()
            
            # torch_grads_distance = chebyshev_distance(old_torch_grads, torch_grads)
            # old_torch_grads = torch_grads
            torch.save(old_torch_state_dict, './model_weights.pth')

            # mindspore
            memory_info = process.memory_info()
            ms_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            ms_time_start = time.time()

            loss_ms, grads = train_step(imgs_ms, targets_ms)
            ms_time_end = time.time()

            ms_time_train = ms_time_end - ms_time_start
            memory_info = process.memory_info()
            ms_memory_train = memory_info.rss / 1024 / 1024 - ms_memory_train_start

            mindspore_grads = {param.name: grad.asnumpy() for param, grad in zip(model_ms.trainable_params(), grads)}
            # mindspore_grads_distance = chebyshev_distance(old_mindspore_grads, mindspore_grads)
            # old_mindspore_grads = mindspore_grads
            # jax
            memory_info = process.memory_info()
            jax_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            jax_time_start = time.time()

            # jaxparams 2 torchparams
            params_jax_numpy = {name: np.array(value) for name, value in params_jax.items()}
            params_torch_updated = {name: torch.from_numpy(value) for name, value in params_jax_numpy.items()}
            # for key,value in params_jax.items():
            #     print(key,value)
            #     print(old_torch_state_dict[key])
            #     break
            # for name, param in model_torch.named_parameters():
            #     param.copy_(params_torch_updated[name])
            # quit(66)
            model_torch.load_state_dict(params_torch_updated)
            outputs_torch_tensor = model_torch(imgs_torch)

            jax_out_put  = outputs_torch_tensor.detach().cpu().numpy()
            jax_out_put_targets  = targets_torch.detach().cpu().numpy()
            loss_jax, jax_grads = jax.value_and_grad(loss_fn)(params_jax, jax_out_put, jax_out_put_targets)

            updates, opt_state = optimizer_jax.update(jax_grads, opt_state, params_jax)
            params_jax = optax.apply_updates(params_jax, updates)
            jax_time_end = time.time()
            jax_time_train = jax_time_end - jax_time_start
            memory_info = process.memory_info()
            jax_memory_train = memory_info.rss / 1024 / 1024 - jax_memory_train_start

            # jax_grads_distance = chebyshev_distance(old_jax_grads, jax_grads)
            # old_jax_grads = jax_grads
            torch_grads = {key: value.detach().cpu().numpy() for key, value in model_torch.state_dict().items()}
            mindspore_grads = {param.name: grad.asnumpy() for param, grad in zip(model_ms.trainable_params(), grads)}


            torch_grads_distance = chebyshev_distance(list(torch_grads.values())[-1], list(mindspore_grads.values())[-1])
            mindspore_grads_distance = chebyshev_distance(list(mindspore_grads.values())[-1], list(params_jax[list(torch_grads.keys())[-1]])[-1])
            jax_grads_distance = chebyshev_distance(list(torch_grads.values())[-1], list(params_jax[list(torch_grads.keys())[-1]])[-1])
            # model_torch.load_state_dict(old_torch_state_dict)
            loaded_state_dict = torch.load('./model_weights.pth')
            model_torch.load_state_dict(loaded_state_dict)

            if batch % per_batch == 0:
                # folder_path = '/data1/ypr/net-sv/output_model/resnet50'
                # os.makedirs(folder_path, exist_ok=True)
                # os.makedirs(folder_path+'/pytorch_model', exist_ok=True)
                # os.makedirs(folder_path+'/mindspore_model', exist_ok=True)
                # torch.save(model_torch.state_dict(),
                #            folder_path + '/pytorch_model/pytorch_model_' + str(epoch) + '_' + str(
                #                batch // per_batch) + '.pth')
                # mindspore.save_checkpoint(model_ms,
                #                           folder_path + '/mindspore_model/mindspore_model' + str(epoch) + '_' + str(
                #                               batch // per_batch) + '.ckpt')
                f = open(ture_log_path, 'a+')
                f.write(
                    f"batch: {batch}, \n \
                    torch_loss: {loss_torch.item()}, ms_loss: {loss_ms.asnumpy()}, jax_loss: {np.array(loss_jax)}, \n \
                    torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, jax_memory:  {jax_memory_train}MB , \n \
                    torch_time: {torch_time_train}, ms_time:  {ms_time_train}, jax_time:  {jax_time_train} , \n \
                    torch_mindsore_distance: {torch_grads_distance},  \n  ms_jax_distance:  {mindspore_grads_distance},  \n jax_troch_distance:  {jax_grads_distance} , \n \
                    ")
                f.close()
                train_logger.info(
                    f"batch: {batch}, \n \
                    torch_loss: {loss_torch.item()}, ms_loss: {loss_ms.asnumpy()}, jax_loss: {np.array(loss_jax)}, \n \
                    torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, jax_memory:  {jax_memory_train}MB , \n \
                    torch_time: {torch_time_train}, ms_time:  {ms_time_train}, jax_time:  {jax_time_train} , \n \
                    torch_mindsore_distance: {torch_grads_distance},  \n  ms_jax_distance:  {mindspore_grads_distance},  \n jax_troch_distance:  {jax_grads_distance} , \n \
                    ")
                # index+=1
                # if index >= 3:
                #     quit(66666)
                if batch == 5000:
                    break

            losses_torch.append(loss_torch.item())
            losses_ms.append(loss_ms.asnumpy())
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
            batch += batch_size

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
                                torch_ms_memory: {cosine_similarity([torch_memorys,ms_memorys])}, torch_jax_memory: {cosine_similarity([torch_memorys,jax_memorys])}, ms_jax_memory:  {cosine_similarity([ms_memorys,jax_memorys])}, \n \
                            ")
        f.close()
        train_logger.info(
            f"epoch: {epoch}, \n, \
                torch_loss_avg: {np.mean(losses_torch)}, ms_loss_avg: {np.mean(losses_ms)}, jax_loss_avg: {np.mean(losses_jax)}, \n,\
                      torch_memory_avg: {np.mean(torch_memory_train)}MB, ms_memory_avg:  {np.mean(ms_memory_train)}MB, jax_memory_avg:  {np.mean(jax_memory_train)}MB, \n,\
                          torch_time_avg: {np.mean(torch_time_train)}, ms_time_avg:  {np.mean(ms_time_train)}, jax_time_avg:  {np.mean(jax_time_train)}, \n,\
                            torch_mindsore_distance_avg: {np.mean(torch_mindsore_distance)},  \n  ms_jax_distance_avg:  {np.mean(ms_jax_distance)},  \n jax_troch_distance_avg:  {np.mean(jax_troch_distance)} , \n \
                                torch_ms_memory: {cosine_similarity([torch_memorys,ms_memorys])}, torch_jax_memory: {cosine_similarity([torch_memorys,jax_memorys])}, ms_jax_memory:  {cosine_similarity([ms_memorys,jax_memorys])}, \n \
                            ")

        # 测试步骤开始
        model_torch.eval()
        model_ms.set_train(False)
        test_data_size = 0
        total_accuracy = 0
        correct_ms = 0

        with torch.no_grad():
            for item in test_iter:
                nums += item['image'].shape[0]

                imgs_array, targets_array = deepcopy(item['image']), deepcopy(item['label'])
                imgs_torch, targets_torch = torch.tensor(imgs_array), torch.tensor(targets_array, dtype=torch.long)
                imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mstype.float32), mindspore.Tensor(targets_array,
                                                                                                     mstype.int32)

                test_data_size += len(imgs_ms)

                imgs_torch = imgs_torch.to(final_device)
                targets_torch = targets_torch.to(final_device)
                outputs_torch_tensor = model_torch(imgs_torch)
                outputs_torch_array = outputs_torch_tensor.cpu().numpy()
                targets_torch_array = targets_torch.cpu().numpy()

                accuracy = (outputs_torch_array.argmax(1) == targets_torch_array).sum()
                total_accuracy = total_accuracy + accuracy

                pred_ms = model_ms(imgs_ms)
                correct_ms += (pred_ms.argmax(1) == targets_ms).asnumpy().sum()

        correct_ms /= test_data_size
        f = open(ture_log_path, 'a+')
        f.write(
            f"Mindspore Test Accuracy: {(100 * correct_ms)}%" + " Pytorch Test Accuracy: {}%".format(
            100 * total_accuracy / test_data_size))
        f.close()
        train_logger.info(f"Mindspore Test Accuracy: {(100 * correct_ms)}%" + " Pytorch Test Accuracy: {}%".format(
            100 * total_accuracy / test_data_size))

        eval_torch.append(total_accuracy / test_data_size)
        eval_ms.append(correct_ms)

    train_logger.generation = train_configs['generation']
    analyze_util = train_result_analyze(model_name=model_name, epochs=epochs, loss_ms=losses_ms_avg,
                                        loss_torch=losses_torch_avg, eval_ms=eval_ms, eval_torch=eval_torch,
                                        memories_ms=ms_memorys, memories_torch=torch_memorys, loss_truth=loss_truth,
                                        acc_truth=acc_truth, memory_truth=memory_truth, train_logger=train_logger)
    analyze_util.analyze_main()


if __name__ == '__main__':
    """
    export CONTEXT_DEVICE_TARGET=GPU
    export CUDA_VISIBLE_DEVICES=0,1
    """

    train_configs = {
        "model_name": "resnet50",
        'dataset_name': "cifar10",
        'batch_size': 5,
        'input_size': (2, 3, 224, 224),
        'test_size': 2,
        'dtypes': ['float'],
        'epoch': 5,
        'loss_name': "CrossEntropy",
        'optimizer': "SGD",
        'learning_rate': 0.02,
        'loss_ground_truth': 2.950969386100769,
        'eval_ground_truth': 0.998740881321355,
        'memory_threshold': 0.01,
        'device_target': 'GPU',
        'device_id': 0,
        "dataset_path": "/data1/pzy/raw/cifar10"

    }

    data = [np.ones(train_configs["input_size"])]
    ms_dtypes = [mindspore.float32]
    torch_dtypes = [torch.float32]
    model_ms_origin, model_torch_origin = get_model(train_configs["model_name"], train_configs["input_size"],
                                                    only_ms=False, scaned=True)

    log_path = '/data1/myz/empirical_exp/common/log/E3/rq2result/patchcore-2023.9.20.8.43.32/mutation.txt'

    model_name = train_configs['model_name']
    logger = Logger(log_file='/data1/czx/net-sv/log/resnet50.log')
    args_opt = argparse.Namespace(
        model=model_name,
        dataset_path=r'/data1/pzy/mindb/ssd/datamind/ssd.mindrecord0',
        batch_size=5,
        epoch=5,
        mutation_iterations=100,
        selected_model_num=1,
        mutation_type=["WS", "NS", "NAI", "NEB", "GF"],
        # "LA","LD","WS","NS","NAI","NEB","GF","LC","LS","RA","WS", "NS", "NAI", "NEB", "GF"
        mutation_log='/data1/myz/netsv/common/log',
        selected_generation=None,
        mutation_strategy="random"
    )
    model_ms, model_torch, train_configs = model_ms_origin, model_torch_origin, train_configs

    start_imageclassification_train(model_ms, model_torch, train_configs, logger.logger)
