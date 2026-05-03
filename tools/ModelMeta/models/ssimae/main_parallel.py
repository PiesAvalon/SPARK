import os
import shutil
import mindspore
import numpy as np
import torch
from mindspore import nn, load_checkpoint
from model_utils.config import config
from src.dataset import Dataloader
from src.eval_utils import apply_eval
from src.network import SSIMLoss, AutoEncoder, NetWithLoss
from src.utils import get_results, get_results_torch


def remove_dir(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, True)






if __name__ == '__main__':
    cfg = config
    loss = SSIMLoss()
    from src.network_torch import SSIMLoss as SSIMLoss_torch

    loss_torch = SSIMLoss_torch()
    device = "cuda:6"
    auto_encoder = AutoEncoder(cfg)
    from src.network_torch import AutoEncoder as AutoEncoder_torch
    from src.network_torch import NetWithLoss as NetWithLoss_torch

    auto_encoder_torch = AutoEncoder_torch(config)
    net_loss = NetWithLoss(auto_encoder, loss)
    net_loss_torch = NetWithLoss_torch(auto_encoder_torch, loss_torch).to(device)
    if os.path.exists(cfg.aug_dir):
        remove_dir(cfg.aug_dir)
    os.makedirs(cfg.aug_dir)
    if os.path.exists(cfg.tmp):
        remove_dir(cfg.tmp)
    os.makedirs(cfg.tmp)
    dataloader = Dataloader()
    train_dataset = dataloader.create_dataset(1, 0)
    cfg.dataset_size = train_dataset.get_dataset_size()
    optimizer = nn.AdamWeightDecay(params=auto_encoder.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.decay)
    optimizer_torch = torch.optim.Adam(params=auto_encoder_torch.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    train_dataset = train_dataset.create_tuple_iterator()


    def forward_fn(data):
        loss = net_loss(data)
        return loss


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)


    def train_step(data):
        (loss), grads = grad_fn(data)
        loss = mindspore.ops.depend(loss, optimizer(grads))
        return loss


    epoch_num = 1
    per_batch = 200
    losses_ms_avg1 = []
    losses_torch_avg1 = []
    current_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = os.path.join(current_path, "ssimae_ascend_v190_mvtecadbottle_official_cv_ok96.8_nok96.8_avg95.2.ckpt")
    load_checkpoint(ckpt_path, net=auto_encoder)
    weight_dict = torch.load("ssimae_ascend_v190_mvtecadbottle_official_cv_ok96.8_nok96.8_avg95.2.pth")
    auto_encoder_torch.load_state_dict(weight_dict)
    for epoch in range(epoch_num):
        nums = 0
        losses_ms = []
        losses_torch = []
        # auto_encoder.set_train(True)
        # auto_encoder_torch.train()
        # for data in train_dataset:
        #     nums += data[0].shape[0]
        #     loss_ms = train_step(data[0])
        #     optimizer_torch.zero_grad()
        #     loss_torch = net_loss_torch(torch.tensor(data[0].asnumpy()).to(device))
        #     optimizer_torch.step()
        #     if nums % per_batch == 0:
        #         print("batch:" + str(nums) + " ms_loss1:" + str(
        #             loss_ms.asnumpy()) + " torch_loss1:" + str(
        #             loss_torch.detach().cpu().numpy()))
        #     losses_ms.append(loss_ms.asnumpy())
        #     losses_torch.append(loss_torch.detach().cpu().numpy())
        #     # break
        # losses_ms_avg1.append(np.mean(losses_ms))
        # losses_torch_avg1.append(np.mean(losses_torch))
        # print("epoch {}: ".format(epoch), " ms_loss1: ",
        #       str(np.mean(losses_ms)), " torch_loss1: ",
        #       str(np.mean(losses_torch)))
        auto_encoder.set_train(False)
        auto_encoder_torch.eval()
        get_results(cfg, auto_encoder)
        print("Generate ms results at", cfg.save_dir)
        apply_eval(cfg)
        get_results_torch(cfg, auto_encoder_torch)
        print("Generate torch results at", cfg.save_dir)
        apply_eval(cfg)
