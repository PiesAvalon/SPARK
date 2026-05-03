import os
import shutil
import mindspore
import numpy as np
import torch
import troubleshooter as ts
from torch.optim import Adam
from comparer import compare_models
from model_utils.config import config
from src.dataset import Dataloader
from src.eval_utils import apply_eval
from src.network_torch import SSIMLoss, AutoEncoder, NetWithLoss
from src.utils import get_results_torch


def remove_dir(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, True)


def compare_ms_torch():
    device = "CPU"
    network_torch = AutoEncoder(config).to(device)
    network_torch.eval()
    weight_dict = torch.load('ssimae_ascend_v190_mvtecadbottle_official_cv_ok96.8_nok96.8_avg95.2.pth')
    network_torch.load_state_dict(weight_dict, strict=False)
    from src.network import AutoEncoder as AutoEncoder_ms
    network_ms = AutoEncoder_ms(config)
    network_ms.set_train(False)
    mindspore.load_checkpoint("ssimae_align.ckpt", network_ms)
    aa = np.ones([128, 3, 256, 256])
    a = torch.tensor(aa, dtype=torch.float32).to(device)
    anp = (a,)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=network_torch, ms_net=network_ms, fix_seed=0,
                                                  auto_conv_ckpt=0)  #
    diff_finder.compare(inputs=anp)
    # mindspore.save_checkpoint(network_ms, "ssimae_align.ckpt")
    compare_models(network_ms, network_torch, np_data=[aa])


if __name__ == '__main__':
    compare_ms_torch()
    device = "cuda:6"
    cfg = config
    loss = SSIMLoss()
    auto_encoder = AutoEncoder(cfg).to(device)
    net_loss = NetWithLoss(auto_encoder, loss).to(device)
    if os.path.exists(cfg.aug_dir):
        remove_dir(cfg.aug_dir)
    os.makedirs(cfg.aug_dir)
    if os.path.exists(cfg.tmp):
        remove_dir(cfg.tmp)
    os.makedirs(cfg.tmp)
    dataloader = Dataloader()
    train_dataset = dataloader.create_dataset(1, 0)
    cfg.dataset_size = train_dataset.get_dataset_size()
    optimizer = Adam(params=auto_encoder.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    train_dataset = train_dataset.create_tuple_iterator(output_numpy=True)

    epoch_num = 200
    per_batch = 200
    losses_torch_avg1 = []
    for epoch in range(epoch_num):
        nums = 0
        losses_torch = []
        auto_encoder.train()
        for data in train_dataset:
            # print("data[0].shape: ", data[0].shape)
            nums += data[0].shape[0]
            optimizer.zero_grad()
            loss_torch = net_loss(torch.tensor(data[0]).to(device))
            optimizer.step()
            if nums % per_batch == 0:
                print("batch:" + str(nums) + " ms_loss1:" + str(
                    loss_torch.detach().cpu().numpy()))
            losses_torch.append(loss_torch.detach().cpu().numpy())
            # break
        losses_torch_avg1.append(np.mean(losses_torch))
        print("epoch {}: ".format(epoch), " torch_loss1: ",
              str(np.mean(losses_torch)))
        auto_encoder.eval()
        get_results_torch(cfg, auto_encoder)
        print("Generate results at", cfg.save_dir)
        apply_eval(cfg)
