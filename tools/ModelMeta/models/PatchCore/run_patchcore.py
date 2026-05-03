# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train"""
import json
import sys
import datetime
import os
import time
from pathlib import Path
import cv2
import faiss
import mindspore
import numpy as np
import torch
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection

from model_ms import wide_resnet50_2
from src.dataset import createDataset
from src.oneStep import OneStepCell
from train_torch import OneStepCell as OneStepCell_torch
from src.operator import embedding_concat, prep_dirs, reshape_embedding, normalize, save_anomaly_map
from src.sampling_methods.kcenter_greedy import kCenterGreedy
from src.config import cfg, merge_from_cli_list

set_seed(1)
opts = sys.argv[1:]
merge_from_cli_list(opts)
cfg.freeze()
print(cfg)


def eval_pathcore(model_ms, model_torch, train_configs):
    device = train_configs['device']
    device_id = train_configs['device_id']
    batch_size = train_configs['batch_size']
    dataset_name = train_configs['dataset_name']

    if device == "CPU":
        final_device = "CPU"
    else:
        final_device = "cuda:" + str(device_id)
    current_path = os.path.abspath(os.path.dirname(__file__))

    # dataset = get_dataset(dataset_name)
    train_dataset, test_dataset, _, test_json_path = createDataset(cfg.dataset_path, cfg.category)
    # test_dataset, test_json_path = dataset(batch_size=batch_size, is_train=False)
    test_iter = test_dataset.create_dict_iterator(output_numpy=True)

    embedding_dir_path, _ = prep_dirs(current_path, cfg.category)

    # network
    model = OneStepCell(model_ms)
    model_torch = OneStepCell_torch(model_torch)

    mean = cfg.mean
    std = cfg.std
    json_path = Path(test_json_path)
    with json_path.open("r") as label_file:
        label = json.load(label_file)

    embedding_dir_path, sample_path = prep_dirs(current_path, cfg.category)

    gt_list_px_lvl_ms_torch = []
    pred_list_px_lvl_ms = []
    pred_list_px_lvl_torch = []
    gt_list_img_lvl_ms_torch = []
    pred_list_img_lvl_ms = []
    pred_list_img_lvl_torch = []
    img_path_list = []
    index_ms = faiss.read_index(os.path.join(embedding_dir_path, "index_ms.faiss"))
    index_torch = faiss.read_index(os.path.join(embedding_dir_path, "index_torch.faiss"))

    for step, data in enumerate(test_iter):
        step_label = label["{}".format(mindspore.Tensor(data["idx"])[0], mindspore.float32)]
        file_name = step_label["name"]
        x_type = step_label["img_type"]
        features_m = model(mindspore.Tensor(data["img"], mindspore.float32))
        features_t = model_torch(torch.tensor(data["img"], dtype=torch.float32).to(final_device))
        embedding = embedding_concat(features_m[0].asnumpy(), features_m[1].asnumpy())
        embedding_t = embedding_concat(features_t[0].detach().cpu().numpy(),
                                       features_t[1].detach().cpu().numpy())
        embedding_test_ms = reshape_embedding(embedding)
        embedding_test_torch = reshape_embedding(embedding_t)

        embedding_test_ms = np.array(embedding_test_ms, dtype=np.float32)
        embedding_test_torch = np.array(embedding_test_torch, dtype=np.float32)
        score_patches_ms, _ = index_ms.search(embedding_test_ms, k=9)
        score_patches_torch, _ = index_torch.search(embedding_test_torch, k=9)

        anomaly_map_ms = score_patches_ms[:, 0].reshape((28, 28))
        anomaly_map_torch = score_patches_torch[:, 0].reshape((28, 28))
        N_b = score_patches_ms[np.argmax(score_patches_ms[:, 0])]
        N_b_torch = score_patches_torch[np.argmax(score_patches_torch[:, 0])]
        w = 1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b)))
        w_torch = 1 - (np.max(np.exp(N_b_torch)) / np.sum(np.exp(N_b_torch)))
        score_ms = w * max(score_patches_ms[:, 0])
        score_torch = w_torch * max(score_patches_torch[:, 0])
        gt_np = data["gt"][0, 0].astype(int)
        anomaly_map_resized_ms = cv2.resize(anomaly_map_ms, (224, 224))
        anomaly_map_resized_torch = cv2.resize(anomaly_map_torch, (224, 224))
        anomaly_map_resized_blur_ms = gaussian_filter(anomaly_map_resized_ms, sigma=4)
        anomaly_map_resized_blur_torch = gaussian_filter(anomaly_map_resized_torch, sigma=4)

        gt_list_px_lvl_ms_torch.extend(gt_np.ravel())
        pred_list_px_lvl_ms.extend(anomaly_map_resized_blur_ms.ravel())
        pred_list_px_lvl_torch.extend(anomaly_map_resized_blur_torch.ravel())
        gt_list_img_lvl_ms_torch.append(data["label"][0])
        pred_list_img_lvl_ms.append(score_ms)
        pred_list_img_lvl_torch.append(score_torch)
        img_path_list.extend(file_name)
        img = normalize(mindspore.Tensor(data["img"]), mean, std)
        input_img = cv2.cvtColor(np.transpose(img, (0, 2, 3, 1))[0] * 255, cv2.COLOR_BGR2RGB)
        save_anomaly_map(sample_path, anomaly_map_resized_blur_ms, input_img, gt_np * 255, file_name, x_type)

    pixel_auc_ms = roc_auc_score(gt_list_px_lvl_ms_torch, pred_list_px_lvl_ms)
    pixel_auc_torch = roc_auc_score(gt_list_px_lvl_ms_torch, pred_list_px_lvl_torch)
    print(np.max(np.abs(np.array(pred_list_px_lvl_ms) - np.array(pred_list_px_lvl_torch))))
    img_auc_ms = roc_auc_score(gt_list_img_lvl_ms_torch, pred_list_img_lvl_ms)
    img_auc_torch = roc_auc_score(gt_list_img_lvl_ms_torch, pred_list_img_lvl_torch)

    print("\ntest_epoch_end")
    print("category is {}".format(cfg.category))
    print("Mindspore img_auc: {}, pixel_auc: {}".format(img_auc_ms, pixel_auc_ms))
    print("PyTorch img_auc: {}, pixel_auc: {}".format(img_auc_torch, pixel_auc_torch))


if __name__ == '__main__':
    train_configs = {'dataset_name': 'patchcoredataset',
                     'batch_size': 5,
                     'input_size': ['(2,3,224,224)'],
                     'test_size': 2,
                     'dtypes': ['float'],
                     'epoch': 100,
                     'model_name': 'openpose',
                     'loss_name': 'bertloss',
                     'device': 'GPU',
                     'device_id': 0,
                     'optimizer': 'adam',
                     'learning_rate': 0.005,
                     'loss_ground_truth': 2.950969386100769,
                     'eval_ground_truth': 0.998740881321355,
                     'memory_threshold': '1e-2',
                     'dataset_path': '/data1/pzy/raw/MVTecAD'}

    model_name = "patchcore"
    device_target = "CPU"
    device_id = 0
    input_size = (2, 3, 224, 224)

    if device_target == "CPU":
        final_device = "cuda:" + str(device_id)

    network = wide_resnet50_2()
    mindspore.load_checkpoint("wide_resnet50_2-95faca4d.ckpt", network)
    for p in network.trainable_params():
        p.requires_grad = False
    model = OneStepCell(network)
    from model_torch import wide_resnet50_2 as wide_resnet50_2_torch

    network_torch = wide_resnet50_2_torch().to(final_device)
    param_dict = torch.load("wide_resnet50_2-95faca4d.pth")
    network_torch.load_state_dict(param_dict)
    print("ckpts successfully loaded")
    for p in network_torch.parameters():
        p.requires_grad = False

    # model_ms_origin1, model_torch_origin1 = get_model(model_name, device_target, device_id, input_size)
    data = [np.ones(input_size)]
    ms_dtypes = [mindspore.float32]
    torch_dtypes = [torch.float32]
    eval_pathcore(network, network_torch, train_configs)
