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

from comparer import compare_models
from src.dataset import createDataset
from src.model import wide_resnet50_2
from src.oneStep import OneStepCell
from src.operator import embedding_concat, prep_dirs, reshape_embedding, normalize, save_anomaly_map
from src.sampling_methods.kcenter_greedy import kCenterGreedy
from src.config import cfg, merge_from_cli_list
set_seed(1)
opts = sys.argv[1:]
merge_from_cli_list(opts)
cfg.freeze()
print(cfg)


if __name__ == "__main__":
    device = "cuda:6"
    current_path = os.path.abspath(os.path.dirname(__file__))
    context.set_context(device_target=cfg.platform, save_graphs=False, mode=context.PYNATIVE_MODE)
    context.set_context(device_id=cfg.device_id)

    train_dataset, test_dataset, _, test_json_path = createDataset(cfg.dataset_path, cfg.category)
    embedding_dir_path, _ = prep_dirs(current_path, cfg.category)

    # network
    network = wide_resnet50_2()
    mindspore.load_checkpoint("/root/patchcore/wide_resnet50_2-95faca4d.ckpt", network)
    for p in network.trainable_params():
        p.requires_grad = False
    model = OneStepCell(network)
    from model_torch import wide_resnet50_2 as wide_resnet50_2_torch
    network_torch = wide_resnet50_2_torch().to(device)
    param_dict = torch.load("/root/patchcore/wide_resnet50_2-95faca4d.pth")
    network_torch.load_state_dict(param_dict)
    print("ckpts successfully loaded")
    for p in network_torch.parameters():
        p.requires_grad = False
    # compare_models(network, network_torch, np_data=[np.ones([1, 3, 224, 224])])
    from train_torch import OneStepCell as OneStepCell_torch
    model_torch = OneStepCell_torch(network_torch)
    # compare_models(model, model_torch, np_data=[np.ones([1, 3, 224, 224])])
    mean = cfg.mean
    std = cfg.std
    json_path = Path(test_json_path)
    with json_path.open("r") as label_file:
        label = json.load(label_file)
    test_iter = test_dataset.create_dict_iterator(output_numpy=True)
    embedding_dir_path, sample_path = prep_dirs(current_path, cfg.category)
    # train
    embedding_list_ms = []
    embedding_list_torch = []
    data_iter = train_dataset.create_dict_iterator(output_numpy=True)
    step_size = train_dataset.get_dataset_size()
    num_epochs = 1
    print("***************start train***************")
    for epoch in range(num_epochs):
        # for step, data in enumerate(data_iter):
        #     # time
        #     start = datetime.datetime.fromtimestamp(time.time())
        #     features_ms = model(mindspore.Tensor(data["img"], dtype=mindspore.float32))
        #     features_torch = model_torch(torch.tensor(data["img"], dtype=torch.float32).to(device))
        #     end = datetime.datetime.fromtimestamp(time.time())
        #     step_time = (end - start).microseconds / 1000.0
        #     print("step: {}, time: {}ms".format(step, step_time))
        #
        #     embedding_ms = embedding_concat(features_ms[0].asnumpy(), features_ms[1].asnumpy())
        #     embedding_list_ms.extend(reshape_embedding(embedding_ms))
        #     embedding_torch = embedding_concat(features_torch[0].detach().cpu().numpy(),
        #                                        features_torch[1].detach().cpu().numpy())
        #     embedding_list_torch.extend(reshape_embedding(embedding_torch))
        #     # break
        #
        # total_embeddings_ms = np.array(embedding_list_ms, dtype=np.float32)
        # total_embeddings_torch = np.array(embedding_list_torch, dtype=np.float32)
        #
        # # Random projection
        # randomprojector = SparseRandomProjection(n_components="auto", eps=0.9)
        # randomprojector.fit(total_embeddings_ms)
        # randomprojector_torch = SparseRandomProjection(n_components="auto", eps=0.9)
        # randomprojector_torch.fit(total_embeddings_torch)
        #
        # # Coreset Subsampling
        # selector_ms = kCenterGreedy(total_embeddings_ms, 0, 0)
        # selector_torch = kCenterGreedy(total_embeddings_torch, 0, 0)
        # selected_idx_ms = selector_ms.select_batch(
        #     model=randomprojector, already_selected=[], N=int(total_embeddings_ms.shape[0] * cfg.coreset_sampling_ratio)
        # )
        # selected_idx_torch = selector_torch.select_batch(
        #     model=randomprojector_torch, already_selected=[],
        #     N=int(total_embeddings_torch.shape[0] * cfg.coreset_sampling_ratio)
        # )
        # embedding_coreset_ms = total_embeddings_ms[selected_idx_ms]
        # embedding_coreset_torch = total_embeddings_torch[selected_idx_torch]
        #
        # print("initial embedding size : {}".format(total_embeddings_ms.shape))
        # print("final embedding size : {}".format(embedding_coreset_ms.shape))
        #
        # # faiss
        # # res = faiss.StandardGpuResources()
        # index_ms = faiss.IndexFlatL2(embedding_coreset_ms.shape[1])
        # # index_ms = faiss.index_cpu_to_gpu(res, 1, index_ms)
        # index_ms.add(embedding_coreset_ms)
        # faiss.write_index(index_ms, os.path.join(embedding_dir_path, "index_ms.faiss"))
        # index_torch = faiss.IndexFlatL2(embedding_coreset_torch.shape[1])
        # # index_torch = faiss.index_cpu_to_gpu(res, 1, index_torch)
        # index_torch.add(embedding_coreset_torch)
        # faiss.write_index(index_torch, os.path.join(embedding_dir_path, "index_torch.faiss"))
        print("***************start eval***************")
        gt_list_px_lvl_ms_torch = []
        pred_list_px_lvl_ms = []
        pred_list_px_lvl_torch = []
        gt_list_img_lvl_ms_torch = []
        pred_list_img_lvl_ms = []
        pred_list_img_lvl_torch = []
        img_path_list = []
        index_ms = faiss.read_index(os.path.join(embedding_dir_path,
                                                 "/root/patchcore/checkpoint/screw/embeddings/index_ms.faiss"))
        index_torch = faiss.read_index(os.path.join(embedding_dir_path,
                                                    "/root/patchcore/checkpoint/screw/embeddings/index_torch.faiss"))
        for step, data in enumerate(test_iter):
            step_label = label["{}".format(mindspore.Tensor(data["idx"])[0])]
            file_name = step_label["name"]
            x_type = step_label["img_type"]
            features_m = model(mindspore.Tensor(data["img"]))
            features_t = model_torch(torch.tensor(data["img"], dtype=torch.float32).to(device))
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

    print("***************train end***************")
    # compare_models(network, network_torch, np_data=[np.ones([1, 3, 224, 224])])
