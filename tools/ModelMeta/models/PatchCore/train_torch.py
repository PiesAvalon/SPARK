import datetime
import os
import time
import faiss
import numpy as np
import torch
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from model_torch import wide_resnet50_2
from src.config import cfg
from src.dataset import createDataset
from src.operator import prep_dirs, embedding_concat, reshape_embedding
import torch.nn.functional as F
from src.sampling_methods.kcenter_greedy import kCenterGreedy


class OneStepCell(nn.Module):
    """OneStepCell"""

    def __init__(self, network):
        super(OneStepCell, self).__init__()
        self.network = network
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

    def forward(self, img):
        output = self.network(img)
        output_one = self.pool(F.pad(output[0], pad=(1, 1, 1, 1), mode="constant"))
        output_two = self.pool(F.pad(output[1], pad=(1, 1, 1, 1), mode="constant"))
        return [output_one, output_two]


if __name__ == "__main__":
    device = "cuda:6"
    current_path = os.path.abspath(os.path.dirname(__file__))
    train_dataset, _, _, _ = createDataset(cfg.dataset_path, cfg.category)
    embedding_dir_path, _ = prep_dirs(current_path, cfg.category)
    network = wide_resnet50_2().to(device)
    for p in network.parameters():
        # print(p.name)
        p.requires_grad = False
    model = OneStepCell(network)
    embedding_list = []
    print("***************start train***************")
    for epoch in range(cfg.num_epochs):
        data_iter = train_dataset.create_dict_iterator(output_numpy=True)
        step_size = train_dataset.get_dataset_size()
        for step, data in enumerate(data_iter):
            # time
            start = datetime.datetime.fromtimestamp(time.time())
            features = model(torch.tensor(data["img"], dtype=torch.float32).to(device))
            end = datetime.datetime.fromtimestamp(time.time())
            step_time = (end - start).microseconds / 1000.0
            print("step: {}, time: {}ms".format(step, step_time))

            embedding = embedding_concat(features[0].detach().cpu().numpy(), features[1].detach().cpu().numpy())
            embedding_list.extend(reshape_embedding(embedding))

        total_embeddings = np.array(embedding_list, dtype=np.float32)

        # Random projection
        randomprojector = SparseRandomProjection(n_components="auto", eps=0.9)
        randomprojector.fit(total_embeddings)

        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(
            model=randomprojector, already_selected=[], N=int(total_embeddings.shape[0] * cfg.coreset_sampling_ratio)
        )
        embedding_coreset = total_embeddings[selected_idx]

        print("initial embedding size : {}".format(total_embeddings.shape))
        print("final embedding size : {}".format(embedding_coreset.shape))

        # faiss
        index = faiss.IndexFlatL2(embedding_coreset.shape[1])
        index.add(embedding_coreset)
        faiss.write_index(index, os.path.join(embedding_dir_path, "index_torch.faiss"))

    print("***************train end***************")
