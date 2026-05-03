import torch

from mutation_torch.cargo import *
import os
from numpy import ndarray
from models.deeplabv3.main import SegDataset
from models.textcnn.dataset import MovieReview


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def create_dataset(net, data_dir, image_size, batch_size, training=False):
    dataset_1 = dataset_cargo[net.__class__.__name__]
    print(dataset_1)
    print("image_size", image_size)
    if net.__class__.__name__ == "YOLOV3DarkNet53":
        return dataset_1(os.path.join(data_dir, 'train2014'), os.path.join(data_dir,
                                                                         'annotations/instances_train2014.json'),
                       batch_size, 1, 0, is_training=True, shuffle=True)
    if net.__class__.__name__ == "DeepLabV3_torch":
        dataset = SegDataset(image_mean=[103.53, 116.28, 123.675],
                             image_std=[57.375, 57.120, 58.395],
                             data_file=data_dir,
                             batch_size=batch_size,
                             crop_size=513,
                             max_scale=2.0,  
                             min_scale=0.5,
                             ignore_label=255,
                             num_classes=21,
                             num_readers=2,
                             num_parallel_calls=4,
                             shard_id=0,
                             shard_num=1)
        dataset = dataset.get_dataset(repeat=1)
        return dataset
    if net.__class__.__name__ == "SentimentNet":
        return dataset_1(data_dir, batch_size=batch_size, device_num=1, rank=0)
    return dataset_1(data_dir, image_size=image_size, batch_size=batch_size, 
                     training=training)

def dataset(net, data_dir, image_size, Mutate_Batch_size):
    data_dir = path_cargo[net.__class__.__name__]
    image_size = size_cargo[net.__class__.__name__]

    shapes = shape_cargo[net.__class__.__name__]
    dtypes = [torch.float32 for _ in shapes] if net.__class__.__name__ not in nlp_cargo \
    else [torch.int32 for _ in shapes]
    if isinstance(image_size, list):
        # print('asdasd')
        # print(data_dir)
        dataset = create_dataset(net, data_dir, image_size, Mutate_Batch_size, training=False)
        # 带有shuffle的数据集， PIOC不可用其进行复现
        # print(dataset)
        if isinstance(dataset, tuple):
            dataset = dataset[0]

        size = dataset.get_dataset_size()
        print("dataset size is:", size)
        ds = dataset.create_tuple_iterator(output_numpy=True)
        for data in ds:
            data0 = data[0]
            print("data0.shape", data0.shape)
            break
        data0 = torch.tensor(data0, dtype=dtypes[0]).to(device)
        return data0
    elif net.__class__.__name__ == "FastText_torch":
        dataset = create_dataset(data_dir, Mutate_Batch_size, Mutate_Batch_size, training=False)
        # 带有shuffle的数据集， PIOC不可用其进行复现
        size = dataset.get_dataset_size()
        print("dataset size is:", size)
        ds = dataset.create_dict_iterator(output_numpy=True)
        for data in ds:
            data0: ndarray = data['src_token_text']
            data1: ndarray = data['src_tokens_text_length']
            # print("type data0", type(data0)) ndarray
            # print("type data1", type(data1))
            break
        # noinspection PyUnboundLocalVariable
        data0: torch.Tensor = torch.tensor(data0, dtype=dtypes[0]).to(device)
        data1: torch.Tensor = torch.tensor(data1, dtype=dtypes[1]).to(device)
        return data0,data1
    elif net.__class__.__name__ == "TextCNN":
        train_dataset = MovieReview(root_dir=data_dir, maxlen=51, split=0.9).create_train_dataset(
            batch_size=Mutate_Batch_size,
            epoch_size=2)
        train_iter = train_dataset.create_dict_iterator(output_numpy=True, num_epochs=2)
        for item in train_iter:
            text_array = item['data']
            # np.save("text_array.npy", text_array)
            # print("successfully saved text_array.npy")
            # exit(0)
            # print(net(imgs_array).shape)
            data0: torch.Tensor = torch.tensor(text_array, dtype=torch.int64).to(device)
            break
        return data0
    else:
        raise NotImplementedError("dataset is not implemented")