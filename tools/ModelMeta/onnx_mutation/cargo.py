import os
import scipy.io as scio

from models.FastText.Fasttext import create_FastText_dataset
from models.FastText.fasttext_torch import FastText_torch
from models.UNet.Unet import create_Unet_dataset
from models.UNet.main_torch import UNetMedical_torch
from models.deeplabv3.main import SegDataset
from models.deeplabv3.main_torch import DeepLabV3_torch
# from models.resnet50.resnet50 import create_cifar10_dataset
# from models.textcnn.dataset import MovieReview
# from models.textcnn.textcnn_torch import TextCNN
# from models.vgg11.vgg11_torch import vgg11
# from models.vgg19.vgg19_torch import vgg19
# from models.yolov3.main_new import create_yolo_dataset
# from models.yolov3.main_torch import YOLOV3DarkNet53
# from models.lstm.lstm_torch import SentimentNet
# from models.lstm.src.dataset import lstm_create_dataset
# from models.resnet50.resnet50_torch import resnet50
from onnx_mutation.insert import insert_ABSOC_A, insert_ABSOC_B, insert_hybrid, insert_pioc, insert_uoc

from arg import init_config, mutation_args

args = init_config(mutation_args)

dataset_cargo = {
    # "ResNet": create_cifar10_dataset,
    # "vgg11": create_cifar10_dataset,
    # "Vgg19": create_cifar10_dataset,
    "UNetMedical_torch": create_Unet_dataset,
    "DeepLabV3_torch": None,
    # "YOLOV3DarkNet53": create_yolo_dataset,
    "FastText_torch": create_FastText_dataset,
    # "SentimentNet": lstm_create_dataset,
    # TextCNN Not compatible with this pattern
}

size_cargo = {
    "ResNet": [224, 224],
    "vgg11": [32, 32],
    "Vgg19": [224, 224],
    "UNetMedical_torch": [572, 572],
    "DeepLabV3_torch": [513, 513],
    "YOLOV3DarkNet53": [576, 576],
    "FastText_torch": ([1, 128], [1, 1]),
    "TextCNN": None,
    "SentimentNet": [16, 500],
}

path_cargo = {
        "vgg19": "./datasets/data_npy/Vgg19_data0.npy",
        # "Vgg19": "/root/MR/data_npy/vgg19_data0.npy",
        "vgg11": "./datasets/data_npy/vgg11_data0.npy",
        "resnet": "./datasets/data_npy/ResNet_data0.npy",
        "UNetMedical": "./datasets/data_npy/UNetMedical_data0.npy",
        "DeepLabV3": "./datasets/data_npy/DeepLabV3_data0.npy",
        "yolov3": "./datasets/data_npy/yolov3_data0.npy",
        "FastText_data0": "./datasets/data_npy/FastText_data0.npy",
        "FastText_data1": "./datasets/data_npy/FastText_data1.npy",
        "textcnn": "./datasets/data_npy/TextCNN_data0.npy",
        "LSTM": "./datasets/data_npy/SentimentNet_data0.npy",
        # "GPT": GPT(),
        "Wide_Deep": './datasets/data_npy/',
            "patchcore":"./datasets/data_npy/patchcore_data0.npy",
    "SSDmobilenetv1":"./datasets/data_npy/SSDmobilenetv1_data0.npy",
    "SSDresnet50fpn":"./datasets/data_npy/SSDresnet50fpn_data0.npy",
    "openpose": "./datasets/data_npy/openpose_data0.npy",
    "AutoEncoder": "./datasets/data_npy/ssimae_data0.npy",
}


# train_c
def get_model(model_name, device):
    net_cargo = {
        # "vgg19": vgg19,
        # "vgg11": vgg11,
        # "resnet": resnet50,
        "UNetMedical": UNetMedical_torch,
        "DeepLabV3": DeepLabV3_torch,
        # "yolov3": YOLOV3DarkNet53,
        "FastText": FastText_torch,
        "TextCNN": TextCNN,
        # "LSTM": SentimentNet,
        # "GPT": GPT(),
        # "Wide_Deep": WideDeepModel,
    }
    if model_name == "vgg19":
        model = net_cargo[model_name](10)
        return model
    elif model_name == "vgg11":
        model = net_cargo[model_name]()
        return model
    elif model_name == "resnet":
        model = net_cargo[model_name]()
        return model
    elif model_name == "UNetMedical":
        model = net_cargo[model_name](1, 2)
        return model
    elif model_name == "DeepLabV3":
        model = net_cargo[model_name](21)
        return model
    elif model_name == "yolov3":
        model = net_cargo[model_name](True)
        return model
    elif model_name == "FastText":
        model = net_cargo[model_name]()
        return model
    elif model_name == "TextCNN":
        model = net_cargo[model_name](vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
        return model
    elif model_name == "LSTM":
        data = scio.loadmat(os.path.join("./datasets/data_npy/embedding_table.mat"))
        embedding_table = data['embedding_table']
        model = net_cargo[model_name](vocab_size=embedding_table.shape[0],
                                      embed_size=300,
                                      num_hiddens=100,
                                      num_layers=2,
                                      bidirectional=True,
                                      num_classes=2,
                                      weight=torch.tensor(embedding_table).to(device))
        return model
    elif model_name == "Wide_Deep":
        model = net_cargo[model_name]
        return model


nlp_cargo = ["FastText_torch", "TextCNN", "SentimentNet"]
classify_cargo = ["ResNet", "vgg11", "Vgg19"]
shape_cargo = {
    "ResNet": [(1, 3, 224, 224)],
    "vgg11": [(1, 3, 32, 32)],
    "Vgg19": [(1, 3, 224, 224)],
    "UNetMedical_torch": [(1, 1, 572, 572)],
    "DeepLabV3_torch": [(1, 3, 513, 513)],
    "YOLOV3DarkNet53": [(1, 3, 416, 416)],
    "FastText_torch": [(1, 64), (1, 1)],
    "TextCNN": [(1, 51)],
    "SentimentNet": [(16, 500)],
}

# banned_ops = ["torch.cat"]
# mutable_ops = ['call_module', 'call_method', 'root']
# mutable_ops = ['call_module', 'root', 'call_funtion', 'call_method']
mutable_ops = ['call_module', 'root']

methods = {"uoc": insert_uoc,
           "pioc": insert_pioc,
           "Hybrid": insert_hybrid,
           "ABSOC_A": insert_ABSOC_A,
           "ABSOC_B": insert_ABSOC_B, }
