
from scripts.run_train_imageclassification import start_imageclassification_train
from scripts.run_textcnn import start_textcnn_train
from scripts.run_yolov4 import start_yolov4_train
from scripts.run_unet import start_unet_train

def get_model_train(model_name):
    train_fun_dict = {
        'vgg16': start_imageclassification_train,
        'resnet50': start_imageclassification_train,
        "textcnn": start_textcnn_train,
        'unet': start_unet_train,
        'unetplus': start_unet_train,
        'yolov4': start_yolov4_train,
    }
    return train_fun_dict[model_name]
