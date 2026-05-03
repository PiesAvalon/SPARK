import argparse
import torch


def mutation_args():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--net_name", choices=['ResNet', 'vgg11', 'Vgg19', 'UNetMedical_torch', "DeepLabV3_torch",
                                                   "YOLOV3DarkNet53", "FastText_torch", "TextCNN", "SentimentNet"],
                            default="vgg11", type=str)

    arg_parser.add_argument("--frame_name", choices=['onnx', 'mindspore', 'pytorch'], default='onnx', type=str)

    arg_parser.add_argument("--frame_name2", choices=['onnx', 'mindspore', 'pytorch'], default='pytorch', type=str)

    arg_parser.add_argument("--result_saving_dir", default="./torch_mutated_net/", type=str)

    arg_parser.add_argument("--mutation_method", choices=['uoc', 'pioc', 'ABSOC_A', 'ABSOC_B', 'Hybrid'],
                            default="ABSOC_A", type=str)

    arg_parser.add_argument("--mutation_times", default=10, type=int)

    arg_parser.add_argument("--distance_MODE", choices=["EuclideanDistance", "ManhattanDistance", "ChebyshevDistance"],
                            default="EuclideanDistance", type=str)

    arg_parser.add_argument("--save_freq", default=1, type=int)

    arg_parser.add_argument("--Mutate_Batch_size", default=8, type=int)

    arg_parser.add_argument("--LOG_FLAG", choices=[True, False], default=False, type=bool)

    arg_parser.add_argument("--model_list", default=[], type=list)

    arg_parser.add_argument("--distance_list", default=[], type=list)

    arg_parser.add_argument("--log_flag", choices=[True, False], default=False, type=bool)

    arg_parser.add_argument("--LOG_PATH",
                            default=r"F:\NEW\比赛\项目\MR2023\torch_mutated_net\vgg11\2023_10_28_21_02_22\LOG_DICT_Windows_cuda_0 - 副本.json",
                            type=str)
    # arg_parser.add_argument("--datasets_path", default='F:/NEW/比赛/项目/MR2023/datasets/cifar10")', type=str)

    return arg_parser


def init_config(arg_init):
    arg_parser = arg_init()

    args = arg_parser.parse_args()

    return args



#