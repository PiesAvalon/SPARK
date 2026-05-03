import time

from arg import init_config, mutation_args
# from mutation_mindspore.run import Mindspore_Mutator
from onnx_mutation.run import onnx_Mutator
# from mutation_torch.run import Torch_Mutator

args = init_config(mutation_args)

time_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

if args.frame_name == 'onnx':

    mutator = onnx_Mutator(net_name=args.net_name, method=args.mutation_method, distance_mode=args.distance_MODE,
                           time_time=time_time, frame_name=args.frame_name2)

    mutator.mutate(args.mutation_times, args.Mutate_Batch_size)
