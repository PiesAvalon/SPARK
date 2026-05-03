import os
import sys

import yaml
model_name = "unet3d"
f = open('./config.txt', 'w')
config_path = f'./config/{model_name}.yaml'
f.write(config_path.lower())
f.close()
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

import argparse
import numpy as np
import time
from copy import deepcopy
import torch
import mindspore
from mindspore import context
from mindspore.ops import operations as P

from common.dataset_utils import get_dataset
from common.model_utils import get_model
from common.log_recoder import Logger
from common.mutation_ms.mutator_selection_logic import Roulette, MCMC
from common.mutation_torch.mutation_main_followlog import analyze_log_torch_followtrace
from common.mutation_ms.model_mutation_operators_followlog import analyze_log_mindspore_followtrace
from common.mutation_ms.model_mutation_generators import generate_model_by_model_mutation
from common.help_utils import rename_parameter
from common.help_utils import ToolUtils
from common.mutation_ms.model_mutation_generators import all_mutate_ops
from common.run_ssd import start_ssd_train
from common.run_yolov3 import start_yolov3_train
from common.run_yolov4 import start_yolov4_train
from common.run_yolov5 import start_yolov5_train
from common.run_unet import start_unet_train
from common.run_unet3d import start_unet3d_train
from common.run_deeplabv3 import start_deeplab_train
from common.run_train_imageclassification import start_imageclassification_train
from common.run_textcnn import start_textcnn_train
from common.run_fasttext import start_fasttext_train
from common.run_bert import start_bert_train
from common.run_transformer_project.run_transformer import start_transformer_train
from common.run_maskrcnn import start_maskrcnn_train
from common.run_FasterRcnn import start_fasterrcnn_train
from common.run_retinaface import start_retinaface_train
from common.run_pangu import start_pangu_train


class YoloUtil:
    def __init__(self):
        super(YoloUtil, self).__init__()

    @staticmethod
    def reformat_outputs_first_generation(multi_outputs):
        reformat_outputs_list = []
        for tuple_output in multi_outputs:
            outputs_list = []
            for tensor_tuple in tuple_output:
                outputs_list.extend(tensor_tuple)
            reformat_outputs_list.append(tuple(outputs_list))
        return reformat_outputs_list

    @staticmethod
    def reformat_outputs_second_generation(multi_outputs):
        reformat_outputs_list = []
        for tuple_output in multi_outputs:
            for tensor_tuple in tuple_output:
                reformat_outputs_list.append(tensor_tuple)
        return tuple(reformat_outputs_list)


class NetworkGeneralization:
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model
        self.dataset_path = args.dataset_path
        self.mutation_type = args.mutation_type
        self.selected_generation = args.selected_generation

        config_path = os.getcwd() + '/config/' + self.model_name + '.yaml'
        with open(config_path.lower(), 'r', encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.train_config = config['train_config']
        if args.batch_size is None:
            self.batch_size = int(config['train_config']['batch_size'])
        else:
            self.batch_size = args.batch_size
            self.train_config['batch_size'] = args.batch_size

        if args.epoch is None:
            self.epoch = int(config['train_config']['epoch'])
        else:
            self.epoch = args.epoch
            self.train_config['epoch'] = args.epoch

        if args.mutation_iterations is None:
            self.mutation_iterations = int(config['mutation_config']['mutation_iterations'])
        else:
            self.mutation_iterations = args.mutation_iterations
            config['mutation_config']['mutation_iterations'] = args.mutation_iterations

        if args.mutation_strategy is None:
            self.mutation_strategy = str(config['mutation_config']['mutator_strategy'])
        else:
            self.mutation_strategy = args.mutation_strategy
            config['mutation_config']['mutator_strategy'] = args.mutation_strategy

        if args.selected_model_num is None:
            self.selected_model_num = int(config['mutation_config']['select_mutmodel_nums'])
        else:
            self.selected_model_num = args.selected_model_num
            config['mutation_config']['select_mutmodel_nums'] = args.selected_model_num

        if self.mutation_iterations < self.selected_model_num:
            err_msg = 'the number of selected mutation model should no more than mutation iteration, while ' \
                      'selected_model_num is {} and mutation_iteration is {}.'.format(self.selected_model_num,
                                                                                      self.mutation_iterations)
            raise ValueError(err_msg)

        # export CONTEXT_DEVICE_TARGET=CPU
        if 'CONTEXT_DEVICE_TARGET' in os.environ:
            self.device_target = os.environ['CONTEXT_DEVICE_TARGET'].upper()
        else:
            self.device_target = 'CPU'

        self.device_target = config['train_config']['device']
        self.device_id = config['train_config']['device_id']

        if 'CONTEXT_MODE' in os.environ and os.environ['CONTEXT_MODE'] == 'GRAPH_MODE':
            context.set_context(mode=context.GRAPH_MODE, device_id=(self.device_id + 1))
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_id=(self.device_id + 1))

        if args.mutation_log is None:
            self.mutation_log = config['mutation_config']['mutlog_path']
        else:
            self.mutation_log = args.mutation_log

        time_tuple = time.localtime(time.time())
        time_stamp = "{}.{}.{}.{}.{}.{}".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                                time_tuple[4], time_tuple[5])
        self.log_path = self.mutation_log + "/" + str(self.model_name) + "-" + time_stamp
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.mut_log_path = self.log_path + '/mutation.txt'

        self.run_log = self.init_logging()
        self.dataset = get_dataset(dataset_name=config['train_config']['dataset_name'])
        self.train_config['dataset_path'] = self.dataset_path
        self.mutation_eval_metric = config['mutation_config']['mutation_eval_metric']
        self.first_generation_tracedict = {}
        self.second_generation_tracedict = {}
        self.data_forcal_size = self.batch_size * 2
        self.test_size = config['train_config']['test_size']
        self.threshold = config['mutation_config']['validation_threshold']

        self.input_size_list = config['train_config']['input_size']
        input_size_lists = []
        for input_size_each in self.input_size_list:
            input_size_str = input_size_each[1:-1].split(",")
            input_size_list = []
            for val in input_size_str:
                if val == "":
                    continue
                input_size_list.append(int(val))
            input_size_list = tuple(input_size_list)
            input_size_lists.append(input_size_list)

        self.input_size = input_size_lists[0]

        self.first_scores = []
        self.second_scores = []
        self.yolo_util = YoloUtil()
        self.mindspore_pass_rate = 0
        self.run_log.info("Mutation config:\n-------------------------------------------------------"
                          "-------\n{}".format(config))

    def init_logging(self):
        log = Logger(log_file=self.log_path + '/run.log')
        return log.logger

    def cal_mutation_score(self, model, data_forcal, origin_outputs):

        if self.mutation_eval_metric == "origin_diff":
            if not isinstance(origin_outputs[0], tuple):
                if isinstance(data_forcal, list):  # mutiply inputs and single outputs
                    out_diffs = []
                    for idx in range(0, data_forcal[0].shape[0], self.test_size):
                        inputs = []
                        for val in data_forcal:
                            inputs.append(val[idx: (idx + self.test_size), :])
                        mutation_output = model(*inputs)
                        origin_output = mindspore.ops.flatten(origin_outputs[int(idx / self.test_size)])
                        mutation_output = mindspore.ops.flatten(mutation_output)
                        out_diff = (origin_output - mutation_output).asnumpy()
                        out_diffs.append(np.linalg.norm(out_diff))

                else:  # single inputs and single outputs
                    out_diffs = []
                    for idx in range(0, data_forcal.shape[0], self.test_size):
                        imgs_ms = data_forcal[idx:(idx + self.test_size), :]

                        outputs = model(imgs_ms)

                        origin_output = mindspore.ops.flatten(origin_outputs[int(idx / self.test_size)])
                        output = mindspore.ops.flatten(outputs)
                        out_diff = (origin_output - output).asnumpy()
                        out_diffs.append(np.linalg.norm(out_diff))



            else:
                if isinstance(data_forcal, list):  # mutiply inputs and mutiply outputs
                    mutation_outputs = []
                    for idx in range(0, data_forcal[0].shape[0], self.test_size):
                        inputs = []
                        for val in data_forcal:
                            inputs.append(val[idx: (idx + self.test_size), :])
                        mutation_output = model(*inputs)
                        mutation_outputs.append(mutation_output)

                else:  # single inputs and mutiply outputs
                    mutation_outputs = []
                    for idx in range(0, data_forcal.shape[0], self.test_size):
                        imgs_ms = data_forcal[idx:idx + self.test_size, :]

                        mutation_output = model(imgs_ms)  # mutation_outputs is a tuple whose element is tensor
                        mutation_outputs.append(mutation_output)

                    if 'yolo' in self.model_name:
                        origin_outputs = self.yolo_util.reformat_outputs_first_generation(origin_outputs)
                        mutation_outputs = self.yolo_util.reformat_outputs_first_generation(mutation_outputs)

                out_diffs = []
                for idx in range(len(mutation_outputs)):
                    origin_output = origin_outputs[idx]
                    mutation_output = mutation_outputs[idx]

                    for idj in range(len(origin_output)):
                        distance_vec = origin_output[idj] - mutation_output[idj]
                        distance_vec = mindspore.ops.flatten(distance_vec)

                        out_diff = np.linalg.norm(mindspore.Tensor(distance_vec).asnumpy())
                        out_diffs.append(out_diff)

            score = np.mean(out_diffs)
        return score

    def get_filter_data(self, test_iter):
        if self.model_name == "fasttext":
            for item in test_iter:
                src_tokens = deepcopy(item[0])
                src_tokens_length = deepcopy(item[1])
                srctokens_ms_forcal = mindspore.Tensor(src_tokens, mindspore.int32)
                srctokenslength_ms_forcal = mindspore.Tensor(src_tokens_length, mindspore.int32)
                break
            imgs_ms_forcal = [srctokens_ms_forcal, srctokenslength_ms_forcal]


        elif self.model_name == "bert":
            for item in test_iter:
                data1 = deepcopy(item[0])
                data2 = deepcopy(item[1])
                data3 = deepcopy(item[2])

                data1_ms = mindspore.Tensor(data1, mindspore.int64)
                data2_ms = mindspore.Tensor(data2, mindspore.int64)
                data3_ms = mindspore.Tensor(data3, mindspore.int64)
                break
            imgs_ms_forcal = [data1_ms, data2_ms, data3_ms]

        elif self.model_name == "maskrcnn":
            for item in test_iter:
                data1 = deepcopy(item[0])
                data2 = deepcopy(item[1])
                data3 = deepcopy(item[2])
                data4 = deepcopy(item[3])
                data5 = deepcopy(item[4])
                data6 = deepcopy(item[5])

                data1_ms = mindspore.Tensor(data1, mindspore.float32)
                data2_ms = mindspore.Tensor(data2, mindspore.float32)
                data3_ms = mindspore.Tensor(data3, mindspore.float32)
                data4_ms = mindspore.Tensor(data4, mindspore.int32)
                data5_ms = mindspore.Tensor(data5, mindspore.bool_)
                data6_ms = mindspore.Tensor(data6, mindspore.bool_)
                break
            imgs_ms_forcal = [data1_ms, data2_ms, data3_ms, data4_ms, data5_ms, data6_ms]

        elif self.model_name == "fasterrcnn":
            for item in test_iter:
                data1 = deepcopy(item[0])
                data2 = deepcopy(item[1])
                data3 = deepcopy(item[2])
                data4 = deepcopy(item[3])
                data5 = deepcopy(item[4])

                data1_ms = mindspore.Tensor(data1, mindspore.float32)
                data2_ms = mindspore.Tensor(data2, mindspore.float32)
                data3_ms = mindspore.Tensor(data3, mindspore.float32)
                data4_ms = mindspore.Tensor(data4, mindspore.int32)
                data5_ms = mindspore.Tensor(data5, mindspore.float32)
                break
            imgs_ms_forcal = [data1_ms, data2_ms, data3_ms, data4_ms, data5_ms]

        elif self.model_name == "transformer":
            for item in test_iter:
                data1 = deepcopy(item[0])
                data2 = deepcopy(item[1])
                data3 = deepcopy(item[4])
                data4 = deepcopy(item[5])

                data1_ms = mindspore.Tensor(data1, mindspore.float32)
                data2_ms = mindspore.Tensor(data2, mindspore.float32)
                data3_ms = mindspore.Tensor(data3, mindspore.float32)
                data4_ms = mindspore.Tensor(data4, mindspore.int32)

                break
            imgs_ms_forcal = [data1_ms, data2_ms, data3_ms, data4_ms]

        elif self.model_name == "pangu":
            for item in test_iter:
                data1 = deepcopy(item[0])
                data2 = deepcopy(item[1])
                data3 = deepcopy(item[2])

                input_ids = mindspore.Tensor(data1, mindspore.int32)
                input_position = mindspore.Tensor(data2, mindspore.int32)
                attention_mask = mindspore.Tensor(data3, mindspore.float32)

                tokens = P.StridedSlice()(input_ids, (0, 0), (self.data_forcal_size, -1), (1, 1))
                input_position = P.StridedSlice()(input_position, (0, 0), (self.data_forcal_size, 1024), (1, 1))
                decoder_attention_masks = P.StridedSlice()(attention_mask, (0, 0, 0),
                                                           (self.data_forcal_size, 1024, 1024), (1, 1, 1))
                break
            imgs_ms_forcal = [tokens, input_position, decoder_attention_masks]


        else:
            for item in test_iter:
                imgs_array = deepcopy(item[0])
                if self.train_config['dtypes'][0] == "float":
                    dtype = mindspore.float32
                elif self.train_config['dtypes'][0] == "int":
                    dtype = mindspore.int32

                imgs_ms_forcal = mindspore.Tensor(imgs_array, dtype)
                break

        return imgs_ms_forcal

    def mindspore_mutation(self):
        origin_model_ms, _ = get_model(self.model_name, self.device_target, device_id=self.device_id,
                                       input_size=tuple(self.input_size))

        test_set = self.dataset(data_dir=self.dataset_path, batch_size=self.data_forcal_size, is_train=True)
        test_iter = test_set.create_tuple_iterator(output_numpy=True)

        imgs_ms_forcal = self.get_filter_data(test_iter)

        origin_outputs = []
        if isinstance(imgs_ms_forcal, list):  # mulptily model inputs
            for idx in range(0, self.data_forcal_size, self.test_size):
                inputs = []
                for val in imgs_ms_forcal:
                    inputs.append(val[idx:idx + self.test_size, :])

                outputs = origin_model_ms(*inputs)
                origin_outputs.append(outputs)

        else:  # single model inputs
            for idx in range(0, imgs_ms_forcal.shape[0], self.test_size):
                imgs_ms = imgs_ms_forcal[idx:idx + self.test_size, :]

                outputs = origin_model_ms(imgs_ms)
                origin_outputs.append(outputs)

        if self.mutation_strategy == "random":
            self.random_mutate(origin_model_ms, imgs_ms_forcal, origin_outputs)
        elif self.mutation_strategy == "MCMC":
            self.MCMC_mutate(origin_model_ms, imgs_ms_forcal, origin_outputs)

        mut_succ = os.popen(f"grep -c mut_result:True {self.mut_log_path}").readlines()[0]
        muttypes = os.popen(f"grep -aF 'Adopt' {self.mut_log_path}").readlines()
        mutresults = os.popen(f"grep -aF 'mut_result:' {self.mut_log_path}").readlines()
        muttype_count = dict(zip(self.mutation_type, [0] * len(self.mutation_type)))

        for i in range(len(muttypes)):
            muttype = muttypes[i].split(' ')[1]
            mutresult = mutresults[i].split(':')[1].split('\n')[0]
            if mutresult == 'True' or mutresult == True:
                muttype_count[muttype] += 1

        self.run_log.info('mutation iteration:{}, mutation success:{}'.format(self.mutation_iterations, mut_succ))
        self.run_log.info('mutation type:{}\n'.format(muttype_count))

    def random_mutate(self, origin_model_ms, imgs_ms_forcal, origin_outputs):
        mutation_scores = []
        for generation in range(1, self.mutation_iterations + 1):
            self.run_log.info(
                "================== start {} generation!({}/{}) ==================".format(generation, generation,
                                                                                           self.mutation_iterations))
            mut_type = np.random.permutation(self.mutation_type)[0]
            mut_result = generate_model_by_model_mutation(origin_model_ms, mut_type, self.input_size, self.mut_log_path,
                                                          generation, self.run_log)



            if mut_result == 'True' or mut_result == True:
                self.mindspore_pass_rate += 1
                mutate_score = self.cal_mutation_score(origin_model_ms, imgs_ms_forcal, origin_outputs)
                mutation_scores.append(mutate_score)
            else:
                mutation_scores.append(-100)
                # execution_traces=[i for i in range(1,generation)]
                # net_ms_seed, _ = get_model(self.model_name, self.device_target, device_id=self.device_id,input_size=tuple(self.input_size))
                # net_ms = analyze_log_mindspore_followtrace(execution_traces, net_ms_seed, self.mut_log_path,self.input_size)
                # origin_model_ms = net_ms

        self.run_log.info("mutation_scores: {}".format(mutation_scores))
        first_select_generations = np.argsort(np.array(mutation_scores))[
                                   (len(mutation_scores) - self.selected_model_num):]
        self.run_log.info("**** stage1 select generations: {} ****\n".format(first_select_generations+1))

        for select_generation in first_select_generations:
            execution_traces = [i for i in range(1, select_generation + 2)]
            self.first_generation_tracedict[str(select_generation + 1)] = execution_traces

        self.first_scores = mutation_scores
        return self.first_generation_tracedict

    def MCMC_mutate(self, origin_model_ms, imgs_ms_forcal, origin_outputs):

        mcmc_help_info = {}  # Record  the current model evolved from which generation of seeds
        mutation_scores = []
        mutator_selector_func, mutant_selector_func = MCMC, Roulette
        mutate_ops = all_mutate_ops()
        mutate_op_history = {k: 0 for k in mutate_ops}
        mutate_num = self.mutation_iterations
        mutator_selector, mutant_selector = mutator_selector_func(mutate_ops), mutant_selector_func(
            [self.model_name + "_seed"], capacity=mutate_num + 1)

        last_used_mutator = None
        last_inconsistency = 0

        generation = 1

        while generation < self.mutation_iterations:
            self.run_log.info("================== start {} generation!({}/{}) ==================".format(generation, generation,self.mutation_iterations))
            new_seed_name = self.model_name + "_" + str(generation)

            net_ms_seed, _ = get_model(self.model_name, self.device_target, device_id=self.device_id,
                                       input_size=tuple(self.input_size))
            picked_seed = ToolUtils.select_mutant(mutant_selector)
            selected_op = ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator)
            mutate_op_history[selected_op] += 1
            last_used_mutator = selected_op
            mutator = mutator_selector.mutators[selected_op]
            mutant = mutant_selector.mutants[picked_seed]

            # create net_ms(selected the "picked_seed"th mutation model)
            if picked_seed == self.model_name + "_seed":
                net_ms = net_ms_seed

            else:
                father_name = picked_seed
                execution_traces = [int(father_name.split("_")[1])]
                while True:
                    father_name = mcmc_help_info[father_name]
                    if father_name == self.model_name + "_seed":
                        break
                    trace_singledot = int(father_name.split("_")[1])

                    execution_traces.append(trace_singledot)

                execution_traces.sort()
                assert len(execution_traces) > 0
                assert net_ms_seed is not None
                net_ms = analyze_log_mindspore_followtrace(execution_traces, net_ms_seed, self.mut_log_path,
                                                           self.input_size)

            mcmc_help_info[
                new_seed_name] = picked_seed  # pick_seed is the father of current seed whose name is new_seed_name

            if net_ms is None:
                raise RuntimeError("seed model is None !")

            mut_result = generate_model_by_model_mutation(net_ms, selected_op, self.input_size, self.mut_log_path,
                                                          generation, self.run_log)



            if mut_result == "True" or mut_result == True:
                mutant.selected += 1
                mutator.total += 1

                accumulative_inconsistency = self.cal_mutation_score(net_ms, imgs_ms_forcal, origin_outputs)
                mutation_scores.append(accumulative_inconsistency)

                delta = accumulative_inconsistency - last_inconsistency
                mutator.delta_bigger_than_zero = mutator.delta_bigger_than_zero + 1 if delta > 0 else mutator.delta_bigger_than_zero
                if delta > 0:
                    if mutant_selector.is_full():
                        mutant_selector.pop_one_mutant()
                    mutant_selector.add_mutant(new_seed_name)
                    last_inconsistency = accumulative_inconsistency
                self.mindspore_pass_rate += 1


            else:
                self.run_log.error("Exception raised when mutate {} with {}".format(picked_seed, selected_op))
                mutation_scores.append(-100)
            generation += 1


        self.run_log.info("mutation_scores: {}".format(mutation_scores))
        first_select_generations = np.argsort(np.array(mutation_scores))[
                                   (len(mutation_scores) - self.selected_model_num):]
        self.run_log.info("**** stage1 select generations: {} ****\n".format(first_select_generations+1))

        for select_generation in first_select_generations:
            start_gen = select_generation
            execution_traces = [start_gen]
            while True:
                father_name = mcmc_help_info[self.model_name + "_" + str(start_gen)]
                if father_name == self.model_name + "_seed":
                    break
                trace_singledot = int(father_name.split("_")[1])
                execution_traces.append(trace_singledot)
                start_gen = trace_singledot

            execution_traces.sort()
            self.first_generation_tracedict[str(select_generation + 1)] = execution_traces

        self.first_scores = mutation_scores
        return self.first_generation_tracedict

    def diff_calculate(self, ):

        first_select_generations_str = list(self.first_generation_tracedict.keys())
        first_select_generations = [int(val) for val in first_select_generations_str]
        select_generations = np.sort(first_select_generations)


        test_set = self.dataset(data_dir=self.dataset_path, batch_size=self.data_forcal_size, is_train=True)
        test_iter = test_set.create_tuple_iterator(output_numpy=True)

        imgs_ms_forcal = self.get_filter_data(test_iter)

        if self.device_target == 'GPU':
            final_device = 'cuda:' + str(self.device_id)
        else:
            final_device = 'cpu'

        for generation in select_generations:
            model_ms_origin, model_torch_origin = get_model(self.model_name, self.device_target,
                                                            device_id=self.device_id, input_size=tuple(self.input_size))
            self.run_log.info('compare mindspore and pytorch with selected mutation model: {}'.format(generation))

            cur_generation_trace = self.first_generation_tracedict[str(generation)]

            model_ms_mutation = analyze_log_mindspore_followtrace(deepcopy(cur_generation_trace), model_ms_origin,
                                                                  self.mut_log_path, self.input_size,self.run_log)
            rename_parameter(model_ms_mutation)
            model_torch_mutation = analyze_log_torch_followtrace(deepcopy(cur_generation_trace), model_torch_origin,
                                                                 self.mut_log_path, self.input_size,self.run_log)


            # calculate the output difference with torch model and mindspore model
            mut_out_diff_metric = []
            if isinstance(imgs_ms_forcal, list):  # mulptily inputs
                for idx in range(0, self.data_forcal_size, self.test_size):
                    mindspore_inputs, torch_inputs = [], []
                    for val in imgs_ms_forcal:

                        val_slice = val[idx:idx + self.test_size, :]
                        val_array = val_slice.asnumpy()
                        if "float32" in str(val.dtype).lower():
                            t_dtype = torch.float32
                        elif "int32" in str(val.dtype).lower():
                            t_dtype = torch.int32
                        elif "int64" in str(val.dtype).lower():
                            t_dtype = torch.int64
                        val_torch = torch.tensor(val_array, dtype=t_dtype).to(final_device)
                        mindspore_inputs.append(val[idx:idx + self.test_size, :])
                        torch_inputs.append(val_torch)
                    with torch.no_grad():
                        if "rcnn" in self.model_name:
                            output_torch = model_torch_mutation.backbone(torch_inputs[0])
                            output_ms = model_ms_mutation.backbone(mindspore_inputs[0])
                        else:
                            output_torch = model_torch_mutation(*torch_inputs)
                            output_ms = model_ms_mutation(*mindspore_inputs)

                    if not isinstance(output_ms, tuple):
                        output_ms = output_ms.asnumpy()
                        output_ms = torch.tensor(output_ms, dtype=output_torch.dtype)

                        output_torch_array = torch.flatten(output_torch).detach().cpu().numpy()
                        output_ms_array = torch.flatten(
                            output_ms).detach().cpu().numpy()  # since torch.flatten is different from that of mindspore, we adpopt torch.flatten

                        fenzi = output_torch_array.dot(output_ms_array)
                        fenmu = np.linalg.norm(output_torch_array) * np.linalg.norm(output_ms_array)

                        cos_dim = 1 - (fenzi / fenmu)
                        mut_out_diff_metric.append(cos_dim)
                    else:
                        for idx in range(len(output_torch)):
                            output_ms_idx = output_ms[idx].asnumpy()
                            output_torch_idx = output_torch[idx]

                            output_ms_fortorch = torch.tensor(output_ms_idx, dtype=output_torch_idx.dtype)

                            output_torch_array = torch.flatten(output_torch_idx).detach().cpu().numpy()
                            output_ms_array = torch.flatten(output_ms_fortorch).detach().cpu().numpy()

                            cos_dim = 1 - output_torch_array.dot(output_ms_array) / (
                                        np.linalg.norm(output_torch_array) * np.linalg.norm(output_ms_array))
                            mut_out_diff_metric.append(cos_dim)

            else:  # single inputs
                for idx in range(0, self.data_forcal_size, self.test_size):

                    ms_input = imgs_ms_forcal[idx:idx + self.test_size, :]
                    array_input = ms_input.asnumpy()

                    if "float32" in str(ms_input.dtype).lower():
                        t_dtype = torch.float32
                    elif "int32" in str(ms_input.dtype).lower():
                        t_dtype = torch.int64

                    torch_input = torch.tensor(array_input, dtype=t_dtype).to(final_device)

                    with torch.no_grad():
                        output_torch = model_torch_mutation(torch_input)
                    output_ms = model_ms_mutation(ms_input)

                    if 'yolo' in self.model_name:
                        output_ms = self.yolo_util.reformat_outputs_second_generation(output_ms)
                        output_torch = self.yolo_util.reformat_outputs_second_generation(output_torch)

                    if not isinstance(output_ms, tuple):
                        output_ms = output_ms.asnumpy()
                        output_ms = torch.tensor(output_ms, dtype=output_torch.dtype)

                        output_torch_array = torch.flatten(output_torch).detach().cpu().numpy()
                        output_ms_array = torch.flatten(
                            output_ms).detach().cpu().numpy()  # since torch.flatten is different from that of mindspore, we adpopt torch.flatten

                        cos_dim = 1 - output_torch_array.dot(output_ms_array) / (
                                    np.linalg.norm(output_torch_array) * np.linalg.norm(output_ms_array))
                        mut_out_diff_metric.append(cos_dim)
                    else:
                        for idx in range(len(output_torch)):
                            output_ms_idx = output_ms[idx].asnumpy()
                            output_torch_idx = output_torch[idx]

                            output_ms_fortorch = torch.tensor(output_ms_idx, dtype=output_torch_idx.dtype)

                            output_torch_array = torch.flatten(output_torch_idx).detach().cpu().numpy()
                            output_ms_array = torch.flatten(output_ms_fortorch).detach().cpu().numpy()

                            cos_dim = 1 - output_torch_array.dot(output_ms_array) / (
                                        np.linalg.norm(output_torch_array) * np.linalg.norm(output_ms_array))
                            mut_out_diff_metric.append(cos_dim)

            self.second_scores.append(np.mean(mut_out_diff_metric))
            if np.mean(mut_out_diff_metric) > self.threshold or np.isnan(np.mean(mut_out_diff_metric)):
                self.second_generation_tracedict[str(generation)] = self.first_generation_tracedict[str(generation)]
            self.run_log.info("selected_mutation_model: {}, ms_torch_diff: {}, diff_threshold: {}".format(generation,
                                                                                                          np.mean(
                                                                                                              mut_out_diff_metric),
                                                                                                          self.threshold))

        if len(self.second_generation_tracedict) == 0:
            self.run_log.info("All selected mutation model check success")
            sys.exit()
        self.run_log.info("**** stage2 select generations: {} ****\n".format(self.second_generation_tracedict.keys()))
        return self.second_generation_tracedict

    def get_train_fun(self, model_name):
        train_fun_dict = {
            'vgg16': start_imageclassification_train,
            'resnet': start_imageclassification_train,
            'mobilenetv2': start_imageclassification_train,
            'vit': start_imageclassification_train,
            'yolov3': start_yolov3_train,
            'yolov4': start_yolov4_train,
            'yolov5': start_yolov5_train,
            'retinaface': start_retinaface_train,
            'SSDmobilenetv1': start_ssd_train,
            'SSDvgg16': start_ssd_train,
            'SSDmobilenetv2': start_ssd_train,
            'SSDmobilenetv1fpn': start_ssd_train,
            'SSDresnet50fpn': start_ssd_train,
            'unet': start_unet_train,
            'unetplus': start_unet_train,
            "fasttext": start_fasttext_train,
            "textcnn": start_textcnn_train,
            "deeplabv3": start_deeplab_train,
            "deeplabv3plus": start_deeplab_train,
            "unet3d": start_unet3d_train,
            "transformer": start_transformer_train,
            "bert": start_bert_train,
            "maskrcnn": start_maskrcnn_train,
            "fasterrcnn": start_fasterrcnn_train,
            "pangu": start_pangu_train,
        }
        return train_fun_dict[model_name]

    def mutation_model_train(self):

        if self.selected_generation is None:
            second_select_generations_str = list(self.second_generation_tracedict.keys())
            second_select_generations = [int(val) for val in second_select_generations_str]
        else:
            second_select_generations = self.selected_generation

        second_select_generations = np.sort(second_select_generations)

        for generation in second_select_generations:
            model_ms_origin, model_torch_origin = get_model(self.model_name, self.device_target,
                                                            device_id=self.device_id,
                                                            input_size=tuple((self.batch_size,) + self.input_size[1:]))

            cur_generation_trace = self.second_generation_tracedict[str(generation)]

            model_ms_mutation = analyze_log_mindspore_followtrace(deepcopy(cur_generation_trace), model_ms_origin,
                                                                  self.mut_log_path,
                                                                  (self.batch_size,) + self.input_size[1:])
            rename_parameter(model_ms_mutation)  # rename the name of all Parameters
            model_torch_mutation = analyze_log_torch_followtrace(deepcopy(cur_generation_trace), model_torch_origin,
                                                                 self.mut_log_path,
                                                                 (self.batch_size,) + self.input_size[1:])

            # Enter train compare stage
            self.run_log.info(
                "==================== start training {} generation mutation model ====================".format(
                    generation))
            self.run_log.info("{} generation mutation trace: {}".format(generation, cur_generation_trace))
            train_log = self.log_path + '/' + str(generation) + 'mutation_model_train.log'

            print("Enter Train Stage")
            start_model_train = self.get_train_fun(self.model_name)
            start_model_train(model_ms_mutation, model_torch_mutation, self.train_config, train_log)


def get_arg_opt():
    parser = argparse.ArgumentParser(description='network generalization')
    parser.add_argument('--model', type=str, choices=['bert', 'deeplabv3', 'deeplabv3plus', 'fasterrcnn', 'fasttext',
                                                      'maskrcnn', 'mobilenetv2', 'resnet', 'retinaface',
                                                      'SSDmobilenetv1', 'SSDmobilenetv1fpn', 'SSDmobilenetv2',
                                                      'SSDresnet50fpn', 'SSDvgg16', 'textcnn', 'transformer', 'unet',
                                                      'unet3d', 'unetplus', 'vgg16', 'vit', 'yolov3', 'yolov4',
                                                      'yolov5'], required=True, help='model name')
    parser.add_argument('--dataset_path', type=str, required=True, default=None, help='dataset path')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--mutation_iterations', type=int, default=None, help='network mutation iterations')
    parser.add_argument('--selected_model_num', type=int, default=None, help='number of selected mutation models')
    parser.add_argument('--mutation_type', type=str, nargs='+', default=['LD', 'PM', 'LA', 'RA', 'CM'],
                        help='mutation type: one or more')  # choices=['LD', 'PM', 'LA', 'RA', 'CM'],
    parser.add_argument('--mutation_strategy', type=str, default='random', choices=['random', 'MCMC'],
                        help='mutation strategy')
    parser.add_argument('--mutation_log', type=str, default=None, help='the path of mutation log')
    parser.add_argument('--selected_generation', type=int, default=None, help='specify generation of mutation')
    return parser.parse_args()


if __name__ == '__main__':
    # args_opt = get_arg_opt()

    args_opt = argparse.Namespace(
        model=model_name,
        dataset_path=r"/data1/pzy/mindb/LUNA16",
        batch_size=8,
        epoch=10,
        mutation_iterations=10,
        selected_model_num=2,
        mutation_type=["PM","LA","RA","CM"],
        mutation_log='/data1/myz/netsv/log',
        selected_generation=None,
        mutation_strategy="random"
    )

    mutate = NetworkGeneralization(args=args_opt)
    mutate.run_log.info("Stage1: start network mutation!")
    mutate.mindspore_mutation()
    mutate.run_log.info('Stage2: select models by mutation score')
    mutate.diff_calculate()
    mutate.run_log.info('Stage3: train mutation model with large output variance between MindSpore and Pytorch')
    mutate.mutation_model_train()
