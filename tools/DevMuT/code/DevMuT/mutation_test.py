import os
import sys
import yaml
import argparse
import numpy as np
from datetime import datetime
from copy import deepcopy
import torch
import mindspore
from mindspore import context
from mindspore.ops import operations as P
from common.dataset_utils import get_dataset
from common.model_utils import get_model
from common.log_recoder import Logger
from common.mutation_ms.mutator_selection_logic import Roulette, MCMC, doubleq_state
from common.mutation_torch.mutation_main_followlog import analyze_log_torch_followtrace,check_ms_failed_trace
from common.mutation_ms.model_mutation_operators_followlog import analyze_log_mindspore_followtrace
from common.mutation_ms.model_mutation_generators import generate_model_by_model_mutation
from common.help_utils import rename_parameter
from common.help_utils import ToolUtils
from common.help_utils import YoloUtil
from common.help_utils import get_filter_data
from common.model_train import get_model_train
import time
from utils.util import QNetwork
from utils.util import check_illegal_mutant

class NetworkGeneralization:
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model

        time_tuple = time.localtime(time.time())
        time_stamp = "{}.{}.{}.{}.{}.{}".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                                time_tuple[4], time_tuple[5])

        self.log_path = args.mutation_log + "/" + str(self.model_name) + "-" + time_stamp
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.mut_log_path = self.log_path + '/mutation.txt'
        self.run_log = self.init_logging()
        self.run_log.log_file = self.log_path + '/run.log'
        self.true_log_path = self.log_path + '/run.txt'
        # self.run_log.log_file = '/data1/czx/net-sv/log/test.log'
        # self.run_log = Logger(log_file='/data1/czx/net-sv/log/test.log')
        if 'CONTEXT_DEVICE_TARGET' not in os.environ:
            self.run_log.warning("Please use 'export CONTEXT_DEVICE_TARGET' to set device target, "
                                 "the default device is CPU if not set")
            os.environ['CONTEXT_DEVICE_TARGET'] = 'CPU'
        elif os.environ['CONTEXT_DEVICE_TARGET'] not in ['CPU', 'GPU', 'Ascend']:
            err_msg = 'CONTEXT_DEVICE_TARGET={} is not expected, please choose in GPU/CPU/Ascend'.format(
                os.environ['CONTEXT_DEVICE_TARGET'])
            raise TypeError(err_msg)

        if os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU' and 'CUDA_VISIBLE_DEVICES' in os.environ:
            self.run_log.warning("For PyTorch and MindSpore running separately, CUDA_VISIBLE_DEVICES should "
                                 "have at least two GPUs, such as 'export CUDA_VISIBLE_DEVICES=0,1'")

        config_path = os.getcwd() + '/config/' + self.model_name + '.yaml'
        with open(config_path, 'r', encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.dataset_path = args.dataset_path
        self.mutation_strategy = args.mutation_strategy
        self.mutation_type = args.mutation_type
        self.train_config = config['train_config']
        if args.batch_size:
            self.train_config['batch_size'] = args.batch_size
        self.batch_size = int(self.train_config['batch_size'])
        if args.epoch:
            self.train_config['epoch'] = args.epoch
        self.epoch = int(self.train_config['epoch'])
        if args.mutation_iterations:
            config['mutation_config']['mutation_iterations'] = args.mutation_iterations
        self.mutation_iterations = int(config['mutation_config']['mutation_iterations'])
        self.selected_model_num = args.selected_model_num if args.selected_model_num \
            else int(np.ceil(self.mutation_iterations/2))

        if self.mutation_iterations < self.selected_model_num:
            err_msg = 'the number of selected mutation model should no more than mutation iteration, while ' \
                      'selected_model_num is {} and mutation_iteration is {}.'.format(self.selected_model_num,
                                                                                      self.mutation_iterations)
            raise ValueError(err_msg)

        device_id = int(os.environ['CUDA_VISIBLE_DEVICES'].split(",")[1])
        if 'CONTEXT_MODE' in os.environ and os.environ['CONTEXT_MODE'] == 'GRAPH_MODE':
            context.set_context(mode=context.GRAPH_MODE, device_target=os.environ['CONTEXT_DEVICE_TARGET'],
                                device_id=device_id)
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=os.environ['CONTEXT_DEVICE_TARGET'],
                                device_id=device_id)

        self.dataset = get_dataset(dataset_name=config['train_config']['dataset_name'])
        self.train_config['dataset_path'] = self.dataset_path
        self.mutation_eval_metric = config['mutation_config']['mutation_eval_metric']
        self.first_generation_tracedict = {}
        self.selected_generation = args.selected_gen
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
        self.mindspore_pass_rate = 0

        config['mutation_config'].update({'log_path': self.log_path, 'mutation_strategy': self.mutation_strategy,
                                          'mutation_type': self.mutation_type,
                                          'selected_model_num': self.selected_model_num})
        config['train_config'].update({'device': os.environ['CONTEXT_DEVICE_TARGET']})
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            config['train_config'].update({'device_id': os.environ['CUDA_VISIBLE_DEVICES']})
        f = open(self.true_log_path, 'a+')
        f.write("Mutation config:\n-------------------------------------------------------"
                          "---------------------------------\n{}\n".format
                          ("\n".join([f"{key}: {value}" for key, value in config.items()])))
        f.close()
        self.run_log.info("Mutation config:\n-------------------------------------------------------"
                          "---------------------------------\n{}\n".format
                          ("\n".join([f"{key}: {value}" for key, value in config.items()])))

    def init_logging(self):
        log = Logger(log_file=self.log_path + '/run.log')
        return log.logger

    def cal_mutation_score(self, model, data_forcal, origin_outputs):
        if self.mutation_eval_metric == "origin_diff":
            if not (isinstance(origin_outputs[0], tuple) or isinstance(origin_outputs[0], list)):
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

                    if 'yolo' in self.model_name or "openpose" in self.model_name:
                        origin_outputs = YoloUtil.reformat_outputs_first_generation(origin_outputs)
                        mutation_outputs = YoloUtil.reformat_outputs_first_generation(mutation_outputs)

                out_diffs = []
                for idx in range(len(mutation_outputs)):
                    self.origin_output = origin_outputs[idx]
                    self.mutation_output = mutation_outputs[idx]

                    for idj in range(len(self.origin_output)):
                        distance_vec = self.origin_output[idj] - self.mutation_output[idj]
                        distance_vec = mindspore.ops.flatten(distance_vec)

                        out_diff = np.linalg.norm(mindspore.Tensor(distance_vec).asnumpy())
                        out_diffs.append(out_diff)

            score = np.mean(out_diffs)
        return score

    def mindspore_mutation(self):
        origin_model_ms = get_model(self.model_name, input_size=tuple(self.input_size),only_ms=True)
        test_set = self.dataset(data_dir=self.dataset_path, batch_size=self.data_forcal_size, is_train=True)
        test_iter = test_set.create_tuple_iterator(output_numpy=True)

        imgs_ms_forcal = get_filter_data(self.model_name, test_iter, self.data_forcal_size, self.train_config['dtypes'])

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
            self.MCMC_mutate(imgs_ms_forcal, origin_outputs)
        elif self.mutation_strategy == "ddqn":
            self.doubleq_mutate(imgs_ms_forcal, origin_outputs)

        mut_succ = os.popen(f"grep -c mut_result:True {self.mut_log_path}").readlines()[0]
        muttypes = os.popen(f"grep -aF 'Adopt' {self.mut_log_path}").readlines()
        mutresults = os.popen(f"grep -aF 'mut_result:' {self.mut_log_path}").readlines()
        muttype_count1 = dict(zip(self.mutation_type, [0] * len(self.mutation_type)))
        muttype_count2 = dict(zip(self.mutation_type, [0] * len(self.mutation_type)))
        for i in range(len(muttypes)):
            muttype = muttypes[i].split(' ')[1]
            muttype_count1[muttype] += 1
            mutresult = mutresults[i].split(':')[1].split('\n')[0]
            if mutresult == 'True' or mutresult is True:
                muttype_count2[muttype] += 1
        f = open(self.true_log_path, 'a+')
        f.write('mutation iteration:{} ({}), mutation success:{} ({})\n'.format(
            self.mutation_iterations, muttype_count1, mut_succ, muttype_count2))
        f.close()
        self.run_log.info('mutation iteration:{} ({}), mutation success:{} ({})\n'.format(
            self.mutation_iterations, muttype_count1, mut_succ, muttype_count2))


    def doubleq_mutate(self, imgs_ms_forcal, origin_outputs):

        self. origin_outputs = origin_outputs

        self.mutation_outputs = []
        input_data = [imgs_ms_forcal[0:(0 + self.test_size), :]]
        model_dtypes_ms = [input_data[0].dtype]


        if os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
            device = devices[-2]
            final_device = 'cuda:' + device
        else:
            final_device = 'cpu'

        if isinstance(origin_outputs, list):
            model_output4q = origin_outputs[0][0:1,:]
        else:
            model_output4q = origin_outputs[0][0:1, :]
        q1, q2 = QNetwork(model_output4q.shape[-1], len(self.mutation_type)).to(final_device), QNetwork(model_output4q.shape[-1], len(self.mutation_type)).to(final_device)

        eplison = 0.3
        alpha = 0.1
        gamma = 0.9

        loss_fun = torch.nn.MSELoss(reduction='none').to(final_device)
        optimizer1 = torch.optim.Adam(q1.parameters(), lr=1e-4)
        optimizer2 = torch.optim.Adam(q2.parameters(), lr=1e-4)
        mutation_scores = []

        net_ms_seed = get_model(self.model_name, input_size=tuple(self.input_size), only_ms=True)


        self.mutants_info = {self.model_name + "_seed": doubleq_state(self.model_name + "_seed", 0, 0, {k: 0 for k in self.mutation_type})}
        self.trace_info = {self.model_name + "_seed":self.model_name + "_seed"}
        current_seed_name = self.model_name + "_seed"

        self.run_log.info("**** Start DoubleQ Learning ****\n")
        self.loss =[]

        for generation in range(1, self.mutation_iterations + 1):
            self.run_log.info("================== start {} generation!({}/{}) ==================".format(generation, generation, self.mutation_iterations))
            self.trace_info[self.model_name+"_"+str(generation)] = current_seed_name
            p = np.random.rand(1)[0]
            if p <= eplison:
                mut_type = np.random.permutation(self.mutation_type)[0]
            else:
                origin_q_input = torch.tensor(model_output4q.asnumpy(), dtype=torch.float32).to(final_device)
                with torch.no_grad():
                    q1_value, q2_value = q1(origin_q_input), q2(origin_q_input)
                q1q2 = (q1_value + q2_value) / 2
                mut_type = self.mutation_type[np.argmax(q1q2.detach().cpu().numpy())]


            mut_result = generate_model_by_model_mutation(net_ms_seed, mut_type, self.input_size, self.mut_log_path, generation, self.run_log, self.train_config)

            self.mutants_info[current_seed_name].selected = self.mutants_info[current_seed_name].selected+1
            self.mutants_info[current_seed_name].mutator_dict[mut_type] = self.mutants_info[current_seed_name].mutator_dict[mut_type] + 1


            if mut_result == 'True' or mut_result is True:
                r = self.cal_mutation_score(net_ms_seed, imgs_ms_forcal, origin_outputs)

                check_result = check_illegal_mutant(net_ms_seed, self.mutation_outputs)
                if not check_result:
                    r = -1
                    self.mutants_info[current_seed_name].reward = r
                    maxuct = -99999
                    maxuct_idx = -1
                    for mut_model_name in list(self.mutants_info.keys()):
                        val = self.mutants_info[mut_model_name]
                        val_score = val.score(mut_type)
                        if val_score > maxuct:
                            maxuct = val_score
                            maxuct_idx = mut_model_name

                    current_seed_name = self.mutants_info[maxuct_idx].name

                    orignial_model = get_model(self.model_name, input_size=tuple(self.input_size), only_ms=True)

                    father_name = deepcopy(current_seed_name)
                    if "seed" in current_seed_name:
                        execution_traces = ["seed"]
                    else:
                        execution_traces = [int(father_name.split("_")[1])]
                    while True:
                        father_name = self.trace_info[father_name]
                        if "_seed" in father_name:
                            break
                        trace_singledot = int(father_name.split("_")[1])
                        execution_traces.append(trace_singledot)

                    execution_traces.sort()
                    assert len(execution_traces) > 0
                    assert net_ms_seed is not None
                    if "seed" not in current_seed_name:
                        net_ms_seed = analyze_log_mindspore_followtrace(execution_traces, orignial_model,self.mut_log_path, self.input_size,self.train_config)
                    mutation_scores.append(-100)

                else:
                    mutants_names = list(self.mutants_info.keys())
                    assert not self.model_name + "_" + str(generation) in mutants_names
                    mutator_dict = {k: 0 for k in self.mutation_type}
                    mutant_info = doubleq_state(self.model_name + "_" + str(generation), 0, r, mutator_dict)

                    self.mutants_info[self.model_name + "_" + str(generation)] = mutant_info
                    current_seed_name = self.model_name + "_" + str(generation)
                    mutation_scores.append(r)

                    # if generation % 2 == 0:
                    #     try:
                    #         model_json = ms_model2json(net_ms_seed, input_data, model_dtypes_ms)
                    #         with open(self.log_path + "/mut_model_json{}.json".format(generation), 'w',
                    #                   encoding='utf-8') as json_file:
                    #             json.dump(model_json[0], json_file, ensure_ascii=False, indent=4)
                    #         json_file.close()
                    #         del model_json
                    #     except Exception as e:
                    #         self.run_log.info(str(e) + "\n")

            else:
                r = -1
                self.mutants_info[current_seed_name].reward = r
                maxuct = -99999
                maxuct_idx = -1
                for mut_model_name in list(self.mutants_info.keys()):
                    val = self.mutants_info[mut_model_name]
                    val_score = val.score(mut_type)
                    if val_score > maxuct:
                        maxuct = val_score
                        maxuct_idx = mut_model_name

                current_seed_name = self.mutants_info[maxuct_idx].name

                orignial_model = get_model(self.model_name, input_size=tuple(self.input_size), only_ms=True)

                father_name = deepcopy(current_seed_name)
                if "seed" in current_seed_name:
                    execution_traces = ["seed"]
                else:
                    execution_traces = [int(father_name.split("_")[1])]
                while True:
                    father_name = self.trace_info[father_name]
                    if "_seed" in father_name:
                        break
                    trace_singledot = int(father_name.split("_")[1])
                    execution_traces.append(trace_singledot)

                execution_traces.sort()
                assert len(execution_traces) > 0
                assert net_ms_seed is not None
                if "seed" not in current_seed_name:
                    net_ms_seed = analyze_log_mindspore_followtrace(execution_traces, orignial_model, self.mut_log_path,self.input_size, self.train_config)
                mutation_scores.append(-100)

            if len(self.mutation_outputs) == 0:
                if isinstance(origin_outputs, list):
                    model_output4q = deepcopy(origin_outputs[0][0:1, :])
                else:
                    model_output4q = deepcopy(origin_outputs[0][0:1, :])


            p = np.random.rand(1)[0]
            q_input = torch.tensor(model_output4q.asnumpy(), dtype=torch.float32).to(final_device)
            self.q_input = q_input
            if p <= 0.5:
                q_values = q1(q_input).detach().cpu().numpy()
                idx = np.argmax(q_values)
                next_q_values = q2(q_input).detach().cpu().numpy()[0][idx]
                target_q_values = r + (gamma * next_q_values)
                q_values = max(q_values) + alpha * (target_q_values - q_values)
                q_values, target_q_values = torch.tensor([np.max(q_values)],dtype=torch.float32,requires_grad=True).to(final_device),torch.tensor([np.max(target_q_values)],dtype=torch.float32,requires_grad=True).to(final_device)
                loss = loss_fun(q_values, target_q_values)
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                self.loss.append(loss.detach().cpu().numpy())
                del loss
                del next_q_values
                del q_values
            else:
                q_values = q2(q_input).detach().cpu().numpy()
                idx = np.argmax(q_values)
                next_q_values = q1(q_input).detach().cpu().numpy()[0][idx]
                target_q_values = r + (gamma * next_q_values)
                q_values = max(q_values) + alpha * (target_q_values - q_values)
                q_values, target_q_values = torch.tensor([np.max(q_values)],dtype=torch.float32,requires_grad=True).to(final_device),torch.tensor([np.max(target_q_values)],dtype=torch.float32,requires_grad=True).to(final_device)
                loss = loss_fun(q_values, target_q_values)
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()
                self.loss.append(loss.detach().cpu().numpy())
                del loss
                del next_q_values
                del q_values


        self.run_log.info("mutation_scores: {}".format(mutation_scores))
        first_select_generations = np.argsort(np.array(mutation_scores))[(len(mutation_scores) - self.selected_model_num):]
        self.run_log.info("**** stage1 select generations: {} ****\n".format(first_select_generations + 1))


        mut_trace = {}
        for gen in range(1, self.mutation_iterations + 1):
            start_gen = gen
            execution_traces = [start_gen]
            while True:
                father_name = self.trace_info[self.model_name + "_" + str(start_gen)]
                if father_name == self.model_name + "_seed":
                    break
                trace_singledot = int(father_name.split("_")[1])
                execution_traces.append(trace_singledot)
                start_gen = trace_singledot

            execution_traces.sort()
            mut_trace[str(gen)] = execution_traces

        self.total_trace_record = mut_trace
        f = open(self.mut_log_path, 'a+')
        f.write("mutation trace: {}\n".format(mut_trace))
        f.close()

        first_select_generations = list(map(str, first_select_generations + 1))
        self.first_generation_tracedict = {gen: trace for gen, trace in mut_trace.items()
                                           if gen in first_select_generations}
        self.first_scores = mutation_scores

        check_times = {k: 0 for k in self.mutation_type}
        for key in list(self.mutants_info.keys()):
            mutator_dictforcheck = self.mutants_info[key].mutator_dict
            for m in list(mutator_dictforcheck.keys()):
                check_times[m] = check_times[m] + mutator_dictforcheck[m]
        sums = 0
        for m in list(check_times.keys()):
            sums = sums + check_times[m]

        assert sums == self.mutation_iterations
        torch.save(q1.state_dict(), './q1_weight.pth')
        torch.save(q2.state_dict(), './q2_weight.pth')
        return self.first_generation_tracedict


    def random_mutate(self, origin_model_ms, imgs_ms_forcal, origin_outputs):
        mutation_scores = []
        for generation in range(1, self.mutation_iterations + 1):
            f = open(self.true_log_path, 'a+')
            f.write("================== start {} generation!({}/{}) ==================".format(generation, generation, self.mutation_iterations))
            f.close()
            self.run_log.info("================== start {} generation!({}/{}) ==================".format(generation, generation, self.mutation_iterations))
            mut_type = np.random.permutation(self.mutation_type)[0]
            mut_result = generate_model_by_model_mutation(origin_model_ms, mut_type, self.input_size, self.mut_log_path, generation, self.run_log, self.train_config)

            if mut_result == 'True' or mut_result is True:
                self.mindspore_pass_rate += 1
                mutate_score = self.cal_mutation_score(origin_model_ms, imgs_ms_forcal, origin_outputs)
                mutation_scores.append(mutate_score)
            else:
                mutation_scores.append(-100)
                execution_traces = [i for i in range(1, generation)]
                net_ms_seed = get_model(self.model_name, input_size=tuple(self.input_size),only_ms=True)
                net_ms = analyze_log_mindspore_followtrace(execution_traces, net_ms_seed, self.mut_log_path,
                                                           self.input_size, self.train_config)
                origin_model_ms = net_ms
        f = open(self.true_log_path, 'a+')
        f.write("mutation_scores: {}".format(mutation_scores))
        f.close()
        self.run_log.info("mutation_scores: {}".format(mutation_scores))
        first_select_generations = np.argsort(np.array(mutation_scores))[(len(mutation_scores) - self.selected_model_num):]
        f = open(self.true_log_path, 'a+')
        f.write("**** stage1 select generations: {} ****\n".format(first_select_generations + 1))
        f.close()
        self.run_log.info("**** stage1 select generations: {} ****\n".format(first_select_generations + 1))

        mut_trace = {str(i): list(range(1, i+1)) for i in range(1, self.mutation_iterations+1)}
        self.total_trace_record = mut_trace
        f = open(self.mut_log_path, 'a+')
        f.write("mutation trace: {}\n".format(mut_trace))
        f.close()
        first_select_generations = list(map(str, first_select_generations+1))
        self.first_generation_tracedict = {gen: trace for gen, trace in mut_trace.items()
                                           if gen in first_select_generations}
        self.first_scores = mutation_scores
        return self.first_generation_tracedict

    def MCMC_mutate(self, imgs_ms_forcal, origin_outputs):
        mcmc_help_info = {}  # Record  the current model evolved from which generation of seeds
        mutation_scores = []
        mutator_selector_func, mutant_selector_func = MCMC, Roulette
        mutate_ops = self.mutation_type
        mutate_op_history = {k: 0 for k in mutate_ops}
        mutate_num = self.mutation_iterations
        mutator_selector, mutant_selector = mutator_selector_func(mutate_ops), mutant_selector_func(
            [self.model_name + "_seed"], capacity=mutate_num + 1)

        last_used_mutator = None
        last_inconsistency = 0
        generation = 1
        while generation <= self.mutation_iterations:
            f = open(self.true_log_path, 'a+')
            f.write(
                "================== start {} generation!({}/{}) ==================".format(generation, generation,
                                                                                           self.mutation_iterations))
            f.close()
            self.run_log.info(
                "================== start {} generation!({}/{}) ==================".format(generation, generation,
                                                                                           self.mutation_iterations))
            new_seed_name = self.model_name + "_" + str(generation)

            net_ms_seed = get_model(self.model_name, input_size=tuple(self.input_size), only_ms=True)
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
                                                           self.input_size, self.train_config)
            # pick_seed is the father of current seed whose name is new_seed_name
            mcmc_help_info[new_seed_name] = picked_seed
            if net_ms is None:
                raise RuntimeError("seed model is None !")
            mut_result = generate_model_by_model_mutation(net_ms, selected_op, self.input_size, self.mut_log_path,
                                                          generation, self.run_log, self.train_config)

            if mut_result == "True" or mut_result is True:
                mutant.selected += 1
                mutator.total += 1

                accumulative_inconsistency = self.cal_mutation_score(net_ms, imgs_ms_forcal, origin_outputs)
                mutation_scores.append(accumulative_inconsistency)

                delta = accumulative_inconsistency - last_inconsistency
                mutator.delta_bigger_than_zero = mutator.delta_bigger_than_zero + 1 if delta > 0 else \
                    mutator.delta_bigger_than_zero
                if delta > 0:
                    if mutant_selector.is_full():
                        mutant_selector.pop_one_mutant()
                    mutant_selector.add_mutant(new_seed_name)
                    last_inconsistency = accumulative_inconsistency
                self.mindspore_pass_rate += 1
            else:
                mutation_scores.append(-100)
            generation += 1
        f = open(self.true_log_path, 'a+')
        f.write("mutation_scores: {}".format(mutation_scores))
        f.close()
        self.run_log.info("mutation_scores: {}".format(mutation_scores))
        first_select_generations = np.argsort(np.array(mutation_scores))[
                                   (len(mutation_scores) - self.selected_model_num):]
        f = open(self.true_log_path, 'a+')
        f.write("**** stage1 select generations: {} ****\n".format(first_select_generations + 1))
        f.close()
        self.run_log.info("**** stage1 select generations: {} ****\n".format(first_select_generations + 1))

        mut_trace = {}
        for gen in range(1, self.mutation_iterations+1):
            start_gen = gen
            execution_traces = [start_gen]
            while True:
                father_name = mcmc_help_info[self.model_name + "_" + str(start_gen)]
                if father_name == self.model_name + "_seed":
                    break
                trace_singledot = int(father_name.split("_")[1])
                execution_traces.append(trace_singledot)
                start_gen = trace_singledot
            execution_traces.sort()
            mut_trace[str(gen)] = execution_traces

        self.total_trace_record = mut_trace
        f = open(self.mut_log_path, 'a+')
        f.write("mutation trace: {}\n".format(mut_trace))
        f.close()

        first_select_generations = list(map(str, first_select_generations+1))
        self.first_generation_tracedict = {gen: trace for gen, trace in mut_trace.items()
                                           if gen in first_select_generations}
        self.first_scores = mutation_scores
        return self.first_generation_tracedict

    def diff_calculate(self):

        _, seed_model_torch = get_model(self.model_name, input_size=tuple(self.input_size))
        self.inconsistency_traces = {}
        if self.mutation_strategy == "random":
            self.inconsistency_traces = check_ms_failed_trace(seed_model_torch, self.mut_log_path, self.input_size, self.train_config, deepcopy(self.total_trace_record[str(self.mutation_iterations)]), mutate_logger=self.run_log)
        else:
            for key in list(self.total_trace_record.keys()):
                traces = self.total_trace_record[key]
                _, seed_model_torch = get_model(self.model_name, input_size=tuple(self.input_size))
                self.inconsistency_traces.update(check_ms_failed_trace(seed_model_torch, self.mut_log_path, self.input_size, self.train_config, deepcopy(traces), mutate_logger=self.run_log))

        del_traces = list(self.inconsistency_traces.keys())
        for trace in del_traces:
            if trace in list(self.first_generation_tracedict.keys()):
                self.first_generation_tracedict.pop(trace)

        # delete the inconsistency trace from the execution trace of each mutation model, like 1 is inconsistency 4 is composed of 1,2,3, 1 will be removed, thus 4 is composed as 2, 3
        first_generation_tracedict_keys = list(self.first_generation_tracedict.keys())
        for key in first_generation_tracedict_keys:
            key_traces = self.first_generation_tracedict[key]
            key_traces_new = []
            for key_trace in key_traces:
                if not key_trace in del_traces:
                    key_traces_new.append(key_trace)
            self.first_generation_tracedict[key] = deepcopy(key_traces_new)
        first_select_generations_str = list(self.first_generation_tracedict.keys())
        first_select_generations = [int(val) for val in first_select_generations_str]
        select_generations = np.sort(first_select_generations)

        test_set = self.dataset(data_dir=self.dataset_path, batch_size=self.data_forcal_size, is_train=True)
        test_iter = test_set.create_tuple_iterator(output_numpy=True)
        imgs_ms_forcal = get_filter_data(self.model_name, test_iter, self.data_forcal_size, self.train_config['dtypes'])


        if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
            device = devices[-2]
            final_device = "cuda:" + device
        else:
            final_device = 'cpu'

        for generation in select_generations:
            model_ms_origin, model_torch_origin = get_model(self.model_name, input_size=tuple(self.input_size))
            f = open(self.true_log_path, 'a+')
            f.write('compare mindspore and pytorch with selected mutation model: {}'.format(generation))
            f.close()
            self.run_log.info('compare mindspore and pytorch with selected mutation model: {}'.format(generation))

            cur_generation_trace = self.first_generation_tracedict[str(generation)]
            model_ms_mutation = analyze_log_mindspore_followtrace(deepcopy(cur_generation_trace), model_ms_origin,
                                                                  self.mut_log_path, self.input_size, self.train_config)
            rename_parameter(model_ms_mutation)
            model_torch_mutation = analyze_log_torch_followtrace(deepcopy(cur_generation_trace), model_torch_origin,
                                                                 self.mut_log_path, self.input_size, self.train_config)

            # calculate the output difference with torch model and mindspore model
            mut_out_diff_metric = []
            if isinstance(imgs_ms_forcal, list):  # mulptily inputs
                for idx in range(0, self.data_forcal_size+1, self.test_size):
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
                        # since torch.flatten is different from that of mindspore, we adopt torch.flatten
                        output_ms_array = torch.flatten(output_ms).detach().cpu().numpy()

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

                    if 'yolo' in self.model_name or 'openpose' in self.model_name:
                        output_ms = YoloUtil.reformat_outputs_second_generation(output_ms)
                        output_torch = YoloUtil.reformat_outputs_second_generation(output_torch)

                    if not (isinstance(output_ms, tuple) or isinstance(output_ms, list)):
                        output_ms = output_ms.asnumpy()
                        output_ms = torch.tensor(output_ms, dtype=output_torch.dtype)

                        output_torch_array = torch.flatten(output_torch).detach().cpu().numpy()
                        # since torch.flatten is different from that of mindspore, we adopt torch.flatten
                        output_ms_array = torch.flatten(output_ms).detach().cpu().numpy()

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
            f = open(self.true_log_path, 'a+')
            f.write("selected_mutation_model: {}, ms_torch_diff: {}, diff_threshold: {}".format(
                generation, np.mean(mut_out_diff_metric), self.threshold))
            f.close()
            self.run_log.info("selected_mutation_model: {}, ms_torch_diff: {}, diff_threshold: {}".format(
                generation, np.mean(mut_out_diff_metric), self.threshold))

        if len(self.second_generation_tracedict) == 0:
            f = open(self.true_log_path, 'a+')
            f.write("All selected mutation model check success")
            f.close()
            self.run_log.info("All selected mutation model check success")
            # sys.exit()
        return self.second_generation_tracedict

    def mutation_model_train(self):
        if self.second_generation_tracedict == {}:
            mut_trace = os.popen(f"grep -aF 'mutation trace:' {self.mut_log_path}").readlines()[0]
            self.second_generation_tracedict = eval(mut_trace.split('mutation trace: ')[1].split('\n')[0])
        if self.selected_generation:
            second_select_generations = self.selected_generation
        else:
            second_select_generations = [int(val) for val in list(self.second_generation_tracedict.keys())]
        second_select_generations = np.sort(second_select_generations)
        f = open(self.true_log_path, 'a+')
        f.write("The generation of mutation model to be trained: {}".format(second_select_generations))
        f.close()
        self.run_log.info("The generation of mutation model to be trained: {}".format(second_select_generations))

        for generation in second_select_generations:
            cur_generation_trace = self.second_generation_tracedict[str(generation)]
            model_ms_origin, model_torch_origin = get_model(self.model_name,
                                                            input_size=tuple((self.batch_size,) + self.input_size[1:]))
            model_ms_mutation = analyze_log_mindspore_followtrace(deepcopy(cur_generation_trace), model_ms_origin,
                                                                  self.mut_log_path,
                                                                  (self.batch_size,) + self.input_size[1:],
                                                                  self.train_config)
            # print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[",type(model_ms_mutation))
            rename_parameter(model_ms_mutation)  # rename the name of all Parameters
            model_torch_mutation = analyze_log_torch_followtrace(deepcopy(cur_generation_trace), model_torch_origin,
                                                                 self.mut_log_path,
                                                                 (self.batch_size,) + self.input_size[1:],
                                                                 self.train_config)

            # Enter train compare stage
            f = open(self.true_log_path, 'a+')
            f.write(
                "==================== start training {} generation mutation model ====================".format(
                    generation))
            f.close()
            self.run_log.info(
                "==================== start training {} generation mutation model ====================".format(
                    generation))
            f = open(self.true_log_path, 'a+')
            f.write("{} generation mutation trace: {}".format(generation, cur_generation_trace))
            f.close()
            self.run_log.info("{} generation mutation trace: {}".format(generation, cur_generation_trace))
            train_log_path = self.log_path + '/' + str(generation) + '_mutation_model'
            if not os.path.exists(train_log_path):
                os.makedirs(train_log_path)

            self.train_config['generation'] = generation
            start_model_train = get_model_train(self.model_name)
            start_model_train(model_ms_mutation, model_torch_mutation, self.train_config, self.run_log)


def get_arg_opt():
    """
    """

    parser = argparse.ArgumentParser(description='network generalization')
    parser.add_argument('--model', type=str, choices=['bert', 'deeplabv3', 'deeplabv3plus', 'fasterrcnn', 'fasttext',
                                                      'maskrcnn', 'mobilenetv2', 'resnet', 'retinaface',
                                                      'SSDmobilenetv1', 'SSDmobilenetv1fpn', 'SSDmobilenetv2',
                                                      'SSDresnet50fpn', 'SSDvgg16', 'textcnn', 'transformer', 'unet',
                                                      'unet3d', 'unetplus', 'vgg16', 'vit', 'yolov3', 'yolov4',
                                                      'yolov5', 'pangu', "openpose", "patchcore", "ssimae", "deepspeech2", "wide_and_deep", "gpt3","lstm"], required=True, help='model name')
    parser.add_argument('--dataset_path', type=str, required=True, default=None, help='dataset path')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--mutation_iterations', type=int, default=None, help='network mutation iterations')
    parser.add_argument('--selected_model_num', type=int, default=None, help='number of selected mutation models')
    parser.add_argument('--mutation_type', type=str, nargs='+',
                        default=['LD', 'PM', 'LA', 'RA', 'CM', 'SM', 'LS', 'LC'],
                        choices=['LD', 'PM', 'LA', 'RA', 'CM', 'SM', 'LS', 'LC', 'WS', 'NS', 'GF', 'NAI', 'NEB',"SM", "DM"],
                        help='mutation type: one or more')
    parser.add_argument('--mutation_strategy', type=str, default='random', choices=['random', 'MCMC'],
                        help='mutation strategy')
    parser.add_argument('--log_path', type=str, default='./log', help='the path of mutation log')
    parser.add_argument('--time_stamp', type=str, default=None)
    parser.add_argument('--mutation_log', type=str, default=None, help='the path of existing mutation log')
    parser.add_argument('--selected_gen', type=int, nargs='+', default=None, help='specify generation of mutation')
    return parser.parse_args()



if __name__ == '__main__':


    """
    export CONTEXT_DEVICE_TARGET=GPU
    export CUDA_VISIBLE_DEVICES=2,3
    """


    """
        resnet50 ./dataset/cifar10
        textcnn r"./dataset/rt-polarity"
        unet r"./dataset/ischanllge"
        ssimae ./dataset/MVTecAD/
    """


    model_name = "vgg16" # resnet50 unet unetplus vgg16 textcnn
    args_opt = argparse.Namespace(
        model=model_name,
        dataset_path=r"./dataset/cifar10",
        batch_size=2,
        epoch=5,
        mutation_iterations=10,
        selected_model_num=1,
        mutation_type=[ 'LD', 'PM', 'LA', 'RA', 'CM', 'SM', 'LC'],
        mutation_strategy="ddqn",
        mutation_log='/data1/czx/net-sv/common/log',
        selected_gen=None,
    )


    """
    batch_size > input_size[0] = test_size
    """
    
    mutate = NetworkGeneralization(args=args_opt)
    # input_size = (2, 3, 224, 224)
    # model1, model2 = get_model(model_name, input_size)
    # mutate.train_config['generation'] = 0
    # from common.model_train import start_textcnn_train
    # start_textcnn_train(model1, model2, mutate.train_config, mutate.run_log)
    f = open(mutate.true_log_path, 'a+')
    f.write("Stage1: start network mutation!")
    f.close()
    mutate.run_log.info("Stage1: start network mutation!")
    mutate.mindspore_mutation()
    f = open(mutate.true_log_path, 'a+')
    f.write('Stage2: select mutation models by calculating distance')
    f.close()
    mutate.run_log.info('Stage2: select mutation models by calculating distance')
    mutate.diff_calculate()
    f = open(mutate.true_log_path, 'a+')
    f.write('Stage3: train mutation model with large output variance between MindSpore and Pytorch')
    f.close()
    mutate.run_log.info('Stage3: train mutation model with large output variance between MindSpore and Pytorch')
    mutate.mutation_model_train()





