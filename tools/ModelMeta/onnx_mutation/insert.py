import random
import torch
import json
import platform
import datetime
from tqdm import tqdm
import os
from onnx_mutation.distance import *

from onnx_mutation.mutations import *
from onnx_mutation.cargo import *
from mutator_selection_logic import *
import utils
from arg import init_config, mutation_args
args = init_config(mutation_args)
from onnx_mutation.deadcode import DeadGenerator
import copy
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
from onnx_mutation.edge_node import *
from onnx_mutation.mutate_utils import *
from onnx_mutation.node_gen import make_node_chain_generator
import onnxruntime as ort
mutable_ops = ['call_module', 'root']
total_Mutate_time = 0

def select_places(sequence, k):
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            print("sequence is", sequence)
            # print("k is", k)
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        # print(chosen, subs_place)
        if max(chosen) != subs_place:
            # print("subs_place is", subs_place)
            # print("chosen is", chosen)
            return subs_place, chosen
    print("Cannot find suitable places")
    return None, None
    # raise Exception("Cannot find suitable places")

def check_node(node):
    if len(node.users) == 0:
        return False
    return True


def log_distance_to_file(f, i, dist, dist_chess):
    f.write("times: " + str(i) + " distance(final output)"
            + str(dist) + " distance_chess" + str(dist_chess) + "\n")
    if dist <= 1e-5 and dist_chess <= 1e-5:
        f.write(" Equal: True")
        print("Equal: True", flush=True)
    else:
        f.write(" Equal: False")
        print("Equal: False", flush=True)
        print("times: ", i, " distance(final output)", dist, " distance_chess", dist_chess, flush=True)


def insert_uoc(seed_model, all_edges, LOG_FLAG, log_dict, net, old_net, time_time, f, times, data0, depth=0):
    """在网络中插入节点"""
    mutate_logger = Logger(level='info')
    model_zoo = {"original_model": seed_model}
    # graph = module.graph
    # inputs = mindspore.Tensor(np.random.randn(1, 3, 32, 32), mindspore.float32)
    op_types = all_mutate_ops()
    # model = copy.copy(self.seed_model)\
    model = copy.deepcopy(seed_model)
    max_node_idx , max_edge_idx  = make_node_chain_generator(seed_model)
    mutate_op_history = {k: 0 for k in op_types}
    mutate_op_invalid_history = {k: 0 for k in op_types}
    mutant_history = []
    mutator_selector = MCMC(op_types)
    mutant_selector = Roulette(["original_model"], capacity=times + 1)
    last_used_mutator = None
    accumulative_inconsistency = 0
    mutant_counter = 0
    last_inconsistency = accumulative_inconsistency

    for i in tqdm(range(times), disable=True if depth > 0 else False):
        max_node_idx , max_edge_idx  = make_node_chain_generator(model)
        # op_type = op_types[random.randint(0, len(op_types) - 1)] if LOG_FLAG is False else log_dict[str(i)]['op_type']
        picked_seed = utils.ToolUtils.select_mutant(mutant_selector) if LOG_FLAG is False \
            else log_dict[str(i)]['picked_seed']
        if LOG_FLAG is False:
            log_dict[i] = {}
            log_dict[i]['picked_seed'] = picked_seed
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator) \
            if LOG_FLAG is False else log_dict[str(i)]['op_type']
        new_seed_name = "{}-{}{}".format(picked_seed[:-3], selected_op, mutate_op_history[selected_op])
        if LOG_FLAG is False and new_seed_name in mutant_selector.mutants.keys():
            print("skipping previous mutant")
            continue
        last_used_mutator = selected_op
        op_type = mutator_selector.mutators[selected_op] if LOG_FLAG is False else log_dict[str(i)]['op_type']
        mutant = mutant_selector.mutants[picked_seed]
        graph = model_zoo[mutant.name].graph
        model_list = []
        distance_list = []
        if LOG_FLAG is False:
            log_dict[i] = {}
            log_dict[i]['op_type'] = op_type.name

        if LOG_FLAG is False:
            log_dict[i]['mutate_type'] = "UOC"
        length = len(all_edges)
        
        if LOG_FLAG is False:
            subs_place, dep_places = select_places(range(0, length - 1), 5)
        else:
            subs_place, dep_places = log_dict[str(i)]['subs_place'], log_dict[str(i)]['dep_places']
        if LOG_FLAG is False:
            log_dict[i]['subs_place'], log_dict[i]['dep_places'] = subs_place, dep_places
        # seat = 0
        if depth > 0 and (subs_place is None or dep_places is None):
            return
        elif subs_place is not None and dep_places is not None:
            dep_places.sort(reverse=True)
            a = all_edges[dep_places[-1]]
            b = all_edges[dep_places[-2]]
            c = all_edges[dep_places[-3]]
            d = all_edges[dep_places[-4]]
            e = all_edges[subs_place]
            # print(a.name, b.name, c.name, d.name, e.name, op_type.name)
            # max_node_idx
            new_uoc = UOC(op_type, max_node_idx , max_edge_idx)
            edges = new_uoc.run_uoc(a, b, c, d, e) # 

            subs_node = e.def_node

            all_edges.remove(e)
            target_node_name = e.name
            for node in model.graph.node:
                if target_node_name in node.output:
                    model.graph.node.remove(node)
                    break
            # model.graph.node.remove(subs_node)

            new_onnx_edges = convert_edge_to_value_info(edges)

            for onnx_edge in new_onnx_edges:

                model.graph.value_info.append(onnx_edge)
            new_onnx_nodes = retrieve_node_from_edges(edges)
            # print('2222222222222222222222222222222222222222')
            for item in reversed(new_onnx_nodes):
                if item != None:
                    # print(item)
                    model.graph.node.insert(subs_place, item)

            for item in reversed(edges):
                if item != None:
                    all_edges.insert(subs_place, item)

            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) + "/uoc/" + "/" + str(time_time) + "/ONNX/"
            # # 动态推理
            # for node in model.graph.node:
            #     # print(node)
            #     if node.op_type == 'AveragePool':
            #         print(node.input)
                # if node.op_type == 'Pad':
                #     print(node.output)
            
            # onnx.checker.check_model(model)
            # print('动态推理')

            # if ort_inputs != None:
            #     print('MT SUCCESS')
            # else:
            #     print('Field')
            # if int(i) % 100 == 0:
            print('动态推理')
            # ort_session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
            # new_outputs = {ort_session.get_inputs()[0].name: data0}
             #print(new_outputs)
            onnx.save(model, os.path.join(model_dir, "%d.onnx" % i))
                # print(model_dir)

        else:
            log_dict[i]['status'] = "Failed"
            print("depth", depth, "mutate failed for Cannot find suitable places", flush=False)

        if LOG_FLAG is False:
            log_dict[i]['status'] = "Success"
        with open("onnx_mutated_net/" + str(net.__class__.__name__) + "/uoc/" + str(time_time) + "/LOG_DICT_"
                  + str(platform.system()) + "_" + str(device).replace(':', '_') + ".json", 'w',
                  encoding='utf-8') as file:
            
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

        # if depth == 0:
        #     model_path = os.path.join(model_dir, "%d.onnx" % i)
        #     model_run = onnx.load(model_path)
        #     # onnx.checker.check_model(model)

        #     print('动态推理')
        #     ort_session = ort.InferenceSession(model_run.SerializeToString(), providers=["CUDAExecutionProvider"])
        #     new_outputs = {ort_session.get_inputs()[0].name: data0}

        #     ort_session1 = ort.InferenceSession(seed_model.SerializeToString(), providers=["CUDAExecutionProvider"])
        #     outputs = {ort_session1.get_inputs()[0].name: data0}


        #     # fpy = open("torch_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/"
        #     #            + new_net.__class__.__name__
        #     #            + str(platform.system()) + "_" + str(device).replace(':', '_') + "_" + str(i) + "times" + ".txt",
        #     #            "w")
        #     # fpy.write(str(module.code))
        #     # fpy.close()
        #     # print("new_net", new_net, flush=True)
        #     # import inspect
        #     # def print_init_code(obj):
        #     #     init_method = obj.__class__.__init__
        #     #     source_code = inspect.getsource(init_method)
        #     #     print(source_code)
        #     #
        #     # print_init_code(new_net)
        #     # print("new_outputs", new_outputs.shape)
        #     if isinstance(outputs, tuple):
        #         print("handling tuple")

        #         # old_avg, new_avg = handle_tuple(outputs), handle_tuple(new_outputs)
        #         dist = distance(outputs, new_outputs)
        #         dist_chess = ChebyshevDistance(outputs, new_outputs)

        #         log_distance_to_file(i, dist, dist_chess)

        #     elif isinstance(outputs, torch.Tensor):
        #         old_out = outputs
        #         # old_out_np = old_out.asnumpy()
        #         # new_out_np = new_outputs.asnumpy()
        #         dist = distance(old_out, new_outputs)
        #         dist_chess = ChebyshevDistance(old_out, new_outputs)
        #         # if np.isnan(dist) or np.isnan(dist_chess):
        #         #     print("dist or dist_chess is nan", flush=True)
        #         #     print(new_net)
        #         #     exit(-10086)
        #         log_distance_to_file(i, dist, dist_chess)

        #     else:
        #         print("len(outputs): ", len(outputs))
        #         print("type(outputs): ", type(outputs))
        #         print("outputs: ", outputs.shape)
        #         raise NotImplementedError("len(outputs) > 2 or len(outputs) == 1 TUPLE!!!")

        #     distance_list.append(dist_chess)
        #     model_list.append(model_run)
        #     model_zoo[new_seed_name] = model_run
        #     if new_seed_name not in mutant_selector.mutants.keys():
        #         mutate_st = datetime.datetime.now()
        #         mutate_status = 0 if log_dict[i]['status'] == "Success" else -1
        #         mutate_et = datetime.datetime.now()
        #         mutate_dt = mutate_et - mutate_st
        #         h, m, s = utils.ToolUtils.get_HH_mm_ss(mutate_dt)
        #         mutate_logger.info("INFO:Mutate Time Used on {} : {}h, {}m, {}s".format(selected_op, h, m, s))
        #         # mutation status code is successful
        #         if mutate_status == 0:
        #             mutant.selected += 1
        #             op_type.total += 1
        #             # execute this model on all platforms
        #             accumulative_inconsistency = dist_chess

        #             mutant_history.append(new_seed_name)

        #             delta = accumulative_inconsistency - last_inconsistency

        #             op_type.delta_bigger_than_zero = op_type.delta_bigger_than_zero + 1 \
        #                 if delta > 0 else op_type.delta_bigger_than_zero

        #             if delta > 0:
        #                 if mutant_selector.is_full():
        #                     mutant_selector.pop_one_mutant()
        #                 mutant_selector.add_mutant(new_seed_name)
        #                 last_inconsistency = accumulative_inconsistency

        #             mutate_logger.info("SUCCESS:{} pass testing!".format(new_seed_name))
        #             mutant_counter += 1
        #         else:
        #             mutate_logger.error("Exception raised when mutate {} with {}".format(picked_seed, selected_op))

        #         mutate_logger.info("Mutated op used history:")
        #         mutate_logger.info(mutate_op_history)

        #         mutate_logger.info("Invalid mutant generated history:")
        #         mutate_logger.info(mutate_op_invalid_history)


def insert_ABSOC_A(seed_model, all_edges, LOG_FLAG, log_dict, net, old_net, time_time, f, times, data0, depth=0):
    """在网络中插入节点"""
    # graph = module.graph
    # inputs = mindspore.Tensor(np.random.randn(1, 3, 32, 32), mindspore.float32)
    op_types = all_mutate_ops()
    model = copy.deepcopy(seed_model)
    max_node_idx , max_edge_idx  = make_node_chain_generator(seed_model)
    mutator_selector = MCMC(op_types)
    last_used_mutator = None
    saving_frequency = args.save_freq
    for i in tqdm(range(times), disable=True if depth > 0 else False):
        max_node_idx , max_edge_idx  = make_node_chain_generator(model)
        # op_type = op_types[random.randint(0, len(op_types) - 1)] if LOG_FLAG is False else log_dict[str(i)]['op_type']
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator) \
            if LOG_FLAG is False else log_dict[str(i)]['op_type']
        last_used_mutator = selected_op
        op_type = mutator_selector.mutators[selected_op] if LOG_FLAG is False else log_dict[str(i)]['op_type']

        if LOG_FLAG is False:
            log_dict[i] = {}
            log_dict[i]['op_type'] = op_type.name

        if LOG_FLAG is False:
            log_dict[i]['mutate_type'] = "ABSOC_A"
        length = len(all_edges)
        if LOG_FLAG is False:
            subs_place, dep_places = select_places(range(0, length - 1), 5)
        else:
            subs_place, dep_places = log_dict[str(i)]['subs_place'], log_dict[str(i)]['dep_places']
        if LOG_FLAG is False:
            log_dict[i]['subs_place'], log_dict[i]['dep_places'] = subs_place, dep_places
        # seat = 0
        if depth > 0 and (subs_place is None or dep_places is None):
            return
        elif subs_place is not None and dep_places is not None:
            dep_places.sort(reverse=True)
            a = all_edges[dep_places[-1]]
            b = all_edges[dep_places[-2]]
            c = all_edges[dep_places[-3]]
            d = all_edges[dep_places[-4]]
            e = all_edges[subs_place] 

            new_uoc = ABSOC_A(op_type, max_node_idx , max_edge_idx)
            # next_node = nodelist[dep_places[-4] + 1]
            # next_node = mindspore.rewrite.api.node.Node(next_node)
            edges = new_uoc.run_ABSOC_A(a, b, c, d, e) # 

            subs_node = e.def_node

            # all_edges.remove(e)
            target_node_name = e.name
            for node in model.graph.node:
                print(node.name,target_node_name)
                if target_node_name in node.output:
                    model.graph.node.remove(node)
                    break

            new_onnx_edges = convert_edge_to_value_info(edges[:-1])

            for onnx_edge in new_onnx_edges:
                model.graph.value_info.append(onnx_edge)
            new_onnx_nodes = retrieve_node_from_edges(edges)

            for item in reversed(new_onnx_nodes):
                if item != None:
                    model.graph.node.insert(subs_place, item)
                
            # for item in reversed(edges):
            #     if item != None:
            #         all_edges.insert(subs_place, item)
                        
            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) + "/ABSOC_A/" + "/" + str(time_time) + "/ONNX/"
            
            # # 动态推理
            # ort_session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            # input_shape = data0.shape
            # input_data = np.random.random(input_shape).astype(np.float32)

            # ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            # if ort_inputs != None:
            #     print('MT SUCCESS')
            # onnx.checker.check_model(model)
            onnx.save(model, os.path.join(model_dir, "%d.onnx" % i))


        else:
            log_dict[i]['status'] = "Failed"
            print("depth", depth, "mutate failed for Cannot find suitable places", flush=False)
        # module.recompile()
        if LOG_FLAG is False:
            log_dict[i]['status'] = "Success"
        with open("onnx_mutated_net/" + str(net.__class__.__name__) + "/ABSOC_A/" + str(time_time) + "/LOG_DICT_"
                  + str(platform.system()) + "_" + str(device).replace(':', '_') + ".json", 'w',
                  encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)
        # if depth == 0:
        #     new_net = module.to(device)
        #     fpy = open("torch_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/"
        #                + new_net.__class__.__name__
        #                + str(platform.system()) + "_" + str(device).replace(':', '_') + "_" + str(i) + "times" + ".txt",
        #                "w")
        #     fpy.write(str(module.code))
        #     fpy.close()
        #     outputs = old_net(*inputs)
        #     # print("outputs", outputs.shape)
        #     new_outputs = new_net(*inputs)
        #     # print("new_outputs", new_outputs.shape)
        #     if isinstance(outputs, tuple):
        #         old_avg, new_avg = handle_tuple(outputs), handle_tuple(new_outputs)
        #         # print("old_avg", old_avg.shape)
        #         # print("new_avg", new_avg.shape)
        #         dist = distance(old_avg, new_avg)
        #         dist_chess = ChebyshevDistance(old_avg, new_avg)
        #         log_distance_to_file(f, i, dist, dist_chess)

        #     elif isinstance(outputs, torch.Tensor):
        #         old_out: torch.tensor = outputs
        #         dist = distance(old_out, new_outputs)
        #         dist_chess = ChebyshevDistance(old_out, new_outputs)
        #         log_distance_to_file(f, i, dist, dist_chess)

        #     else:
        #         print("len(outputs): ", len(outputs))
        #         print("type(outputs): ", type(outputs))
        #         print("outputs: ", outputs.shape)
        #         raise NotImplementedError("len(outputs) > 2 or len(outputs) == 1 TUPLE!!!")
        #     args.distance_list.append(dist_chess)
        #     args.model_list.append(new_net)



def insert_ABSOC_B(seed_model, all_edges, LOG_FLAG, log_dict, net, old_net, time_time, f, times, data0, depth=0):
    """在网络中插入节点"""
    # graph = module.graph
    # inputs = mindspore.Tensor(np.random.randn(1, 3, 32, 32), mindspore.float32)
    op_types = all_mutate_ops()
    model = copy.deepcopy(seed_model)
    max_node_idx , max_edge_idx  = make_node_chain_generator(seed_model)
    mutator_selector = MCMC(op_types)
    last_used_mutator = None
    saving_frequency = args.save_freq
    for i in tqdm(range(times), disable=True if depth > 0 else False):
        max_node_idx , max_edge_idx  = make_node_chain_generator(model)
        # op_type = op_types[random.randint(0, len(op_types) - 1)] if LOG_FLAG is False else log_dict[str(i)]['op_type']
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator) \
            if LOG_FLAG is False else log_dict[str(i)]['op_type']
        last_used_mutator = selected_op
        op_type = mutator_selector.mutators[selected_op] if LOG_FLAG is False else log_dict[str(i)]['op_type']

        if LOG_FLAG is False:
            log_dict[i] = {}
            log_dict[i]['op_type'] = op_type.name

        if LOG_FLAG is False:
            log_dict[i]['mutate_type'] = "ABSOC_B"
        length = len(all_edges)
        if LOG_FLAG is False:
            subs_place, dep_places = select_places(range(0, length - 1), 5)
        else:
            subs_place, dep_places = log_dict[str(i)]['subs_place'], log_dict[str(i)]['dep_places']
        if LOG_FLAG is False:
            log_dict[i]['subs_place'], log_dict[i]['dep_places'] = subs_place, dep_places
        # seat = 0
        if depth > 0 and (subs_place is None or dep_places is None):
            return
        elif subs_place is not None and dep_places is not None:
            dep_places.sort(reverse=True)
            a = all_edges[dep_places[-1]]
            b = all_edges[dep_places[-2]]
            c = all_edges[dep_places[-3]]
            d = all_edges[dep_places[-4]]
            e = all_edges[subs_place] 

            new_uoc = ABSOC_B(op_type, max_node_idx , max_edge_idx)
            # next_node = nodelist[dep_places[-4] + 1]
            # next_node = mindspore.rewrite.api.node.Node(next_node)
            edges = new_uoc.run_ABSOC_B(a, b, c, d, e) # 

            subs_node = e.def_node

            # all_edges.remove(e)
            target_node_name = e.name
            for node in model.graph.node:
                print(node.name,target_node_name)
                if target_node_name in node.output:
                    model.graph.node.remove(node)
                    break

            new_onnx_edges = convert_edge_to_value_info(edges[:-1])

            for onnx_edge in new_onnx_edges:
                model.graph.value_info.append(onnx_edge)
            new_onnx_nodes = retrieve_node_from_edges(edges)

            for item in reversed(new_onnx_nodes):
                if item != None:
                    model.graph.node.insert(subs_place, item)
                
            # for item in reversed(edges):
            #     if item != None:
            #         all_edges.insert(subs_place, item)
                        
            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) + "/ABSOC_B/" + str(time_time) + "/ONNX/"
            
            # # 动态推理
            # ort_session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            # input_shape = data0.shape
            # input_data = np.random.random(input_shape).astype(np.float32)

            # ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            # if ort_inputs != None:
            #     print('MT SUCCESS')
            # onnx.checker.check_model(model)
            onnx.save(model, os.path.join(model_dir, "%d.onnx" % i))


        else:
            log_dict[i]['status'] = "Failed"
            print("depth", depth, "mutate failed for Cannot find suitable places", flush=False)
        # module.recompile()
        if LOG_FLAG is False:
            log_dict[i]['status'] = "Success"
        with open("onnx_mutated_net/" + str(net.__class__.__name__) + "/ABSOC_B/" + str(time_time) + "/LOG_DICT_"
                  + str(platform.system()) + "_" + str(device).replace(':', '_') + ".json", 'w',
                  encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)
        # if depth == 0:
        #     new_net = module.to(device)
        #     fpy = open("torch_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/"
        #                + new_net.__class__.__name__
        #                + str(platform.system()) + "_" + str(device).replace(':', '_') + "_" + str(i) + "times" + ".txt",
        #                "w")
        #     fpy.write(str(module.code))
        #     fpy.close()
        #     outputs = old_net(*inputs)
        #     # print("outputs", outputs.shape)
        #     new_outputs = new_net(*inputs)
        #     # print("new_outputs", new_outputs.shape)
        #     if isinstance(outputs, tuple):
        #         old_avg, new_avg = handle_tuple(outputs), handle_tuple(new_outputs)
        #         # print("old_avg", old_avg.shape)
        #         # print("new_avg", new_avg.shape)
        #         dist = distance(old_avg, new_avg)
        #         dist_chess = ChebyshevDistance(old_avg, new_avg)
        #         log_distance_to_file(f, i, dist, dist_chess)

        #     elif isinstance(outputs, torch.Tensor):
        #         old_out: torch.tensor = outputs
        #         dist = distance(old_out, new_outputs)
        #         dist_chess = ChebyshevDistance(old_out, new_outputs)
        #         log_distance_to_file(f, i, dist, dist_chess)

        #     else:
        #         print("len(outputs): ", len(outputs))
        #         print("type(outputs): ", type(outputs))
        #         print("outputs: ", outputs.shape)
        #         raise NotImplementedError("len(outputs) > 2 or len(outputs) == 1 TUPLE!!!")
        #     args.distance_list.append(dist_chess)
        #     args.model_list.append(new_net)

def insert_pioc(seed_model, all_edges, LOG_FLAG, log_dict, net, old_net, time_time, f, times, data0, depth=0):
    """在网络中插入节点"""
    # graph = module.graph

    # op_types = ['Add', 'Sub', 'Mul', 'Conv', 'Dense']
    op_types = all_mutate_ops()
    model = copy.deepcopy(seed_model)
    max_node_idx , max_edge_idx  = make_node_chain_generator(seed_model)
    mutator_selector = MCMC(op_types)
    last_used_mutator = None
    saving_frequency = args.save_freq
    for i in tqdm(range(times), disable=True if depth > 0 else False):
        max_node_idx , max_edge_idx  = make_node_chain_generator(model)
        # op_type = op_types[random.randint(0, len(op_types) - 1)] if LOG_FLAG is False else log_dict[str(i)]['op_type']
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator) \
            if LOG_FLAG is False else log_dict[str(i)]['op_type']
        last_used_mutator = selected_op
        op_type = mutator_selector.mutators[selected_op] if LOG_FLAG is False else log_dict[str(i)]['op_type']

        if LOG_FLAG is False:
            log_dict[i] = {}
            log_dict[i]['op_type'] = op_type.name

        # mutate_type = PIOC
        if LOG_FLAG is False:
            log_dict[i]['mutate_type'] = "PIOC"
        length = len(all_edges)
        if LOG_FLAG is False:
            subs_place, dep_places = select_places(range(0, length - 1), 5) 
        else:
            subs_place, dep_places = log_dict[str(i)]['subs_place'], log_dict[str(i)]['dep_places']
        if LOG_FLAG is False:
            log_dict[i]['subs_place'], log_dict[i]['dep_places'] = subs_place, dep_places
        if depth > 0 and (subs_place is None or dep_places is None):
            return
        elif subs_place is not None and dep_places is not None:
            dep_places.sort(reverse=True)
            a = all_edges[dep_places[-1]]
            b = all_edges[dep_places[-2]]
            c = all_edges[dep_places[-3]]
            d = all_edges[dep_places[-4]]
            e = all_edges[subs_place] 
            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/ONNX/"
            new_pioc = PIOC(op_type, max_node_idx , max_edge_idx, data0, model_dir, str(net.__class__.__name__))
            # seat = 0
            edges = new_pioc.run_PIOC(a, b, c, d, e)

            subs_node = e.def_node

            # all_edges.remove(e)
            target_node_name = e.name
            for node in model.graph.node:
                if target_node_name in node.output:
                    model.graph.node.remove(node)
                    break
            new_onnx_edges = convert_edge_to_value_info(edges[:-1])

            for onnx_edge in new_onnx_edges:
                model.graph.value_info.append(onnx_edge)
            new_onnx_nodes = retrieve_node_from_edges(edges)

            for item in reversed(new_onnx_nodes):
                if item != None:
                    model.graph.node.insert(subs_place, item)
                
            # for item in reversed(edges):
            #     if item != None:
            #         all_edges.insert(subs_place, item)
                        
            
            
            # # 动态推理
            # ort_session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            # input_shape = data0.shape
            # input_data = np.random.random(input_shape).astype(np.float32)

            # ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            # if ort_inputs != None:
            #     print('MT SUCCESS')
            # onnx.checker.check_model(model)
            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) +"/pioc/"  + "/" + str(time_time) + "/ONNX/"
            onnx.save(model, os.path.join(model_dir, "%d.onnx" % i))


        else:
            log_dict[i]['status'] = "Failed"
            print("depth", depth, "mutate failed for Cannot find suitable places")

        if LOG_FLAG is False:
            log_dict[i]['status'] = "Success"
        with open("onnx_mutated_net/" + str(net.__class__.__name__) + "/ABSOC_B/" + str(time_time) + "/LOG_DICT_"
                  + str(platform.system()) + "_" + str(device).replace(':', '_') + ".json", 'w',
                  encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)
#         if depth == 0:
#             new_net = module.to(device)
#             fpy = open("torch_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/"
#                        + new_net.__class__.__name__
#                        + str(platform.system()) + "_" + str(device).replace(':', '_') + "_" + str(i) + "times" + ".txt",
#                        "w")
#             fpy.write(str(module.code))
#             fpy.close()
#             # print(*inputs)
#             outputs = old_net(*inputs)
#             # print("outputs", outputs, flush=True)
#             new_outputs = new_net(*inputs)
#             # print("new_outputs", new_outputs, flush=True)
#             if isinstance(outputs, tuple):
#                 old_avg, new_avg = handle_tuple(outputs), handle_tuple(new_outputs)
#                 dist = distance(old_avg, new_avg)
#                 dist_chess = ChebyshevDistance(old_avg, new_avg)
#                 log_distance_to_file(f, i, dist, dist_chess)

#             elif isinstance(outputs, torch.Tensor):
#                 old_out = outputs
#                 dist = distance(old_out, new_outputs)
#                 dist_chess = ChebyshevDistance(old_out, new_outputs)
#                 log_distance_to_file(f, i, dist, dist_chess)

#             else:
#                 print("len(outputs): ", len(outputs))
#                 print("type(outputs): ", type(outputs))
#                 print("outputs: ", outputs.shape)
#                 raise NotImplementedError("len(outputs) > 2 or len(outputs) == 1 TUPLE!!!")
#             args.distance_list.append(dist_chess)
#             print('7987987')
#             print(args.distance_list)
#             args.model_list.append(new_net)
            


def insert_hybrid(seed_model, all_edges, LOG_FLAG, log_dict, net, old_net, time_time, f, times, data0, depth=0):
    op_types = all_mutate_ops()
    model = copy.deepcopy(seed_model)
    max_node_idx , max_edge_idx  = make_node_chain_generator(seed_model)
    mutator_selector = MCMC(op_types)
    last_used_mutator = None
    saving_frequency = args.save_freq
    for i in tqdm(range(times), disable=True if depth > 0 else False):
        max_node_idx , max_edge_idx  = make_node_chain_generator(model)
        # op_type = op_types[random.randint(0, len(op_types) - 1)] if LOG_FLAG is False else log_dict[str(i)]['op_type']
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator) \
            if LOG_FLAG is False else log_dict[str(i)]['op_type']
        last_used_mutator = selected_op
        op_type = mutator_selector.mutators[selected_op] if LOG_FLAG is False else log_dict[str(i)]['op_type']

        if LOG_FLAG is False:
            log_dict[i] = {}
            log_dict[i]['op_type'] = op_type.name

        # for node in graph.nodes:
        #     if node.op in mutable_ops:
        #         nodelist.append(node)
        # mutate_type = "ABSOC_A" if LOG_FLAG is False \
        #     else log_dict[i]['mutate_type']
        if LOG_FLAG is False:
            log_dict[i]['mutate_type'] = "hybrid"
        length = len(all_edges)
        if LOG_FLAG is False:
            subs_place, dep_places = select_places(range(0, length - 1), 5)
        else:
            subs_place, dep_places = log_dict[str(i)]['subs_place'], log_dict[str(i)]['dep_places']
        if LOG_FLAG is False:
            log_dict[i]['subs_place'], log_dict[i]['dep_places'] = subs_place, dep_places
        # seat = 0
        if depth > 0 and (subs_place is None or dep_places is None):
            return
        elif subs_place is not None and dep_places is not None:
            dep_places.sort(reverse=True)
            a = all_edges[dep_places[-1]]
            b = all_edges[dep_places[-2]]
            c = all_edges[dep_places[-3]]
            d = all_edges[dep_places[-4]]
            e = all_edges[subs_place] 
            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/ONNX/"
            new_uoc = Hybrid(op_type, max_node_idx , max_edge_idx, data0, model_dir, str(net.__class__.__name__), )
            # next_node = nodelist[dep_places[-4] + 1]
            # next_node = mindspore.rewrite.api.node.Node(next_node)
            edges = new_uoc.run_Hybrid(a, b, c, d, e) # 

            subs_node = e.def_node

            # all_edges.remove(e)
            target_node_name = e.name
            for node in model.graph.node:
                print(node.name,target_node_name)
                if target_node_name in node.output:
                    model.graph.node.remove(node)
                    break

            new_onnx_edges = convert_edge_to_value_info(edges[:-1])

            for onnx_edge in new_onnx_edges:
                model.graph.value_info.append(onnx_edge)
            new_onnx_nodes = retrieve_node_from_edges(edges)

            for item in reversed(new_onnx_nodes):
                if item != None:
                    model.graph.node.insert(subs_place, item)
                
            for item in reversed(edges):
                if item != None:
                    all_edges.insert(subs_place, item)
                        
            model_dir = "onnx_mutated_net/" + str(net.__class__.__name__) + "/Hybrid/"  + "/" + str(time_time) + "/ONNX/"
            
            # # 动态推理
            # ort_session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            # input_shape = data0.shape
            # input_data = np.random.random(input_shape).astype(np.float32)

            # ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            # if ort_inputs != None:
            #     print('MT SUCCESS')
            # onnx.checker.check_model(model)
            onnx.save(model, os.path.join(model_dir, "%d.onnx" % i))
        else:
            log_dict[i]['status'] = "Failed"
            print("depth", depth, "mutate failed for Cannot find suitable places", flush=False)

        if LOG_FLAG is False:
            log_dict[i]['status'] = "Success"
        with open("onnx_mutated_net/" + str(net.__class__.__name__) + "/Hybrid/" + str(time_time) + "/LOG_DICT_"
                  + str(platform.system()) + "_" + str(device).replace(':', '_') + ".json", 'w',
                  encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)
#         if depth == 0:
#             new_net = module.to(device)

#             fpy = open("torch_mutated_net/" + str(net.__class__.__name__) + "/" + str(time_time) + "/MUTANTS/"
#                        + new_net.__class__.__name__
#                        + str(platform.system()) + "_" + str(device).replace(':', '_') + "_" + str(i) + "times" + ".txt",
#                        "w")
#             fpy.write(str(module.code))
#             fpy.close()
#             # print("new_net", new_net, flush=True)
#             print('123123')
#             print(*inputs)
#             print('123')
#             print(len(*inputs))
#             outputs = old_net(*inputs)
#             # print("outputs", outputs.shape)
#             new_outputs = new_net(*inputs)
#             # import inspect
#             # def print_init_code(obj):
#             #     init_method = obj.__class__.__init__
#             #     source_code = inspect.getsource(init_method)
#             #     print(source_code)
#             #
#             # print_init_code(new_net)
#             # print("new_outputs", new_outputs.shape)
#             if isinstance(outputs, tuple):
#                 print("handling tuple")
#                 old_avg, new_avg = handle_tuple(outputs), handle_tuple(new_outputs)
#                 dist = distance(old_avg, new_avg)
#                 dist_chess = ChebyshevDistance(old_avg, new_avg)

#                 log_distance_to_file(f, i, dist, dist_chess)

#             elif isinstance(outputs, torch.Tensor):
#                 old_out = outputs
#                 # old_out_np = old_out.asnumpy()
#                 # new_out_np = new_outputs.asnumpy()
#                 dist = distance(old_out, new_outputs)
#                 dist_chess = ChebyshevDistance(old_out, new_outputs)
#                 # if np.isnan(dist) or np.isnan(dist_chess):
#                 #     print("dist or dist_chess is nan", flush=True)
#                 #     print(new_net)
#                 #     exit(-10086)
#                 log_distance_to_file(f, i, dist, dist_chess)

#             else:
#                 print("len(outputs): ", len(outputs))
#                 print("type(outputs): ", type(outputs))
#                 print("outputs: ", outputs.shape)
#                 raise NotImplementedError("len(outputs) > 2 or len(outputs) == 1 TUPLE!!!")

#             args.distance_list.append(dist_chess)
#             args.model_list.append(new_net)
#             model_zoo[new_seed_name] = new_net
#             if new_seed_name not in mutant_selector.mutants.keys():
#                 mutate_st = datetime.datetime.now()
#                 mutate_status = 0 if log_dict[i]['status'] == "Success" else -1
#                 mutate_et = datetime.datetime.now()
#                 mutate_dt = mutate_et - mutate_st
#                 h, m, s = utils.ToolUtils.get_HH_mm_ss(mutate_dt)
#                 mutate_logger.info("INFO:Mutate Time Used on {} : {}h, {}m, {}s".format(selected_op, h, m, s))
#                 # mutation status code is successful
#                 if mutate_status == 0:
#                     mutant.selected += 1
#                     op_type.total += 1
#                     # execute this model on all platforms
#                     accumulative_inconsistency = dist_chess

#                     mutant_history.append(new_seed_name)

#                     delta = accumulative_inconsistency - last_inconsistency

#                     op_type.delta_bigger_than_zero = op_type.delta_bigger_than_zero + 1 \
#                         if delta > 0 else op_type.delta_bigger_than_zero

#                     if delta > 0:
#                         if mutant_selector.is_full():
#                             mutant_selector.pop_one_mutant()
#                         mutant_selector.add_mutant(new_seed_name)
#                         last_inconsistency = accumulative_inconsistency

#                     mutate_logger.info("SUCCESS:{} pass testing!".format(new_seed_name))
#                     mutant_counter += 1
#                 else:
#                     mutate_logger.error("Exception raised when mutate {} with {}".format(picked_seed, selected_op))

#                 mutate_logger.info("Mutated op used history:")
#                 mutate_logger.info(mutate_op_history)

#                 mutate_logger.info("Invalid mutant generated history:")
#                 mutate_logger.info(mutate_op_invalid_history)