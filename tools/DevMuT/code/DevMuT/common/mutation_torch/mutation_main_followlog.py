import torch
from common.mutation_torch.Other_utils import *
from common.mutation_torch.model_mutation_operators import LD_mut, PM_mut, LA_mut, RA_mut, CM_mut, WS_mut, NS_mut, \
    GF_mut, NAI_mut, NEB_mut, LS_mut, LC_mut, SM_mut, DM_mut
from common.model_utils import get_model


def check_ms_failed_trace(model, log_path, input_size, train_configs, execution_traces, mutate_logger):

    origin_traces = deepcopy(execution_traces)
    model_name = log_path.split("/")[-2].split("-")[0]
    inconsistency_traces = {}
    f = open(log_path)
    log = f.readlines()
    i = -1
    while i < len(log):
        ms_mut_result = None
        pt_mut_result = None
        generation = 0
        i += 1
        if i >= len(log):
            break
        line = log[i]
        if "\n" == line:
            continue
        elif "generation" in line:
            continue
        j = i
        if "LD mut_strategy" in line:
            while "mut_result" not in log[j]:
                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]

            if "not enough layers to delete!" in log[j]:
                continue

            elif "set layers failure" in log[j]:
                continue

            elif "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"

            l_name = log[i + 1].split(":")[-1]
            mutate_layer_indice = int(log[i + 2][:-1].split(":")[-1])
            pt_mut_result = LD_mut(model, input_size, del_layer_name=l_name[:-1],
                                   mutate_layer_indice=mutate_layer_indice, train_configs=train_configs)

        elif "LS mut_strategy" in line:
            while "mut_result" not in log[j]:
                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]

            if "suitable" in log[j] or "no suitable op" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"

            l_name1 = log[i + 1].split(":")[-1]
            l_name2 = log[i + 2].split(":")[-1]

            pt_mut_result = LS_mut(model, input_size, l_name1[:-1], l_name2[:-1], train_configs=train_configs)

        elif "Adopt WS mut_strategy" in line or "Adopt NS mut_strategy" in line or "Adopt GF mut_strategy" in line or \
                "Adopt NAI mut_strategy" in line or "Adopt NEB mut_strategy" in line:
            mut_type_flag = -1
            if "WS mut_strategy" in line:
                mut_type_flag = 0
            elif "NS mut_strategy" in line:
                mut_type_flag = 1
            elif "GF mut_strategy" in line:
                mut_type_flag = 2
            elif "NAI mut_strategy" in line:
                mut_type_flag = 3
            elif "NEB mut_strategy" in line:
                mut_type_flag = 4

            while "mut_result" not in log[j]:
                if "mutation_ratio" in log[j]:
                    mutation_ratio = float(log[j][:-1].split(":")[1])
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "select layer" in log[j]:
                    l_name = log[j][:-1].split(":")[-1]

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]

            if "suitable" in log[j] or "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            if mut_type_flag == 0:
                pt_mut_result = WS_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                                       train_configs=train_configs)
            elif mut_type_flag == 1:
                pt_mut_result = NS_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                                       train_configs=train_configs)
            elif mut_type_flag == 2:
                pt_mut_result = GF_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                                       train_configs=train_configs)
            elif mut_type_flag == 3:
                pt_mut_result = NAI_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                                        train_configs=train_configs)
            elif mut_type_flag == 4:
                pt_mut_result = NEB_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                                        train_configs=train_configs)

        elif "Adopt LA mut_strategy" in line or "Adopt RA mut_strategy" in line or \
                "Adopt CM mut_strategy" in line or "Adopt LC mut_strategy" in line:
            mut_type_flag = -1
            if "LA mut_strategy" in line:
                mut_type_flag = 0
            elif "RA mut_strategy" in line:
                mut_type_flag = 1
            elif "CM mut_strategy" in line:
                mut_type_flag = 2
            elif "LC mut_strategy" in line:
                mut_type_flag = 3

            activation_name = None
            add_layer_type = None
            while "mut_result" not in log[j]:
                if "select layer: " in log[j]:
                    end1 = log[j].index(" layer_type:")
                    mut_layer_name = log[j][len("select layer: "):end1]
                elif "add Basic layer" in log[j]:
                    add_layer_type = (log[j][len("add Basic layer : "):-1])
                elif "mut Basic type:" in log[j]:
                    mut_layer_isBasic = (log[j][len("mut Basic layer : ") - 2:-1] == "True")
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "select insert layer: " in log[j]:
                    end2 = log[j].index("<")
                    insert_layer_info = log[j][len("select insert layer: "):end2]
                    if "Dense" == insert_layer_info:
                        insert_layer_info = "Linear"
                    if "Transpose" in insert_layer_info:
                        dimension = insert_layer_info[4:6]
                        insert_layer_info = "ConvTranspose" + dimension

                    k = 0
                    if "dwpw_group" == insert_layer_info:
                        k = j
                        while "dwpw_activation" not in log[k]:
                            k += 1
                        activation_name = log[k][:-1].split(": ")[1][:-2]

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]


            if "No suitable ops for" in log[j] or "Create illegal layer" in log[j] or "set layers failure" in log[j]:
                continue
            ms_mut_result = log[j].split(":")[1][:-1] == "True"

            if mut_type_flag == 0:
                pt_mut_result = LA_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type,
                                       insert_layer_info, activation_name, mut_layer_indice,
                                       train_configs=train_configs)
            elif mut_type_flag == 1:
                pt_mut_result = RA_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type,
                                       insert_layer_info, activation_name, mut_layer_indice,
                                       train_configs=train_configs)
            elif mut_type_flag == 2:
                pt_mut_result = CM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, insert_layer_info,
                                       activation_name, mut_layer_indice, train_configs=train_configs)
            elif mut_type_flag == 3:
                pt_mut_result = LC_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type,
                                       insert_layer_info, activation_name, train_configs=train_configs)


        elif "Adopt PM mut_strategy" in line:
            m_value = None
            while "mut_result" not in log[j]:
                if "Edit value:" in log[j]:
                    m_value = log[j][len("Edit value: "):log[j].index(" new_inshape")]
                    if "(" in m_value and ")" in m_value:
                        tmp = m_value[1: -1].split(",")
                        m_value = tuple([int(val) for val in tmp])
                    elif m_value in "True False":
                        m_value = bool(m_value)
                    elif ("." in m_value or "e" in m_value):
                        m_value = float(m_value)
                    elif not ("." in m_value or "e" in m_value):
                        m_value = int(m_value)

                elif "mutlayers_indice:" in log[j]:
                    start3 = log[j].index(":")
                    mutate_layer_indice = int(log[j][(start3 + 1):-1])

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]


            if "Parameter Miss" in log[j] or "PM Create illegal layer" in log[j] or "set layers failure" in log[j]:
                continue
            ms_mut_result = log[j].split(":")[1][:-1] == "True"

            assert m_value is not None
            end1 = log[i + 2].index(" layer_type")
            end2 = log[i + 2].index(" input_shape:")
            start2 = log[i + 2].index("selected param:") + len("selected param:")
            sel_layer_name = log[i + 2][11:end1]
            mutate_param_selname = log[i + 2][start2:end2]

            if "group" in mutate_param_selname:
                mutate_param_selname = "groups"

            pt_mut_result = PM_mut(model, input_size, sel_layer_name, mutate_layer_indice, mutate_param_selname,
                                   m_value, train_configs=train_configs)

        elif "Adopt SM mut_strategy" in line:
            while "mut_result" not in log[j]:
                if "select layer: " in log[j]:
                    end1 = log[j].index(" layer_type:")
                    mut_layer_name = log[j][len("select layer: "):end1]
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "mut Basic type:" in log[j]:
                    mut_layer_isBasic = (log[j][len("mut Basic layer : ") - 2:-1] == "True")
                elif "mutate state: " in log[j]:
                    mutate_state = log[j][len("mutate state: "):-1]
                    if mutate_state == "all":
                        mutate_input_shape_str = log[j + 1][log[j + 1].index("[") + 1:log[j + 1].index("]")]
                        mutate_input_shape_str = mutate_input_shape_str.replace(" ", "")
                        mutate_input_shape_str = mutate_input_shape_str.split(",")
                        mutate_input_shape = tuple([int(val) for val in mutate_input_shape_str])

                        mutate_output_shape_str = log[j + 2][log[j + 2].index("[") + 1:log[j + 2].index("]")]
                        mutate_output_shape_str = mutate_output_shape_str.replace(" ", "")
                        mutate_output_shape_str = mutate_output_shape_str.split(",")
                        mutate_output_shape = tuple([int(val) for val in mutate_output_shape_str])
                        mut_state = 2

                    elif mutate_state == "before":
                        mutate_input_shape_str = log[j + 1][log[j + 1].index("[") + 1:log[j + 1].index("]")]
                        mutate_input_shape_str = mutate_input_shape_str.replace(" ", "")
                        mutate_input_shape_str = mutate_input_shape_str.split(",")
                        mutate_input_shape = tuple([int(val) for val in mutate_input_shape_str])
                        mutate_output_shape = None
                        mut_state = 0

                    elif mutate_state == "after":
                        mutate_output_shape_str = log[j + 1][log[j + 1].index("[") + 1:log[j + 1].index("]")]
                        mutate_output_shape_str = mutate_output_shape_str.replace(" ", "")
                        mutate_output_shape_str = mutate_output_shape_str.split(",")
                        mutate_output_shape = tuple([int(val) for val in mutate_output_shape_str])
                        mutate_input_shape = None
                        mut_state = 1

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]

            if "Create illegal layer" in log[j] or "set layers failure" in log[j] or "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            pt_mut_result = SM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, mut_state, mutate_input_shape,
                                   mutate_output_shape, mut_layer_indice, train_configs=train_configs)

        elif "Adopt DM mut_strategy" in line:
            while "mut_result" not in log[j]:
                if "select layer: " in log[j]:
                    end1 = log[j].index(" layer_type:")
                    mut_layer_name = log[j][len("select layer: "):end1]
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "mut Basic type:" in log[j]:
                    mut_layer_isBasic = (log[j][len("mut Basic layer : ") - 2:-1] == "True")

                elif "in_dtype:" in log[j]:
                    dtype_str = str(log[j][:-1]).split(":")[1]
                    if "float16" in dtype_str.lower():
                        t_dtype = torch.float16
                    elif "float32" in dtype_str.lower():
                        t_dtype = torch.float32
                    elif "int32" in dtype_str.lower():
                        t_dtype = torch.int32
                    elif "int16" in dtype_str.lower():
                        t_dtype = torch.int16

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in execution_traces:
                    continue
                else:
                    generation_idx = execution_traces.index(generation)
                    del execution_traces[generation_idx]

            if "Create illegal layer" in log[j] or "set layers failure" in log[j] or "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            pt_mut_result = DM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, t_dtype, mut_layer_indice,
                                   train_configs=train_configs)

        if pt_mut_result==None:
            continue

        if not pt_mut_result == ms_mut_result or not pt_mut_result:
            _, seed_model_torch = get_model(model_name, input_size)

            model_traces = sorted(list(set(origin_traces) - set(execution_traces)))

            if len(model_traces)==0:
                _, model = get_model(model_name, input_size)
            else:
                model = analyze_log_torch_followtrace(model_traces, seed_model_torch, log_path, input_size, train_configs)

        if 'Adopt' in line and 'mut_strategy' in line:


            if not mutate_logger == "":
                mutate_logger.info("torch_mut_result: " + str(pt_mut_result))
                mutate_logger.info("generation: " + str(generation))

            if ms_mut_result != pt_mut_result:
                mutate_logger.error(f"For {generation} generation mutation model, the results of MindSpore and PyTorch "
                                    f"are inconsistent, MindSpore: {ms_mut_result}, PyTorch: {pt_mut_result}")
                inconsistency_traces[str(generation)] = 'MindSpore: ' + str(ms_mut_result) + ', PyTorch: ' + \
                                                        str(pt_mut_result)
            else:
                mutate_logger.debug(f"For {generation} generation mutation model, the results of MindSpore and PyTorch "
                                    f"are consistent, MindSpore: {ms_mut_result}, PyTorch: {pt_mut_result}")
        i = j
    f.close()
    return inconsistency_traces


def analyze_log_torch_followtrace(traces, model, log_path, input_size, train_configs):
    f = open(log_path)
    log = f.readlines()
    i = -1

    while i < len(log):
        i += 1
        if len(traces) == 0:
            return model
        if i >= len(log):
            break
        line = log[i]
        if "\n" == line:
            continue
        elif "generation" in line:
            continue

        if "LD mut_strategy" in line:
            j = i
            while "mut_result" not in log[j]:
                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "not enough layers to delete!" in log[j] or "set layers failure" in log[j] or \
                    "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            l_name = log[i + 1].split(":")[-1]
            mutate_layer_indice = int(log[i + 2][:-1].split(":")[-1])
            if ms_mut_result is False:
                i = j
                continue

            LD_mut(model, input_size, del_layer_name=l_name[:-1], mutate_layer_indice=mutate_layer_indice,
                   train_configs=train_configs)

        elif "LS mut_strategy" in line:
            j = i
            while "mut_result" not in log[j]:
                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])

                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "suitable" in log[j] or "no suitable op" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            l_name1 = log[i + 1].split(":")[-1]
            l_name2 = log[i + 2].split(":")[-1]
            if ms_mut_result is False:
                i = j
                continue

            LS_mut(model, input_size, l_name1[:-1], l_name2[:-1], train_configs=train_configs)

        elif "Adopt WS mut_strategy" in line or "Adopt NS mut_strategy" in line or "Adopt GF mut_strategy" in line or \
                "Adopt NAI mut_strategy" in line or "Adopt NEB mut_strategy" in line:
            mut_type_flag = -1
            if "WS mut_strategy" in line:
                mut_type_flag = 0
            elif "NS mut_strategy" in line:
                mut_type_flag = 1
            elif "GF mut_strategy" in line:
                mut_type_flag = 2
            elif "NAI mut_strategy" in line:
                mut_type_flag = 3
            elif "NEB mut_strategy" in line:
                mut_type_flag = 4

            j = i
            while "mut_result" not in log[j]:
                if "mutation_ratio" in log[j]:
                    mutation_ratio = float(log[j][:-1].split(":")[1])
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "select layer" in log[j]:
                    l_name = log[j][:-1].split(":")[-1]
                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "suitable" in log[j] or "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            if ms_mut_result is False:
                i = j
                continue
            if mut_type_flag == 0:
                WS_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                       train_configs=train_configs)
            elif mut_type_flag == 1:
                NS_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                       train_configs=train_configs)
            elif mut_type_flag == 2:
                GF_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                       train_configs=train_configs)
            elif mut_type_flag == 3:
                NAI_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                        train_configs=train_configs)
            elif mut_type_flag == 4:
                NEB_mut(model, input_size, l_name, mutation_ratio, mut_layer_indice,
                        train_configs=train_configs)

        elif "Adopt LA mut_strategy" in line or "Adopt RA mut_strategy" in line or \
                "Adopt CM mut_strategy" in line or "Adopt LC mut_strategy" in line:
            mut_type_flag = -1
            if "LA mut_strategy" in line:
                mut_type_flag = 0
            elif "RA mut_strategy" in line:
                mut_type_flag = 1
            elif "CM mut_strategy" in line:
                mut_type_flag = 2
            elif "LC mut_strategy" in line:
                mut_type_flag = 3

            j = i
            activation_name = None
            add_layer_type = None
            while "mut_result" not in log[j]:
                if "select layer: " in log[j]:
                    end1 = log[j].index(" layer_type:")
                    mut_layer_name = log[j][len("select layer: "):end1]
                elif "add Basic layer" in log[j]:
                    add_layer_type = (log[j][len("add Basic layer : "):-1])
                elif "mut Basic type:" in log[j]:
                    mut_layer_isBasic = (log[j][len("mut Basic layer : ") - 2:-1] == "True")
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "select insert layer: " in log[j]:
                    end2 = log[j].index("<")
                    insert_layer_info = log[j][len("select insert layer: "):end2]
                    if "Dense" == insert_layer_info:
                        insert_layer_info = "Linear"
                    if "Transpose" in insert_layer_info:
                        dimension = insert_layer_info[4:6]
                        insert_layer_info = "ConvTranspose" + dimension
                    k = 0
                    if "dwpw_group" == insert_layer_info:
                        k = j
                        while "dwpw_activation" not in log[k]:
                            k += 1
                        activation_name = log[k][:-1].split(": ")[1][:-2]
                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "No suitable ops for" in log[j] or "Create illegal layer" in log[j] or "set layers failure" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            if ms_mut_result is False:
                i = j
                continue

            if mut_type_flag == 0:
                LA_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type,
                       insert_layer_info, activation_name, mut_layer_indice, train_configs=train_configs)
            elif mut_type_flag == 1:
                RA_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type,
                       insert_layer_info, activation_name, mut_layer_indice, train_configs=train_configs)
            elif mut_type_flag == 2:
                CM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, insert_layer_info,
                       activation_name, mut_layer_indice, train_configs=train_configs)
            elif mut_type_flag == 3:
                LC_mut(model, input_size, mut_layer_isBasic, mut_layer_name, add_layer_type,
                       insert_layer_info, activation_name, train_configs=train_configs)

        elif "Adopt PM mut_strategy" in line:
            j = i
            m_value = None
            while "mut_result" not in log[j]:
                if "Edit value:" in log[j]:
                    m_value = log[j][len("Edit value: "):log[j].index(" new_inshape")]
                    if "(" in m_value and ")" in m_value:
                        tmp = m_value[1: -1].split(",")
                        m_value = tuple([int(val) for val in tmp])
                    elif m_value in "True False":
                        m_value = bool(m_value)
                    elif ("." in m_value or "e" in m_value):
                        m_value = float(m_value)
                    elif not ("." in m_value or "e" in m_value):
                        m_value = int(m_value)

                elif "mutlayers_indice:" in log[j]:
                    start3 = log[j].index(":")
                    mutate_layer_indice = int(log[j][(start3 + 1):-1])

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])

                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "Parameter Miss" in log[j] or "PM Create illegal layer" in log[j] or "set layers failure" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            assert m_value is not None
            end1 = log[i + 2].index(" layer_type")
            end2 = log[i + 2].index(" input_shape:")
            start2 = log[i + 2].index("selected param:") + len("selected param:")
            sel_layer_name = log[i + 2][11:end1]
            mutate_param_selname = log[i + 2][start2:end2]
            if "group" in mutate_param_selname:
                mutate_param_selname = "groups"

            if ms_mut_result is False:
                i = j
                continue

            PM_mut(model, input_size, sel_layer_name, mutate_layer_indice, mutate_param_selname, m_value,
                   train_configs=train_configs)

        elif "Adopt SM mut_strategy" in line:
            mut_type_name = "SM"
            j = i
            while "mut_result" not in log[j]:
                if "select layer: " in log[j]:
                    end1 = log[j].index(" layer_type:")
                    mut_layer_name = log[j][len("select layer: "):end1]
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "mut Basic type:" in log[j]:
                    mut_layer_isBasic = (log[j][len("mut Basic layer : ") - 2:-1] == "True")
                elif "mutate state: " in log[j]:
                    mutate_state = log[j][len("mutate state: "):-1]
                    if mutate_state == "all":
                        mutate_input_shape_str = log[j + 1][log[j + 1].index("[") + 1:log[j + 1].index("]")]
                        mutate_input_shape_str = mutate_input_shape_str.replace(" ", "")
                        mutate_input_shape_str = mutate_input_shape_str.split(",")
                        mutate_input_shape = tuple([int(val) for val in mutate_input_shape_str])

                        mutate_output_shape_str = log[j + 2][log[j + 2].index("[") + 1:log[j + 2].index("]")]
                        mutate_output_shape_str = mutate_output_shape_str.replace(" ", "")
                        mutate_output_shape_str = mutate_output_shape_str.split(",")
                        mutate_output_shape = tuple([int(val) for val in mutate_output_shape_str])
                        mut_state = 2

                    elif mutate_state == "before":
                        mutate_input_shape_str = log[j + 1][log[j + 1].index("[") + 1:log[j + 1].index("]")]
                        mutate_input_shape_str = mutate_input_shape_str.replace(" ", "")
                        mutate_input_shape_str = mutate_input_shape_str.split(",")
                        mutate_input_shape = tuple([int(val) for val in mutate_input_shape_str])
                        mutate_output_shape = None
                        mut_state = 0

                    elif mutate_state == "after":
                        mutate_output_shape_str = log[j + 1][log[j + 1].index("[") + 1:log[j + 1].index("]")]
                        mutate_output_shape_str = mutate_output_shape_str.replace(" ", "")
                        mutate_output_shape_str = mutate_output_shape_str.split(",")
                        mutate_output_shape = tuple([int(val) for val in mutate_output_shape_str])
                        mutate_input_shape = None
                        mut_state = 1

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "Create illegal layer" in log[j] or "set layers failure" in log[j] or "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            if ms_mut_result is False:
                i = j
                continue

            SM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, mut_state, mutate_input_shape,
                   mutate_output_shape, mut_layer_indice, train_configs=train_configs)

        elif "Adopt DM mut_strategy" in line:
            j = i
            while "mut_result" not in log[j]:
                if "select layer: " in log[j]:
                    end1 = log[j].index(" layer_type:")
                    mut_layer_name = log[j][len("select layer: "):end1]
                elif "mutlayers_indice" in log[j]:
                    mut_layer_indice = (int(log[j][:-1].split(":")[-1]))
                elif "mut Basic type:" in log[j]:
                    mut_layer_isBasic = (log[j][len("mut Basic layer : ") - 2:-1] == "True")
                elif "in_dtype:" in log[j]:
                    dtype_str = str(log[j][:-1]).split(":")[1]
                    if "float16" in dtype_str.lower():
                        t_dtype = torch.float16
                    elif "float32" in dtype_str.lower():
                        t_dtype = torch.float32
                    elif "int32" in dtype_str.lower():
                        t_dtype = torch.int32
                    elif "int16" in dtype_str.lower():
                        t_dtype = torch.int16

                j += 1

            assert "generation" in log[j + 1]
            if "generation" in log[j + 1]:
                generation = int(log[j + 1][:-1].split(" ")[0])
                if generation not in traces:
                    continue
                else:
                    generation_idx = traces.index(generation)
                    del traces[generation_idx]

            if "Create illegal layer" in log[j] or "set layers failure" in log[j] or "No suitable ops for" in log[j]:
                continue

            ms_mut_result = log[j].split(":")[1][:-1] == "True"
            if ms_mut_result is False:
                i = j
                continue

            DM_mut(model, input_size, mut_layer_isBasic, mut_layer_name, t_dtype, mut_layer_indice,
                   train_configs=train_configs)

    f.close()


if __name__ == '__main__':
    from common.log_recoder import Logger
    from common.model_utils import get_model

    model_name = "crnn"
    import argparse

    args_opt = argparse.Namespace(
        model=model_name,
        dataset_path=r'/data1/pzy/mindb/IC03/processed',
        batch_size=1,
        epoch=5,
        mutation_iterations=50,
        selected_model_num=1,
        mutation_strategy="random",
        mutation_type=['LD', 'PM', 'LA', 'RA', 'CM', 'SM', 'LC', 'WS', 'NS', 'GF', 'NAI', 'NEB'],
        mutation_log='/data1/myz/net-sv/common/log',
        selected_gen=None,
    )
    from mutation_test import NetworkGeneralization

    mutate = NetworkGeneralization(args=args_opt)

    log_path, input_size = '/data1/myz/net-sv/common/log/crnn-2023.11.28.17.56.22/mutation.txt', (1, 3, 32, 100)
    model1, model2 = get_model(model_name, input_size)
    logger = Logger(log_file='./log/debug.log')
    execution_traces = [i for i in range(1, 51)]
    traces_dict = {}
    for i in range(1,51):
        traces_dict[str(i)] = [i for i in range(1,i+1)]
    traces_dict['1'] = [1]
    result = check_ms_failed_trace(model2, log_path, input_size, mutate.train_config, execution_traces, logger.logger)

