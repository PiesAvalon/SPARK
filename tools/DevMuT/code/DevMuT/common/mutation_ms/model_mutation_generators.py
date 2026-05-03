import random
from common.mutation_ms.model_mutation_operators import PM_mut, RA_mut, LD_mut, LA_mut, LC_mut, CM_mut, WS_mut, \
    NS_mut, GF_mut, NAI_mut, NEB_mut, LS_mut, SM_mut, DM_mut


def generate_model_by_model_mutation(model, operator, input_size, mut_file_path, generations, mutate_logger,
                                     train_configs):
    layer_names = list(model.layer_names.keys())
    # Parameter mutation
    if operator == 'PM':
        return PM_mut(model=model, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Replace operators
    elif operator == 'RA':
        add_layer_type = random.choice(["Basic_op", "Cascade_op"])
        return RA_mut(model=model, layer_names=layer_names, input_size=input_size, add_layer_type=add_layer_type,
                      mut_file_path=mut_file_path, generations=generations, mutate_logger=mutate_logger,
                      train_configs=train_configs)
    # Layer deletion
    elif operator == "LD":
        del_layer_type = random.choice(["Basic_op", "Cascade_op"])
        if len(model.get_Cascade_OPs()) < 10:
            del_layer_type = "Basic_op"
        return LD_mut(model=model, layer_names=layer_names, input_size=input_size, del_layer_type=del_layer_type,
                      mut_file_path=mut_file_path, generations=generations, mutate_logger=mutate_logger,
                      train_configs=train_configs)
    # Layer Increment
    elif operator == "LA":
        add_layer_type = random.choice(["Basic_op", "Cascade_op"])
        return LA_mut(model=model, layer_names=layer_names, input_size=input_size, add_layer_type=add_layer_type,
                      mut_file_path=mut_file_path, generations=generations, mutate_logger=mutate_logger,
                      train_configs=train_configs)
    # Layer Copy
    elif operator == "LC":
        return LC_mut(model=model, layer_names=layer_names, input_size=input_size, add_layer_type="",
                      mut_file_path=mut_file_path, generations=generations, mutate_logger=mutate_logger,
                      train_configs=train_configs)
    # Connection Modifiction
    elif operator == "CM":
        return CM_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Layer Switch
    elif operator == "LS":
        return LS_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Shape Modifiction
    elif operator == "SM":
        return SM_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mut_layer_isBasic="", mutate_logger=mutate_logger,
                      train_configs=train_configs)

    # Shape Modifiction
    elif operator == "DM":
        return DM_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mut_layer_isBasic="", mutate_logger=mutate_logger,
                      train_configs=train_configs)

    # Weights Shuffling
    elif operator == "WS":
        return WS_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Neuron Switch
    elif operator == "NS":
        return NS_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Gaussian Fuzzing
    elif operator == "GF":
        return GF_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                      generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Neuron Activation Inverse
    elif operator == "NAI":
        return NAI_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                       generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)
    # Neuron Effect Block
    elif operator == "NEB":
        return NEB_mut(model=model, layer_names=layer_names, input_size=input_size, mut_file_path=mut_file_path,
                       generations=generations, mutate_logger=mutate_logger, train_configs=train_configs)


def all_mutate_ops():
    return ['LA', 'RA', 'LD', 'PM', 'CM', "WS", "NS", "GF", "NAI", "NEB", "LS", "LC", "SM","DM"]
