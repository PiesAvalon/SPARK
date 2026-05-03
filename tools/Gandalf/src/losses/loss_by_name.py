import tensorflow as tf
import torch
import jittor

losses_built_in = ['MAE', 'MSE', 'CrossEntropy', 'BCE', 'BCEwithLogits', 'SmoothMAE']


def loss_by_name(name, framework):
    if name not in losses_built_in:
        raise Exception('No support loss name.')
    if framework == 'TensorFlow':
        tf_loss_table = {
            'MAE': 'tf.keras.losses.MeanAbsoluteError()',
            'MSE': 'tf.keras.losses.MeanSquaredError()',
            'CrossEntropy': 'tf.keras.losses.CategoricalCrossentropy()',
            # 'Poisson': 'tf.keras.losses.Poisson()',
            # 'KLD': 'tf.keras.losses.KLDivergence()',
            'BCE': 'tf.keras.losses.BinaryCrossentropy()',
            'BCEwithLogits': 'tf.keras.losses.BinaryCrossentropy(from_logits=True)',
            'SmoothMAE': 'tf.keras.losses.Huber(delta=1.0)',
        }
        loss = tf_loss_table[name]
    elif framework == 'PyTorch':
        torch_loss_table = {
            'MAE': 'torch.nn.L1Loss()',
            'MSE': 'torch.nn.MSELoss()',
            'CrossEntropy': 'torch.nn.CrossEntropyLoss()',
            # 'Poisson': 'torch.nn.PoissonNLLLoss()',
            # 'KLD': 'torch.nn.KLDivLoss()',
            'BCE': 'torch.nn.BCELoss()',
            'BCEwithLogits': 'torch.nn.BCEWithLogitsLoss()',
            'SmoothMAE': 'torch.nn.SmoothL1Loss()',
        }
        loss = torch_loss_table[name]
    elif framework == 'Jittor':
        jittor_loss_table = {
            'MAE': 'jittor.nn.L1Loss()',
            'MSE': 'jittor.nn.MSELoss()',
            'CrossEntropy': 'jittor.nn.CrossEntropyLoss()',
            # 'Poisson': 'jittor.nn.PoissonNLLLoss()',
            # 'KLD': 'torch.nn.KLDivLoss()',
            'BCE': 'jittor.nn.BCELoss()',
            'BCEwithLogits': 'jittor.nn.BCEWithLogitsLoss()',
            'SmoothMAE': 'jittor.nn.SmoothL1Loss()',
        }
        loss = jittor_loss_table[name]
    else:
        raise Exception('No support DL framework.')

    return eval(loss)