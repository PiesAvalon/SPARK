import tensorflow as tf
import torch
import jittor
import re

import src.metircs.torch_expansion as exp_t
import src.metircs.jittor_expansion as exp_j

tf_metric_table = {
    'Accuracy': 'tf.keras.metrics.Accuracy()',
    'OneHotAccuracy': 'tf.keras.metrics.CategoricalAccuracy()',
    'MSE': 'tf.keras.metrics.MeanSquaredError()',
    'MAE': 'tf.keras.metrics.MeanAbsoluteError()',
    'TopKOneHotAccuracy': 'tf.keras.metrics.TopKCategoricalAccuracy()',
    'TopKAccuracy': 'tf.keras.metrics.SparseTopKCategoricalAccuracy()',
}

torch_metric_table = {
    'Accuracy': 'exp_t.accuracy.Accuracy()',
    'OneHotAccuracy': 'exp_t.one_hot_accuracy.OneHotAccuracy()',
    'MSE': 'exp_t.mse.MSE()',
    'MAE': 'exp_t.mae.MAE()',
    'TopKOneHotAccuracy': 'exp_t.top_k_one_hot_accuracy.TopKOneHotAccuracy()',
    'TopKAccuracy': 'exp_t.top_k_accuracy.TopKAccuracy()',
}

jittor_metric_table = {
    'Accuracy': 'exp_j.accuracy.Accuracy()',
    'OneHotAccuracy': 'exp_j.one_hot_accuracy.OneHotAccuracy()',
    'MSE': 'exp_j.mse.MSE()',
    'MAE': 'exp_j.mae.MAE()',
    'TopKOneHotAccuracy': 'exp_j.top_k_one_hot_accuracy.TopKOneHotAccuracy()',
    'TopKAccuracy': 'exp_j.top_k_accuracy.TopKAccuracy()',
}


def get_tf_metrics(name):
    top_k_pattern = r'^Top[1-9][0-9]*Accuracy$'
    top_k_one_hot_pattern = r'^Top[1-9][0-9]*OneHotAccuracy$'
    if re.match(top_k_pattern, name):
        t = re.search(r'[1-9][0-9]*', name).span()
        k = name[t[0]:t[1]]
        mat = tf_metric_table['TopKAccuracy']
        return eval(mat[:-1] + 'k={0}'.format(k) + mat[-1])
    elif re.match(top_k_one_hot_pattern, name):
        t = re.search(r'[1-9][0-9]*', name).span()
        k = name[t[0]:t[1]]
        mat = tf_metric_table['TopKOneHotAccuracy']
        return eval(mat[:-1] + 'k={0}'.format(k) + mat[-1])
    else:
        return eval(tf_metric_table[name])


def get_torch_metrics(name):
    top_k_pattern = r'^Top[1-9][0-9]*Accuracy$'
    top_k_one_hot_pattern = r'^Top[1-9][0-9]*OneHotAccuracy$'
    if re.match(top_k_pattern, name):
        t = re.search(r'[1-9][0-9]*', name).span()
        k = name[t[0]:t[1]]
        mat = torch_metric_table['TopKAccuracy']
        return eval(mat[:-1] + 'k={0}'.format(k) + mat[-1])
    elif re.match(top_k_one_hot_pattern, name):
        t = re.search(r'[1-9][0-9]*', name).span()
        k = name[t[0]:t[1]]
        mat = torch_metric_table['TopKOneHotAccuracy']
        return eval(mat[:-1] + 'k={0}'.format(k) + mat[-1])
    else:
        return eval(torch_metric_table[name])


def get_jittor_metrics(name):
    top_k_pattern = r'^Top[1-9][0-9]*Accuracy$'
    top_k_one_hot_pattern = r'^Top[1-9][0-9]*OneHotAccuracy$'
    if re.match(top_k_pattern, name):
        t = re.search(r'[1-9][0-9]*', name).span()
        k = name[t[0]:t[1]]
        mat = jittor_metric_table['TopKAccuracy']
        return eval(mat[:-1] + 'k={0}'.format(k) + mat[-1])
    elif re.match(top_k_one_hot_pattern, name):
        t = re.search(r'[1-9][0-9]*', name).span()
        k = name[t[0]:t[1]]
        mat = jittor_metric_table['TopKOneHotAccuracy']
        return eval(mat[:-1] + 'k={0}'.format(k) + mat[-1])
    else:
        return eval(jittor_metric_table[name])