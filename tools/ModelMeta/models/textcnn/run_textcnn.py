import os
import platform
from datetime import datetime
from pprint import pprint

import numpy as np
import psutil
import torch
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell

from models.textcnn.dataset import MovieReview


def train_eval_TextCNN(model_old, model_mutant, data_dir, batch_size, now_time):
    instance = MovieReview(root_dir=data_dir, maxlen=51, split=0.9)
    epoch_num = 6
    train_dataset = instance.create_train_dataset(batch_size=batch_size, epoch_size=epoch_num)
    test_dataset = instance.create_train_dataset(batch_size=batch_size, epoch_size=epoch_num)

    train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    loss_ms = SoftmaxCrossEntropyExpand(sparse=True)
    model_name = "TextCNN"
    model_old_trainable_params = model_old.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in model_old_trainable_params:
        modelms_trainable_param.name = model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    model_mutant_trainable_params = model_mutant.trainable_params()
    mutant_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in model_mutant_trainable_params:
        modelms_trainable_param.name = model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        mutant_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    f = open(os.path.join("mutated_net/" + str(model_old.__class__.__name__)+"/", str(now_time),
                          "/loss_" + str(platform.platform()) + str(mindspore.get_context('device_target')) + ".txt"),
             "w")
    opt_old = nn.Adam(filter(lambda x: x.requires_grad, model_old.get_parameters()), learning_rate=1e-5,
                      weight_decay=float(3e-5))
    opt_mutant = nn.Adam(filter(lambda x: x.requires_grad, model_mutant.get_parameters()), learning_rate=1e-5,
                         weight_decay=float(3e-5))

    def forward_fn_old(data, label):
        outputs = model_old(data)
        loss = loss_ms(outputs, label)
        return loss

    def forward_fn_mutant(data, label):
        outputs = model_mutant(data)
        loss = loss_ms(outputs, label)
        return loss

    grad_fn_old = mindspore.ops.value_and_grad(forward_fn_old, None, opt_old.parameters, has_aux=False)
    grad_fn_mutant = mindspore.ops.value_and_grad(forward_fn_mutant, None, opt_mutant.parameters, has_aux=False)

    def train_step_old(data, label):
        (loss), grads = grad_fn_old(data, label)
        loss = mindspore.ops.depend(loss, opt_old(grads))
        return loss

    def train_step_mutant(data, label):
        (loss), grads = grad_fn_mutant(data, label)
        loss = mindspore.ops.depend(loss, opt_mutant(grads))
        return loss

    losses_ms_avg = []
    losses_ms_avg_new = []
    eval_ms = []
    eval_ms_new = []

    for epoch in range(epoch_num):
        print('----------------------------')
        print(f"epoch: {epoch}/{epoch_num}")
        model_old.set_train(True)
        model_mutant.set_train(True)

        batch = 0
        losses_ms = []
        losses_ms_new = []
        for item in train_iter:
            text_array, targets_array = item['data'].asnumpy(), item['label'].asnumpy()

            text_tensor, targets_tensor = mindspore.Tensor(text_array, dtype=mstype.int32), mindspore.Tensor(
                targets_array, dtype=mstype.int32)
            # print("text_tensor: ", text_tensor.shape)
            # print("targets_tensor: ", targets_tensor.shape)
            loss_ms_result = train_step_old(text_tensor, targets_tensor)
            loss_ms_result_new = train_step_mutant(text_tensor, targets_tensor)

            if batch % 500 == 0:
                print("batch: {}, ms_loss: {}, ms_loss_new: {}".format(
                    str(batch), str(loss_ms_result), str(loss_ms_result_new)))
                f.write("batch: {}, ms_loss: {}, ms_loss_new: {}".format(
                    str(batch), str(loss_ms_result), str(loss_ms_result_new))+"\n")

            losses_ms.append(loss_ms_result.asnumpy())
            losses_ms_new.append(loss_ms_result_new.asnumpy())

            batch += batch_size
            # break

        losses_ms_avg.append(np.mean(losses_ms))
        losses_ms_avg_new.append(np.mean(losses_ms_new))
        print("epoch: {}, ms_loss_avg_old: {}, losses_ms_avg_new: {}".format(epoch,
                                                                             str(np.mean(losses_ms)),
                                                                             str(np.mean(losses_ms_new))))
        f.write("epoch: {}, ms_loss_avg_old: {}, losses_ms_avg_new: {}".format(epoch,
                                                                               str(np.mean(losses_ms)),
                                                                               str(np.mean(losses_ms_new)))+"\n")
        # 测试步骤开始
        model_old.set_train(False)
        model_mutant.set_train(False)

        test_data_size = 0
        correct_ms = 0
        correct_ms_new = 0

        for item in test_iter:
            text, targets = item['data'], item['label']
            test_data_size += text.shape[0]

            output_ms = model_old(text)
            output_ms_new = model_mutant(text)
            indices_ms = np.argmax(output_ms.asnumpy(), axis=1)
            indices_ms_new = np.argmax(output_ms_new.asnumpy(), axis=1)
            result_ms = (np.equal(indices_ms, targets.asnumpy()) * 1).reshape(-1)
            result_ms_new = (np.equal(indices_ms_new, targets.asnumpy()) * 1).reshape(-1)
            accuracy_ms = result_ms.sum()
            accuracy_ms_new = result_ms_new.sum()
            correct_ms = correct_ms + accuracy_ms
            correct_ms_new = correct_ms_new + accuracy_ms_new

        eval_ms.append(correct_ms / test_data_size)
        eval_ms_new.append(correct_ms_new / test_data_size)
        print("OLD Mindpsore Test Accuacy: {}".format(
            100 * correct_ms / test_data_size))
        f.write("OLD Mindpsore Test Accuacy: {}".format(
            100 * correct_ms / test_data_size)+"\n")
        print("NEW Mindpsore Test Accuacy: {}".format(
            100 * correct_ms_new / test_data_size))
        f.write("NEW Mindpsore Test Accuacy: {}".format(
            100 * correct_ms_new / test_data_size)+"\n")


def loss_com_ms(logit, label):
    # class_nums=mindspore.Tensor([class_nums],mindspore.int32)
    class_nums = 2
    exp = ops.Exp()
    reduce_sum = ops.ReduceSum(keep_dims=True)
    onehot = ops.OneHot()
    on_value = mindspore.Tensor(1.0, mindspore.int32)
    off_value = mindspore.Tensor(0.0, mindspore.int32)
    div = ops.Div()
    log = ops.Log()
    sum_cross_entropy = ops.ReduceSum(keep_dims=False)
    mul = ops.Mul()
    reduce_mean = ops.ReduceMean(keep_dims=False)
    reduce_max = ops.ReduceMax(keep_dims=True)
    sub = ops.Sub()

    logit_max = reduce_max(logit, -1)
    exp0 = exp(sub(logit, logit_max))
    exp_sum = reduce_sum(exp0, -1)
    softmax_result = div(exp0, exp_sum)

    label = onehot(label, class_nums, on_value, off_value)
    softmax_result_log = log(softmax_result)
    loss = sum_cross_entropy((mul(softmax_result_log, label)), -1)
    loss = mul(ops.scalar_to_tensor(-1.0), loss)
    loss = reduce_mean(loss, -1)
    return loss



class SoftmaxCrossEntropyExpand(Cell):
    r"""
    Computes softmax cross entropy between logits and labels. Implemented by expanded formula.

    This is a wrapper of several functions.

    .. math::
        \ell(x_i, t_i) = -log\left(\frac{\exp(x_{t_i})}{\sum_j \exp(x_j)}\right),
    where :math:`x_i` is a 1D score Tensor, :math:`t_i` is the target class.

    Note:
        When argument sparse is set to True, the format of label is the index
        range from :math:`0` to :math:`C - 1` instead of one-hot vectors.

    Args:
        sparse(bool): Specifies whether labels use sparse format or not. Default: False.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **label** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, a scalar tensor including the mean loss.
    """

    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.div = ops.Div()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.cast = ops.Cast()
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.reduce_max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, logit, label):
        """
        construct
        """
        # print("logit", logit.shape, "label", label.shape)
        logit_max = self.reduce_max(logit, -1)
        # print(logit_max)
        exp = self.exp(self.sub(logit, logit_max))
        # print(exp)
        exp_sum = self.reduce_sum(exp, -1)
        # print(exp_sum)
        softmax_result = self.div(exp, exp_sum)
        # print(softmax_result)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
            # print(label)
        softmax_result_log = self.log(softmax_result)
        # print("softmax_result_log.shape2", softmax_result_log.shape)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        # print(loss)
        loss = self.mul2(ops.scalar_to_tensor(-1.0), loss)
        loss = self.reduce_mean(loss, -1)

        return loss
