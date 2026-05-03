import os
import torch
import torch.nn as nn_t
import mindspore
import mindspore as ms
from mindspore import ops


class loss_torch(nn_t.Module):
    def __init__(self):
        super(loss_torch, self).__init__()
        if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
            self.final_device = 'cuda:0'
        else:
            self.final_device = 'cpu'

    def forward(self, logit, label, class_nums):
        exp = torch.exp
        reduce_sum = torch.sum
        onehot = nn_t.functional.one_hot
        div = torch.div
        log = torch.log
        sum_cross_entropy = torch.sum
        mul = torch.Tensor.mul
        reduce_mean = torch.std_mean
        reduce_max = torch.max  # keep_dims=True
        sub = torch.Tensor.sub
        logit_max, _ = reduce_max(logit, -1, keepdim=True)
        exp0 = exp(sub(logit, logit_max))
        exp_sum = reduce_sum(exp0, -1, keepdim=True)
        softmax_result = div(exp0, exp_sum)
        label = onehot(label, class_nums)
        softmax_result_log = log(softmax_result)
        loss = sum_cross_entropy((mul(softmax_result_log, label)), -1, keepdim=False)
        loss = mul(torch.Tensor([-1.0]).to(self.final_device), loss)
        loss, _ = reduce_mean(loss, -1, keepdim=False)
        return loss


def loss_com_ms(logit, label, class_nums):
    # class_nums=mindspore.Tensor([class_nums],mindspore.int32)
    exp = ops.Exp()
    reduce_sum = ops.ReduceSum(keep_dims=True)
    onehot = ops.OneHot()
    on_value = ms.Tensor(1.0, mindspore.int32)
    off_value = ms.Tensor(0.0, mindspore.int32)
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

# if __name__ == '__main__':
#     device = "GPU"
#     ms.set_context(mode=ms.GRAPH_MODE, device_target=device.upper())
#     #ms.set_context(mode=ms.PYNATIVE_MODE)
#     epoch_num, batch_size = 50, 32
#     num_classes = 2
#     instance = MovieReview(root_dir="/data1/pzy/mindb/rt-polarity/rt-polaritydata", maxlen=51, split=0.9)
#     print("-------------")
#     print(instance.get_dict_len())
#
#
#     test_dataset = instance.create_test_dataset(batch_size=batch_size)
#
#     train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
#     test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
#
#
#     model_t = textcnn_torch(vocab_len=instance.get_dict_len(), word_len=51, num_classes=num_classes, vec_length=40).to(final_device)
#     model_ms = textcnn_ms(vocab_len=instance.get_dict_len(), word_len=51, num_classes=num_classes,vec_length=40)
#
#     loss_com_t=loss_torch()
#     loss_com_t.to(final_device)
#
#     learning_rate =float(1e-5)
#
#     opt_t = torch.optim.Adam(filter(lambda x: x.requires_grad, model_t.parameters()), lr=learning_rate,weight_decay=float(3e-5))
#     opt_ms= nn_ms.Adam(filter(lambda x: x.requires_grad, model_ms.get_parameters()),learning_rate=learning_rate, weight_decay=float(3e-5))
#
#
#     def forward_fn(data, label,num_classes):
#         outputs = model_ms(data)
#         loss = loss_com_ms(outputs, label,num_classes)
#         return loss, outputs
#
#         # Get gradient function
#
#
#     grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt_ms.parameters, has_aux=True)
#
#
#     # Define function of one-step training
#     def train_step(data, label,num_classes):
#         (loss, _), grads = grad_fn(data, label,num_classes)
#         loss = mindspore.ops.depend(loss, opt_ms(grads))
#         return loss
#
#     for epoch in range(epoch_num):
#         model_t.train()
#         model_ms.set_train(True)
#         print("epoch", epoch, "/", epoch_num)
#         batch = 0
#
#
#         for item in train_iter:
#             text_array, targets_array = item['data'].asnumpy(), item['label'].asnumpy()
#
#             opt_t.zero_grad()
#             output_torch = model_t(torch.LongTensor(text_array).to(final_device))
#             loss_t = loss_com_t(output_torch, torch.LongTensor(targets_array).to(final_device),num_classes)
#             loss_t.backward()
#             opt_t.step()
#
#             text_tensor, targets_tensor = ms.Tensor(text_array, dtype=mindspore.int32), ms.Tensor(targets_array,dtype=mindspore.int32)
#             loss_ms = train_step(text_tensor, targets_tensor,num_classes)
#
#             if batch % 100 == 0:
#                 print("batch:" + str(batch) + " torch_loss:" + str(loss_t.item())+" mindspre_loss:"+str(loss_ms))
#             batch += batch_size
#
#
#
#
#         # 测试步骤开始
#         model_t.eval()
#         model_ms.set_train(False)
#
#         test_data_size = 0
#         correct_torch = 0
#         correct_ms = 0
#
#
#         with torch.no_grad():
#             for item in test_iter:
#                 text, targets = item['data'], item['label']
#                 test_data_size +=  text.shape[0]
#                 text_array, targets_array = text.asnumpy(), targets.asnumpy()
#
#                 text_tensor,targets_tensor=torch.LongTensor(text_array).to(final_device),torch.LongTensor(targets_array).to(final_device)
#
#                 output_torch = model_t(text_tensor)
#                 output_ms = model_ms(text)
#                 indices_ms=np.argmax(output_ms.asnumpy(),axis=1)
#                 result_ms = (np.equal(indices_ms, targets.asnumpy()) * 1).reshape(-1)
#                 accuracy_ms = result_ms.sum()
#                 correct_ms = correct_ms + accuracy_ms
#
#                 indices = torch.argmax(output_torch.to(final_device), dim=1)
#                 result = (np.equal(indices.detach().cpu().numpy(), targets_tensor.detach().cpu().numpy()) * 1).reshape(-1)
#                 accuracy = result.sum()
#                 correct_torch = correct_torch + accuracy
#
#         print("Pytorch Test Accuracy: {}%".format(100 * correct_torch / test_data_size) + " " + "Mindpsore Test Accuacy: {}".format(correct_ms / test_data_size))
