import math
import numpy as np
import torch
from torch import Tensor, nn, LongTensor,FloatTensor
from tqdm import tqdm

from src.dataset import MovieReview
from src.textcnn_torch import TextCNN


device="cpu"


def loss_com(logit, label):
    """
    construct
    """
    # print("logit", logit.shape, "label", label.shape)
    exp = torch.exp
    reduce_sum = torch.sum
    onehot = nn.functional.one_hot
    # on_value = Tensor([1.0])
    # off_value = Tensor([0.0])
    div = torch.div
    log = torch.log
    sum_cross_entropy = torch.sum
    mul = torch.Tensor.mul
    reduce_mean = torch.std_mean
    reduce_max = torch.max  # keep_dims=True
    sub = torch.Tensor.sub
    logit_max, _ = reduce_max(logit, -1, keepdim=True)
    # print(logit_max)
    exp0 = exp(sub(logit, logit_max))
    # print(exp0)
    exp_sum = reduce_sum(exp0, -1, keepdim=True)
    # print(exp_sum)
    softmax_result = div(exp0, exp_sum)
    # print(softmax_result)
    label = onehot(label, num_classes)
    # print(label)
    softmax_result_log = log(softmax_result)
    # print("softmax_result_log.shape1", softmax_result_log.shape)
    loss = sum_cross_entropy((mul(softmax_result_log, label)), -1, keepdim=False)
    # print(loss)
    loss = mul(Tensor([-1.0]).to(device), loss)
    # loss, _ = reduce_mean(loss, -1, keepdim=False)
    loss, _ = reduce_mean(loss, -1, keepdim=False)
    return loss


if __name__ == '__main__':
    epoch_num, batch_size = 5, 64
    num_classes=2
    instance = MovieReview(root_dir="data/rt-polaritydata", maxlen=51, split=0.9)
    train_dataset = instance.create_train_dataset(batch_size=batch_size, epoch_size=epoch_num)
    test_dataset = instance.create_train_dataset(batch_size=batch_size, epoch_size=epoch_num)
    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=51,
                  num_classes=num_classes, vec_length=40).to(device)
    train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=float(1e-3),
                           weight_decay=float(3e-5))

    from torchsummaryDemo import summary

    summary(net,(51,))

    for epoch in range(epoch_num):
        net.train()
        print("epoch", epoch, "/", epoch_num)
        batch = 0
        for item in train_iter:
            text_array, targets_array = item['data'].numpy(), item['label'].numpy()
            print(text_array.shape)
            print(targets_array.shape)

            break
    #         output_torch = net(torch.LongTensor(text_array).to(device))
    #         loss1 = loss_com(output_torch, torch.LongTensor(targets_array).to(device))
    #         opt.zero_grad()
    #         loss1.backward()
    #         opt.step()
    #         if batch % 100 == 0:
    #             print("batch:" + str(batch) + " torch_loss:" + str(loss1.item()))
    #         batch += batch_size
    #         # 测试步骤开始
    #     net.eval()
    #     total_test_loss = 0
    #     total_accuracy = 0
    #     test_data_size = 0
    #     with torch.no_grad():
    #         for item in test_iter:
    #             text, targets = item['data'], item['label']
    #             text_array, targets_array = text.numpy(), targets.numpy()
    #             # text_ms, targets_ms = mindspore.Tensor(text_array, mindspore.float32), mindspore.Tensor(targets_array,
    #             #                                                                                         mindspore.int32)
    #             test_data_size +=  text.shape[0]
    #             output_torch = net(LongTensor(text_array).to(device))
    #             loss_torch = loss_com(output_torch, LongTensor(targets_array).to(device))
    #             total_test_loss = total_test_loss + loss_torch.item()
    #             # print("output_torch.argmax(1)", output_torch.argmax(1))
    #             # print("targets", targets)
    #             # 此处accuracy存在bug
    #             indices = torch.argmax(output_torch.to("cpu"), dim=1)
    #             result = (np.equal(indices, targets) * 1).reshape(-1)
    #             accuracy = result.sum()
    #             total_accuracy = total_accuracy + accuracy
    #
    #     print("Pytorch Test Accuracy: {}%".format(
    #         100 * total_accuracy / test_data_size) + " " + "Pytorch Test Loss: {}".format(
    #         total_test_loss / test_data_size))



