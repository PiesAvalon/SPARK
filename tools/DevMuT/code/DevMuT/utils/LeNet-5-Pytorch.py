import mindspore
import mindspore.nn as nn2
import torch
import torch.nn as nn1
import torchvision


class CNN_torch(nn1.Module):

    def __init__(self):
        # 调用父类的构造函数
        super(CNN_torch, self).__init__()
        # 第一层卷积池化， Sequential内的函数顺序执行
        # 原文中激活函数都是用的sigmoid，这里使用更好的ReLu
        self.conv_pool1 = nn1.Sequential(
            nn1.Conv2d(in_channels=1,  # input (1, 28, 28) padding to(1,32,32)
                       # 这里的input和output的值都是针对一个样本来说的，而训练时是一次输入一个batch
                       out_channels=6,
                       kernel_size=(5, 5),
                       padding=2
                       ),  # output(6, 28, 28)
            nn1.ReLU(),  # 激活函数
            nn1.MaxPool2d(2, stride=2)  # output(6, 14, 14)
        )

        self.conv_pool2 = nn1.Sequential(
            nn1.Conv2d(in_channels=6,
                       out_channels=16,
                       kernel_size=(5, 5)
                       ),  # output(16, 10, 10)
            nn1.ReLU(),
            nn1.MaxPool2d(2, stride=2)  # output(16, 5, 5)
        )

        # 全连接层
        self.fc1 = nn1.Sequential(  # 这里用全连接层代替原文的卷积层
            nn1.Linear(16 * 5 * 5, 120),
            nn1.ReLU()
        )

        # 全连接层
        self.fc2 = nn1.Sequential(
            nn1.Linear(120, 84),
            nn1.ReLU()
        )
        # 输出层
        self.out = nn1.Sequential(
            nn1.Linear(84, 10),

        )

    # 前向传播
    def forward(self, x):
        x = self.conv_pool1(x)
        x = self.conv_pool2(x)
        x = x.view(x.size(0), -1)  # resize to 2-dims(batch_size, 16*5*5) 展平成1维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


class CNN_ms(nn2.Cell):

    def __init__(self):
        # 调用父类的构造函数
        super(CNN_ms, self).__init__()
        # 第一层卷积池化， Sequential内的函数顺序执行
        # 原文中激活函数都是用的sigmoid，这里使用更好的ReLu
        self.conv_pool1 = nn2.SequentialCell(
            [nn2.Conv2d(in_channels=1,
                        pad_mode="pad",
                        padding=2,
                        out_channels=6,
                        kernel_size=(5, 5),
                        ),
             nn2.ReLU(),
             nn2.MaxPool2d(2, stride=2)]
        )
        self.reshape = mindspore.ops.Reshape()
        self.conv_pool2 = nn2.SequentialCell(
            [nn2.Conv2d(in_channels=6,
                        out_channels=16,
                        pad_mode="valid",
                        kernel_size=(5, 5)
                        ),
             nn2.ReLU(),
             nn2.MaxPool2d(2, stride=2)]
        )

        # 全连接层
        self.fc1 = nn2.SequentialCell(  # 这里用全连接层代替原文的卷积层
            [nn2.Dense(16 * 5 * 5, 120),
             nn2.ReLU()]
        )

        # 全连接层
        self.fc2 = nn2.SequentialCell([nn2.Dense(120, 84), nn2.ReLU()])
        # 输出层
        self.out = nn2.SequentialCell([nn2.Dense(84, 10), ])

    # 前向传播
    def construct(self, x):
        x = self.conv_pool1(x)
        x = self.conv_pool2(x)
        x = self.reshape(x, (-1, 400))  # resize to 2-dims(batch_size, 16*5*5) 展平成1维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


def data_loader(batch_size):
    # 将数据类型转换成tensor的函数
    transform = torchvision.transforms.ToTensor()

    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=True)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=True)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loaders, test_loaders


# def train_loop(model, x,y, loss_fn, optimizer):
#     # Define forward function
#     def forward_fn(data, label):
#         logits = model(data)
#         loss = loss_fn(logits, label)
#         return loss, logits
#
#     # Get gradient function
#     grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
#
#     # Define function of one-step training
#     def train_step(data, label):
#         (loss, _), grads = grad_fn(data, label)
#         loss =  mindspore.ops.depend(loss, optimizer(grads))
#         return loss
#
#
#     loss = train_step(x,y)
#     return loss


if __name__ == '__main__':
    model1 = CNN_torch()
    model2 = CNN_ms()
    epochs = 1

    batch_size = 100
    train_loader, test_loader = data_loader(batch_size)

    # 损失函数
    loss_fun1 = nn1.CrossEntropyLoss()
    loss_fun2 = nn2.CrossEntropyLoss()

    # 优化器
    learning_rate = 1e-2
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
    optimizer2 = nn2.SGD(params=model2.trainable_params(), learning_rate=learning_rate)


    def forward_fn(data, label):
        logits = model2(data)
        loss = loss_fun2(logits, label)
        return loss, logits

        # Get gradient function


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer2.parameters, has_aux=True)


    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer2(grads))
        return loss, grads


    for epoch in range(epochs):
        model1.train()
        model2.set_train()
        print("epoch:" + str(epoch))
        batch = 0

        for data in train_loader:
            imgs, targets = data
            imgs_array, targets_array = imgs.numpy(), targets.numpy()
            imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mindspore.float32), mindspore.Tensor(targets_array,
                                                                                                    mindspore.int32)

            output_torch = model1(imgs)
            loss_torch = loss_fun1(output_torch, targets)

            print(loss_torch.grad)

            # 优化器优化模型
            optimizer1.zero_grad()
            loss_torch.backward()
            optimizer1.step()

            loss_ms, gradient = train_step(imgs_ms, targets_ms)

            # print("gradinet: "+str(gradient))

            # loss_ms=train_loop(model2, imgs_ms,targets_ms, loss_fun2, optimizer2)
            if batch % 10000 == 0:
                print("batch:" + str(batch) + " torch_loss:" + str(loss_torch.item()) + " ms_loss:" + str(
                    loss_ms.asnumpy()))
            batch += batch_size

        # 测试步骤开始
        model1.eval()
        model2.set_train(False)
        test_data_size = 0
        total_test_loss = 0
        total_accuracy = 0
        test_loss_ms, correct_ms = 0, 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                imgs_array, targets_array = imgs.numpy(), targets.numpy()
                imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mindspore.float32), mindspore.Tensor(targets_array,
                                                                                                        mindspore.int32)

                test_data_size += len(imgs_ms)

                outputs_torch = model1(imgs)
                loss_torch = loss_fun1(outputs_torch, targets)

                pred_ms = model2(imgs_ms)
                test_loss_ms += loss_fun2(pred_ms, targets_ms).asnumpy()
                correct_ms += (pred_ms.argmax(1) == targets_ms).asnumpy().sum()

                total_test_loss = total_test_loss + loss_torch.item()
                accuracy = (outputs_torch.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        test_loss_ms /= test_data_size
        correct_ms /= test_data_size

        print("Pytorch Test Accuracy: {}%".format(
            100 * total_accuracy / test_data_size) + " " + "Pytorch Test Loss: {}".format(
            total_test_loss / test_data_size))
        print(f"Mindspore Test Accuracy: {(100 * correct_ms)}%, Mindspore Test Loss: {test_loss_ms}")  #
