import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from newFormat import get_data
from newFormat import reformatDataset

# # 2. 下载mnist数据集
# trainsets = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor()) # 格式转换
# Dataset = []
# testDataset = []
#
# print(trainsets)
#
#
# # for trainset in trainsets:
# #     Dataset.append(trainset)
# #
# # print(Dataset[0])
# #
# testsets = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# #
# #
# # for testset in testsets:
# #     testDataset.append(testset)
#
# # print(testsets)
# #%%
#
# class_names = trainsets.classes # 查看类别/标签
#
# print(class_names)
#
# #%%
#
# # 3. 查看数据集的大小shape
#
# print(trainsets.data.shape)
#
# #%%
#
# print(trainsets.targets.shape)
#
# #%%
#
# print(testsets.data.shape)
#
# #%%
#
# print(testsets.targets.shape)

#%%

# 4.定义超参数
BATCH_SIZE = 4 # 每批读取的数据大小
EPOCHS = 10 # 训练10轮

#%%

# 5. 创建数据集的可迭代对象，也就是说一个batch一个batch的读取数据

trainsets, testsets = get_data()

l = []
for data in trainsets:
    if data[1] not in l:
        l.append(data[1])
print(l)
trainsets = reformatDataset(trainsets, l)
testsets = reformatDataset(testsets, l)



train_loader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testsets, batch_size=BATCH_SIZE, shuffle=True)

#%%

# 查看一批batch的数据

images, labels = next(iter(test_loader))

#%%

print(images.shape)

#%%

print(labels.shape)

print('end')
# 1. 定义模型
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()  # 初始化父类中的构造方法
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 构建LSTM模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐层状态全为0
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 初始化cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 分离隐藏状态，以免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = self.fc(out[:, -1, :])
        return out


# %%

# 2. 初始化模型
input_dim = 28  # 输入维度，图片是28X28
hidden_dim = 100  # 隐层维度100
layer_dim = 1  # 1个隐层
output_dim = len(l)  # 输出维度：10， 即0-9共十个数字

model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)

# 判断是否有GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# %%

# 3. 循环查看
for i in range(len(list(model.parameters()))):
    print("参数：%d" % (i + 1))
    print(list(model.parameters())[i].size())

# %%

# 4. 初始化损失函数
criterion = nn.CrossEntropyLoss()

# 5. 初始化优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%

# 6. 模型训练

sequence_dim = 28  # 序列长度
loss_list = []  # 保存loss
accuracy_list = []  # 保存accuracy
iteration_list = []  # 循环次数
iter = 0
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # 声明模型训练
        # 一个batch的数据转换为RNN的输入维度
        images = images.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device)
        # 梯度清零（否则会不断累积）
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        # print(labels)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算器自增
        iter += 1
        # 模型验证
        if iter % 500 == 0:
            model.eval()
            # 计算accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集
            for images, labels in test_loader:
                # 一个batch的数据转换为RNN的输入维度
                images = images.view(-1, sequence_dim, input_dim).to(device)
                # 模型预测
                outputs = model(images)
                # 获取预测概率最大值的下标
                predict = torch.max(outputs.data, 1)[1]
                print(predict)
                # 统计label的数量
                total += labels.size(0)  # labels.size(0) = 32 ，即一个batchsize的大小
                # 统计预测正确的数量
                if torch.cuda.is_available():
                    correct += (predict.cuda() == labels.cuda()).sum()
                else:
                    correct += (predict == labels).sum()
            # 计算accuracy
            accuracy = correct / total * 100
            # 保存loss， accuracy，
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print('loop : {} Loss : {} Accuracy : {}'.format(iter, loss.item(), accuracy))

# %%

# 可视化loss
plt.plot(iteration_list, loss_list)
plt.xlabel('Number of Iteration')
plt.ylabel('Loss')
plt.title('LSTM')
plt.show()

# %%

# 可视化accuracy
plt.plot(iteration_list, accuracy_list, color='b')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('LSTM')
plt.savefig('LSTM_accuracy.png')
plt.show()