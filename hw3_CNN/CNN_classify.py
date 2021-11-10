import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

from torch.utils.data.dataset import T_co


def readfile(path, label):
    image_dir = sorted(os.listdir(path))  # 返回一个文件名称列表
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):  # i 为索引下标，file为路径
        img = cv2.imread(os.path.join(path, file))  # 返回一个三维np数组（height，width，3）
        x[i, :, :] = cv2.resize(img, (128, 128))  # 给第i个图片的0-128行与0-128列 的每一个RGB赋值
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


print("reading data...")
workspace_dir = './food-11'
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print(f'size of training data: {len(train_x)}')
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print(f'size of val data: {len(val_x)}')
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print(f'size of test data: {len(test_x)}')

# 数据增强，将图片进行变换后作为训练数据
# 调用时会compose类遍历每个列表，对图片依次进行列表中的变换后返回
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),  # 转换为张量返回
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# 重写dataset类
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = self.x[idx]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[idx]
            return X, Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(

            # 第一层
            nn.Conv2d(3, 64, 3, 1, 1),  # 设置二维卷积参数,输入3(R,G,B) 输出64，卷积核3， 步长1， padding 1。因为设置了padding 因此不会改变图片的宽高
            nn.BatchNorm2d(64),  # 对每一层进行归一处理
            nn.ReLU(),  # 同归激活函数
            nn.MaxPool2d(2, 2, 0),  # 第一层的池化层参数 池化层窗口大小， 步长， 填充

            # 通过第一层后维度 64*64*64

            # 第二层
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 池化后图片 32 * 32

            # 第三层
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 池化后图片 16 * 16

            # 第四层
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 8*8

            # 第五层
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 4*4
        )

        # 用全连接层降维
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),  # 该线性全连接层将通过5层后的512 * 4 * 4 降维乘1024
            nn.ReLU(),
            nn.Linear(1024, 512),  # 1024 降维至512
            nn.ReLU(),
            nn.Linear(512, 11)  # 512 降维至11个类别
        )

    def forward(self, x):  # 这里由于继承了nn.module类。会在call方法中调用该方法
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # 转换成全连接网络的数据
        return self.fc(out)


model = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())  # 预测值和标签必须都在cuda上
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())  # argmax 返回nparray的某个轴的最大值下标
        train_loss += batch_loss.item()

    model.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新。
    with torch.no_grad():  # 防止GPU爆
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

train_val_x = np.concatenate((train_x, val_x), axis=0)  # 行拼接训练集和验证集
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())  # 预测的结果
        batch_loss = loss(train_pred, data[1].cuda())  # data[1]代表真实值
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # 將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time, \
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
