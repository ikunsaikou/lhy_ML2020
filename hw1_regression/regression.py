import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

data = pd.read_csv('/dataset/train.csv', encoding='big5')

data = data.iloc[:, 3:]

data[data == 'NR'] = 0
raw_data = data.to_numpy()

# 提取特征 维度转化 12*18*20*24的math_data
month_data = np.zeros((12, 18, 480))
for month in range(12):
    sample = np.zeros([18, 480])  # 18个检测量，一个月取20天*24小时
    for day in range(20):
        # sample选取18行，每天的0-23列.后面的raw_data取该天的18个检测值的24小时的数据
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample # month_data 是12*18*480的数组

# 12*18*480 -->（471*12）*（18*9）
# 滑动窗口的公式计算 480-10+1= 471
# x的行有12*471个对应滑动窗口产生的数据集。x的列对应18个数据9个小时 18*9的特征值

train_data = np.zeros((471*12, 18*9))
label_data = np.zeros((471*12, 1))

for month in range(12):
    for i in range(471):
        train_data[i + 471*month, :] = month_data[month][:, i:i+9].flatten()
        label_data[i + 471*month, :] = month_data[month][9, i+9]

# 标准化转化为标准正态分布的数据
# （x - mean）/ std
mean_td = np.mean(train_data, axis = 0)
std_td = np.std(train_data, axis = 0)
for i in range(len(train_data)):
    for j in range(len(train_data[0])):
        if std_td[j] != 0:
            train_data[i][j] = (train_data[i][j] - mean_td[j]) / std_td[j]


# training

dim = 18 * 9 + 1 # 18*9个偏置w， 1个bias
w = 0.01 * np.random.randn(dim, 1) # 用正太分布初始化
train_data = np.concatenate((np.ones([12 * 471, 1]), train_data), axis = 1).astype(float) # 训练数据最前面接一列1,
learning_rate = 5
iter_time = 10000 # 迭代次数
adagrad = np.zeros([dim, 1]) #梯度累加,用来使用adagrad算法更新学习率
eps = 1e-10 # 因为新的学习率是learning_rate/sqrt(sum_of_pre_grads**2),而adagrad=sum_of_grads**2,所以处在分母上而迭代时adagrad可能为0，所以加上一个极小数，使其不除0

a_loss = []
a_x = np.arange(10000)
for i in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(train_data, w) - label_data, 2)) / 471 / 12) #损失函数，均方误差
    a_loss.append(loss)
    if i % 100 == 0: # 每迭代100次输出一次损失
        print(str(i) + ":" + str(loss))
    gradient = 2 * np.dot(train_data.transpose(), np.dot(train_data, w) - label_data) # 西瓜书梯度计算公式3.10
    adagrad += gradient ** 2 # adagrad用于保存前面使用到的所有gradient的平方，进而在更新时用于调整学习率
    w = w - learning_rate / np.sqrt(adagrad + eps) * gradient # 更新权重
np.save('weight.npy', w)
print(len(w))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(a_x,a_loss)
plt.xlabel('训练次数')
plt.ylabel('loss')
plt.axis([0, 10000, 0, 20])
plt.show()

# 训练集读入与处理

test_data = pd.read_csv('/dataset/test.csv', header = None, encoding ='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_td[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_td[j]) / std_td[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

#写文件
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

