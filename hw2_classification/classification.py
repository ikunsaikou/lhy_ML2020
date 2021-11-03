import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X_train_fpath = './dataset/X_train'
Y_train_fpath = './dataset/Y_train'
X_test_fpath = './dataset/X_test'
output_fpath = './output/output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)  # 返回文件下一行
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

print(X_train)
print(X_test.shape[0])
print(X_test.shape[1])


def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # 这个函数对X中的特定列标准化.
    # 训练集的平均值与标准差会在测试集中再次被使用
    #
    # 参数:
    #     X: 被预测数据
    #     train: 'True' 训练集, 'False' 测试集
    #     specific_column: 要被标准化的列索引, 'None'代表标准化所有列.
    #     X_mean: 训练集的平均值, 当train = 'False'即测试集中使用.
    #     X_std: 训练集的标准差, 当train = 'False'即测试集中使用.
    # 输出:
    #     X: 标准化后数据
    #     X_mean: 训练集的平均值
    #     X_std: 训练集的平均值

    if specified_column == None:
        specified_column = np.arange(X.shape[1])  # shape[1]代表矩阵的列数 = 510
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1,
                                                            -1)  # 将x的所有行和特定列传入，axis = 0 对个列求均值 reshape(1,-1)转换为行数1, 列数根据X数据列数计算（1*510）
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)  # (x-μ)/σ  X_std加入一个很小的数防止分母除以0
    return X, X_mean, X_std


# 将训练集拆成训练集和验证集，默认值是0.25，可以调
def _train_dev_split(X, Y, dev_ratio=0.25):
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]  # 省略对列的操作，仅对行进行分割列不变


# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, X_mean=X_mean, X_std=X_std)  # '_'做变量名合法, 作为无用的临时变量

# 分割训练集, 发展集 按9:1划分
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


def _shuffle(X, Y):
    # 该函数对两个等长的列表或数组洗牌(XY同步).
    randomize = np.arange(len(X))  # arange(len(X))返回0,1...len(X)
    np.random.shuffle(randomize)  # shuffle()方法将行的随机排序，打乱randomize中的元素
    return (X[randomize], Y[randomize])


def _sigmoid(z):
    # Sigmoid函数用来计算概率.
    # 为避免溢出, 限定输出的最大最小值.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))  # clip将数限定到范围1e-8和1-(1e-8)中


def _f(X, w, b):
    # 对率回归函数, 用w b作为参数
    #
    # 参数:
    #     X: 输入数据, shape = [小批次的尺寸batch_size, data_dimension]
    #     w: weight 向量, shape = [data_dimension, ]
    #     b: bias, 标量
    # 输出:
    #     每一行X为正例的预测概率, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)  # matmul矩阵相乘


def _predict(X, w, b):
    # 通过把对率回归函数的结果四舍五入(round), 并转换类型, 得到每行X真正的预测值.
    return np.round(_f(X, w, b)).astype(int)


def _accuracy(Y_pred, Y_label):
    # 该函数计算预测精确度
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))  # 误差1/n*Σ|y*-y|
    return acc


def _cross_entropy_loss(y_pred, Y_label):
    # 该函数计算交叉熵.
    #
    # 参数:
    #     y_pred: 预测概率, float vector
    #     Y_label: 标准答案(ground truth)标记, bool vector
    # Output:
    #     cross entropy交叉熵, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))  # log默认以e为底
    return cross_entropy


def _gradient(X, Y_label, w, b):
    # 关于weight w和bias b, 用交叉熵损失计算梯度.
    y_pred = _f(X, w, b)  # 对率回归
    pred_error = Y_label - y_pred  # 误差
    w_grad = -np.sum(pred_error * X.T, 1)  # sum参数axis=1是压缩列,即将每一列的元素相加
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


# 将weights和bias初始化为0
w = np.zeros((data_dim,))  # zeros第一个参数为形状
b = np.zeros((1,))

# 训练超参数
max_iter = 272
batch_size = 128
learning_rate = 0.1
regularization_weight = 0.005

# 保存每一次迭代的损失与精确度, 用来画图
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# 计参数更新次数(步数)
step = 1

# 迭代式训练
for epoch in range(max_iter):
    # 每一代(epoch)随机洗牌
    X_train, Y_train = _shuffle(X_train, Y_train)

    # 小批训练
    for idx in range(int(np.floor(train_size / batch_size))):  # np.floor()返回不大于输入参数的最大整数。（向下取整）
        # 取这一小批的X, Y
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # 计算梯度
        w_grad, b_grad = _gradient(X, Y, w, b)

        # 利用梯度下降更新参数wb
        # 学习率随时间(step)减少
        w = w * (1 - (learning_rate / np.sqrt(step)) * regularization_weight) - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step = step + 1

    # 计算训练集与发展集的损失与精确度
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)  # / train_size消除训练集与发展集大小不同带来的影响

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))  # [-1]表示数组中最后一位
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

# Loss curve
plt.plot(train_loss)  # x可省略,默认[0,1..,N-1]递增
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])  # 默认参数: 图例的名称
plt.savefig('./output/loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# 预测测试集的标记
predictions = _predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):  # enumerate列出数据和生成数据下标
        f.write('{},{}\n'.format(i, label))

# 筛选最大(显著)的几个weight, 即得到最有用的特征
ind = np.argsort(np.abs(w))[::-1]  # argsort从小到大排序, ::-1从后向前读取
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
