import numpy as np

np.random.seed(0)
X_train_fpath = './dataset/X_train'
Y_train_fpath = './dataset/Y_train'
X_test_fpath = './dataset/X_test'
output_fpath = './output_{}.csv'

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
print(X_train.shape[0])
print(X_train.shape[1])


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
                                                            -1)  # 将x的所有行和特定列传入，axis = 0 对个列求均值 reshape(1,-1)转换为行数1, 列数根据行数计算
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


def _shuffle(X, Y):
    # 该函数对两个等长的列表或数组洗牌(XY同步).
    randomize = np.arange(len(X))  # arange(len(X))返回0,1...len(X)
    np.random.shuffle(randomize)  # shuffle()方法将序列的所有元素随机排序，打乱randomize中的元素
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
    return np.round(_f(X, w, b)).astype(np.int)


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
    w_grad = -np.sum(pred_error * X.T, 1)  # sum参数axis=1是压缩列,即将每一行的元素相加,将矩阵压缩为一列
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad
