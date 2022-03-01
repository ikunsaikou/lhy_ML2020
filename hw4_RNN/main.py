import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import pandas as pd
from hw4_RNN.data import TwitterDataset
from hw4_RNN.model import LSTM_Net
from hw4_RNN.preprocess import Preprocess
from hw4_RNN.train import training
from hw4_RNN.test import testing
from hw4_RNN.utils import load_training_data, load_testing_data

path_prefix = './'
device = torch.device("cuda")

train_with_label = os.path.join(path_prefix, 'data/training_label.txt')
train_no_label = os.path.join(path_prefix, 'data/training_nolabel.txt')
testing_data = os.path.join(path_prefix, 'data/testing_data.txt')

w2v_path = os.path.join(path_prefix, './w2v_all.model')

sen_len = 20
fix_embedding = True
batch_size = 128
epoch = 5
lr = 0.001

model_dir = path_prefix

print("读取数据...")
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)

preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)  # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把 data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

# 開始測試模型並做預測
print("loading testing data ...")
test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
model.to(device)
outputs = testing(batch_size, test_loader, model, device)

# 寫到 csv 檔案供上傳 Kaggle
tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
print("save csv ...")
tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
print("Finish Predicting")
