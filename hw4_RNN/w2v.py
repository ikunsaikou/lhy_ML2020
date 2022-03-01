import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec  # 导入gensim NLP开发包
from hw4_RNN.utils import load_training_data, load_testing_data

path_prefix = './'
def train_word2vec(x):
    model = word2vec.Word2Vec(x, vector_size=250, window=5, min_count=5, workers=12, epochs=10, sg=1)
    return model


if __name__ == '__main__':
    print("读取训练数据中 ...")
    train_x, y = load_training_data('./data/training_label.txt')
    train_x_no_label = load_training_data('./data/training_nolabel.txt')

    print("读取测试数据中 ...")
    test_x = load_testing_data('./data/testing_data.txt')

    model = train_word2vec(train_x + test_x)

    print("保存模型 ...")
    model.save(os.path.join(path_prefix, 'w2v_all.model'))