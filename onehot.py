import pickle

import pandas
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import csv


def get_csv_inf():
    df = pandas.read_csv('./train/clean_cut.csv')
    # 创建文本样本
    text_samples = df['News']
    index = df['id']
    return text_samples, index


# 实例化Tokenizer

def get_onehot():
    text_samples, index = get_csv_inf()
    tokenizer = Tokenizer(num_words=6000)

    # 拟合Tokenizer并构建词汇表
    tokenizer.fit_on_texts(text_samples)

    # 将文本转换为整数序列
    sequences = tokenizer.texts_to_sequences(text_samples)

    # 将文本转换为One-Hot编码矩阵
    one_hot_matrix = tokenizer.texts_to_matrix(text_samples)
    word_index = tokenizer.word_index
    return one_hot_matrix, word_index, index


def save_onehot(word_index):
    with open("./word_index.pkl", 'wb') as f:
        pickle.dump(word_index, f)


def save2csv(index, one_hot_matrix=None):
    one_hot_matrix_list = one_hot_matrix.tolist()
    with open("./train/word_index.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "one_hot_vector"])
        for i in range(one_hot_matrix.shape[0]):
            one_hot_vector = one_hot_matrix_list[i]
            writer.writerow([index[i], one_hot_vector])


if __name__ == '__main__':
    # _, word_index = get_onehot()
    # save_onehot(word_index)
    one_hot_matrix, word_index,index = get_onehot()
    save_onehot(word_index)
    save2csv(index, one_hot_matrix)
