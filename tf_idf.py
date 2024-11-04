import csv

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import pandas
from gensim.models import Word2Vec
import tensorflow as tf

warnings.filterwarnings("ignore",
                        message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")


# 加载停用词列表
def get_stopword():
    with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    return stop_words


# 自定义分词函数，同时过滤停用词
def jieba_tokenize(text):
    words = jieba.lcut(text)
    stop_words = get_stopword()
    return [word for word in words if word not in stop_words]


def get_tfidf(text):
    # 初始化TfidfVectorizer，传入自定义分词函数
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, tokenizer=jieba_tokenize)
    # 计算TF-IDF矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    tfidf_array = tfidf_matrix.toarray()
    tfidf_names = tfidf_vectorizer.get_feature_names_out()

    return tfidf_array, tfidf_names


def csv2vec(csv_path):
    df = pandas.read_csv(csv_path)
    data = df["News Url"]
    get_str = pandas.Series(data).fillna('').tolist()
    tfidf_array, feature_names = get_tfidf(get_str)

    # print(tfidf_array.shape)
    return tfidf_array, feature_names


def get_keywords(tfidf_array, feature_names):
    sum_tfidf = tfidf_array.sum(axis=0)
    words_tfidf = zip(feature_names, sum_tfidf)
    # 按TF-IDF值排序

    sorted_words_tfidf = sorted(words_tfidf, key=lambda x: x[1], reverse=True)
    keywords, _ = zip(*sorted_words_tfidf)
    return keywords[0:14999]


def sentence_to_avg(sentence, model):
    words = [word for word in sentence if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words], axis=0)


def my_word2vec():
    df = pandas.read_csv('./train/table_clean.csv', encoding='utf-8')
    get_str = pandas.Series(df['News']).fillna('').tolist()
    get_label = df['label']
    get_str2word = [jieba_tokenize(item) for item in get_str]
    get_index = df['id']
    filter_str2word = []
    with open('keyword.csv', 'r', encoding='utf-8') as f:
        rider = csv.reader(f)
        keyword = next(rider)
        for i in get_str2word:
            temp_str2word = [item for item in i if item in keyword]
            filter_str2word.append(temp_str2word)
    empty_index = [index for index, element in enumerate(filter_str2word) if element == []]
    filter_str = [element for element in filter_str2word if element != []]

    filter_id = [i for i in get_index if i not in empty_index]

    filter_label = [item for index, item in enumerate(get_label) if index not in empty_index]

    model = Word2Vec(filter_str, vector_size=10000, window=5, min_count=1, workers=4)
    get_vec = [sentence_to_avg(item, model) for item in filter_str]
    labels = ['id', 'vec', 'label']
    with open('./vec2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        for i in range(len(filter_str)):
            write_value = [filter_id[i], ",".join(map(str, get_vec[i])), filter_label[i]]
            writer.writerow(write_value)
        f.close()


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(102534,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_cut():
    # with open('./train/table_clean.csv', 'r', encoding='utf-8') as f:
    #     rider = csv.reader(f)
    df = pandas.read_csv('./train/table_clean.csv', encoding='utf-8')
    get_str = pandas.Series(df['News']).fillna('').tolist()
    get_label = df['label']
    get_id = df['id']
    get_office = df['office']
    get_title = df['title']
    str_list = []
    for i in get_str:
        temp_var = jieba_tokenize(i)
        temp_str = " ".join(temp_var)
        str_list.append(temp_str)
    with open('./train/clean_cut.csv', 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "office", "title", "News", "label"])
        for i in range(len(get_id)):
            value = [get_id[i], get_office[i], get_title[i], str_list[i], get_label[i]]
            writer.writerow(value)
        f.close()


if __name__ == "__main__":
    print(" ")
    get_cut()
    # df = pandas.read_csv('./train/table_clean.csv', encoding='utf-8')
    # get_str = pandas.Series(df['News']).fillna('').tolist()
    # get_label = df['label']
    # get_id = df['id']
    # get_office = df['office']
    # get_title = df['title']
    # print(type(get_str))
    # print(get_office[0])
    # with open('./train/clean_cut.csv', 'w', newline="", encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "office", "title", "News", "label"])
    #     for i in range(len(get_id)):
    #         value = [get_id[i], get_office[i], get_title[i], str_list[i], get_label[i]]
    #         writer.writerow(value)
    # f.close()
    # tfidf_array, feature_names = csv2vec("./train/train.csv")
    # # print(len(feature_names))
    # # reader = pandas.read_csv('./output.csv', encoding='utf-8')
    # df = pandas.read_csv('./train/train.csv', encoding='utf-8')
    # get_id = df['id']
    # get_label = df['label']
    # get_labels = tf.expand_dims(get_label, axis=-1)
    # train_data, test_data = tfidf_array[0:600], tfidf_array[600:1000]
    # train_labels, test_labels = get_labels[0:600], get_labels[600:1000]
    # # get_id = reader['id']
    # # get_vec = reader.iloc[:,1].tolist()
    # # print(len(get_vec[2]))
    # # print(get_vec)
    # # get_label = reader['label']
    # # get_labels = [float(i) for i in get_label]
    # # train_data, test_data = get_vec[0:600], get_label[600:1000]
    # # train_labels, test_labels = get_labels[0:600], get_labels[600:1000]
    # model = build_model()
    # model.fit(train_data, train_labels, epochs=100)
    # test_loss, test_acc = model.evaluate(train_data, train_labels)
    # predictions = model.predict(test_data)
    # print(predictions)
    # predicted_classes = np.argmax(predictions, axis=1)
    # print(predicted_classes)
    # accuracy = np.sum(predicted_classes == test_labels) / len(test_labels)
    # print(accuracy)
# keywords = get_keywords(tfidf_array, feature_names)
#     reader = pandas.read_csv('./train/train.csv', encoding='utf-8')
#     labels = reader['label']
#     with open('./output.csv', 'w',newline="", encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['id', feature_names, 'label'])
#         for i in range(len(tfidf_array)):
#             writer.writerow([i, tfidf_array[i],labels[i]])
