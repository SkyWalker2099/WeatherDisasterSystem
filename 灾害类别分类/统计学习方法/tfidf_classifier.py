# -*- coding:UTF-8 -*-
# editor: zzh
# date: 2020/10/22

from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
# from keras.utils import np_utils
import pickle
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

# def lables2onehot(data, num_class):
#     """
#             字符串标签类别转为one_hot类型数据
#             :param data:
#             :param num_class:
#             :return:
#             """
#     le = preprocessing.LabelEncoder()
#     target = le.fit_transform(data)
#     target = np_utils.to_categorical(target, num_classes=num_class)
#     return target

# def build_model():
#     train_data = pd.read_csv("DATAS\\DATA6\\train_data.csv")
#     newss = train_data["news"]
#     labels = train_data["label"]
#     corpus = newss.apply(lambda x: " ".join(list(jieba.cut(x)))).values
#     print(corpus)
#
#     stop_words = []
#     with open("stop_words_ch-停用词表.txt", 'r', encoding='gbk') as f:
#         words = f.readlines()
#         for w in words:
#             stop_words.append(w.strip())
#
#     print(stop_words)
#     vectorizer = TfidfVectorizer(stop_words=stop_words)
#     X = vectorizer.fit_transform(corpus)
#     Y = lables2onehot(labels, num_class=12)
#     Y = np.argmax(Y, axis=1)
#     svc = SVC(max_iter=20000)
#
#     print(X.shape)
#     print(Y.shape)
#
#     svc.fit(X=X, y=Y)
#     pickle.dump(svc, open("MODELS\\svc.pk", 'wb'))
#     pickle.dump(vectorizer, open("MODELS\\tfidf_vectorizer.pk", 'wb'))
#
# def test():
#     test_data = pd.read_csv("DATAS\\DATA6\\test_data.csv")
#     titles = test_data["title"]
#     newss = test_data["news"]
#     labels = test_data["label"]
#     corpus = newss.apply(lambda x: " ".join(list(jieba.cut(x)))).values
#     # print(corpus)
#     stop_words = []
#     with open("stop_words_ch-停用词表.txt", 'r', encoding='gbk') as f:
#         words = f.readlines()
#         for w in words:
#             stop_words.append(w.strip())
#     # print(stop_words)
#     vectorizer = TfidfVectorizer(stop_words=stop_words)
#     Y = lables2onehot(labels, num_class=12)
#     Y = np.argmax(Y, axis=1)
#     svc = SVC(max_iter=20000)
#     vectorizer = pickle.load(open("MODELS\\tfidf_vectorizer.pk", "rb"))
#     svc = pickle.load(open("MODELS\\svc.pk", "rb"))
#     results = svc.predict(vectorizer.transform(corpus))
#     c = 0
#     w = 0
#     for t, r, y in zip(titles, results, Y):
#         if r == y:
#             c += 1
#         else:
#             w += 1
#     print(c)
#     print(w)
#     print(c / (c + w))


tv = pickle.load(open("../MODELS/统计学习/tfidf_vectorizer.pk", "rb"))
svc = pickle.load(open("../MODELS/统计学习/svc.pk", "rb"))

def predict(zw):
    wlist = " ".join(jieba.lcut(zw))
    corpu = tv.transform([wlist])
    result = svc.predict(corpu)
    return result[0]

if __name__ == '__main__':

    path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第二次测试"
    for classification in classifications:
        files = os.listdir(path + '\\' + classification)
        for file in files:
            with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace(
                    '\u3000', '') for content in contents if len(content) > 2])
                # print(len(zw))
                index = predict(title+zw)
                try:
                    print("文件名：", file.strip())
                    print("新闻标题：", title.strip())
                except Exception:
                    pass
                print("模型结果：", classifications[index], " 正确结果：", classification,
                      classifications[index] == classification, '\n')


