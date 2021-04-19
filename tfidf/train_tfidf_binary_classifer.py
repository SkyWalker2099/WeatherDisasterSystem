# -*- coding: UTF-8 -*-
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle
import numpy as np
def fenci(data):
    words_df = data.apply(lambda x: ' '.join(jieba.cut(x)))
    return words_df

def tfidf_build(data,savepath):
    """
    根据输入的文本训练一个tfidf模型并保存
    :param data: 输入文本,已分完词（列表）
    :param savepath: 保存路径
    """

    print("加载停用词")
    with open("stopwords.txt","r",encoding="utf8") as f:
        lines = f.readlines()
        stopwords = [line.strip() for line in lines]
    print(stopwords)
    # tv = CountVectorizer(stop_words=stopwords,max_features=5000,lowercase=False)
    tv = TfidfVectorizer(stop_words=stopwords,max_features=5000,lowercase=False)
    print("训练模型")
    tv.fit(data)
    print("保存tfidf模型")
    # print("保存cv模型")
    with open(savepath,'wb') as f:
        pickle.dump(tv,f)


def tfidf_clf_builder(news,labels,savepath):


    # 将label转为one_hot
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    print(labels)
    # labels = np.array([labels]).reshape(-1, 1)
    # print(labels)
    # labels = preprocessing.OneHotEncoder().fit_transform(labels)
    # labels = labels.toarray()
    # print(labels,labels.shape)

    tv = pickle.load(open("MODELS\\tfidf\\tfidf.pk","rb"))
    # tv = pickle.load(open("MODELS\\tfidf\\cv.pk", "rb"))
    news = fenci(news)
    news_tfidf = tv.transform(news)
    news_tfidf = news_tfidf.toarray()
    print(news_tfidf,news_tfidf.shape)

    # clf = MultinomialNB()
    clf = SVC(max_iter=10000)
    clf.fit(news_tfidf, labels)
    with open(savepath,"wb") as f:
        pickle.dump(clf,f)
    print(clf.score(news_tfidf,labels))


def test(titles,news,labels,model_path):
    classifications = ["disaster","not_disaster"]
    labels_ = labels
    labels_ = preprocessing.LabelEncoder().fit_transform(labels_)


    tv = pickle.load(open("MODELS\\tfidf\\tfidf.pk", "rb"))
    # tv = pickle.load(open("MODELS\\tfidf\\cv.pk", "rb"))
    news = fenci(news)
    news_tfidf = tv.transform(news)
    news_tfidf = news_tfidf.toarray()

    clf = pickle.load(open(model_path,"rb"))

    r = 0
    w = 0

    results = clf.predict(news_tfidf)

    for title,news,label,result in zip(titles,news,labels_,results):
        print(title[:10],classifications[label],classifications[result])
        if label == result:
            r+=1
        else:
            w+=1

    print("right:",r)
    print("wrong:",w)
    print("accurcy:",r/(r+w))


if __name__ == '__main__':
    train_data = pd.read_csv("DATA\\train_datas.csv")
    news = train_data["news"]
    titles = train_data["title"]
    labels = train_data["label"]

    tfidf_build(fenci(news),"MODELS\\tfidf\\tfidf.pk")
    tfidf_clf_builder(news,labels,"MODELS\\tfidf\\tfidf_svm_clf.pk")


    # test_data = pd.DataFrame(columns=["title","news","label"])
    # data1 = {"title":"test",
    #          "news":"""test""",
    #          "label":"disaster"}
    # test_data = test_data.append(data1,ignore_index=True)

    test_data = pd.read_csv("DATA\\test_datas.csv")
    news = test_data["news"]
    titles = test_data["title"]
    labels = test_data["label"]
    test(titles,news,labels,"MODELS\\tfidf\\tfidf_svm_clf.pk")





