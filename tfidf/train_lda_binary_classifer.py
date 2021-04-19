# -*- coding: UTF-8 -*-
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import preprocessing
import pickle
import numpy as np
def fenci(data):
    words_df = data.apply(lambda x: ' '.join(jieba.cut(x)))
    return words_df


def lda_build(data,savepath,n_topic):
    """
    在原有tfidf或cv的基础上训练lda
    :param data:
    :return:
    """

    tv = pickle.load(open("MODELS\\tfidf\\tfidf.pk", "rb"))
    # tv = pickle.load(open("MODELS\\tfidf\\cv.pk", "rb"))
    data = fenci(data)
    data_tfidf = tv.transform(data)
    data_tfidf = data_tfidf.toarray()
    print(data_tfidf.shape)
    lda = LatentDirichletAllocation(n_components=n_topic,max_iter=1000,verbose=True)
    lda.fit(data_tfidf)

    with open(savepath,"wb") as f:
        pickle.dump(lda,f)
    print(lda.perplexity(data_tfidf))


def lda_clf_build(news,labels,savepath):
    labels = preprocessing.LabelEncoder().fit_transform(labels)

    tv = pickle.load(open("MODELS\\tfidf\\tfidf.pk", "rb"))
    # tv = pickle.load(open("MODELS\\tfidf\\cv.pk", "rb"))
    lda = pickle.load(open("MODELS\\lda\\lda.pk", "rb"))
    # lda = pickle.load(open("MODELS\\lda\\lda2.pk","rb"))

    news_fenci = fenci(news)
    news_tfidf = tv.transform(news_fenci)
    news_lda = lda.transform(news_tfidf)

    print(news_lda,news_lda.shape)
    print(labels,labels.shape)

    # clf = MultinomialNB()
    clf = SVC(max_iter=10000)
    clf.fit(news_lda,labels)
    with open(savepath,"wb") as f:
        pickle.dump(clf,f)
    print(clf.score(news_lda,labels))


def accuracy_test(titles,news,labels,model_path):
    classifications = ["disaster","not_disaster"]
    labels_ = labels
    labels_ = preprocessing.LabelEncoder().fit_transform(labels_)



    tv = pickle.load(open("MODELS\\tfidf\\tfidf.pk", "rb"))
    # tv = pickle.load(open("MODELS\\tfidf\\cv.pk", "rb"))
    lda = pickle.load(open("MODELS\\lda\\lda.pk", "rb"))
    # lda = pickle.load(open("MODELS\\lda\\lda2.pk", "rb"))
    news = fenci(news)
    news_tfidf = tv.transform(news)
    news_tfidf = news_tfidf.toarray()
    news_lda = lda.transform(news_tfidf)
    print(news_lda.shape)

    clf = pickle.load(open(model_path,"rb"))

    r = 0
    w = 0

    results = clf.predict(news_lda)

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
    #
    #
    lda_build(fenci(news),"MODELS\\lda\\lda.pk",n_topic=100)
    lda_clf_build(news, labels, "MODELS\\lda\\lda_svm_clf.pk")

    test_data = pd.read_csv("DATA\\test_datas.csv")
    news = test_data["news"]
    titles = test_data["title"]
    labels = test_data["label"]
    accuracy_test(titles, news, labels, "MODELS\\lda\\lda_svm_clf.pk")
