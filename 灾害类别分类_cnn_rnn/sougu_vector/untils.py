# -*- coding:utf-8 -*-
# editor: zzh

import numpy as np
import pandas as pd
from collections import *
import jieba
import pickle
import re
import jieba.posseg as psg

pattern = re.compile('[\u4e00-\u9fa5]+')

unk = "<UNK>"
pad = "<PAD>"

m = "<M>"
ns = "<NS>"
nt = "<NT>"
v = "<V>"

default_dict = ['m','ns','nt','v']
stop_dict = []
with open("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\stop_words_ch-停用词表.txt", 'r', encoding='gbk') as f:
    lines = f.readlines()
    for l in lines:
        stop_dict.append(l.strip())

word2id = pickle.load(open("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\sougu_vector\word2ids.pkl",'rb'))

def sentence2ids(sentence, max_length = 15):
    xs = pcut(sentence)
    ids = []
    # print(word2id)
    for x in xs:
        if(x[0] in word2id.keys()):
            ids.append(word2id[x[0]])
        elif(x[1] in default_dict):
            ids.append(word2id['<'+x[1].upper()+'>'])
        else:
            ids.append(word2id['<UNK>'])

    while(len(ids) < max_length):
        ids.append(word2id['<PAD>'])
    return np.array(ids[:max_length])

def pcut(sentence):
    sentence = sentence.replace(" ","")
    words = psg.cut(sentence)
    words = [(x.word, x.flag) for x in words if x.word not in stop_dict]
    return words

def lcut(content):
    content = content.replace(" ", "")
    words = jieba.lcut(content, cut_all=False)
    words = [w for w in words if w not in stop_dict]
    return words

def build_vocab_tokenizer():
    # datas = pd.read_csv("../灾害类别分类/DATAS/DATA6/train_data.csv")
    # titles = datas["title"]
    # newss = datas["news"]
    # result = []
    # for t,n in zip(titles,newss):
    #     twords = lcut(t)
    #     nwords = lcut(n)
    #     result = result + twords
    #     result = result + nwords
    #
    # counter = Counter(result)
    # pickle.dump(counter,open("word_counter.pkl",'wb'))
    # print(counter)
    dim = 300
    wc = pickle.load(open("../word_counter.pkl", 'rb'))
    sogu_w2v = pickle.load(open("word2vec.pkl", 'rb'))
    sogu_keys = sogu_w2v.keys()
    print(len(wc.keys()))
    word2id = {}
    vectors = []

    word2id[pad] = len(vectors)
    vectors.append(np.zeros(sogu_w2v["北京"].shape))

    word2id[unk] = len(vectors)
    vectors.append(np.random.random(vectors[0].shape))

    word2id[ns] = len(vectors)
    vectors.append(sogu_w2v["北京"])

    word2id[nt] = len(vectors)
    vectors.append(sogu_w2v["政府"])

    word2id[v] = len(vectors)
    vectors.append(sogu_w2v["袭击"])

    word2id[m] = len(vectors)
    vectors.append(sogu_w2v["一百"])

    for k in wc.keys():
        if (wc[k] >= 10 and pattern.match(k)):
            if k in sogu_keys:
                word2id[k] = len(vectors)
                vectors.append(sogu_w2v[k])

    print(word2id)
    vectors = np.array(vectors,dtype=float)
    print(vectors.shape)

    np.save("vectors",vectors)
    pickle.dump(word2id,open("word2ids.pkl",'wb'))

def load_pretrain_w2v(path):
    with open(path,encoding='utf8') as f:
        first_line = f.readline()
        len = int(first_line.split()[0])
        dim = int(first_line.split()[1])
        print(len,dim)
        lines = f.readlines()
        word2vec = {}
        for line in lines:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.array([float(t) for t in tokens[1:]])
            word2vec[word] = vector
            print(word)
        pickle.dump(word2vec, open("word2vec.pkl", 'wb'))


if __name__ == '__main__':
    s = "萧县一名村民不幸被雷电击中"
    # print(sentence2ids(s))
    # print(pcut(s))
    # build_vocab_tokenizer()
    pass