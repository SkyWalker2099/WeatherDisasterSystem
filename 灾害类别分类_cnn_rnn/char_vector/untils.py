# -*- coding:utf-8 -*-
# editor: zzh

import numpy as np
import pandas as pd
from collections import *
import pickle
import re

pattern = re.compile('[\u4e00-\u9fa5]+')

char2id = pickle.load(open("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\char_vector\char2ids.pkl", 'rb'))

# vector = np.load("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\char_vector\vectors.npy.npy")

def sentence2ids(sentence, max_length = 20):
    # print(char2id['<PAD>'],char2id['<UNK>'],char2id['</s>'])
    xs = ['</s>'] + list(sentence)
    # print(xs)
    ids = []
    # print(word2id)
    for x in xs:
        if(x in char2id.keys()):
            ids.append(char2id[x])
        else:
            ids.append(char2id['<UNK>'])

    while(len(ids) < max_length):
        ids.append(char2id['<PAD>'])
    return np.array(ids[:max_length])

def build_vocab_tokenizer():

    c2v = pickle.load(open("char2vec.pkl",'rb'))

    char2ids = {}
    vectors = []

    char2ids["<PAD>"] = len(vectors)
    vectors.append(np.zeros([50]))

    char2ids["<UNK>"] = len(vectors)
    vectors.append(np.random.random(vectors[0].shape))

    for k in c2v.keys():
        char2ids[k] = len(vectors)
        vectors.append(c2v[k])


    vectors = np.array(vectors)
    print(vectors.shape)

    pickle.dump(char2ids,open("char2ids.pkl",'wb'))
    np.save("vectors",vectors)

def load_pretrain_c2v(path):
    with open(path,encoding='utf8') as f:
        lines = f.readlines()
        char2vec = {}
        for line in lines[:10000]:
            tokens = line.rstrip().split(' ')
            token = tokens[0]
            vector = np.array([float(t) for t in tokens[1:]])
            char2vec[token] = vector
            print(token)
        pickle.dump(char2vec, open("char2vec.pkl", 'wb'))


if __name__ == '__main__':
    # load_pretrain_c2v("gigaword_chn.all.a2b.uni.ite50.vec")
    # build_vocab_tokenizer()
    s = "萧县一名村民不幸被雷电击中"
    print(sentence2ids(s))
    # print(pcut(s))
    # build_vocab_tokenizer()
    pass