# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/12/3
# -*- coding: UTF-8 -*-
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert import gen_batch_inputs

import keras.backend as K

import codecs
from random import choice
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
from sklearn import preprocessing

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import os,shutil
import re
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 目的是扩充vocab.txt 文件 ， 实现这个类中的方法
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R  # 匹配字典集


def lables2onehot(data, num_class):
    """
            字符串标签类别转为one_hot类型数据
            :param data:
            :param num_class:
            :return:
            """
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(data)
    target = np_utils.to_categorical(target, num_classes=num_class)
    return target

classifications = ["disaster","not_disaster"]

classifications2 = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'

class HierarchicalBertWeatherRecongizer(object):

    def build_model(self,trainable = False):
        bert_model = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,seq_len=None)
        for l in bert_model.layers:
            l.trainable = trainable #设定为BERT可训练
        x1_in = Input(shape=(self.max_piece_num,768))
        x2_in = Input(shape=(self.max_piece_num,768))

        lam = Lambda(lambda x:x[:,0])

        x = []

        for i in range(self.max_piece_num):
            o = bert_model([x1_in[i],x2_in[i]])
            x.append(o)

        x = lam(x)
        x_in2 = Masking(mask_value=0,input_shape=(self.max_piece_num,768))(x)
        encoded_text = Bidirectional(LSTM(units=100, return_sequences=False))(x_in2)
        # encoded_text = LSTM(100)(x_in)
        out_dense = Dense(30,activation='relu')(encoded_text)
        out = Dense(2,activation='sigmoid')(out_dense)
        self.model = Model([x1_in,x2_in],out)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()


    def __init__(self,config_path,checkpoint_path,dict_path,split_len=200,overlap_len=30):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path  # 三个BERT文件，需修改路径
        self.token_dict = {}
        self.token_list = []
        self.split_len = split_len
        self.overlap_len = overlap_len
        self.maxlen = split_len
        self.max_piece_num = 10
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)  # 读取BERT字典
                self.token_list.append(token)
        self.tokenizer = OurTokenizer(self.token_dict)
        self.build_model()


    def encode(self,content):
        content = content[:self.maxlen]
        lines = re.split(r'[，,。.？?!！]',content)
        x1 = [self.token_dict["[CLS]"]]
        x2 = [0]
        for idx,line in enumerate(lines):
            line_encode,_ = self.tokenizer.encode(first=line)
            line_encode = line_encode[1:]
            x1 = x1 + line_encode
            x2 = x2 + [idx%2]*len(line_encode)

        if len(x1) >= self.maxlen:
            x1 = x1[:self.maxlen]
            x2 = x2[:self.maxlen]
        else:
            x1 += [0]*(self.maxlen-len(x1))
            x2 += [0]*(self.maxlen-len(x2))

        return x1,x2

    def get_encode2(self,data):
        all_data = data
        X1 = []
        X2 = []
        for i in all_data:
            # x1, x2 = self.tokenizer.encode(first=i, max_len=self.maxlen)
            x1,x2 = self.encode(i)
            X1.append(x1)
            X2.append(x2)

        return [X1, X2]

    def get_encode(self,data):
        all_data = data
        X1 = []
        X2 = []
        for i in all_data:
            x1, x2 = self.tokenizer.encode(first=i, max_len=self.maxlen)
            X1.append(x1)
            X2.append(x2)
        return [X1, X2]

    def mask_data(self,datas,mask_rate):
        """
        随机在句子中添加mask
        :param data:
        :return:
        """
        # print(datas)
        datas_ = []
        def mask_data(data, mask_rate):
            len = data.index(self.token_dict["[SEP]"]) - 1
            mask_pos = random.sample(range(1, len + 1), int(len * mask_rate))
            lis = [0]*self.maxlen
            for pos in mask_pos:
                data[pos] = self.token_dict["[MASK]"]
                lis[pos] = 1
            return data
        for data in datas:
            data = mask_data(data, mask_rate=mask_rate)
            datas_.append(data)
        return np.array(datas_)

    def generater(self,x_train,y_train,batch_size,mask_rate=0.1):
        while True:
            x1_train = x_train[0]
            x2_train = x_train[1]
            len1 = len(x1_train)
            len2 = len(y_train)
            if(len1 != len2):
                raise Exception("len1 != len2")
            i = 0
            while i*batch_size + batch_size < len1:
                x1_in = x1_train[i*batch_size:i*batch_size + batch_size]
                x1_in = self.mask_data(x1_in,mask_rate=mask_rate)
                x2_in = np.array(x2_train[i*batch_size:i*batch_size + batch_size])
                y = y_train[i*batch_size:i*batch_size + batch_size]
                i+=1
                yield({'input_1':x1_in,'input_2':x2_in},{'dense_1':y})

    def get_split_text(self,text, split_len=None, overlap_len=None):
        max_piece_num = self.max_piece_num
        if split_len == None:
            split_len = self.split_len
        if overlap_len == None:
            overlap_len = self.overlap_len
        split_texts = []
        window_len = split_len - overlap_len
        for w in range(min(len(text) // split_len + 1,max_piece_num)):
            if w == 0:
                text_piece = text[:split_len]
            else:
                text_piece = text[w * window_len:w * window_len + split_len]
            split_texts.append(text_piece)
        return split_texts


    def predict_str(self,news):
        pad = np.zeros(shape=(768,))
        text_splits = self.get_split_text(news)
        texts = []
        for text in text_splits:
            x1, x2 = self.tokenizer.encode(text, max_len=None)
            text_embedding = self.bert.predict([[x1], [x2]])
            texts.append(text_embedding[0])
        while len(texts) < 10:
            texts.append(pad)
        texts = np.array(texts)
        result = self.lstm_model.predict([[texts]])
        index = np.argmax(result[0])
        return index




    def load_weight(self,weights_path):
        print("loading...")
        self.model.load_weights(weights_path)
        print("completed")



if __name__ == '__main__':
    hbc = HierarchicalBertWeatherRecongizer(config_path,checkpoint_path,dict_path,split_len=200,overlap_len=30)
    # hbc.model.load_weights("MODELS\\h_bert\\hbbc2.h5")

    # X = np.load('DATA\\NPY\\train_X.npy')
    # Y = np.load('DATA\\NPY\\train_Y.npy')
    # #
    # #
    # train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=0.1,random_state=2020)
    #
    # print(train_x.shape)
    # print(test_x.shape)
    #
    # print(train_y.shape)
    # print(test_y.shape)
    #
    # hbc.build_lstm()
    # hbc.lstm_model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),batch_size=32,epochs=5)
    # hbc.lstm_model.save("hbbc2.h5")





    # ndatas = pd.read_csv("DATA/CSV/test.csv")
    # titles = ndatas["title"]
    # newss = ndatas["news"]
    # labels = ndatas["label"]
    # count = 0
    #
    # final_result = {"disaster": {"disaster": 0, "not_disaster": 0}, "not_disaster": {"disaster": 0, "not_disaster": 0}}
    #
    # r = 0
    # w = 0
    # for t, n, l in zip(titles, newss, labels):
    #     if len(n) < 500:
    #         continue
    #     index = hbc.predict_str(n)
    #     result = classifications[index]
    #     if result == l:
    #         r += 1
    #     else:
    #         w += 1
    #     final_result[l][result] += 1
    #     print(t, l, result, result == l)
    #
    # print(r, w, r / (r + w))
    #
    # print(final_result)
    # precise = final_result["disaster"]["disaster"] / (
    #             final_result["disaster"]["disaster"] + final_result["not_disaster"]["disaster"])
    # recall = final_result["disaster"]["disaster"] / (
    #             final_result["disaster"]["disaster"] + final_result["disaster"]["not_disaster"])
    # f1 = (final_result["disaster"]["disaster"] + final_result["not_disaster"]["not_disaster"]) / (len(test_data))
    # print(precise, recall, f1)





