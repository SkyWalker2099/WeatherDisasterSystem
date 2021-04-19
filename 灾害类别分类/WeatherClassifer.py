# -*- coding: UTF-8 -*-

from keras_bert import load_trained_model_from_checkpoint, Tokenizer

import codecs
from random import choice

import numpy as np
import pandas as pd
import os
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn import preprocessing

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import os,shutil
from keras.callbacks import ModelCheckpoint

import random
import re

import pickle

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

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'

class BertWeatherClassifer(object):
    def build_bert(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,seq_len=None)
        for l in bert_model.layers:
            l.trainable = True #设定为BERT可训练
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        # x1_in = Lambda(self.mask_data,arguments={'mask_rate':0.1})(x1_in)
        x = bert_model([x1_in, x2_in])
        # print((K.shape(x)))
        x = Lambda(lambda x: x[:, 0])(x)
        # print((K.shape(x)))
        # x2 = Dense(100, activation='tanh')(x)
        # x3 = Dropout(0.5)(x2)
        p = Dense(12, activation='sigmoid')(x)
        # p = Dense(12, activation='softmax')(x)
        self.model = Model([x1_in, x2_in], p)
        self.model.compile(
            # loss = 'binary_crossentropy',
            loss='categorical_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.model.summary()

    def __init__(self,config_path,checkpoint_path,dict_path,maxlen=200,):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path  # 三个BERT文件，需修改路径
        self.maxlen = maxlen
        self.token_dict = {}
        self.token_list = []
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)  # 读取BERT字典
                self.token_list.append(token)
        self.tokenizer = OurTokenizer(self.token_dict)
        self.build_bert()

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

    def get_encode(self,data):
        all_data = data
        X1 = []
        X2 = []
        for i in all_data:
            # x1, x2 = self.tokenizer.encode(first=i, max_len=self.maxlen)
            x1,x2 = self.encode(i)
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
            lis = [0] * self.maxlen
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


    def fit(self,x_train,y_train,savepath,overwrite=False):
        self.model.fit(x_train,y_train)
        self.model.save(savepath,overwrite=overwrite)

    def score(self,x_test,y_test):
        score = self.model.evaluate(x_test,y_test,verbose=0)
        return score

    def predict_str(self,x):
        """
        对字符串文本进行分类
        :param x: 输入一个文本
        :return: 文本类别
        """
        x = x.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','')
        s = self.get_encode([x])
        result = self.model.predict(s)
        index = result.argmax(axis=1)
        return index[0],result

    def load_weight(self,weights_path):
        print("loading...")
        self.model.load_weights(weights_path)
        print("completed")

if __name__ == '__main__':


     bwc = BertWeatherClassifer(config_path,checkpoint_path,dict_path,maxlen=400)

     train_data = pd.read_csv("DATAS\\DATA6\\train_data.csv")
     test_data = pd.read_csv("DATAS\\DATA6\\test_data.csv")

     test_data1 = test_data[:int(len(test_data)/2)]
     test_data = test_data[int(len(test_data)/2):]

     train_data = pd.concat([train_data,test_data1])



     # train_data = train_data[:int(len(train_data)/1000)]
     print(len(train_data),len(test_data))

     titles = train_data["title"]
     newss = train_data["news"]
     labels = train_data["label"]

     Xs = [t+n for t,n in zip(titles,newss)]
     X = bwc.get_encode(Xs)
     Y = lables2onehot(labels,12)

     #
     checkpoint = ModelCheckpoint('bert_masked_{epoch:03d}.h5',save_weights_only=True,period=1)
     history = bwc.model.fit_generator(generator=bwc.generater(X,Y,batch_size=2,mask_rate=0.1),epochs=5,steps_per_epoch=int(len(Y)/2),callbacks=[checkpoint])
     bwc.model.save("MODELS\\bert_masked_12_4.h5")
     pickle.dump(history, open("训练记录/WC/history2.pk", 'wb'))






















