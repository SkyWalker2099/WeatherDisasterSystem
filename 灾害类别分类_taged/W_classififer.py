# -*- coding:utf-8 -*-
# editor: zzh

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

import tensorflow as tf
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



token_dict = {}
token_list = []
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)  # 读取BERT字典
        token_list.append(token)
tokenizer = OurTokenizer(token_dict)
maxlen = 200

classifications_text =" "+"  ".join(classifications)+" "

def gather(x):
    data = x[0]
    pos = x[1]
    return tf.gather(data,pos,batch_dims=1)

class BertWeatherClassifer(object):
    def build_bert(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,seq_len=None)
        for l in bert_model.layers:
            l.trainable = True #设定为BERT可训练
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x3_in = Input(shape=(12,),dtype='int32')
        # x1_in = Lambda(self.mask_data,arguments={'mask_rate':0.1})(x1_in)
        x = bert_model([x1_in, x2_in])
        print(K.shape(x),x.shape)
        print(K.shape(x3_in),x3_in.shape)
        # x_gather = tf.gather(x,x3_in,batch_dims=1)
        x_gather = Lambda(lambda x:gather(x))([x,x3_in])
        print(K.shape(x_gather),x_gather.shape)
        p = Dense(50,activation='relu')(x_gather)
        print(p.shape)
        p = Dense(1, activation='sigmoid')(p)
        print(p.shape)
        p = Lambda(lambda x: K.squeeze(x,axis=2))(p)
        print(p.shape)
        self.model = Model([x1_in, x2_in, x3_in], p)
        self.model.compile(
            loss = 'binary_crossentropy',
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

    @staticmethod
    def get_encode(data):
        all_data = data
        X1 = []
        X2 = []
        X3 = []
        for i in all_data:
            i = i[:maxlen - 80]
            x1, x2 = tokenizer.encode(first=i,second=classifications_text,max_len=maxlen)
            x3 = []
            ilen = len(i)
            for j in range(1,13):
                x1[ilen + 1 + j * 6] = token_dict["[unused2]"]
                x3.append(ilen + 1 + j * 6)
            X1.append(x1)
            X2.append(x2)
            X3.append(x3)


        print(X1,X2,X3)
        return [X1, X2, X3]

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
            x3_train = x_train[2]

            # print(x1_train.shape)
            # print(x2_train.shape)
            # print(x3_train.shape)
            #
            print(x1_train)
            print(x2_train)
            print(x3_train)

            len1 = len(x1_train)
            len2 = len(y_train)
            if(len1 != len2):
                raise Exception("len1 != len2")
            i = 0
            while i*batch_size + batch_size < len1:
                x1_in = x1_train[i*batch_size:i*batch_size + batch_size]
                x1_in = self.mask_data(x1_in,mask_rate=mask_rate)
                x2_in = np.array(x2_train[i*batch_size:i*batch_size + batch_size])
                x3_in = np.array(x3_train[i*batch_size:i*batch_size + batch_size])
                y = y_train[i*batch_size:i*batch_size + batch_size]
                i+=1
                yield({'input_1':x1_in,'input_2':x2_in,'input_3':x3_in},{'lambda_2':y})

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
        return index[0],result[0][index[0]]

    def load_weight(self,weights_path):
        print("loading...")
        self.model.load_weights(weights_path)
        print("completed")

if __name__ == '__main__':

    train_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\train_data.csv")
    test_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\test_data.csv")

    test_data1 = test_data[:int(len(test_data)/2)]
    test_data = test_data[int(len(test_data)/2):]

    train_data = pd.concat([train_data,test_data1])



    # train_data = train_data[:int(len(train_data)/1000)]
    print(len(train_data),len(test_data))

    titles = train_data["title"]
    newss = train_data["news"]
    labels = train_data["label"]

    bwc = BertWeatherClassifer(config_path, checkpoint_path, dict_path, maxlen=400)

    Xs = [t+n for t,n in zip(titles,newss)]
    X = bwc.get_encode(Xs)
    Y = lables2onehot(labels,12)
    print(Y.shape)
    #
    checkpoint = ModelCheckpoint('bert_taged_{epoch:03d}.h5',save_weights_only=True,period=1)
    history = bwc.model.fit_generator(generator=bwc.generater(X,Y,batch_size=2,mask_rate=0.0),epochs=5,steps_per_epoch=int(len(Y)/2),callbacks=[checkpoint])
    bwc.model.save("MODELS\\bert_taged.h5")
    pickle.dump(history, open("训练记录/history.pk", 'wb'))



     # text = "中新网济南5月7日电(刘佳)山东日照附近黄海海域4日发现大面积赤潮，赤潮分布面积达780平方公里。"
     # ans_coded = BertWeatherClassifer.get_encode([text])
     #
     # print(ans_coded)
     # print(len(text),len(ans_coded[0][0]),len(ans_coded[1][0]))
     # for a,b,c, in zip("_"+text+"_"+classifications_text,ans_coded[0][0],ans_coded[1][0]):
     #    print(a,b,c)
























