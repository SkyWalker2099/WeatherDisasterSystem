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
from untils import *

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
maxlen = 450
max_piece = 10


classifications_text =" "+"  ".join(classifications)+" "



def gather(x):
    data = x[0]
    pos = x[1]
    return tf.gather(data,pos,batch_dims=1)

class AutomaticSummary(object):
    def build_bert(self,hidden_size):
        bert_model = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,seq_len=None)
        for l in bert_model.layers:
            l.trainable = True #设定为BERT可训练
        x1_in = Input(shape=(None,)) # encoded_tokens
        x2_in = Input(shape=(None,)) # seg_emb
        x3_in = Input(shape=(10,),dtype='int32') # idx_pos
        x4_in = Input(shape=(10,)) # mask

        x = bert_model([x1_in, x2_in])
        print(K.shape(x),x.shape)
        print(K.shape(x3_in),x3_in.shape)
        # x_gather = tf.gather(x,x3_in,batch_dims=1)
        x_gather = Lambda(lambda x:gather(x))([x,x3_in])
        print(K.shape(x_gather),x_gather.shape)

        x_cls = Lambda(lambda x: x[:, 0])(x)
        x_hidden_start = Dense(hidden_size)(x_cls)
        x_cell_start = Dense(hidden_size)(x_cls)

        mask = Lambda(lambda x:K.expand_dims(x,-1))(x4_in)

        print(x_gather.shape, mask.shape)
        x_gather_masked = Multiply()([x_gather, mask])
        x_gather_masked = Masking(mask_value=0)(x_gather_masked)
        # x_lstm = Bidirectional(LSTM(units=hidden_size,return_sequences=True,dropout=0.5))(x_gather_masked,initial_state=[x_hidden_start,x_cell_start])
        x_lstm = LSTM(units=hidden_size, return_sequences=True, dropout=0.5)(x_gather_masked,initial_state=[x_hidden_start,x_cell_start])
        p = Dense(1, activation='sigmoid')(x_lstm)
        print(p.shape)
        p = Lambda(lambda x: K.squeeze(x, axis=2))(p)
        # p = Lambda(lambda x: K.squeeze(x,axis=2))(p)
        print(p.shape)
        self.model = Model([x1_in, x2_in, x3_in, x4_in], p)
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.model.summary()

    def __init__(self,config_path,checkpoint_path,dict_path,maxlen=200,hidden_size = 100):
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
        self.build_bert(hidden_size=hidden_size)

    @staticmethod
    def get_encode(contents):

        X1 = [] # encoded_tokens
        X2 = [] # seg_embed
        X3 = [] # idx
        X4 = [] # mask

        for content in contents:
            all_splited_lines = split_contents(content, max_len=maxlen, max_piece=max_piece)
            # print(all_splited_lines)

            for all_splited_line in all_splited_lines:
                idx = []
                encoded_tokens = [token_dict["[CLS]"]]
                seg_embed = [0]
                for splited_line in all_splited_line:
                    encoded_line,_ = tokenizer.encode(first=splited_line)
                    idx.append(len(encoded_tokens))
                    encoded_tokens.append(token_dict["[unused1]"])
                    encoded_tokens = encoded_tokens + encoded_line[1:]
                    seg_embed += [(1 + len(idx))%2] * (len(encoded_line))

                mask = [0]*max_piece
                for i in range(len(idx)):
                    mask[i] = 1

                while(len(idx) < max_piece):
                    idx.append(0)

                while(len(encoded_tokens) < maxlen):
                    encoded_tokens.append(0)

                while(len(seg_embed) < maxlen):
                    seg_embed.append(0)

                X1.append(encoded_tokens[:maxlen])
                X2.append(seg_embed[:maxlen])
                X3.append(idx[:max_piece])
                X4.append(mask[:max_piece])

        return [np.array(X1),np.array(X2),np.array(X3),np.array(X4)]



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
            x4_train = x_train[3]

            len1 = len(x1_train)
            len2 = len(y_train)

            if(len1 != len2):
                raise Exception("len1 != len2")
            i = 0
            while i*batch_size + batch_size < len1:
                x1_in = x1_train[i*batch_size:i*batch_size + batch_size]
                x2_in =  x2_train[i * batch_size:i * batch_size + batch_size]
                x3_in =  x3_train[i * batch_size:i * batch_size + batch_size]
                x4_in =  x4_train[i * batch_size:i * batch_size + batch_size]
                y = y_train[i*batch_size:i*batch_size + batch_size]
                # print(type(y),y.shape)
                i+=1
                yield({'input_1':x1_in,'input_2':x2_in,'input_3':x3_in,'input_4':x4_in},{'lambda_4':y})

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


def build_train_data(json_datas,max_piece = 10):
    datas = json2label(json_datas)

    texts = []
    labels = []

    for data in datas:
        lines = [d[0] for d in data]
        label = [d[1] for d in data]

        text = '。'.join(lines)
        while(len(label) < max_piece):
            label.append(0)

        texts.append(text)
        labels.append(label)

    texts_encoded = AutomaticSummary.get_encode(texts)

    # print(len(texts_encoded[0]),len(texts_encoded[1]),len(texts_encoded[2]),len(texts_encoded[3]),len(labels))

    return texts_encoded,np.array(labels)








if __name__ == '__main__':
    asr = AutomaticSummary(config_path,checkpoint_path,dict_path,maxlen=450,hidden_size=100)

     # text1 = "中新网济南5月7日电(刘佳)山东日？中新网济方公里!中新网济南公里"
     # text2 = "中新网济南5月7日电(刘佳)山东日？中新网济方公!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试!测试测试测试测试"
     # ans_coded = AutomaticSummary.get_encode([text1,text2])
     #
     #
     # xs1,xs2,xs3,xs4 = ans_coded[0],ans_coded[1],ans_coded[2],ans_coded[3]
     #
     # for j in range(len(xs1)):
     #     print("*"*20)
     #     idx = [" "]*len(xs1[j])
     #     for i in xs3[j]:
     #         idx[i] = "<-"
     #
     #     for x1,x2,i in zip(xs1[j],xs2[j],idx):
     #         print(x1,x2,token_list[x1+1],i)
     #
     #     print(xs3[j],xs4[j])

    import json

    datas = []
    with open("weather.jsonl", 'r', encoding='utf8') as f:
        for l in f:
            data = json.loads(l)
            if (len(data["annotations"]) > 0):
                datas.append(data)

    ans_coded,Ys = build_train_data(datas,max_piece=10)

    xs1,xs2,xs3,xs4 = ans_coded[0],ans_coded[1],ans_coded[2],ans_coded[3]

    # for j in range(len(xs1)):
    #     print(len(xs1[j]))
    # for j in range(len(xs2)):
    #     print(len(xs2[j]))
    # for j in range(len(xs3)):
    #     print(len(xs3[j]))
    # for j in range(len(xs4)):
    #     print(len(xs4[j]))


    asr.model.fit_generator(generator=asr.generater(ans_coded,Ys,batch_size=2,mask_rate=0.0),epochs=5,steps_per_epoch=int(len(Ys)/2))
    asr.model.save("models\\asr.h5")




























