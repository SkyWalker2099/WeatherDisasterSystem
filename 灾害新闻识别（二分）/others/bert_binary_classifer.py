# -*- coding: UTF-8 -*-
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert import gen_batch_inputs

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

config_path = r'chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'chinese_L-12_H-768_A-12\vocab.txt'

class BertWeatherRecongizer(object):

    def build_bert(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,seq_len=None)
        for l in bert_model.layers:
            l.trainable = True #设定为BERT可训练
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        # x1_in = Lambda(self.mask_data,arguments={'mask_rate':0.1})(x1_in)
        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        p = Dense(2, activation='sigmoid')(x)
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
        s = self.get_encode([x])
        result = self.model.predict(s)
        index = result.argmax(axis=1)
        return index,result

    def load_weight(self,weights_path):
        print("loading...")
        self.model.load_weights(weights_path)
        print("completed")

bc1 = BertWeatherRecongizer(config_path,checkpoint_path,dict_path,maxlen=500)
bc1.load_weight("MODELS\\bert2\\bert_masked_1.h5")

bc2 = BertWeatherRecongizer(config_path,checkpoint_path,dict_path,maxlen=30)
bc2.load_weight("MODELS\\bert\\bert-2.h5")

def calc_result(result1,result2):
    result1 = result1[0]
    result2 = result2[0]

    def hege(result):
        if (result[0] > 0.85 and result[1] < 0.1) or (result[1] > 0.85 and result[0] < 0.1):
            return True
        else:
            return False

    if hege(result1):
        return 0 if result1[0] > result1[1] else 1
    elif hege(result2):
        return 0 if result2[0] > result2[1] else 1
    else:
        return 1

def predict(title,content):
    index1, result1 = bc1.predict_str(content)
    index2, result2 = bc2.predict_str(title)
    index = calc_result(result1=result1, result2=result2)
    return index




if __name__ == '__main__':

    path = "C:\\Users\Zzh\Desktop\第一次测试\邢-测试用例\\"
    classifications2 = ["冰雹灾害", "台风灾害", "城市内涝", "大风灾害", "暴雨洪涝", "雷电灾害"]
    for classification in classifications2:
        fpath = path + classification
        files = os.listdir(fpath)
        for file in files:
            if file == "_新闻文本说明.txt":
                continue
            with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()

                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                zw = ''.join([content for content in contents])

                index = predict(title,zw)
                print(title.strip(),index)

    path = "C:\\Users\Zzh\Desktop\第一次测试\张-测试用例\\"
    classifications2 = ["冰雹灾害", "台风灾害", "城市内涝", "大风灾害", "暴雨洪涝", "雷电灾害"]
    for classification in classifications2:
        fpath = path + classification
        files = os.listdir(fpath)
        for file in files:
            if file == "_新闻文本说明.txt":
                continue
            with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()

                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                zw = ''.join([content for content in contents])

                index = predict(title, zw)
                print(title.strip(), index)

    path = "C:\\Users\Zzh\Desktop\第一次测试\邢-测试用例2\\"
    classifications2 = ["冰雹灾害", "台风灾害", "城市内涝", "大风灾害", "暴雨洪涝", "雷电灾害"]
    for classification in classifications2:
        fpath = path + classification
        files = os.listdir(fpath)
        for file in files:
            if file == "说明文档.xlsx":
                continue
            with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()

                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                zw = ''.join([content for content in contents])

                index = predict(title, zw)
                print(title.strip(), index)



























