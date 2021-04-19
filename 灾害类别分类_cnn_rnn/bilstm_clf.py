# -*- coding:utf-8 -*-
# editor: zzh
# date: 2021/3/24

# -*- coding:utf-8 -*-
# editor: zzh

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing


import keras.backend as k
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import os,shutil
from keras.callbacks import ModelCheckpoint

from sougu_vector.untils import *

vectors = np.load("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\sougu_vector\\vectors.npy")

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

class BiLstmClf(object):
    def build_model(self,maxlen,num_class,vocab_len,dim,hidden_size):
        x_input = Input(shape=(maxlen,))
        x_embed = Embedding(input_dim=vocab_len, output_dim=dim, input_length=maxlen,weights=[vectors], mask_zero=True)(x_input)

        hs = Bidirectional(LSTM(units=hidden_size,return_sequences=True,dropout=0.5))(x_embed)

        #####
        # hs = Activation(activation='tanh')(hs)
        # ms = Dense(1,use_bias=False)(hs)
        #####
        ms = Dense(1,activation='tanh')(hs)
        #####


        # print(ms.shape)
        alpha = Softmax()(ms)
        print(alpha.shape, hs.shape)
        out = Multiply()([alpha,hs])
        # print(out.shape)
        out = Lambda(lambda x: k.sum(x,axis=1))(out)
        out = ReLU()(out)
        out = Dense(64)(out)
        output = Dense(num_class,activation='softmax')(out)

        self.model = Model(x_input,output)

        self.model.compile(loss='categorical_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy'])

        self.model.summary()

    def __init__(self,num_class,maxlen = 15):
        self.build_model(maxlen,num_class,vocab_len=vectors.shape[0],dim = vectors.shape[1],hidden_size=256)
        self.maxlen = maxlen

    def sentences_encode(self,sentences):
        X = []
        for sentence in sentences:
            X.append(sentence2ids(sentence,self.maxlen))
        return np.array(X)

    def generater(self,x_train,y_train,batch_size,mask_rate=0.1):
        while True:
            len1 = len(x_train)
            len2 = len(y_train)
            if(len1 != len2):
                raise Exception("len1 != len2")
            i = 0
            while i*batch_size + batch_size < len1:
                x1_in = x_train[i*batch_size:i*batch_size + batch_size]
                y = y_train[i*batch_size:i*batch_size + batch_size]
                i+=1
                yield({'input_1':x1_in},{'dense_3':y})

    def predict_str(self,x):
        """
        对字符串文本进行分类
        :param x: 输入一个文本
        :return: 文本类别
        """
        x = x.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','')
        s = self.sentences_encode([x])
        result = self.model.predict(s)
        index = result.argmax(axis=1)
        return index[0],result

if __name__ == '__main__':

    tcc = BiLstmClf(num_class=12)
    # train_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\train_data.csv")
    # test_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\test_data.csv")
    # test_data1 = test_data[:int(len(test_data) / 2)]
    # test_data = test_data[int(len(test_data) / 2):]
    # train_data = pd.concat([train_data, test_data1])
    #
    # # Xs = tcc.sentences_encode(train_data["title"])
    # # Ys = lables2onehot(train_data["label"],num_class=12)
    # #
    # # checkpoint = ModelCheckpoint('models\\bilstmclf_{epoch:03d}.h5', save_weights_only=True, period=10)
    # # tcc.model.load_weights("models\\bilstmclf_000.h5")
    # # tcc.model.fit_generator(generator=tcc.generater(Xs,Ys,batch_size=16),epochs=100,steps_per_epoch=int(len(Ys)/16),callbacks=[checkpoint])
    #
    #
    # Xs = tcc.sentences_encode(test_data["title"])
    # Ys = lables2onehot(test_data["label"], num_class=12)
    #
    # tcc.model.load_weights("models\\bilstmclf_100.h5")
    # score = tcc.model.evaluate(Xs,Ys)
    # print(score)


