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

from char_vector.untils import *
import pickle

vectors = np.load("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\char_vector\\vectors.npy")

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
        x_embed = Embedding(input_dim=vocab_len, output_dim=dim, input_length=maxlen,weights=[vectors])(x_input)

        x_cov1 = Conv1D(filters=256, kernel_size=2, strides=1, activation='relu', padding='same')(x_embed)
        x_cov2= Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')(x_embed)
        x_cov3 = Conv1D(filters=256, kernel_size=4, strides=1, activation='relu', padding='same')(x_embed)

        x_cov = Concatenate(axis=-1)([x_cov1,x_cov2,x_cov3])
        print(x_cov.shape)
        x_masked = Masking(mask_value=0.,)(x_cov)

        hs = Bidirectional(LSTM(units=hidden_size,return_sequences=True,dropout=0.5))(x_masked)
        ms = Dense(1,activation='tanh')(hs)
        # print(ms.shape)
        alpha = Softmax()(ms)
        # print(alpha.shape)
        out = Multiply()([alpha,hs])
        # print(out.shape)
        out = Lambda(lambda x: k.sum(x,axis=1))(out)
        out = ReLU()(out)
        out = Dense(64)(out)
        out = Dropout(rate=0.2)(out)
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

if __name__ == '__main__':

    tcc = BiLstmClf(num_class=12,maxlen=20)
    train_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\train_data.csv")
    test_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\test_data.csv")
    test_data1 = test_data[:int(len(test_data) / 2)]
    test_data = test_data[int(len(test_data) / 2):]
    train_data = pd.concat([train_data, test_data1])

    # Xs = tcc.sentences_encode(train_data["title"])
    # Ys = lables2onehot(train_data["label"],num_class=12)
    #
    # checkpoint = ModelCheckpoint('models\\rnn_cnn\\rnn_cnn_clf_{epoch:03d}.h5', save_weights_only=True, period=10)
    # his = tcc.model.fit_generator(generator=tcc.generater(Xs,Ys,batch_size=16),epochs=300,steps_per_epoch=int(len(Ys)/16),callbacks=[checkpoint])
    # pickle.dump(his,open("models\\rnn_cnn\\history.pkl",'rb'))

    Xs = tcc.sentences_encode(test_data["title"])
    Ys = lables2onehot(test_data["label"], num_class=12)
    #
    for i in range(1,10):
        tcc.model.load_weights("models\\rnn_cnn\\rnn_cnn_clf_0{}0.h5".format(i))
        score = tcc.model.evaluate(Xs,Ys)
        print(score)


