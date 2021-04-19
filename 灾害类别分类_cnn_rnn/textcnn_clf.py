# -*- coding:utf-8 -*-
# editor: zzh

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing

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

class TextCnnClf(object):
    def build_model(self,maxlen,num_class,vocab_len,dim):
        x_input = Input(shape=(maxlen,))
        x_embed = Embedding(input_dim=vocab_len, output_dim=dim, input_length=maxlen,weights=[vectors])(x_input)

        x_cov1 = Conv1D(filters=128,kernel_size=3,strides=1,activation='relu')(x_embed)
        x_pool1 = MaxPooling1D(pool_size=maxlen - 2)(x_cov1) # 1*128

        x_cov2 = Conv1D(filters=128,kernel_size=4,strides=1,activation='relu')(x_embed)
        x_pool2 = MaxPooling1D(pool_size=maxlen - 3)(x_cov2)

        x_cov3 = Conv1D(filters=128,kernel_size=5,strides=1,activation='relu')(x_embed)
        x_pool3 = MaxPooling1D(pool_size=maxlen - 4)(x_cov3) #

        x_cnn = Concatenate(axis=-1)([x_pool1,x_pool2,x_pool3])

        x_flat = Flatten()(x_cnn)

        x_dense1 = Dense(128,use_bias=True,activation='relu')(x_flat)

        x_dropout = Dropout(rate=0.1)(x_dense1)

        output = Dense(num_class,activation='softmax')(x_dropout)

        self.model = Model(x_input,output)

        self.model.compile(loss='categorical_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy'])

        self.model.summary()

    def __init__(self,num_class,maxlen = 15):
        self.build_model(maxlen,num_class,vocab_len=vectors.shape[0],dim = vectors.shape[1])
        self.maxlen = maxlen

    def sentences_encode(self,sentences):
        X = []
        # d = 0
        for sentence in sentences:
            # print(d)
            # d+=1
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
                yield({'input_1':x1_in},{'dense_2':y})

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

    tcc = TextCnnClf(num_class=12)
    train_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\train_data.csv")
    test_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\test_data.csv")
    test_data1 = test_data[:int(len(test_data) / 2)]
    test_data = test_data[int(len(test_data) / 2):]
    train_data = pd.concat([train_data, test_data1])

    Xs = tcc.sentences_encode(train_data["title"])
    Ys = lables2onehot(train_data["label"],num_class=12)

    
    Xs2 = tcc.sentences_encode(test_data["title"])
    Ys2 = lables2onehot(test_data["label"], num_class=12)

    checkpoint = ModelCheckpoint('models\\text_cnn\\textcnnclf_{epoch:03d}.h5', save_weights_only=True, period=1)
    tcc.model.fit_generator(generator=tcc.generater(Xs,Ys,batch_size=16),validation_data=(Xs2,Ys2),epochs=100,steps_per_epoch=int(len(Ys)/16),callbacks=[checkpoint])

    # tcc.model.load_weights("models\\text_cnn\\textcnnclf.h5")
    score = tcc.model.evaluate(Xs2,Ys2)
    print(score)

    # tcc = TextCnnClf(num_class=2,maxlen=300)
    # train_data = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\CSV\\train.csv")
    # test_data = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\CSV\\test.csv")
    #
    # train_titles = train_data["title"]
    # train_news = train_data["news"]
    # strs = [str(t) + str(n) for t,n in zip(train_titles,train_news)]
    # Xs = tcc.sentences_encode(strs)
    # Ys = lables2onehot(train_data["label"], num_class=2)
    #
    # test_titles = test_data["title"]
    # test_news = test_data["news"]
    # strs2 = [str(t) + str(n) for t, n in zip(test_titles, test_news)]
    # Xs2 = tcc.sentences_encode(strs2)
    # Ys2 = lables2onehot(test_data["label"], num_class=2)
    #
    # checkpoint = ModelCheckpoint('models\\textcnn_2分类\\textcnn_2_clf_{epoch:03d}.h5', save_weights_only=True, period=1)
    # tcc.model.fit_generator(generator=tcc.generater(Xs,Ys,batch_size=16),validation_data=(Xs2,Ys2),epochs=100,steps_per_epoch=int(len(Ys)/16),callbacks=[checkpoint])
    # tcc.model.load_weights('models\\textcnn_2分类\\textcnn_2_clf.h5')




