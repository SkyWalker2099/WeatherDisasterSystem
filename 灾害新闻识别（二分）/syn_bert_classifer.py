# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/11/21

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from re_assisstant import *
import keras.backend as K
from sklearn.utils import shuffle
import codecs
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
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

# classifications2 = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'

class SynBertWeatherRecongizer(object):

    def build_bert(self,trainable = True):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        for l in bert_model.layers:
            l.trainable = trainable  # 设定为BERT可训练
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        # x1_in = Lambda(self.mask_data,arguments={'mask_rate':0.1})(x1_in)
        x = bert_model([x1_in, x2_in])
        x2 = Lambda(lambda x: x[:, 0])(x)
        x2 = Dropout(0.5)(x2)
        p = Dense(2,activation='softmax')(x2)
        self.bert = Model([x1_in, x2_in], p)

        self.bert.compile(
            # loss = 'binary_crossentropy',
            loss='categorical_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.bert.summary()

        self.mid_layer = K.function([self.bert.layers[0].input,self.bert.layers[1].input],[self.bert.layers[3].output])


    def bert_middle_output(self,news):
        encoded_news = self.get_encode([news])
        mid_out = self.mid_layer(encoded_news)[0][0]
        return mid_out

    def build_lstm(self):
        x_in = Input(shape=(None,768,))
        x_in2 = Masking(mask_value=0,input_shape=(None,768,))(x_in)
        encoded_text = Bidirectional(LSTM(units=100, return_sequences=False))(x_in2)
        # encoded_text = LSTM(100)(x_in)
        encoded_text = Dropout(0.5)(encoded_text)
        out_dense = Dense(30,activation='relu')(encoded_text)
        out = Dense(2,activation='softmax')(out_dense)
        self.lstm_model = Model(x_in,out)
        self.lstm_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.lstm_model.summary()


    def __init__(self,config_path,checkpoint_path,dict_path,max_len = 500,split_len=200,overlap_len=30):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path  # 三个BERT文件，需修改路径
        self.token_dict = {}
        self.token_list = []
        self.split_len = split_len
        self.overlap_len = overlap_len
        self.maxlen = max_len
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)  # 读取BERT字典
                self.token_list.append(token)
        self.tokenizer = OurTokenizer(self.token_dict)
        self.build_bert()
        # self.build_lstm()


    def get_encode(self,data):
        all_data = data
        X1 = []
        X2 = []
        for i in all_data:
            x1, x2 = self.tokenizer.encode(first=i, max_len=self.maxlen)
            X1.append(x1)
            X2.append(x2)
        return [X1,X2]

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

    def get_split_text(self,text, split_len=None, overlap_len=None, max_piece_num = 10):
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

    def predict_short(self,news):
        news = news.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '')
        s = self.get_encode([news])
        result = self.bert.predict(s)
        index = result.argmax(axis=1)
        return index[0],result[0][index[0]]

    def predict_long(self,news):
        pad = np.zeros(shape=(768,))
        text_splits = self.get_split_text(news)
        texts = []
        for text in text_splits:
            text_feature = self.bert_middle_output(text)
            texts.append(text_feature)
        while len(texts) < 10:
            texts.append(pad)
        texts = np.array(texts)
        result = self.lstm_model.predict([[texts]])
        index = np.argmax(result[0])
        return index,result[0][index]

    def predict_str(self,news):
        if len(news) > 500:
            index,result = self.predict_long(news)
        else:
            index,result = self.predict_short(news)

        # if(index == 0 and result < 0.5):
        #     index = 1

        return index,result



if __name__ == '__main__':

    pass
#训练短文本
    # train_data = pd.read_csv("DATA\\BALANCED\\CSV\\train.csv")
    # train_data = shuffle(train_data)
    # print(train_data["label"].value_counts())
    # titles = train_data["title"]
    # newss = train_data["news"]
    # labels = train_data["label"]
    #
    # # for t,n,l in zip(titles,newss,labels):
    # #     if type(t) == float or type(n) == float or type(l) == float:
    # #         print(t,n,l)
    # sbwr = SynBertWeatherRecongizer(config_path, checkpoint_path, dict_path, max_len=400)
    # Xs = [str(t)+n for t,n in zip(titles,newss)]
    # Y = lables2onehot(labels,2)
    # train_x, test_x, train_y, test_y = train_test_split(Xs, Y, random_state=2021, test_size=0.05)
    # train_x = sbwr.get_encode(train_x)
    # test_x = sbwr.get_encode(test_x)
    #
    # print(len(train_x[0]))
    # print(len(test_x[0]))
    # print(len(train_y))
    # print(len(test_y))
    #
    # checkpoint = ModelCheckpoint('MODELS\\balanced\\classifier_{epoch:03d}.h5', save_weights_only=True, period=1)
    #
    # sbwr.bert.fit_generator(validation_data=(test_x,test_y),generator=sbwr.generater(train_x,train_y,batch_size=2,mask_rate=0.1),epochs=5,steps_per_epoch=int(len(train_y)/2),callbacks=[checkpoint])
    #
    # sbwr.bert.save("MODELS\\balanced\\bert_short.h5")

#
    pass
#开始处理长文本生成npy
    # sbwr.bert.load_weights("MODELS\\all_data\\classifier_003.h5")
    # ldatas = pd.read_csv("DATA\\ALL_DATA\\CSV\\ltrain.csv")
    # titles = ldatas["title"]
    # newss = ldatas["news"]
    # labels = ldatas["label"]
    #
    # xs = [str(t)+'\n'+n for t,n, in zip(titles,newss)]
    # X = []
    # for text in xs:
    #     pad = np.zeros(shape=(768,))
    #     text_splits = sbwr.get_split_text(text)
    #     texts = []
    #     for text in text_splits:
    #         text_feature = sbwr.bert_middle_output(text)
    #         texts.append(text_feature)
    #     while len(texts) < 10:
    #         texts.append(pad)
    #     texts = np.array(texts)
    #     X.append(texts)
    #     print(len(X))
    # Y = lables2onehot(labels,num_class=2)
    #
    # X = np.array(X)
    # Y = np.array(Y)
    #
    # print(X.shape,Y.shape)
    #
    # np.save("train_X",X)
    # np.save("train_Y",Y)

#
    pass
#训练lstm
    #(13616, 10, 768) (13616, 2)
    #(33741, 10, 768)(33741, 2)
    # X = np.load("DATA\\ALL_DATA\\NPY\\train_X.npy")
    # Y = np.load("DATA\\ALL_DATA\\NPY\\train_Y.npy")
    #
    # train_x, test_x, train_y, test_y = train_test_split(X,Y,random_state=2020, test_size=0.03)
    #
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    #
    # checkpoint = ModelCheckpoint('MODELS\\all_data\\lstm_{epoch:03d}.h5', save_weights_only=True, period=1)
    #
    # sbwr.lstm_model.fit(validation_data=(test_x,test_y),x = train_x,y = train_y,batch_size=32,epochs=30,callbacks=[checkpoint])
    #
    # sbwr.lstm_model.save("MODELS\\all_data\\lstm_finnal.h5")

#
    pass

# 文本测试
#     sbwr.bert.load_weights("MODELS\\all_data\\classifier_003.h5")
#     sbwr.lstm_model.load_weights("MODELS\\all_data\\lstm_029.h5")
#
#     test_data = pd.read_csv("DATA\\ALL_DATA\\CSV\\test.csv")
#     titles = test_data["title"]
#     newss = test_data["news"]
#     labels = test_data["label"]
#
#     r = 0
#     w = 0
#
#     final_result = {"disaster":{"disaster":0,"not_disaster":0},"not_disaster":{"disaster":0,"not_disaster":0}}
#
#     for t,n,l in zip(titles,newss,labels):
#         try:
#             # index = sbwr.predict_str(str(t) + n)
#             index = sbwr.predict_short(str(t) + n)
#             result = classifications[index]
#             if result == l:
#                 r += 1
#             else:
#                 w += 1
#             final_result[l][result] += 1
#             print(t, l, result, result == l)
#         except Exception as e:
#             pass
#
#
#     print(r,w,r/(r+w))
#
#
#     print(final_result)
#     precise = final_result["disaster"]["disaster"]/(final_result["disaster"]["disaster"] + final_result["not_disaster"]["disaster"])
#     recall = final_result["disaster"]["disaster"]/(final_result["disaster"]["disaster"] + final_result["disaster"]["not_disaster"])
#     f1 = 2*precise*recall/(precise + recall)
#     print(precise,recall,f1)
    # 0.9593432369038312 0.9504260263361735 0.9548638132295719
    # {'disaster': {'disaster': 1227, 'not_disaster': 64}, 'not_disaster': {'disaster': 52, 'not_disaster': 1465}}
