# -*- coding: UTF-8 -*-
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras_contrib.layers import CRF
from keras.layers import GRU
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
import TimeExtract
import re
import codecs

import numpy as np
import pandas as pd
import os

from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# config_path = 'chinese_L-12_H-768_A-12\\bert_config.json'
# checkpoint_path = 'chinese_L-12_H-768_A-12\\bert_model.ckpt'
# vocab_path = 'chinese_L-12_H-768_A-12\\vocab.txt'

config_path = 'chinese_L-12_H-768_A-12_pruned\\bert_config2.json'
checkpoint_path = 'chinese_L-12_H-768_A-12_pruned\\bert_pruning_9_layer.ckpt'
vocab_path = 'chinese_L-12_H-768_A-12_pruned\\vocab.txt'
#以上是使用减少隐藏层层数后的bert预训练模型，需要微调参数


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


class TimeTagger(object):

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        w = 0
        for l in bert_model.layers:
            if w in range(0,4):
                l.trainable = True  # 设定BERT为可训练
            w += 1
        crf = CRF(self.num_tag,sparse_target=True)
        bigru = Bidirectional(GRU(units=128,return_sequences=True,name='layer_bgru'))
        x1_in = Input(shape=(self.maxlen,))
        x2_in = Input(shape=(self.maxlen,))
        x = bert_model([x1_in,x2_in])
        print(x)
        x = bigru(x)
        x = Dense(units=64,activation="tanh")(x)
        x = Dense(self.num_tag)(x)
        output = crf(x)

        self.model = Model([x1_in,x2_in],output)
        self.model.summary()
        self.model.compile(optimizer="adam",loss=crf_loss,metrics=[crf_accuracy])


    def __init__(self, config_path, checkpoint_path, dict_path,num_tag,maxlen=200):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path  # 三个BERT文件，需修改路径
        self.num_tag = num_tag
        self.maxlen = maxlen
        self.token_dict = {}
        self.token_list = []
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)  # 读取BERT字典
                self.token_list.append(token)
        self.tokenizer = OurTokenizer(self.token_dict)
        self.build_model()


    def get_encode(self,data):
        all_data = data
        X1 = []
        X2 = []
        for i in all_data:
            if len(i) >= self.maxlen:
                i = i[:self.maxlen]
            x1 = [self.token_dict.get(j,0) for j in i]
            x1 = x1 + [0]*(self.maxlen - len(x1))
            x2 = [0]*self.maxlen
            if len(x1) != len(x2) or len(x1) != self.maxlen:
                raise Exception("长度出错")
            X1.append(x1)
            X2.append(x2)

        return np.array(X1),np.array(X2)

    def tags_complete(self,data):
        all_data = data
        tags = []
        for i in all_data:
            i = [int(j) for j in i]
            if len(i) >= self.maxlen:
                i = i[:self.maxlen]
            else:
                i = i + [0]*(self.maxlen - len(i))
            tags.append(i)
        return np.array(tags)


    def predict_contents_1(self,contents):
        """
        返回一个列表包含按时间顺序排列的元组，其中包含日期信息
        :param contents:
        :return:
        """
        all_results = []
        types = ["","DS","TS","DO","TO"]
        for content in contents:
            content = content.replace('｡', '。').replace(',', '，')
            lines = re.split('[。！？!?.]', content)
            for line in lines:
                if re.search("([一二三四五六七八九十千百万零点两]+|[\d.]+)", line) == None:
                    continue
                line = "(新闻)" + line + "(结束)"
                line1, line2 = self.get_encode([line])
                output = self.model.predict([line1, line2])
                output = [np.argmax(i) for i in output[0]]
                last = 0
                one_time = ""
                for l,o in zip(line,output):
                    if o == last:
                        if last != 0:
                            one_time = one_time + l
                    else:
                        if last != 0:
                            one_result = (types[last],one_time)
                            all_results.append(one_result)
                            one_time = ""
                        if o != 0:
                            one_time = one_time + l
                    last = o
                if one_time != "":
                    one_result = (type[last],one_time)
                    all_results.append(one_result)
        return all_results


    def predict_contents_2(self,contents):
        """
        返回四个列表，分别包含 DS,TS,DO,TO
        :param contents:
        :return:
        """
        results = self.predict_contents_1(contents)
        all_result = [[], [], [], []]
        tdict = {"DS": 0, "TS": 1, "DO": 2, "TO": 3}
        for result in results:
            all_result[tdict[result[0]]].append(result[1])
        return all_result

    def predict_contents_3(self,contents,time):
        """
        获取开始时间和结束时间
        :param contents:
        :return:
        """
        datetimes = self.predict_contents_1(contents)
        print(datetimes)
        standard_time = TimeExtract.DrawStandardTime(time)
        if standard_time == None:
            return None,None
        stime,otime = TimeExtract.KeyTimeDecide(datetimes=datetimes,standard_time = standard_time)
        return stime,otime



if __name__ == '__main__':

    tt = TimeTagger(config_path=config_path,checkpoint_path=checkpoint_path,dict_path=vocab_path,num_tag=5)

    # data = pd.read_csv("DATA\\train_data3.csv")
    # X = data["line"]
    # Y = data["tags"]
    #
    # x_train_1,x_train_2 = tt.get_encode(X)
    # y_train = tt.tags_complete(Y)
    #
    # print(x_train_1.shape)
    # print(x_train_2.shape)
    # print(y_train.shape)
    #
    # tt.model.fit([x_train_1,x_train_2],np.expand_dims(y_train,2),epochs=3,batch_size=16)
    #
    # tt.model.save("pruned_bert_bigru_crf.h5")
    #
    # test_data = pd.read_csv("DATA\\test_data3.csv")
    # x_test_1, x_test_2 = tt.get_encode(X)
    # y_test = tt.tags_complete(Y)
    #
    # score = tt.model.evaluate([x_test_1,x_test_2],np.expand_dims(y_test,2))



# Total params: 102,382,790
# Trainable params: 705,734
# Non-trainable params: 101,677,056

# Total params: 81,119,174
# Trainable params: 705,734
# Non-trainable params: 80,413,440