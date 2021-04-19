# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/10/15

import numpy as np
# import tensorflow as tf
# from sklearn.metrics import confusion_matrix

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'


import bilstm_char_clf
import bilstm_clf
import textcnn_clf

import os
classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
classifications = ['ff']
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':

    #
    # clf = bilstm_char_clf.BiLstmClf(12)
    # clf = bilstm_clf.BiLstmClf(2)
    clf = textcnn_clf.TextCnnClf(num_class=2,maxlen=300)

    for i in range(1,2):

        # clf.model.load_weights("models\\bilstm_char\\_bilstm_char_clf090.h5")
        clf.model.load_weights("D:\MyProject\天气灾害分类及信息提取\灾害类别分类_cnn_rnn\models\\textcnn_2分类\\textcnn_2_clf_035.h5")

        r = 0
        w = 0
        path = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\爬取测试\\test2\新建文件夹"
        for classification in classifications:
            files = os.listdir(path + '\\' + classification)
            for file in files:
                with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
                    url = f.readline()
                    title = f.readline()
                    time = f.readline()
                    source = f.readline()
                    contents = f.readlines()
                    zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ',
                                                                                                        '').replace(
                        '\u3000', '') for content in contents if len(content) > 2])
                    # print(len(zw))
                    zw = title + zw
                    # index,result = bwc.predict_str(title+zw)

                    this_classififcaton = classifications
                    index, result = clf.predict_str(zw)

                    if True:
                    # if classifications[index] != classification:
                        try:
                            print("文件名：", file.strip())
                            print("新闻标题：", title.strip())
                        except Exception as e:
                            print("文件名：None")
                            print("新闻标题：None")
                            pass
                        print("模型结果：", index, index == 0)

                    if index == 0:
                    # if classifications[index] == classification:
                        r+=1
                    else:
                        w+=1
        print(r,w,r/(r+w))

