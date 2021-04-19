# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/10/15

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'

# import WeatherClassifer
# from WeatherClassifer2 import *
# import WeatherClassifer_baoyu_neilao_dizhi
import W_classififer
import pandas as pd
import os
# classifications2 = ['低温灾害',"雪灾灾害"]
classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']


if __name__ == '__main__':


    bwc = W_classififer.BertWeatherClassifer(config_path, checkpoint_path, dict_path, maxlen=400)
    for i in range(1,2):
        p = "bert_taged_005.h5"

        bwc.model.load_weights(p)

        test_data = pd.read_csv("..\\灾害类别分类\\DATAS\\DATA6\\test_data.csv")

        test_data = test_data[int(len(test_data) / 2):]

        # test_data = test_data[test_data["label"].isin(classifications2)]

        titles = test_data["title"]
        newss = test_data["news"]
        labels = test_data["label"]

        zws = [title + news for title,news in zip(titles,newss)]

        X = bwc.get_encode(zws)

        Y = W_classififer.lables2onehot(labels, 12)



        # score = bwc.model.evaluate(X, Y)
        # print(score)
        # r = 0
        # w = 0
        # for zw,label in zip(zws,labels):
        #     index,result = bwc.predict_str(zw)
        #     if classifications[index] == label:
        #         r+=1
        #     else:
        #         w+=1
        #     print(classifications[index],label,classifications[index] == label)
        #
        # print(r, w, r / (r + w))

        r = 0
        w = 0
        path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第2次测试\测试文本"
        for classification in classifications:
            files = os.listdir(path + '\\' + classification)
            for file in files:
                with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
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
                    index, result = bwc.predict_str(zw)

                    if True:
                    # if classifications[index] != classification:
                        try:
                            print("文件名：", file.strip())
                            print("新闻标题：", title.strip())
                        except Exception as e:
                            print("文件名：None")
                            print("新闻标题：None")
                            pass

                        if result < 0.6:
                            print("模型结果：", this_classififcaton[index], " 正确结果：", "无结果",
                                  False, '\n')
                        else:
                            print("模型结果：", this_classififcaton[index], " 正确结果：", classification,
                                  this_classififcaton[index] == classification,'\n')

                    if this_classififcaton[index] == classification and result > 0.6:
                    # if classifications[index] == classification:
                        r+=1
                    else:
                        w+=1
        print(r,w,r/(r+w))

        # r2 = 0
        # w2 = 0
        # path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料2\其他灾害\新闻文本"
        # for classification in [""]:
        #     files = os.listdir(path)
        #     for file in files:
        #         with open(path + '\\' + file, 'r', encoding='utf8') as f:
        #             title = f.readline()
        #             time = f.readline()
        #             source = f.readline()
        #             contents = f.readlines()
        #             zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ',
        #                                                                                                 '').replace(
        #                 '\u3000', '') for content in contents if len(content) > 2])
        #             # print(len(zw))
        #             zw = title + zw
        #             # index,result = bwc.predict_str(title+zw)
        #
        #             this_classififcaton = classifications
        #             index, result = bwc.predict_str(zw)
        #
        #             if True:
        #                 # if classifications[index] != classification:
        #                 try:
        #                     print("文件名：", file.strip())
        #                     print("新闻标题：", title.strip())
        #                 except Exception as e:
        #                     print("文件名：None")
        #                     print("新闻标题：None")
        #                     pass
        #
        #                 print("模型结果：", result < 0.6, '\n', classifications[index])
        #
        #             if result < 0.6:
        #                 # if classifications[index] == classification:
        #                 r2 += 1
        #             else:
        #                 w2 += 1
        #
        # print(r, w, r / (r + w))
        # print(r2, w2, r2 / (r2 + w2))

