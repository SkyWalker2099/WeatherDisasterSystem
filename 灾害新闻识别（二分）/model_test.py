# -*- coding:utf-8 -*-
# editor: zzh

import os
from syn_bert_classifer import *
from re_assisstant import *
import pandas as pd

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
classifications2 = ["disaster","not_disaster"]

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'

if __name__ == '__main__':

    sbwr = SynBertWeatherRecongizer(config_path, checkpoint_path, dict_path, max_len=400)

    sbwr.bert.load_weights("MODELS\\balanced\\classifier_003.h5")
    # sbwr.lstm_model.load_weights("MODELS\\all_data\\lstm_029.h5")

    # sbwr.bert.load_weights("MODELS\\syn_model\\bert_short.h5")
    # sbwr.lstm_model.load_weights("MODELS\\syn_model\\lstm_model.h5")

    # r = 0
    # w = 0
    # path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第2次测试\测试文本"
    # for classification in classifications:
    #     files = os.listdir(path + '\\' + classification)
    #     for file in files:
    #         with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
    #             title = f.readline()
    #             time = f.readline()
    #             source = f.readline()
    #             contents = f.readlines()
    #             zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ',
    #                                                                                                 '').replace(
    #                 '\u3000', '') for content in contents if len(content) > 2])
    #             # print(len(zw))
    #             zw = title + zw
    #             print(title + zw)
    #             # index,result = bwc.predict_str(title+zw)
    #
    #             this_classififcaton = classifications2
    #             index,result = sbwr.predict_short(title + zw)
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
    #             print("模型结果：", this_classififcaton[index], " 正确结果：", "disaster",
    #                   this_classififcaton[index] == "disaster", '\n')
    #
    #             if index == 0:
    #                 # if classifications[index] == classification:
    #                 r += 1
    #             else:
    #                 w += 1
    # print(r, w, r / (r + w))

# 598 16 0.9739413680781759 灾情

    # r2 = 0
    # w2 = 0
    # # path = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\爬取测试\新建文件夹\灾情"
    # path = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\爬取测试\\test2\新建文件夹\\tf"
    # for classification in [""]:
    #     files = os.listdir(path)
    #     for file in files:
    #         with open(path + '\\' + file, 'r', encoding='utf8') as f:
    #             url = f.readline()
    #             title = f.readline()
    #             time = f.readline()
    #             source = f.readline()
    #             contents = f.readlines()
    #             zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ',
    #                                                                                                 '').replace(
    #                 '\u3000', '') for content in contents if len(content) > 2])
    #             # print(len(zw))
    #
    #             # zw = title.replace("title","").replace("text","") + zw.replace("text","")
    #             print(title.replace("title\t","") + zw.replace("text",""))
    #
    #             this_classififcaton = classifications2
    #             index,result = sbwr.predict_short(title.replace("title\t","").replace(" ","").strip() + zw.replace("text",""))
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
    #                 print("模型结果：", this_classififcaton[index],
    #                   index == 0, '\n')
    #
    #             if index == 0 or disaster_info_num(contents) > 4:
    #                 r2 += 1
    #             else:
    #                 w2 += 1
    #
    # print(r2, w2, r2 / (r2 + w2))

    # r = 0
    # w = 0
    #
    # p = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\CSV\\test.csv"
    # datas = pd.read_csv(p)
    # titles = datas["title"]
    # newss = datas["news"]
    # labels = datas["label"]
    #
    # for t,n,l in zip(titles,newss,labels):
    #     index,result = sbwr.predict_short(str(t)+str(n))
    #     print(t,l,classifications2[index] == l)
    #     if(classifications2[index] == l):
    #         r += 1
    #     else:
    #         w += 1
    #
    # print(r, w, r / (r + w))

#not_disaster  1418/1486  68/1418





