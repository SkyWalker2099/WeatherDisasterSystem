# -*- coding:utf-8 -*-
# editor: zzh

import pandas as pd
# from syn_bert_classifer import *
# from short_classifier import *
import os
import re
from re_assisstant import *
config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'
import shutil
# sbwr = SynBertWeatherRecongizer(config_path,checkpoint_path,dict_path,max_len=400)
# sbwr.bert.load_weights("MODELS\\syn_model\\bert_short.h5")
# sbwr.lstm_model.load_weights("MODELS\\syn_model\\lstm_model.h5")

# bwc = BertWeatherRecongizer(config_path,checkpoint_path,dict_path,maxlen=200)
# bwc.model.load_weights("MODELS\\short_model\\bert_short_004.h5")

classificitions = ["disaster","not_disaster"]
classifications2 = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']


# def func1():
#     """这个是用自己的模型给甲方提供的数据预测一波的代码"""
#     p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\z新灾情文本\气象灾情.csv"
#     path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\z新灾情文本\非灾情文本模型结果"
#     datas = pd.read_csv(p)
#
#     titles = datas["title"]
#     contents = datas["Text_data"]
#
#     for idx in range(0, 1):
#
#         disaster_datas = pd.DataFrame(columns=["title", "news"])
#         not_disaster_datas = pd.DataFrame(columns=["title", "news"])
#
#         # for title,content in zip(titles[idx:idx+1000],contents[idx:idx+1000]):
#         for title, content in zip(titles, contents):
#
#             try:
#                 index = bwc.predict_str(str(title) + content)
#                 one_news = {"title": title, "news": content}
#                 result = classificitions[index]
#
#                 if index == 0:
#                     disaster_datas = disaster_datas.append(one_news, ignore_index=True)
#                 else:
#                     not_disaster_datas = not_disaster_datas.append(one_news, ignore_index=True)
#
#                 print(title, result)
#
#             except Exception as e:
#                 e.with_traceback()
#                 pass
#
#         print(disaster_datas.shape)

        # disaster_datas.to_csv(path + "\\" + "disaster_datas_{}.csv".format(idx), encoding="utf8", index=False)
        # not_disaster_datas.to_csv(path + "\\" + "not_disaster_datas_{}.csv".format(idx), encoding="utf8", index=False)

        # 对于甲方灾害文本测试的结果：2307/170

# def func2():
#     p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料"
#     for classification in classifications2:
#         files = os.listdir(os.path.join(p,classification))
#         for file in files:
#             print("***"*20)
#             print(os.path.join(p,classification,file))
#             with open(os.path.join(p,classification,file),'r',encoding='utf8') as f:
#                 title = f.readline()
#                 time = f.readline()
#                 source = f.readline()
#                 contents = f.readlines()
#
#                 lines = []
#                 total_len = 0
#                 for content in contents:
#                     content = content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '')
#                     total_len += len(content)
#                     lines = lines + re.split('[。！!？?]',content)
#
#                 if(total_len > 150):
#                     for line in lines:
#                         try:
#                             index,_ = bwc.predict_str(line.replace('\n',''))
#                             print(classificitions[index],line)
#                         except Exception as e:
#                             e.with_traceback()


# def func3():
#     p = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\爬取测试\新建文件夹\\ff"
#     for classification in [""]:
#         files = os.listdir(p)
#         for file in files:
#             print("***"*20)
#             print(os.path.join(p,file))
#             with open(os.path.join(p,classification,file),'r',encoding='utf8') as f:
#                 title = f.readline()
#                 time = f.readline()
#                 source = f.readline()
#                 contents = f.readlines()
#
#                 lines = []
#                 total_len = 0
#                 for content in contents:
#                     content = content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '').replace("text	","")
#                     total_len += len(content)
#                     lines = lines + re.split('[。｡！!？?]',content)
#
#                 if(total_len > 150):
#                     for line in lines:
#                         if(len(line)<2):
#                             continue
#                         try:
#                             index,_ = bwc.predict_str(line.replace('\n',''))
#                             print(classificitions[index],line)
#                         except Exception as e:
#                             e.with_traceback()

#
# def func4():
#     p = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\爬取测试\\test2\\not_disaster"
#     p2 = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\爬取测试\\test2\\temp"
#     files = os.listdir(p)
#     for file in files:
#         print("***" * 20)
#         print(os.path.join(p, file))
#         with open(os.path.join(p, file), 'r', encoding='utf8') as f:
#             title = f.readline()
#             time = f.readline()
#             source = f.readline()
#             contents = f.readlines()
#
#             if(disaster_info_num(contents) > 4):
#                 shutil.copy(os.path.join(p, file),os.path.join(p2,file))


if __name__ == '__main__':
    pass
    # func4()

