# -*- coding:utf-8 -*-
# editor: zzh
import numpy as np
import pandas as pd
import re
import os

def content_cut(content):
    lines =  re.split("[。｡！!？?]",content)
    return [line for line in lines if len(line) > 3]

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
classifications = ['暴雨洪涝']

if __name__ == '__main__':

    # p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\z新灾情文本\气象灾情.csv"
    #
    # datas = pd.read_csv(p)
    #
    # titles = datas["title"]
    # newss = datas["Text_data"]
    #
    # for t,n in zip(titles,newss):
    #     print("****"*20)
    #     print(t)
    #     print(n)
    #     for l in content_cut(n):
    #         print(l)

    # p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料"
    # for classification in classifications:
    #     files = os.listdir(os.path.join(p,classification))
    #     for file in files:
    #         print("*****"*20)
    #         print(os.path.join(p,classification,file))
    #         with open(os.path.join(p,classification,file),'r',encoding='utf8') as f:
    #             title = f.readline()
    #             time = f.readline()
    #             source = f.readline()
    #             contents = f.readlines()
    #             # print(contents)
    #             print("-----"*20)
    #             lines = []
    #             for c in contents:
    #                 lines = lines + content_cut(c)
    #
    #             for l in lines:
    #                 print(l)



