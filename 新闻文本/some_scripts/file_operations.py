# -*- coding:utf-8 -*-
# editor: zzh

#用来随便处理一些数据

import os
import pandas as pd

def func1():
    path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\z新灾情文本\非灾情文本模型结果"

    files = os.listdir(path)

    for k in ["disaster","not_disaster"]:
        datas = []
        for file in files:
            if file.startswith(k):
                data = pd.read_csv(path + '\\' + file)
                datas.append(data)
                print(len(datas))
        d = pd.concat(datas)
        d.to_csv(path + '\\' + k + '_datas.csv',encoding='utf8')

def func2():
    pass




