
import os
import re
from re_extract import *

paths = ["D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\暴雨洪涝\未标记暴雨洪涝",
         "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\冰雹\未标记冰雹典型",
         "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\城市内涝\未标记城市内涝",
         "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\大风\未标记大风典型",
         "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\雷电\未标记雷电",
         "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\台风\未标记台风recover"]

paths2 = ["D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\大雾\大雾无标",
        "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\低温\低温无标",
        "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\地质灾害\地质灾害无标",
        "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\干旱\干旱无标",
        "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\森林草原火灾\森林草原火灾无标",
        "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\雪灾\雪灾无标"]

paths = paths + paths2

for p in paths:
    files = os.listdir(p)
    tnum = 0
    for file in files:
        with open(p+'\\'+file,'r',encoding='utf8') as f:
            t = f.readline()
            time = f.readline()
            source = f.readline()
            contents = f.readlines()
            info = info_extract(contents)

            tnum += len(info["ASD"])
    print(p,tnum)