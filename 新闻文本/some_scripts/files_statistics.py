# -*- coding:utf-8 -*-
# editor: zzh
# 用于对现有文本数据进行一些统计

import re
import os
import pandas as pd

def func1():
    """统计一下句子平均长度"""
    classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
    path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料"
    c = 0
    for classification in classifications:
        files = os.listdir(path+"\\"+classification)
        for file in files:
            try:
                with open(path+"\\"+classification+"\\"+file,'r',encoding='utf8') as f:
                    title = f.readline().strip()
                    time = f.readline()
                    source = f.readline()
                    contents = f.readlines()

                    lines = []
                    total_len = 0
                    for content in contents:
                        content = content.replace('\r', '').replace('\n', '').replace('\u3000', '')
                        total_len += len(content)
                        lines = lines + re.split('[。｡！!？?]',content)
                    print(file,total_len,len(lines),int(total_len/len(lines)))
                    if(int(total_len/len(lines)) > 100):
                        c+=1
            except Exception as e:
                pass
    print(c)



def func2():
    p = """D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\SYN\CSV\\train.csv"""

    datas = pd.read_csv(p)

    titles = datas["title"]
    newss = datas["news"]
    labels = datas["label"]
    c = 0

    for t,n,l in zip(titles,newss,labels):
        if(l != 'not_disaster'):
            continue
        lines = re.split('[。｡！!？?\t]',n)
        print(t,len(n),len(lines),int(len(n)/len(lines)))
        if(int(len(n)/len(lines)) > 100):
            c+=1

    print(c)


if __name__ == '__main__':
    # func2()
    pass