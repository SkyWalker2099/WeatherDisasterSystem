# -*- coding:utf-8 -*-
# editor: zzh

import pandas as pd
p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\z新灾情文本\非气象灾情.csv"

datas = pd.read_csv(p)

titles = datas["title"]
contents = datas["Text_data"]

for content in contents:
    print(len(content),content[:20])
