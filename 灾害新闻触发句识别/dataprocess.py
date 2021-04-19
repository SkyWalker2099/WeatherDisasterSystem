# -*- coding:utf-8 -*-
# editor: zzh

from untils import *
import os

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料"

c = 0
with open("all_lines.txt",'w',encoding='utf8') as g:
    for classification in classifications:
        files = os.listdir(path + "\\" + classification)
        for file in files:
            with open(path + "\\" + classification + "\\" + file, 'r', encoding='utf8') as f:
                title = f.readline().strip()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                lines = split_contents(contents,max_piece = 10, max_len = 450)
                for line in lines:
                    g.write('。'.join(line) + '\n')

