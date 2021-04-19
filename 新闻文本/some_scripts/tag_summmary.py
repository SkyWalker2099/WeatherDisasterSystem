# -*- coding:utf-8 -*-
# editor: zzh
# date: 2021/3/1
#用处是将所标注的标签集中到一个txt中

import re
import os

tags = ['/DS','/TS','/DO','/TO']
patterns = [re.compile(' \S*?'+tag+' ') for tag in tags]
classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料2"
path2 = "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\\12类别"
path3 = "D:\MyProject\天气灾害分类及信息提取\新闻文本\语料库2.0\时间\\all"

tpath = "D:\MyProject\天气灾害分类及信息提取\新闻文本\范野学长毕设数据\统合数据1"

def tag_detect(content,pattern,tag):
    results = pattern.findall(content)
    results = [result.strip().replace(tag,"") for result in results]
    return results



for classification in classifications:
    files = os.listdir(path+"\\"+classification)
    files_loc_taged = os.listdir(path2+'\\'+classification+"\\位置")
    files_time_taged = os.listdir(path3+"\\"+classification)

    for file in files:
        if file in files_loc_taged and file in files_time_taged:
            with open(path2+'\\'+classification+"\\位置\\"+file,'r',encoding='utf8') as f1:
                with open(path3+"\\"+classification+'\\'+file,'r',encoding='utf8') as f2:
                    t1 = f1.readline()
                    t2 = f2.readline()
                    time1 = f1.readline()
                    time2 = f2.readline()
                    s1 = f1.readline()
                    s2 = f2.readline()

                    # contents1 = f1.read()
                    contents1 = f1.readlines()
                    contents2 = f2.readlines()
                    print(len(contents1) == len(contents2))
                    # for pattern,tag in zip(patterns,tags):
                    #     for content in contents2:
                    #         results = tag_detect(content,pattern,tag)
                    #         for result in results:
                    #             contents1 = contents1.replace(result," "+result+tag+" ").replace(tag+" "+tag, tag)
                    #
                    # with open(tpath + '\\' + classification + '\\' + file,'w',encoding='utf8') as g:
                    #     g.write(t1)
                    #     g.write(time1)
                    #     g.write(s1)
                    #     g.write(contents1)





