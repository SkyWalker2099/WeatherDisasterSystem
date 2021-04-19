# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/12/30
# 脚本功能，筛选掉非国内新闻

import jieba
import jieba.posseg as pseg
import re
import pickle
import os

p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料"
# p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\国际新闻文本\spider-data"
# p = "D:\MyProject\天气灾害分类及信息提取\测试数据\第5次测试\测试文本"
# p = "test"
classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
# classifications = [""]
foriegns = pickle.load(open("countries.pk",'rb'))

cities_in_china = pickle.load(open("cities_in_china.pk",'rb'))

def c_count(contents):
    i = 0
    f = 0
    for content in contents:
        # content = re.sub("[省|市|县]","",content)
        words = pseg.cut(content)
        for word,flag in words:
            if flag == 'ns':
                print(word)
                word = re.sub("[省|市|县|区]", "", word)
                if word in foriegns:
                    f+=1
                if word in cities_in_china:
                    i+=1
    print("*"*10)
    return f,i

def country_filter(title,contents):
    print(title)
    tf,ti = c_count([title])
    cf1,ci1 = c_count(contents[:1])
    cf,ci = c_count(contents)

    print(tf, ti)
    print(cf1, ci1)
    print(cf, ci)
    print("*****"*10)
    if(tf > ti):
        return True
    elif(tf < ti):
        return False
    else:
        if(cf1 > ci1):
            return True
        elif(cf1 < ci1):
            return False
        else:
            if(cf > ci):
                return True
            else:
                return False



if __name__ == '__main__':
    for classification in classifications:
        files = os.listdir(p + '\\' + classification)
        # files = os.listdir(p)
        for file in files:
            try:
                with open(p + '\\' + classification + '\\' + file,'r',encoding='utf8') as f:
                # with open(p + '\\' + file, 'r', encoding='utf8') as f:
                #     url = f.readline()
                    title = f.readline()
                    time = f.readline()
                    source = f.readline()
                    contents = f.readlines()
                    # cs = contents[:1].append(title)
                    contents.append(title)
                    if country_filter(title,contents) == True:
                        print(title + '\t' + 'yes')
                        with open("test" + '\\' + file,'w',encoding='utf8') as g:
                            # g.write(url)
                            g.write(title)
                            g.write(time)
                            g.write(source)
                            for c in contents:
                                g.write(c)
                    else:
                        print(title + '\t' + 'no')
            except Exception as e:
                e.with_traceback()
                pass

    # file = "D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\country_filter\\test\大雾_报告称雾霾成中国入境游主要影响因素之一.txt"
    # with open(file,'r',encoding='utf8') as f:
    #     title = f.readline()
    #     time = f.readline()
    #     source = f.readline()
    #     contents = f.readlines()
    #     print(country_filter(title,contents))
