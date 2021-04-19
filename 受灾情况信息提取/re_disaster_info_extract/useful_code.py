# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/11/5
import pandas as pd
import re


classifications = ['低温', '冰雹', '台风', '地质灾害', '城市内涝', '大雾', '大风', '干旱灾害', '暴雨洪涝', '森林草原火灾', '雪灾', '雷电']
# classifications = ['低温']
path = r"D:\MyProject\天气灾害分类及信息提取\测试数据\第3次测试\正确结果"
path2 = "D:\MyProject\天气灾害分类及信息提取\测试数据\第2次测试\测试文本"

def get_sentence(type,infos,file):
    result = []
    with open(file,'r',encoding='utf8') as f:
        title = f.readline()
        time = f.readline()
        source = f.readline()
        contents = f.readlines()
        for content in contents:
            lines = re.split('[，。！？；,?!;、()（）]', content)
            # for line in lines:
            #     for info in infos:
            #         if line.find(info) != -1:
            #             result.append(line)
            for idx in range(1,len(lines)-1):
                for info in infos:
                    if lines[idx].find(info) != -1:
                        result.append((type,info,lines[idx-1]+","+lines[idx]+","+lines[idx+1]))
    return result

def get_all_miss_point():
    for classification in classifications:
        datas = pd.read_excel(path+"\\"+classification+"-测试结果.xlsx", header=None)
        # print(datas)

        types = datas[0]
        infos = datas[1]
        titles = datas[4]

        lines = []
        for type,i,t in zip(types,infos,titles):
            try:
                t = t.strip()
                ifos = re.split("[,，、]",i)
                # print(ifos)
                # if type(ifos) == str:
                #     ifos = [ifos]
                result = get_sentence(type,ifos,path2+"\\"+classification+"\\"+classification+"_"+t+".txt")
                lines += result
            except Exception as e:
                pass
        for l in lines:
            print(l)



if __name__ == '__main__':
    get_all_miss_point()





