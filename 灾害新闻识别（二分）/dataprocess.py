# -*- coding: UTF-8 -*-
import os
import random
import pandas as pd
import shutil
from sklearn.utils import shuffle

def random_move_files(fpath, tpath, per):
    """
    随机将某一文件夹内一定比例的文件转移到另一文件夹内
    :param fpath: 源文件夹
    :param tpath: 目的文件夹
    :param per: 比例
    :return:
    """
    files = os.listdir(fpath)
    num = len(files)
    target_num = int(num*per)
    target_files = random.sample(files,target_num)
    for file in target_files:
        try:
            shutil.move(fpath+"\\"+file, tpath+"\\"+file)
        except Exception as e:
            pass

def t_process1():
    """
    将新闻文本转换为csv文件
    :return:
    """
    path = "新闻文本"
    types = ["test", "train"]
    labels = ["disaster", "not_disaster"]
    datas = pd.DataFrame(columns=[])

    for t in types:
        datas = pd.DataFrame(columns=["title", "news", "label"])
        for label in labels:
            p = path + '\\' + t + '\\' + label
            files = os.listdir(p)
            for file in files:
                try:
                    with open(p + '\\' + file, 'r', encoding='utf8') as f:
                        title = f.readline().strip()
                        time = f.readline()
                        source = f.readline()
                        contents = f.readlines()

                        zw = ''.join([content.strip() for content in contents if len(content) > 2])
                        if (len(zw) > 5):
                            one_news = {"title": title, "news": zw, "label": label}
                            datas = datas.append(one_news, ignore_index=True)
                        else:
                            print(title)
                            print("***"*10)
                except Exception as e:
                    pass
        datas = shuffle(datas)
        print(datas, datas.shape)
        datas.to_csv("DATA\\" + t + "_datas.csv", encoding="utf8", index=False)

if __name__ == '__main__':
    # random_move_files("G:\PycharmProject\tfidf\新闻文本\\train\\not_disaster","G:\PycharmProject\tfidf\新闻文本\\test\\not_disaster",0.2)
    # random_move_files("G:\PycharmProject\tfidf\新闻文本\\train\\disaster","G:\PycharmProject\tfidf\新闻文本\\test\\disaster", 0.2)
    # t_process1()
    # test_data = pd.read_csv("DATA\\train_datas.csv")
    # tls = list(test_data.values)
    # tls.sort(key=lambda x:len(x[0]))
    # for tl in tls:
    #     print(tl[0],tl[2])


#生成非灾害csv
    # datas = pd.DataFrame(columns=["title", "news", "label"])
    # path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\非灾害文本"
    # kinds = ["冰雹非灾情","内涝非灾情","其他非灾情","其他非灾情2","其他非灾情1","台风非灾情"] #5899
    # count = 0
    # for k in kinds:
    #     files = os.listdir(path+"\\"+k)
    #     for file in files:
    #         print(count)
    #         try:
    #             with open(path+"\\"+k+"\\"+file,'r',encoding='utf8') as f:
    #                 title = f.readline().strip()
    #                 time = f.readline()
    #                 source = f.readline()
    #                 print(source)
    #                 contents = f.readlines()
    #                 zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '') for content in contents if len(content) > 2])
    #                 if len(zw) > 10:
    #                     one_news = {"title": title, "news": zw, "label": "not_disaster"}
    #                     datas = datas.append(one_news, ignore_index=True)
    #                 count += 1
    #         except Exception as e:
    #             e.with_traceback()
    #
    #     # datas.to_csv("DATA\\not_disaster.csv", encoding="utf8", index=False)
    #
    # # datas = pd.read_csv("DATA\\not_disaster.csv")
    # new_datas = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\新闻文本\非灾害文本\非气象灾害文本.csv") # 10498
    # titles = new_datas['Title']
    # contents = new_datas['Text_data']
    # count = 0
    # for t,c in zip(titles,contents):
    #     try:
    #         if type(c) == str and len(c) > 10:
    #             one_news = {"title": t, "news": c.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', ''), "label": "not_disaster"}
    #             datas = datas.append(one_news, ignore_index=True)
    #     except Exception as e:
    #         pass
    #     count = count+1
    #     print(count)
    #
    # print(len(datas)) #15403
    # datas.to_csv("DATA\\CSV\\not_disaster.csv", encoding="utf8", index=False)


#生成灾害csv
    # datas = pd.DataFrame(columns=["title", "news", "label"])
    # classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
    # path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料 - 副本" # 10375
    # count = 0
    # for classification in classifications:
    #     files = os.listdir(path+"\\"+classification)
    #     for file in files:
    #         with open(path+"\\"+classification+"\\"+file,'r',encoding='utf8') as f:
    #             title = f.readline().strip()
    #             time = f.readline()
    #             source = f.readline()
    #             if len(source) > 50:
    #                 continue
    #             print(source)
    #             contents = f.readlines()
    #             zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '') for content in contents if len(content) > 2])
    #             if len(zw) > 10:
    #                 one_news = {"title": title, "news": zw, "label": "disaster"}
    #                 datas = datas.append(one_news, ignore_index=True)
    #         print(count)
    #         count+=1
    #
    # new_datas = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\新闻文本\气象灾害文本.csv") #  2566
    # titles = new_datas['Title']
    # contents = new_datas['Text_data']
    # count = 0
    # for t,c in zip(titles,contents):
    #     try:
    #         if type(c) == str and len(c) > 10:
    #             one_news = {"title": t, "news": c.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', ''), "label": "disaster"}
    #             datas = datas.append(one_news, ignore_index=True)
    #     except Exception as e:
    #         pass
    #     count = count+1
    #     print(count)
    #
    # print(len(datas)) #2566
    # datas.to_csv("DATA\\CSV\\disaster.csv", encoding="utf8", index=False)
    # print(len(datas)) # 12667


# 看看数据
#     datas = pd.read_csv("DATA\\CSV\\disaster.csv")
#     titles = datas["title"]
#     for i,t in enumerate(titles):
#         print(i,t)

    # ndatas = pd.read_csv("DATA\\CSV\\not_disaster.csv")
    # titles = ndatas["title"]
    # for i, t in enumerate(titles):
    #     print(i, t)

# 合并后分为训练集和测试集
#     datas = pd.read_csv("DATA\\BALANCED\\CSV\\disaster.csv",index_col=0)
#     ndatas = pd.read_csv("DATA\\BALANCED\\CSV\\not_disaster.csv",index_col=0)
#
#     fdatas = pd.concat([datas,ndatas])
#     fdatas = shuffle(fdatas)
#     titles = fdatas["title"]
#     labels = fdatas["label"]
#     for t,l in zip(titles,labels):
#         print(t,l)
#
#     leng = len(fdatas)
#     print(leng)
#
#     train_datas = fdatas[0:int(leng*0.95)]
#     test_datas = fdatas[int(leng*0.95):]
#
#     print(len(train_datas))
#     print(len(test_datas))
#
#     train_datas.to_csv("DATA\\BALANCED\\CSV\\train.csv",encoding='utf8')
#     test_datas.to_csv("DATA\\BALANCED\\CSV\\test.csv",encoding='utf8')

#将train.csv中的新闻分为长短两类
    datas = pd.read_csv("DATA\\BALANCED\\CSV\\train.csv")

    ldatas = pd.DataFrame(columns=["title", "news", "label"])
    sdatas = pd.DataFrame(columns=["title", "news", "label"])

    titles = datas["title"]
    newss = datas["news"]
    labels = datas["label"]
    for t,n,l in zip(titles,newss,labels):
        if len(n) > 500:
            ldatas = ldatas.append({"title": t, "news": n, "label": l},ignore_index=True)
            print(t,l,len(n))
        else:
            sdatas = sdatas.append({"title": t, "news": n, "label": l},ignore_index=True)
            print(t,l,len(n))

    print(len(ldatas))
    print(len(sdatas))

    ldatas.to_csv("DATA\\BALANCED\\CSV\\ltrain.csv", encoding='utf8')
    sdatas.to_csv("DATA\\BALANCED\\CSV\\strain.csv", encoding='utf8')

    print(ldatas)
    print(sdatas)



    # files = os.listdir("DATA\\CSV")
    # for file in files:
    #     datas = pd.read_csv("DATA\\CSV\\" + file)
    #     print(file,len(datas))
    pass

# disaster.csv 12668
# ltrain.csv 13616
# not_disaster.csv 15403
# strain.csv 11647
# test.csv 2808
# train.csv 25263


##########################################
#找出train.csv中，长度小于100的新闻
    # datas = pd.read_csv("DATA\\SYN\\CSV\\test.csv")
    #
    # # ldatas = pd.DataFrame(columns=["title", "news", "label"])
    # sdatas = pd.DataFrame(columns=["title", "news", "label"])
    #
    # titles = datas["title"]
    # newss = datas["news"]
    # labels = datas["label"]
    # for t,n,l in zip(titles,newss,labels):
    #     if len(n) > 150:
    #         pass
    #         # ldatas = ldatas.append({"title": t, "news": n, "label": l},ignore_index=True)
    #         # print(t,l,len(n))
    #     else:
    #         sdatas = sdatas.append({"title": t, "news": n, "label": l},ignore_index=True)
    #         print(t,l,len(n),n)
    #
    # # print(len(ldatas))
    # print(len(sdatas))
    #
    # # ldatas.to_csv("DATA\\CSV\\ltrain.csv", encoding='utf8')
    # sdatas.to_csv("DATA\\超短句\\stest.csv", encoding='utf8')
    #
    # # print(ldatas)
    # print(sdatas)


########################################
#把甲方给的数据与现有数据整合

    # import re
    # from syn_bert_classifer import *

    # sbwr = SynBertWeatherRecongizer(config_path, checkpoint_path, dict_path, max_len=450)

    # sbwr.bert.load_weights("MODELS\\syn_model\\bert_short.h5")
    # sbwr.lstm_model.load_weights("MODELS\\syn_model\\lstm_model.h5")

    # data1 = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\新闻文本\z新灾情文本\气象灾情.csv")
    # data2 = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\SYN\CSV\\disaster.csv")
    #
    # titles = data1["title"]
    # contents = data1["Text_data"]
    #
    # lt = int(len(titles)/3*2)
    # lc = int(len(contents)/3*2)
    #
    # for t,content in zip(titles[:lt],contents[:lc]):
    #     try:
    #         t = str(t)
    #         t = t.split("_")[0]
    #         t = t.split("-")[0]
    #         content = str(content)
    #         # content = content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000','').replace("????","")
    #         content = re.sub("(/[a-zA-Z])|(\r)|(\n)|(\t)|(\s)|(&nbsp)|(\u3000)|([a-zA-Z])",'',content)
    #         if(len(content) < 10 or len(content) > 2500 or content.startswith("很抱歉，没有找到")):
    #             continue
    #         # index,result = sbwr.predict_str(t+content)
    #         # if(index == 0):
    #         #     continue
    #         print(t)
    #         print(content)
    #         data2 = data2.append({"title": t, "news": content, "label": "disaster"}, ignore_index=True)
    #         print(len(data2))
    #         # if(len(data2) > 30000):
    #         #     break
    #     except Exception as e:
    #         pass
    #
    # data2.to_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\\disaster.csv",encoding='utf8')

    # 甲方提供数据，灾害新闻使用了2/3，非灾害新闻使用了 约1万五千条

    # d1 = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\\disaster.csv",index_col=0)
    # d2 = pd.read_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\\disaster.csv",index_col=0)
    # d = pd.concat([d1,d2])
    # d = shuffle(d)
    # print(d)
    # d.to_csv("D:\MyProject\天气灾害分类及信息提取\灾害新闻识别（二分）\DATA\BALANCED\\disaster.csv",encoding='utf8')

