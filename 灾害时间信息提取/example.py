# import kashgari
# from kashgari.embeddings import BERTEmbedding
# from kashgari.tasks.labeling import BiGRU_CRF_Model
# from kashgari.tasks.labeling import BiLSTM_CRF_Model
# from sklearn.model_selection import train_test_split
import os
import TimeTagModel2
import pandas as pd
import numpy as np
# config_path = 'chinese_L-12_H-768_A-12\\bert_config.json'
# checkpoint_path = 'chinese_L-12_H-768_A-12\\bert_model.ckpt'
# vocab_path = 'chinese_L-12_H-768_A-12\\vocab.txt'

config_path = 'chinese_L-12_H-768_A-12_pruned\\bert_config2.json'
checkpoint_path = 'chinese_L-12_H-768_A-12_pruned\\bert_pruning_9_layer.ckpt'
vocab_path = 'chinese_L-12_H-768_A-12_pruned\\vocab.txt'

import re


# model1 = kashgari.utils.load_model("bert_bigru_crf2.h5")
#
# def test1():
#     model1 = kashgari.utils.load_model('bert_bigru_crf2.h5')
#     datas = pd.read_csv("DATA\\test_data.csv")
#     lines = datas["line"]
#     tags = datas["tags"]
#     for l, t in zip(lines, tags):
#         input = [i for i in l]
#         output = model1.predict([input])[0]
#         print(l)
#         for type in range(1, 5):
#             answer = ""
#             answers = list()
#             for i, o in zip(input, output):
#                 if o == str(type):
#                     answer = answer + i
#                 else:
#                     if len(answer) != 0:
#                         answers.append(answer)
#                     answer = ""
#             if len(answer) != 0:
#                 answers.append(answer)
#             print(answers)
#
#
# def test_zw1(contents):
#     print("****" * 10)
#     print("")
#
#     all_results = [[], [], [], []]
#
#     for content in contents:
#         content = content.replace('｡', '。').replace(',', '，')
#         lines = re.split('[。！？!?.]', content)
#         for line in lines:
#             input = [i for i in line]
#             output = model1.predict([input])[0]
#             for type in range(1,5):
#                 answer = ""
#                 for i,o in zip(input,output):
#                     if o == str(type):
#                         answer = answer + i
#                     else:
#                         if len(answer) != 0:
#                             all_results[type-1].append(answer)
#                         answer = ""
#                 if len(answer) != 0:
#                     all_results[type-1].append(answer)
#     print(contents)
#     print(all_results)
#     print("")


# def test2():
#     model2 = TimeTagModel2.TimeTagger(config_path=config_path, checkpoint_path=checkpoint_path,
#                                       dict_path=vocab_path,num_tag=5)
#     model2.model.load_weights("bert_crf.h5")
#     datas = pd.read_csv("DATA\\test_data.csv")
#     lines = datas["line"]
#     tags = datas["tags"]
#
#     lines1,lines2 = model2.get_encode(lines)
#
#     print(lines1.shape)
#     print(lines2.shape)
#
#     tags = model2.tags_complete(tags)
#
#     print(tags.shape)
#
#     for l0,l1,l2,t in zip(lines,lines1,lines2,tags):
#         result = model2.model.predict([[l1],[l2]])
#         # print(l0)
#         # print(list(t))
#         # print([np.argmax(i) for i in result[0]])
#         # print("")
#
#         print(l0)
#         for type in range(1, 5):
#                 answer = ""
#                 answers = list()
#                 for i, o in zip(l0, [np.argmax(i) for i in result[0]]):
#                     if o == type:
#                         answer = answer + i
#                     else:
#                         if len(answer) != 0:
#                             answers.append(answer)
#                         answer = ""
#                 if len(answer) != 0:
#                     answers.append(answer)
#                 print(answers)






tt = TimeTagModel2.TimeTagger(config_path=config_path, checkpoint_path=checkpoint_path,dict_path=vocab_path, num_tag=5,maxlen=100)
tt.model.load_weights("pruned_bert_bigru_crf.h5")
def test_zw2(contents):

    result1 = tt.predict_contents_1(contents)
    result2 = tt.predict_contents_2(contents)
    print(result1)
    print(result2)
    print("\n","****" * 10,"\n")



if __name__ == '__main__':
    classifications = ["冰雹灾害", "台风灾害", "城市内涝", "大风灾害", "暴雨洪涝", "雷电灾害"]

    path = "C:\\Users\Zzh\Desktop\第一次测试\张-测试用例\\"
    for classification in classifications:
        fpath = path + classification
        files = os.listdir(fpath)
        for file in files:
            if file == "_新闻文本说明.txt":
                continue
            with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                print("*****" * 10)
                print("文件名：", "张-测试用例\\" + file.strip())
                print("新闻标题：", title.strip(), '\t类别：', classification)

                print(time)
                print(contents)

                stime, otime = tt.predict_contents_3(contents, time)
                if stime == None or otime == None:
                    continue
                """
                数据结构如下
                {
                    "D":{
                        "yyyy":年,
                        "mm":月,
                        "dd":日
                    }
                    "T":{
                        "hh":时,
                        "mm":分
                    }
                }
                """

                print("开始时间：%s年%s月%s日 %s时%s分" % (
                    stime["D"]["yyyy"], stime["D"]["mm"], stime["D"]["dd"], stime["T"]["hh"], stime["T"]["mm"]))
                print("结束时间：%s年%s月%s日 %s时%s分" % (
                    otime["D"]["yyyy"], otime["D"]["mm"], otime["D"]["dd"], otime["T"]["hh"], otime["T"]["mm"]))
                print("*****" * 10)

    path = "C:\\Users\Zzh\Desktop\第一次测试\邢-测试用例\\"
    for classification in classifications:
        fpath = path + classification
        files = os.listdir(fpath)
        for file in files:
            if file == "_新闻文本说明.txt":
                continue
            with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                print("*****" * 10)
                print("文件名：", "邢-测试用例\\" + file.strip(), '\t类别：', classification)
                print("新闻标题：", title.strip())
                stime, otime = tt.predict_contents_3(contents, time)
                if stime == None or otime == None:
                    continue
                print("开始时间：%s年%s月%s日 %s时%s分" % (
                stime["D"]["yyyy"], stime["D"]["mm"], stime["D"]["dd"], stime["T"]["hh"], stime["T"]["mm"]))
                print("结束时间：%s年%s月%s日 %s时%s分" % (
                otime["D"]["yyyy"], otime["D"]["mm"], otime["D"]["dd"], otime["T"]["hh"], otime["T"]["mm"]))
                print("*****" * 10)
    path = "C:\\Users\Zzh\Desktop\第一次测试\邢-测试用例2\\"
    for classification in classifications:
        fpath = path + classification
        files = os.listdir(fpath)
        for file in files:
            try:
                if file == "说明文档.xlsx":
                    continue
                with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
                    title = f.readline()
                    time = f.readline()
                    source = f.readline()
                    contents = f.readlines()
                    print("*****" * 10)
                    print("文件名：", "邢-测试用例2\\" + file.strip(), '\t类别：', classification)
                    print("新闻标题：", title.strip())
                    stime, otime = tt.predict_contents_3(contents, time)
                    if stime == None or otime == None:
                        continue
                    print("开始时间：%s年%s月%s日 %s时%s分" % (
                        stime["D"]["yyyy"], stime["D"]["mm"], stime["D"]["dd"], stime["T"]["hh"], stime["T"]["mm"]))
                    print("结束时间：%s年%s月%s日 %s时%s分" % (
                        otime["D"]["yyyy"], otime["D"]["mm"], otime["D"]["dd"], otime["T"]["hh"], otime["T"]["mm"]))
                    print("*****" * 10)
            except Exception as e:
                pass



