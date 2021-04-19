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
    test_data = pd.read_csv("DATA\\train_datas.csv")
    tls = list(test_data.values)
    tls.sort(key=lambda x:len(x[0]))
    for tl in tls:
        print(tl[0],tl[2])
