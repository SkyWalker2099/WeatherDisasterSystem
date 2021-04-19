
import pandas as pd
import re
import os
from sklearn.utils import shuffle
tags = ['/DS','/TS','/DO','/TO']
patterns = [re.compile(' \S*?'+tag+' ') for tag in tags]
# tags =

def file2csv():
    all_data = pd.DataFrame(columns=["line","tags"])
    classifications = ["冰雹灾害", "台风灾害", "城市内涝", "大风灾害", "暴雨洪涝", "雷电灾害"]

    y = 0
    n = 0

    for classification in classifications:
        files = os.listdir("新闻文本\\"+classification)
        for file in files:
            with open("新闻文本\\"+classification+'\\'+file,'r',encoding="utf8") as f:
                title = f.readline()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                for content in contents:
                    content = content.replace('｡', '。').replace(',', '，').strip()
                    lines = re.split('[。！？!?.]',content)
                    for line in lines:
                        if re.search("(/DS|/TS|/DO|/TO)",line) != None:
                            l,t = line_tag_detect(line)
                            one_data = {"line":l,"tags":t}
                            all_data = all_data.append(one_data,ignore_index=True)
                            y+=1
                        elif re.search("([一二三四五六七八九十千百万零点两]+|[\d.]+)[余多]*万*亿*(元|块|人)",line) != None and len(line) > 10 and n <= y:
                            l, t = line_tag_detect(line)
                            one_data = {"line":l,"tags":t}
                            all_data = all_data.append(one_data, ignore_index=True)
                            n+=1


    all_data = shuffle(all_data)

    length = len(all_data)
    train_length = int(length * (4 / 5))

    train_data = all_data[0:train_length]
    test_data = all_data[train_length:length]

    train_data.to_csv("DATA\\train_data3.csv",encoding='utf8',index=False)
    test_data.to_csv("DATA\\test_data3.csv", encoding='utf8', index=False)


def line_tag_detect(line):
    global tags
    line = line.replace('\n','')
    all_results = []
    for i,pattern in enumerate(patterns):
        tag = tags[i]
        results = re.findall(pattern,line)
        results = [result.replace(tag,'').strip() for result in results]
        all_results.append(results)
    print(all_results)
    line = line.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','')
    for tag in tags:
        line = line.replace(tag,'')
    line2 = [0]*len(line)

    for type,results in enumerate(all_results):
        start = 0
        for result in results:
            idx = line.find(result,start)
            for i in range(idx,idx+len(result)):
                line2[i] = type+1
            start = idx+len(result)


    line3 = ""
    for i in line2:
        line3 = line3 + str(i)

    # print(line)
    # print(line3)
    if len(line) != len(line3):
        raise Exception("程序出错")
    return line,line3


    # tags = [0]*length

def test():
    datas = pd.read_csv("data.csv")
    lines = datas["line"]
    tags = datas["tags"]
    for l,t in zip(lines,tags):
        if len(l) != len(t):
            raise Exception('长度不一致')
        print(l)
        for i in range(1,5):
            idxs = [j for j in range(len(t)) if t[j] == str(i)]
            res = [l[j] for j in idxs]
            print(res)

if __name__ == '__main__':
    file2csv()