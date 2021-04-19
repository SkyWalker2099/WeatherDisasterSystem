import os
import pandas as pd
from sklearn.utils import shuffle


data_path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料"
# classifications = ["冰雹灾害","台风灾害","城市内涝","大风灾害","暴雨洪涝","雷电灾害"]
classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

def data2csv(maxlen,minlen,csv_path):

    all_data = pd.DataFrame(columns=["title","news","label"])
    for classification in classifications:
        files = os.listdir(data_path+"\\"+classification)
        for file in files:
            try:
                with open(data_path+"\\"+classification+"\\"+file,"r",encoding="utf8") as f:
                    title = f.readline()
                    time = f.readline()
                    source = f.readline()
                    content = f.readlines()
                    contents = [line.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','') for line in content if(len(line)>2)]
                    zw = "".join(contents)
                    if(len(zw)>minlen and len(zw)<maxlen):
                        news = zw
                        one_data = {"title":title,"news":news,"label":classification}
                        print(one_data)
                        all_data = all_data.append(one_data,ignore_index=True)

            except Exception as e:
                e.with_traceback()
    all_data = shuffle(all_data)

    length = len(all_data)
    train_length = int(length*(4/5))

    train_data = all_data[0:train_length]
    test_data = all_data[train_length:length]
    #
    #
    train_path = csv_path+"\\train_data.csv"
    test_path = csv_path+"\\test_data.csv"
    #
    train_data.to_csv(train_path,encoding="utf8",index=False)
    test_data.to_csv(test_path,encoding="utf8",index=False)




if __name__ == '__main__':
    data2csv(maxlen=2000,minlen=20,csv_path="DATAS\\DATA6")
    # data = pd.read_csv("DATAS\\DATA3\\test.csv")
    # print(data["title"].values)
    # print(data["title"].values.shape)
    #
    # print(data["news"].values)
    # print(data["news"].values.shape)
    #
    # print(data["label"].values)
    # print(data["label"].values.shape)
    pass