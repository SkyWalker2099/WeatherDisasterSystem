# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/12/11

import re_extract,syn_bert_classifer,WeatherClassifer,WeathreClassifer3
import os

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'


# bwc = WeatherClassifer.BertWeatherClassifer(config_path, checkpoint_path, dict_path, maxlen=400)
# bwc.model.load_weights("MODELS\\bert_masked_005.h5")
# #灾害类别分类模型


sbwr = syn_bert_classifer.SynBertWeatherRecongizer(config_path, checkpoint_path, dict_path, max_len=400)
sbwr.bert.load_weights("MODELS\\syn_model\\bert_short.h5")
sbwr.lstm_model.load_weights("MODELS\\syn_model\\lstm_model.h5")
#是否为灾害，二分类模型


classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

if __name__ == '__main__':
    r = 0
    w = 0
    path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第5次测试\测试文本"
    for classification in classifications:
        files = os.listdir(path + '\\' + classification)
        for file in files:
            with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
                title = f.readline()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ',
                                                                                                    '').replace(
                    '\u3000', '') for content in contents if len(content) > 2])
                # print(len(zw))
                index1 = sbwr.predict_str(zw)
                # index = bwc.predict_str(title + zw)
                # infos = re_extract.info_extract(contents)
                try:
                    print("文件名：", file.strip())
                    print("新闻标题：", title.strip())
                except Exception as e:
                    pass
                # print("模型结果：", classifications[index], " 正确结果：", classification,
                #       classifications[index] == classification, '\t灾害？：', index1 == 0)
                print("灾害？：",index1 == 0)

                # print(infos)

                # if classifications[index] == classification and index1 == 0:
                if index1 == 0:
                    r += 1
                else:
                    w += 1
    print(r, w, r / (r + w))
