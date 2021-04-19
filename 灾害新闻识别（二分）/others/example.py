# -*- coding: UTF-8 -*-
# import bert_binary_classifer
import hierarchical_bert_binary_classifer

# import lda_binary_classifer
# import tfidf_binary_classifer
classifications = ["冰雹灾害", "台风灾害", "城市内涝", "大风灾害", "暴雨洪涝", "雷电灾害"]

config_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\MyProject\天气灾害分类及信息提取\chinese_L-12_H-768_A-12\vocab.txt'

if __name__ == '__main__':

    hbc = hierarchical_bert_binary_classifer.HierarchicalBertWeatherRecongizer(config_path, checkpoint_path, dict_path, split_len=200, overlap_len=30)
    hbc.build_lstm()
    hbc.lstm_model.load_weights("hbbc2.h5")

    #
    # path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第一次测试\张-测试用例\\"
    # for classification in classifications:
    #      fpath = path + classification
    #      files = os.listdir(fpath)
    #      for file in files:
    #          if file == "_新闻文本说明.txt":
    #              continue
    #          with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
    #              title = f.readline()
    #              time = f.readline()
    #              source = f.readline()
    #              contents = f.readlines()
    #              zw = ''.join([content.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','') for content in contents if len(content) > 2])
    #              result = hbc.predict_str(zw)
    #              print("文件名：", "张-测试用例\\" + file.strip())
    #              print("新闻标题：", title.strip())
    #              print("模型结果：", result, " 正确结果：", 0, result==0, '\n')
    #
    #
    #
    # path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第一次测试\邢-测试用例\\"
    # for classification in classifications:
    #     fpath = path + classification
    #     files = os.listdir(fpath)
    #     for file in files:
    #         if file == "_新闻文本说明.txt":
    #             continue
    #         with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
    #             title = f.readline()
    #             time = f.readline()
    #             source = f.readline()
    #             contents = f.readlines()
    #             zw = ''.join(
    #                 [content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '')
    #                  for content in contents if len(content) > 2])
    #             result = hbc.predict_str(zw)
    #             print("文件名：", "邢-测试用例\\" + file.strip())
    #             print("新闻标题：", title.strip())
    #             print("模型结果：", result, " 正确结果：", 0, result==0,'\n')
    #
    #
    #
    # path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第一次测试\邢-测试用例2\\"
    # for classification in classifications:
    #      fpath = path + classification
    #      files = os.listdir(fpath)
    #      for file in files:
    #          if file == "说明文档.xlsx":
    #              continue
    #          with open(fpath + '\\' + file, 'r', encoding='utf8') as f:
    #              title = f.readline()
    #              time = f.readline()
    #              source = f.readline()
    #              contents = f.readlines()
    #              zw = ''.join(
    #                  [content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000',
    #                                                                                                          '')
    #                   for content in contents if len(content) > 2])
    #              result = hbc.predict_str(zw)
    #              print("文件名：", "邢-测试用例2\\" + file.strip())
    #              print("新闻标题：", title.strip())
    #              print("模型结果：", result, " 正确结果：", 0,result==0, '\n')

#

    # datas = pd.read_csv('D:\MyProject\天气灾害分类及信息提取\新闻文本\非灾害文本\非气象灾害文本.csv')
    # titles = datas['Title']
    # contents = datas['Text_data']
    # count = 0
    # for t,c in zip(titles,contents):
    #     try:
    #         c = c.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','')
    #         if len(contents) < 10:
    #             result = 1
    #         else:
    #             result = hbc.predict_str(c)
    #         print(count,t,"result:", result)
    #     except Exception as e:
    #         print(count,t,"result:", 1)
    #     count = count+1

#468/10494

#
    # kinds = ["暴雨洪涝","冰雹灾害","城市内涝","大风灾害","雷电灾害","台风灾害"]
    # path = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料\\"
    # for k in kinds:
    #     files = os.listdir(path+k)
    #     for file in files:
    #         with open(path+k+"\\"+file,'r',encoding='utf8') as f:
    #             title = f.readline()
    #             time = f.readline()
    #             source = f.readline()
    #             contents = f.readlines()
    #             zw = ''.join([content.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '').replace('\u3000', '') for content in contents if len(content) > 2])
    #             result = hbc.predict_str(zw)
    #             print("新闻标题：", title.strip())
    #             print("模型结果：", result, " 正确结果：", 0, result == 0, '\n')

# 5598/264

    # datas = pd.read_csv('D:\MyProject\天气灾害分类及信息提取\新闻文本\气象灾害文本.csv')
    # titles = datas['Title']
    # contents = datas['Text_data']
    # count = 0
    # for t,c in zip(titles,contents):
    #     try:
    #         c = c.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','').replace('&nbsp','')
    #         if len(contents) < 10:
    #             result = 1
    #         else:
    #             result = hbc.predict_str(c)
    #         print(count,t,"result:", result == 0)
    #     except Exception as e:
    #         print(count,t,"result:", "False2")
    #     count = count+1











