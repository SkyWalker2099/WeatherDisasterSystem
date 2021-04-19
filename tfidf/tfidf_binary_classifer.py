# -*- coding: UTF-8 -*-
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle
import numpy as np
import os
import shutil

tfidf_path = "MODELS\\tfidf\\tfidf.pk"
clf_path = "MODELS\\tfidf\\tfidf_svm_clf.pk"

classifications = ["disaster","not_disaster"]

class TfidfClassifer(object):
    def __init__(self,tfidf_path,clf_path):
        self.tfidf_path = tfidf_path
        self.clf_path = clf_path
        self.tv = pickle.load(open(self.tfidf_path,'rb'))
        self.clf = pickle.load(open(self.clf_path,'rb'))

    def fenci(self,data,type):
        """根据输入数据类型不同进行不同的运算
        :param data:
        :param type: 1: pandas.core.series.Series   2: list    3 and else: string
        :return:
        """
        if type == 1:
            words_df = data.apply(lambda x:' '.join(jieba.cut(x)))
            return words_df
        elif type == 2:
            words_dfs = []
            for line in data:
                words_dfs.append(' '.join(jieba.cut(line)))
            return words_dfs
        else:
            return [' '.join(jieba.cut(data))]

    def predict(self,data):
        """
        :param x: 可以为单独新闻字符串，或字符串列表，series
        :return: 根据输入类型返回值类型不同
        """
        if isinstance(data,pd.core.series.Series):
            type = 1
        elif isinstance(data,list):
            type = 2
        elif isinstance(data,str):
            type = 3
        # 判断输入数据类型
        data_cuted = self.fenci(data,type)
        data_tfidf = self.tv.transform(data_cuted).toarray()
        results = self.clf.predict(data_tfidf)

        if type == 3:
            return results[0]
        else:
            return results

tc = TfidfClassifer(tfidf_path,clf_path)

if __name__ == '__main__':



#     test_data = pd.read_csv("DATA\\test_datas.csv")
#     test_data = test_data["news"][:5]
#
#     s1 = """陕西遭受暴雨、冰雹袭击 部分地区农作物绝收,西部网讯（陕西广播电视台《今日点击》记者 王勇）7月中旬以来，由于受强对流天气影响，陕西省中北部的咸阳、渭南、铜川、延安、榆林等地，先后遭受大风、暴雨、冰雹灾害袭击，造成农作物严重受损，有的地方甚至绝收。陕西遭受暴雨、冰雹袭击 部分地区农作物绝收进入7月中旬，陕西省气象台连续发布短时雷雨大风冰雹警报，7月17号，铜川市3小时内遭遇了3次冰雹袭击，7月18号下午4点，榆林市子洲县突降大暴雨并伴有冰雹，100分钟内降雨量达到114毫米，强降雨造成了当地群众一死一伤。受极端天气影响，陕西省的咸阳、渭南、延安等地也都不同程度受灾。宜君县彭镇赵塬村村民赵仓印：今年投了4万将近5万块钱，这下打得整个没有了，套了将近20万袋，不要说苹果打到地里，连这辛辛苦苦经营了十四、五年的树，面临着拔树，可以说绝收。7月17号下午4点左右，铜川市宜君县部分塬区遭受20分钟的暴雨冰雹袭击，共造成了8个乡镇群众的农作物受损，7月18号，记者在彭镇赵塬村看到，冰雹灾害已造成部分村民的苹果、玉米等农作物绝收。村民赵仓印告诉记者，他家经营了十几年的苹果园，几乎毁在了这场灾害上，损失将近5万元。宜君县彭镇赵塬村村民赵仓印：这打得，你看这，根本剩不下一个半个。在赵塬村记者看到，除了苹果，一些玉米、核桃等农作物也受损严重，据了解，在宜君县8个受灾乡镇中，太安、彭镇、五里、尧生四个乡镇灾情较重。经初步核实，宜君县受灾面积达11530公顷，受灾人数达37156人。宜君县民政局局长张印全：主要受灾作物有玉米、苹果、核桃，预计共造成经济损失1.92亿元。,disaster
# """
#     s2 = """重庆多地遭特大暴雨袭击 多个受灾乡镇停水停电,中新社重庆9月1日电 (记者 连肖)截至9月1日晚，重庆暴雨已持续整整两天。其中，重庆西部的大足区等地降雨量为1958年有气象记录以来之最。当地政府已启动应急机制抢险救灾。暴雨从8月30日晚开始袭击重庆。9月1日上午，重庆大足区的濑溪河河水猛涨，沿河的龙水镇复隆村被洪水分割成大大小小30多个“孤岛”，700多村民被洪水围困。消防官兵乘冲锋舟、橡皮艇、木船等，经过5小时才将被困人员转移到安全地带。截至9月1日10时，大足区16个镇街累计降雨量超过250毫米，城区降雨量达268.1毫米。据大足区官方统计，此次大暴雨致该区12.7万人受灾，垮塌房屋160户333间，大量企业、农田、公路、水利基础设施受损。该区消防官兵称，他们的抢险救援工作已持续40余小时。而在大足毗邻的重庆荣昌县，降雨量一度超过300毫米，当地气象局发布暴雨红色预警信号，该县内广顺街道多处山体出现滑坡。城区街道积水严重，最深处达1.5米，居民出行不敢乘车，只能挽起裤管涉水慢行。降雨也使农民损失惨重，仅重庆合川区古楼镇农作物受灾面积就达到53.5公顷。古鼓楼镇农民王昌河受访时称，暴雨使当地乡间公路部分受损，通往县城的班车也被迫取消，出行成为一大难。截至记者发稿时，重庆多个受灾乡镇停水、停电，当地政府已启动应急救灾机制，向灾民发放了馒头、饼干、方便面、饮用水等生活必需品。重庆市气象局1日晚间发布暴雨橙色预警信号，预计从1日19时到2日20时，重庆大部分地区将持续暴雨，局部地区雨量可达100毫米以上。,disaster
# """
#     s3 = """第三届《超新星运动会》落幕 张峰蝉联男子武术冠军,中新网8月17日电 16日，第三届《超新星运动会》落下帷幕。作为上一届男子武术、男子150米和4X150米接力的“三冠王”张峰，此次在武术比赛中夺得首金，蝉联男子武术项目冠军。此外，他还获得男子跳高银牌，男子团体4X150米接力银牌的好成绩。《超新星运动会》至今已走过三年，第二届时主办方将武术纳入新增比赛项目，对于传统武术的传承与普及具有极大的推广意义。作为第二季《超新星全运会》的“三冠王”，张峰此次作为演员赛区选手参与其中。去年，张峰的太极拳《沧海一声笑》表演给观众留下了极为深刻的印象，而在今年的决赛中，一首《道道道》出场便现“大侠风采”，亮剑出招都干净利落，随后半段《倩女幽魂》音乐拉起，他动作陡然舒缓，现场掌声雷动，主持人也称赞道：“看张峰演绎这一段，有一种他在书写自己武侠人生的感觉，完全把我带入了一个武侠世界。”此外，在最后一日的男子跳高比赛中，张峰更是与李昀锐、木子洋角逐，最终获得银牌。据悉，张峰从小就因为成龙电影而与中国功夫结缘、大学报考武汉体育学院武术专业，毕业又作为老师进入“武汉成龙影视传媒学院”教授武术，他曾参演《紧急救援》《成化十四年》等影片。,not_disaster
# """
#     s = [s1,s2,s3]
#
#
#
#     result1 = tc.predict(test_data)  #输入数据为pandas.core.series.Series类型
#     print(result1)
#
#     result2 = tc.predict(s)   #输入数据为list类型
#     print(result2)
#
#     result3 = tc.predict(s1)  #输入数据为string类型
#     print(result3)

#0为灾害， 1为非灾害

    fpath = "C:\\Users\\Zzh\Desktop\新建文件夹 (2)\非灾情\台风非灾情"
    tpath = "C:\\Users\\Zzh\Desktop\新建文件夹 (2)\非灾情\新建文件夹"
    files = os.listdir(fpath)
    for file in files:
        with open(fpath + "\\" + file,'r',encoding="utf8") as f:
            title = f.readline()
            time = f.readline()
            source = f.readline()
            contents = f.readlines()
            zw = ''.join(contents)

        result = tc.predict(zw)
        if result == 0:
            shutil.move(fpath+'\\'+file,tpath+"\\"+file)


