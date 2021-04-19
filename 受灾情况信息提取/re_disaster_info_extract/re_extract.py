# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/10/11

import re
import os
from decimal import *

pattern_num = re.compile("([点一两二三四五六七八九十千百万亿零]|[\d\．.])+[多余]?[亿万千百]?")
# pattern_num = re.compile("([一两二三四五六七八九十千百万亿零多余]+[点]*[一两二三四五六七八九十千百万亿零多余]*|[\d．.多余]+[万亿]*)")

pattern_unit = re.compile("(平方公里|平方千米|亩|元|米|厘米)")

pattern_aiac = re.compile("((受灾)(.{0,5})(总?面积)?(已)?(合计|超过|.{0,2}至|达到?了?|.*共.?|近|逾|有|大约为|约|为|扩大到|.{0,2}估计.?)?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米)[\s]*)|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米).{0,6}(不同程度)?(受灾))")
pattern_aiac2 = re.compile("((火蔓延|受冻|被淹|淹没|倒伏|影响|受损|受旱|缺水|干旱|霜冻|旱|缺墒|过火|烧毁|火场|干枯|报废|枯死|毁坏|毁|受害|损失|烂秧|死苗|遭.{0,2}害|受冻|生.{0,2}病)(.{0,5})(总?面积)?(已)?(合计|超过|.{0,2}至|达到?了?|.*共.?|近|逾|有|大约为|约|为|在|扩大到|.{0,2}估计.?)?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米)[\s]*)|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米).{0,6}(不同程度)?(受损|受旱|缺水|干旱|霜冻|.旱|缺墒|过火|烧毁|受影响|受害|受冻|干枯|报废|枯死|毁坏|被害|倒伏|受冻|被淹|生.{0,2}病|被砸))")

pattern_asac = re.compile("((成灾)(.{0,5})(总?面积)?(已)?(合计|超过|.{0,2}至|达到?了?|.*共.?|近|逾|有|大约为|约|为|扩大到|.{0,2}估计.?)?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米)[\s]*)|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米).{0,6}(不同程度)?(成灾))")
pattern_atac = re.compile("((绝收)(.{0,5})(总?面积)?(已)?(合计|超过|达到?了?|.*共.?|近|逾|有|大约为|约|为|扩大到|.{0,2}估计.?)?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米)[\s]*)|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(亩|公顷|平方公里|平方千米).*(绝收))")

pattern_aimp = re.compile("(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(名|个)?.{0,5}(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)[次口]?(不同程度)?(.{0,5})(受.{0,10}害|受灾|受困|被困|滞留|受影响|被.*埋|.{0,6}饮水(发生|受到|受)?(困难|影响)|被.{0,3}困|无法出行|受损失|生活困难)[\s]*|(受灾|受困|滞留|受影响|被埋|饮水困难|人饮困难|被.{0,3}困|影响).{0,5}(人口|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)?.{0,3}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(个?人|名)[次口]?[\s]*)|(受灾(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)[次口]?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(名|个)?)")
pattern_adp = re.compile("(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(名|个)?.{0,5}(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)(不幸)?(因.+)?.{0,5}(当场)?(而)?(死亡|遇难|丧生|罹难|牺牲|死|身亡)[\s]*|(死|死亡|遇难|丧生|罹难|牺牲).{0,5}(人口|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)?.{0,3}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(个?人|名)[\s]*)|((死|死亡|遇难|丧生|罹难|牺牲)(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)[次口]?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(名|个)?)|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)(具|位|个)?(遇难者|尸体|遗体))")
pattern_amp = re.compile("(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(名|个)?.{0,5}(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子).{0,4}(失踪|失联|下落不明)[\s]*|(失踪|失联|下落不明).{0,5}(人口|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)?.{0,3}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(个?人|名)[\s]*)|((失踪|失联|下落不明)(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)[次口]?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(名|个)?)")
pattern_ainp = re.compile("(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(名|个)?.{0,5}(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子).{0,2}(不同程度)?受?.{0,3}(受伤|重伤|轻伤|伤)[\s]*|受?(受伤|重伤|轻伤|伤).{0,5}(人口|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)?.{0,3}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(个?人|名)[\s]*)|((受伤|重伤|轻伤|伤)(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)[次口]?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(名|个)?)")
pattern_atp = re.compile("(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(名|个)?.{0,5}(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子).*(转移|撤离|被疏散)+[\s]*|(撤离|转移|疏散|安置|转运).{0,5}(人口|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)?.{0,3}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*[余多]*(个?人|名)[次口]?[\s]*)|((撤离|转移|疏散|安置|转运)(人|群众|人员|学生|旅客|市民|工人|居民|村民|儿童|游客|民众|牧民|老妇|同志|民工|男子|女子)[次口]?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(名|个)?)")

pattern_ae = re.compile("(经济)?(损失|亏损|(受损|损失)金额)(.{0,5})?([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*[千万亿]*[余多]*(元|人民币)*[\s]*")
pattern_ahc = re.compile("((近)?(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(处|栋|间|幢|户|座)?)+(居民)?(房间|房屋|建筑|.{0,4}房).{0,3}(不同程度)?.{0,2}(倒塌|垮塌|压垮)[\s]*|(居民)?(房间|房屋|建筑|.*房)?(倒塌|垮塌)(居民)?(房间|房屋|建筑|.*房)?(近)?(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(栋|间|幢|户|座|处))+(居民)?(房间|房屋|建筑|.*房)?[\s]*)|((房间|房屋|建筑|.*房)(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(处|栋|间|幢|户|座)?)+(倒塌|垮塌|被?压垮))")
pattern_ahc2 = re.compile("((近)?(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(处|栋|间|幢|户|座)?)+(居民)?(房间|房屋|建筑|.{0,4}房).{0,3}(不同程度)?.{0,2}(损伤|受损|损毁|损坏|毁坏|倒损|损失|损破|被.{0,2}坏|.毁)[\s]*|(房间|房屋|建筑|.*房)?(损|受损|损毁|损坏|毁坏|倒损|损失|损破|.坏|.毁)(居民)?(房间|房屋|建筑|.*房)?.{0,2}(近|累计)?(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(栋|间|幢|户|座))+(房间|房屋|建筑|.*房)?[\s]*)|((房间|房屋|建筑|.*房)(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)[余多]*万*亿*[余多]*(栋|间|幢|户|座)?)+(受损|损毁|损坏|毁坏|倒损|损失|损破|.坏|.毁))")

pattern_awd = re.compile("((积水|渍水|水深|被淹)(深|最深处|深处|最深(地段)?)?.{0,5}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+).{0,1}(米|厘米|公分|cm|CM|m))|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+).{0,1}(米|厘米|公分|cm|CM|m).{0,3}(的)?(积水|渍水|淹水))")
pattern_asd = re.compile("((降雪|积雪|雪深)(厚度|深度)?.{0,5}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+).{0,1}(米|厘米|公分|cm|CM|m))|(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+).{0,1}(米|厘米|公分|cm|CM|m).{0,3}(的)?(积雪))")

pattern_extra1 = re.compile("(([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)(死|死亡|遇难|丧生|罹难|牺牲)([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+)(受伤|重伤|轻伤|伤))")

pattern_extra2 = [re.compile("(积水|渍水|水深|被淹)"),re.compile("(深|最深处|深处|最深(地段)?).{0,5}([点一两二三四五六七八九十千百万亿零]+|[\d\.．]+).{0,1}(米|厘米|公分|cm|CM|m)")]

pattern_dict = {"AIAC":pattern_aiac,"AIAC2":pattern_aiac2,"ASAC":pattern_asac,"ATAC":pattern_atac,
                "AIMP":pattern_aimp,"ADP":pattern_adp,"AMP":pattern_amp,"AINP":pattern_ainp,"ATP":pattern_atp,
                "AHC":pattern_ahc,"AHC2":pattern_ahc2,"AE":pattern_ae,
                "AWD":pattern_awd,"ASD":pattern_asd}

sub_aimp = re.compile("([一两二三四五六七八九十千百万亿零]+|[\d\.]+)[余多]*万*亿*(户|头)")
# sub_pattern_dict = {"AIAC":sub_aiac,"ASAC":sub_asac,"ATAC":sub_atac,
                # "AIMP":sub_aimp,"ADP":sub_adp,"AMP":sub_amp,"AINP":sub_ainp,"ATP":sub_atp,
                # "AE":sub_ae,"AHC":sub_ahc,
                # "AWD":sub_awd,"ASD":sub_asd}
sub_pattern_dict = {"AIMP":sub_aimp,"ATP":sub_aimp,"AHC":sub_aimp,"AHC2":sub_aimp}

limit_dict = {"AWD":[0,300],"ASD":[0,300],"AE":[1,100000000],"AIMP":[1,14e8],"ADP":[1,14e8],"AMP":[1,14e8],"AINP":[1,14e8],"ATP":[1,14e8],"AIAC2":[1,100000000],"AIAC2":[1,100000000]}

# pattern_dict = {"AWD":pattern_awd,"ASD":pattern_asd}

unit_dict = {"元":0.0001,"厘米":1,"米":100,"亩":0.06666666666666666666667,"平方千米":100,"平方公里":100}

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']

v_k = {'受灾人口': 'AIMP', '失踪人口': 'AMP', '受伤人口': 'AINP', '转移人口': 'ATP', '死亡人口': 'ADP', '受灾面积': 'AIAC', '受灾面积（类似）': 'AIAC2', '成灾面积': 'ASAC', '绝收面积': 'ATAC', '倒塌房屋': 'AHC', '损坏房屋': 'AHC2', '经济损失': 'AE', '积水深度': 'AWD', '积雪深度': 'ASD'}

before_control = {"AMP":"ADP","AINP":"ADP"}

def str2num(str):
    # print(" >"+str)

    str = str.replace('．', '.')
    str = re.sub("[余多]", "", str)
    # print(str)
    dd = 1
    if str[-1] == '万':
        if len(str) == 1:
            return 10000
        dd = 10000
        dd2 = "10000"
        str = str[0:len(str) - 1]
    elif str[-1] == '亿':
        if len(str) == 1:
            return 100000000
        dd = 100000000
        dd2 = "100000000"
        str = str[0:len(str) - 1]
    # elif str[-1] == '千' and len(str) > 1 and str[-2] in ['1','2','3','4','5','6','7','8','9','0']:
    elif str[-1] == '千' and len(str) > 1 and re.match('\d',str[-2]):
        dd = 1000
        dd2 = "1000"
        str = str[0:len(str) - 1]

    def s2n(str):
        # print(str)
        zhong = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '两': 2}
        danwei = {'十': 10, '百': 100, '千': 1000, '万': 10000}
        num = 0
        if len(str) == 0:
            return 0
        if len(str) == 1:
            if str == '十':
                return 10
            elif str == '百':
                return 100
            elif str == '千':
                return 1000
            elif str == '万':
                return 10000
            elif str == '亿':
                return 100000000
            num = zhong[str]
            return num
        temp = 0
        if str[0] == '十':
            num = 10

        xsd = 1

        for i in str:
            if i == '零':
                temp = zhong[i]
            elif i == '一':
                temp = zhong[i]
            elif i == '二':
                temp = zhong[i]
            elif i == '两':
                temp = zhong[i]
            elif i == '三':
                temp = zhong[i]
            elif i == '四':
                temp = zhong[i]
            elif i == '五':
                temp = zhong[i]
            elif i == '六':
                temp = zhong[i]
            elif i == '七':
                temp = zhong[i]
            elif i == '八':
                temp = zhong[i]
            elif i == '九':
                temp = zhong[i]
            if i == '十':
                temp = temp * danwei[i]
                num += temp
                temp = 0
            elif i == '百':
                temp = temp * danwei[i]
                num += temp
                temp = 0
            elif i == '千':
                temp = temp * danwei[i]
                num += temp
                temp = 0
            elif i == '万':
                temp = temp * danwei[i]
                num += temp
                temp = 0
            elif i == '点':
                num+=temp
                xsd = 0.1

        if str[len(str) - 1] != '十' and str[len(str) - 1] != '百' and str[len(str) - 1] != '千' and str[len(str) - 1] != '万':
            num += temp*xsd
        # print(num)
        return num

    # print(str)
    if re.match("[\d\.]+",str):
        # ans = float(str)
        # print(ans)
        ans = Decimal(str)*Decimal(dd2)
        # ans = ans*dd
        # print(ans)
        return ans
    elif re.match("[一二三四五六七八九十千百万零点两]+",str):
        # print(str)
        s = ""
        num = 0
        for i in str:
            if i == "万":
                if s != "":
                    n = s2n(s)
                    num+=n*10000
                    s = ""
            elif i == "亿":
                if s != "":
                    n = s2n(s)
                    num+=n*100000000
                    s = ""
            elif i in ['一','二','两','三','四','五','六','七','八','九','十','点','千','百']:
                s+=i
        if s!="":
            n = s2n(s)
            num += n
        return num*dd
        # return num*Decimal(dd2)

def extract_num_from_str(str):
    num_strs = re.finditer(pattern_num,str)
    num_strs = list(num_strs)
    num_str = num_strs[-1]
    # print(num_strs)
    num_str = num_str.group()
    # print(num_str)
    num = str2num(num_str)
    # print(num)
    return num

def type_control(num,key):
    if key in ["AIMP","ADP","AMP","AINP","ATP","AHC","AHC2"]:
        if type(num) != int:
            num = int(num)
            return num
        elif type(num) == int:
            return num
    else:
        num = round(num,4)
        # num = decimal(num)
        return num

def info_extract(contents):
    info_dict = {"AIAC":[],"AIAC2": [],"ASAC":[], "ATAC":[],
                    "AIMP":[], "ADP":[], "AMP":[], "AINP":[],
                    "ATP":[],
                    "AE":[], "AHC":[],"AHC2":[],
                    "AWD":[],"ASD":[]}

    visited_in_key = {"AIAC": [],"AIAC2": [], "ASAC": [], "ATAC": [],
                 "AIMP": [], "ADP": [], "AMP": [], "AINP": [],
                 "ATP": [],
                 "AE": [], "AHC": [],"AHC2":[],
                 "AWD": [], "ASD": []}

    have_info = {"AIAC": [],"AIAC2": [], "ASAC": [], "ATAC": [],
                 "AIMP": [], "ADP": [], "AMP": [], "AINP": [],
                 "ATP": [],
                 "AE": [], "AHC": [],"AHC2":[],
                 "AWD": [], "ASD": []}

    for content in contents:
        content = re.sub("([\(\（].*?[\)\）])","",content)
        # print(content)
        content = content.replace('\r','').replace('\n','').replace('\t','').replace(' ','').replace('\u3000','')
        lines = re.split('[，。！？；,?!;、()（）]', content)
        # print(lines)
        for key in info_dict.keys():
            # print(key)
            try:
                for idx,line in enumerate(lines):
                    try:
                        visited = []
                        if (line.strip() in visited):
                            # print("X")
                            continue
                        line = line.strip()
                        visited.append(line)
                        if idx != 0 and lines[idx-1] in have_info[key] and re.search("(其中|包括)",line) != None:
                            # print("     xsx",lines[idx-1],line)
                            continue
                        if key in sub_pattern_dict.keys():
                            line = re.sub(sub_pattern_dict[key],"",line)
                            # print(line)
                        # print(key+" >"+line)
                        finds = re.search(pattern_dict[key],line)
                        if finds:
                            ss = finds.group()
                            # print("     >"+ss)
                            num = extract_num_from_str(ss)
                            # print(num)
                            if num == None:
                                continue
                            unit = re.search(pattern_unit,line)
                            if unit:
                                unit = unit.group()
                                num = num*unit_dict[unit]
                            else:
                                if key == 'AE':
                                    num = num * 0.0001
                            if num not in visited_in_key[key]:
                                if key in limit_dict.keys():
                                    if num>=limit_dict[key][0] and num <= limit_dict[key][1]:
                                        info_dict[key].append(type_control(num,key))
                                        have_info[key].append(line)
                                else:
                                    info_dict[key].append(type_control(num,key))
                                    have_info[key].append(line)
                                visited_in_key[key].append(num)
                    except Exception as e:
                        # e.with_traceback()
                        pass

            except Exception as e:
                # e.with_traceback()
                pass

        for idx,line in enumerate(lines):
            try:
                finds = pattern_extra1.search(line)
                if finds:
                    # print("     >"+finds.group())
                    ss = re.split("(死|死亡|遇难|丧生|罹难|牺牲)",finds.group())
                    ds = ss[0]
                    ins = ss[2]
                    dnum = extract_num_from_str(ds)
                    dnum = type_control(dnum,"ADP")
                    if dnum not in visited_in_key["ADP"]:
                        info_dict["ADP"].append(dnum)
                        have_info["ADP"].append(line)
                        visited_in_key["ADP"].append(dnum)

                    inum = extract_num_from_str(ins)
                    inum = type_control(inum, "AINP")
                    if inum not in visited_in_key["AINP"]:
                        info_dict["AINP"].append(inum)
                        have_info["AINP"].append(line)
                        visited_in_key["AINP"].append(inum)
            except Exception as e:
                # e.with_traceback()
                pass

        for i in range(1,len(lines)):
            l1 = lines[i-1]
            l2 = lines[i]
            f1 = re.search(pattern_extra2[0],l1)
            f2 = re.search(pattern_extra2[1],l2)
            if f1 and f2:
                # print("     >",l1,l2)
                wnum = extract_num_from_str(f2.group())
                if wnum == None:
                    continue
                unit = re.search(pattern_unit,l2)
                if unit:
                    unit = unit.group()
                    wnum = wnum * unit_dict[unit]
                if wnum not in visited_in_key["AWD"]:
                    if wnum >= limit_dict["AWD"][0] and wnum <= limit_dict["AWD"][1]:
                        info_dict["AWD"].append(type_control(wnum,"AWD"))
                        have_info["AWD"].append(l2)
                    visited_in_key["AWD"].append(wnum)


    return info_dict

if __name__ == '__main__':

    # s = ["截至目前，暴雨洪涝灾害已造成2900余万人受灾，江西省抚州抚河唱凯堤发生决口，威胁下游14.5万人口、京福高速公路、316国道以及12万亩粮田的安全。 　　暴雨倾城，江河决堤，6月13日以来，中国南方出现的强降雨过程导致浙江、福建、江西、湖北、湖南、广东、广西、重庆、四川、贵州10省区遭受严重的洪涝灾害。为应对长江流域日益严峻的防汛形势，长江防总从22日8时开始，将防汛应急响应由Ⅲ级提高至Ⅱ级。　　面对重大洪涝灾害，党中央、国务院高度重视，胡锦涛总书记、温家宝总理多次作出重要指示，要求把保障人民群众生命安全放在第一位，逐级落实抢险和救灾责任，及时转移受威胁的群众，妥善安置好灾区群众生活，努力把灾害造成的损失减少到最低程度。6月22日，中央财政紧急下拨洪涝灾害应急救灾资金2.53亿元，用于支持福建、江西、湖南、广西和贵州5个重点受灾省份紧急转移安置受灾群众等救灾工作。　　国家防总相关负责人昨日在江西抚河干流决口现场表示，此次水情突出表现为三个特点：一是主要江河洪水量级大，部分河流超过1998年。江西信江、抚河发生超历史纪录特大洪水，重现期50年；福建闽江发生30年一遇的大洪水。二是发生洪水河流众多。6月13日以来的暴雨洪水涉及长江、闽江、西江三个流域，江西、福建等110余条河流发生超警洪水，9条河流发生超历史纪录洪水。三是闽江、湘江、资水等南方11条主要江河同时发生洪水，近年来少见。　　据中央气象台预报，23日起到26日，新一轮强降水将席卷长江沿江及其以南地区。暴雨对南方地区的经济生产、居民生活的影响可能还将持续。　　据民政部网站消息，截至6月22日11时统计，此次暴雨洪涝灾害过程已造成中国南方10省份2913.5万人受灾，因灾死亡199人，失踪123人，紧急转移安置人口237.6万人；农作物受灾面积1608.9千公顷，其中绝收面积242千公顷；倒塌房屋19.5万间，损坏房屋56.8万间；因灾直接经济损失421.2亿元。　　受洪涝灾害影响，南方地区的生产、生活遭遇重大影响。农业生产方面，湖南、湖北等粮食主产区粮田被淹，农田遭受重大毁坏。交通运输方面，受灾地区多条国道、高速公路、铁路等因水毁、塌方、桥梁垮塌等不能正常通行，特别是21日晚抚河(抚州市境内)唱凯堤溃决，抚河沿线村庄、道路被淹，造成抚州市境内交通受到严重影响。电力、通讯设施方面，部分灾区电力设施被毁，通讯基站遭破坏，出现断电、通讯不畅等现象。　　此外，受南方洪涝灾害影响，部分地区的鲜活农副产品价格上涨明显，有的叶菜类价格涨幅超过了30%。而随着北方即将进入主汛期，市场对于农产品价格上涨有了更多的担忧。专家指出，暴雨带来的菜价上涨是短时间的，而相比旱灾和低温灾害，洪涝灾害对粮食生产的影响要小得多。 "]
    s = ["临时饮水困难人口202.07万人"]
    results = info_extract(s)
    print(results)
    #
    # path = "D:\MyProject\天气灾害分类及信息提取\测试数据\第2次测试\测试文本"
    # for classification in classifications:
    #     files = os.listdir(path + '\\' + classification)
    #     for file in files:
    #         with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
    #             title = f.readline()
    #             time = f.readline()
    #             source = f.readline()
    #             contents = f.readlines()
    #             print("***"*20)
    #             print("文件名：", file.strip())
    #             # print("新闻标题：", title.strip())
    #             infos = info_extract(contents)
    #             c = 0
    #             for key in infos.keys():
    #                 c = c+len(infos[key])
    #             if c == 0:
    #                 pass
    #                 # with open("D:\MyProject\天气灾害分类及信息提取\测试数据\无数据文本"+'\\'+file,'w',encoding='utf8') as g:
    #                 #     g.write(title)
    #                 #     g.write(time)
    #                 #     g.write(source)
    #                 #     for content in contents:
    #                 #         g.write(content)
    #             print(c)
    #             print(infos)











