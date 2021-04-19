# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/10/21

import pandas as pd
import re_extract
import os
from xpinyin import Pinyin

path = r"D:\MyProject\天气灾害分类及信息提取\测试数据\第2次测试\测试文本"
classifications = re_extract.classifications
# classifications = ["冰雹灾害"]
wufiles = ["大雾迷眼车撞树 小货车看不清路撞树上[图]"
,"合宁高速发生5车相撞事故 4人死亡"
,"京港澳高速发生多起交通事故 致2人死20多人伤"
,"江西九景高速发生多起追尾事故致10死19伤"
,"沪渝高速安徽芜湖境内4车追尾致7人死亡"
,"山东平邑发生一起交通事故 4人死亡2人重伤"
,"沪渝高速安徽芜湖段四车追尾致7人死亡"
,"沈海高速连云港段发生7起交通事故致5死11伤"
,"杭浦高速20辆车“雾中”相撞 已致2人死亡"
,"京台高速安徽宿州段连环车祸致5死11伤"
,"大雾弥漫 杭浦高速20辆车雾中相撞"
,"洛阳环城高速3辆大货车连环相撞 2人受伤(图)"
,"京福高速枣庄段发生十余辆车相撞事故 致2死8伤"
,"四川成绵高速因大雾53车追尾 2死多伤"
,"浙江杭浦高速20辆车大雾中相撞 致2死8伤"
,"京藏高速吴忠段34车连撞3人轻伤 已基本恢复通行"
,"江西鄱阳湖大桥连环车祸 已造成十三人死亡"
,"京福高速德州段近20辆车连环相撞致1死8伤"
,"四川成乐高速大雾引发连环车祸 造成2死4伤(图)"
,"沪昆高速7车因大雾追尾 造成2人死亡25人受伤"
,"蓉遵高速贵州习水段发生多车连环相撞事故 涉及37辆车"
,"沪昆高速贵州境内7车连环相撞造成2死25伤"
,"湖南交通事故现场2名救援人员因二次车祸遇难"
,"山东滨莱高速6车相撞致6人死亡"
,"江苏南京宁合高速发生重大车祸 5人死亡3人重伤"
,"山东济菏高速发生追尾事故 已造成9人死亡(图)"
,"山西一高速公路因大雾发生交通事故 死伤升至44人"
,"湖北襄荆高速8车相撞造成两死六伤"
,"京港澳高速航海路十几辆车相撞 至少2人死亡"
,"大雾弥漫 呼和浩特快速路上21辆车连环追尾"
,"京福高速徐州段连续追尾事故已致6死30多伤"
,"贵州剑河12车连环相撞 致2人死亡30余人受伤"
,"山西太长高速多车相撞六人身亡 全国劳模张家胜遇难"
,"京福高速徐州段双向60余车相撞 已致6死32伤(图)"
,"山东济广高速追尾事故死亡人数上升至5人"
,"京台高速宿州段15车连环撞 致5死11伤"
,"因大雾道路湿滑 河南信阳发生交通事故致4死多伤"
,"南京大雾两车相撞造成5死3伤"
,"一小型客车在贵州毕节追尾大客车 致7人遇难"
,"除夕沪昆高速上海段11车追尾 3人遇难"
,"沪渝高速芜宣段发生四车追尾事故 导致7人死亡"
,"四川内宜高速多车连环相撞 致2死30余人受伤"
,"辽宁大雾致14车连撞 造成4死3重伤"
,"京港澳高速邢台段汽车连续追尾致3死6伤"
,"京港澳高速重大追尾事故造成3人死亡"
,"四川内宜高速多车连环相撞 2人死30余人受伤"
,"安徽境内高速路发生8车追尾致8死1伤(图)"
,"沪昆高速江西段大雾 引发连环追尾事故4死4伤(图)"
,"沪昆高速湖南境内因大雾发生9车追尾致3死8伤"
,"京台高速山东段120车连环相撞 已致7死35伤(图)"
]

def sort_(lis):
    pin = Pinyin()
    result = []
    for item in lis:
        result.append((pin.get_pinyin(item.split('_')[1]), item))
    print(result)
    result.sort()
    for i in range(len(result)):
        result[i] = result[i][1]
    return result

def tostr(data):
    if len(data) == 0:
        return ""
    else:
        return "".join([str(d)+"\n" for d in data])

def all_to_excel(filespath,savepath):
    classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']
    # classifications = ['大雾灾害']

    for classification in classifications:
        writer = pd.ExcelWriter(classification + ".xlsx")
        files = os.listdir(filespath + '\\' + classification)
        files = sort_(files)
        # files = wufiles
        data_new = pd.DataFrame()
        for file in files:
            try:
                # with open(path + '\\' + classification + '\\大雾_' + file+'.txt', 'r', encoding='utf8') as f:
                with open(path + '\\' + classification + '\\' + file, 'r', encoding='utf8') as f:
                    title = f.readline()
                    time = f.readline()
                    source = f.readline()
                    contents = f.readlines()
                    print("***" * 20)
                    print("文件名：", file.strip())
                    # print("新闻标题：", title.strip())
                    infos = re_extract.info_extract(contents)

                    data = pd.DataFrame(columns=['实际值', '输出值', '测试结果'],
                                        index=[[title, title, title, title, title, title, title, title, title, title,
                                                title, title, title, title, title, title, title, title, title, title,
                                                title],
                                               ['灾情类别', '开始日期', '开始时间', '结束日期', '结束时间', '位置', '承载体',
                                                '受灾人口', '失踪人口', '受伤人口', '转移人口', '死亡人口',
                                                '受灾面积', '受灾面积（类似）', '成灾面积', '绝收面积', '倒塌房屋', '损坏房屋', '经济损失', '积水深度',
                                                '积雪深度']])

                    print(infos)
                    k_v = {"AIMP": "受灾人口", "AMP": "失踪人口",
                           "AINP": "受伤人口", "ATP": "转移人口", "ADP": "死亡人口", "AIAC": "受灾面积", "AIAC2": "受灾面积（类似）",
                           "ASAC": "成灾面积",
                           "ATAC": "绝收面积", "AHC": "倒塌房屋", "AHC2": "损坏房屋", "AE": "经济损失", "AWD": "积水深度", "ASD": "积雪深度"}

                    for k in infos.keys():
                        data["输出值"][title][k_v[k]] = tostr(infos[k])
                    # data.to_excel(writer,sheet_name=title[:30])
                    data_new = data_new.append(data)
            except Exception as e:
                e.with_traceback()
                pass

        data_new.to_excel(writer,classification+'.xlsx')
        writer.save()


if __name__ == '__main__':
    all_to_excel(path,"测试结果")
    pass
