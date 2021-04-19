# -*- coding:utf-8 -*-
# editor: zzh
# date: 2021/3/1
#根据提取的信息进行数据标注

import re_extract
import os
import re

patterns = re_extract.pattern_dict
target_tag = [tag for tag in patterns.keys()]
classifications = re_extract.classifications
def has_tag(line,type):  #句子是不是与该模式匹配
    pattern = patterns[type]
    return pattern.search(line) != None


def get_line_replace(line,type):   #匹配出关键信息
    print('\n',line)
    pattern = patterns[type]
    find = pattern.search(line)
    if find:
        print("     >"+find.group())
        return find.group()
    else:
        print("     >"+line)
        return line


def tag_paragraph(content):
    content_taged = content
    lines = re.split('[，。！？：；,!;、?()（）??]',content)
    visited = []
    for line in lines:
        if(line.strip() in visited):
            continue
        line = line.strip()

        for tag in target_tag:
            if has_tag(line,tag):
                target_line = get_line_replace(line, tag)
                content_taged = content_taged.replace(target_line, ' '+target_line + '/' + tag + ' ').replace('/' + tag + ' ' +'/' + tag , '/' + tag)
    # print(content_taged)
    return content_taged

def tag_dir(path,savepath):
    files = os.listdir(path)
    for file in files:
        print("**"*30)
        print(file)
        filepath = path + "\\" + file
        savefilepath = savepath + "\\" + file
        with open(filepath, "r", encoding="utf8") as f:
            with open(savefilepath, "w", encoding="utf8") as g:
                title = f.readline()
                time = f.readline()
                source = f.readline()
                contents = f.readlines()
                g.write(title+time+source)
                for content in contents:
                    content_taged = tag_paragraph(content)
                    g.write(content_taged)


if __name__ == '__main__':
    for classification in classifications:
        tag_dir("D:\MyProject\天气灾害分类及信息提取\新闻文本\范野学长毕设数据\原标注数据\\"+classification,"D:\MyProject\天气灾害分类及信息提取\新闻文本\范野学长毕设数据\统合数据1\\"+classification)
    # 将一个文件夹中的所有新闻文件标注后放到另一文件夹
    # content = """ 中新网惠州八月二十五日电　(康孝娟 白剑涛 严初)广东惠州惠城区陈江镇水围村梧村水库，四名青年二十三日在岛上钓鱼时，突然遭到雷击，一人当场死亡，三人昏倒在地。而在当天，惠州市两个镇出现了雷击事故，共造成一死四伤。  　　笔者在惠州市第一人民医院住院部见到了被雷击中的伤者。在十楼烧伤科病房，伤者黄世波静静躺在病床上，无法说话，胸部和左手上臂都有灼伤痕迹。  　　据了解，二十三日是处暑，当天下午，一声秋雷响起，惠州市区下起了暴雨，暑气悄然褪去。惠州市惠城区陈江街道办事处、三栋镇都出现雷击事故，已造成一人死亡，四人受伤。  　　惠州市气象局防雷所陈所长说，受副热带高压控制，每年七、八月为雷雨高发季节，全国百分之八十的雷区都集中在广东。而陈江水围村梧村水库属空旷地，易引雷，加之小岛在高处，成为典型的雷区。  　　黄世波是惠城区三栋镇某工地建筑工人。据其妻子告诉记者，二十三日下午四时许，黄世波骑乘摩托车到镇隆青草窝朋友家闲聚，在路上忽然听到一声雷响，他便被雷击，呆坐在地上，路人马上报警，将他送院治疗。 据悉，梧村水库一带经常发生雷击事件，去年水库旁几个种菜妇女和建筑工也被雷击致死。  　　针对已发生的两起雷击事故，惠州市气象局防雷所陈所长希望借助媒体告知市民在行雷时切勿在空旷地上跑步、拨打手机等易引雷的行为，避免再发生类似事故。"""
    # tag_paragraph(content)