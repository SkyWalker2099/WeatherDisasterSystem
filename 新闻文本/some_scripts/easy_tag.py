# -*- coding:utf-8 -*-
# editor: zzh

import re
import os

#
files = os.listdir("D:\MyProject\天气灾害分类及信息提取\新闻文本\范野学长毕设数据\\temp\低温灾害2")

for file in files:

    with open("D:\MyProject\天气灾害分类及信息提取\新闻文本\范野学长毕设数据\\temp\低温灾害2" + '\\' + file,'r',encoding='utf8') as f:
        title = f.readline()
        time = f.readline()
        source = f.readline()
        contents = f.readlines()

        with open("D:\MyProject\天气灾害分类及信息提取\新闻文本\范野学长毕设数据\\temp\低温灾害" + '\\' + file,'w',encoding='utf8') as g:

            g.write(title)
            g.write(time)
            g.write(source)
            for content in contents:
                content = re.sub("(/TO 电|/TS 电|/DO 电|/DS 电)","电",content)

                finds = re.findall("(低温冰冻灾害|低温冷冻灾害|低温冷冻|霜冻|持续降温|低温灾害|低温天气)", content)

                for find in finds:
                    content = content.replace(find,' '+find+'/COLD ')
                g.write(content)

