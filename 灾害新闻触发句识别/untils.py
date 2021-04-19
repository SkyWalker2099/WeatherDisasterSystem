# -*- coding:utf-8 -*-
# editor: zzh

import re
import json
def split_contents(contents,max_piece,max_len):
    #这个是用来返回一个新闻分句的函数
    lines = []
    if type(contents) == list:
        for content in contents:
            content = re.sub("(/[a-zA-Z])|(\r)|(\n)|(\t)|(\s)|(&nbsp)|(\u3000)|([a-zA-Z])",'',content)
            cs = re.split('[。｡！!？?]',content)
            for c in cs:
                if(len(c) > 5):
                    lines.append(c)
    elif type(contents) == str:
        content = re.sub("(/[a-zA-Z])|(\r)|(\n)|(\t)|(\s)|(&nbsp)|(\u3000)|([a-zA-Z])", '', contents)
        cs = re.split('[。｡！!？?]', content)
        for c in cs:
            if (len(c) > 5):
                lines.append(c)
    else:
        raise Exception("分句时只能输入 str list 或者 str 类型")


    l = 0
    one_splited_lines = []
    all_splited_lines = []
    for idx in range(len(lines)):
        one_splited_lines.append(lines[idx])
        l += len(lines[idx])
        if(idx == len(lines) - 1 or l + len(lines[idx + 1]) + len(one_splited_lines) + 1 > max_len or len(one_splited_lines) == max_piece):
            all_splited_lines.append(one_splited_lines)
            one_splited_lines = []
            l = 0

    return all_splited_lines

def json2label(json_datas):
    datas = []
    for json_data in json_datas:
        one_data = []
        text = json_data["text"]
        annotations = json_data["annotations"]
        ones = [0]*len(text)
        for annotation in annotations:
            for j in range(annotation["start_offset"], annotation["end_offset"]):
                ones[j] = 1
        split_idx_list = [i.start() for i in re.finditer("。",text)]

        last = 0
        for split_idx in split_idx_list:
            split_text = text[last:split_idx]
            split_ones = ones[last:split_idx]

            if(sum(split_ones) > 0):
                one_data.append((split_text,1))
            else:
                one_data.append((split_text,0))
            last = split_idx + 1

        split_text = text[last:]
        split_ones = ones[last:]
        if (sum(split_ones) > 0):
            one_data.append((split_text, 1))
        else:
            one_data.append((split_text, 0))

        datas.append(one_data)

    return datas







if __name__ == '__main__':
    # p = "D:\MyProject\天气灾害分类及信息提取\新闻文本\生语料\城市内涝\城市内涝_东多数水文站点仍超警戒水位(组图).txt"
    # with open(p,'r',encoding='utf8') as f:
    #     title = f.readline()
    #     source = f.readline()
    #     time = f.readline()
    #     contents = f.readlines()
    #     all_splited_lines = split_contents(contents,max_piece=10,max_len=450)
    #
    #     print(title)
    #     print(contents)
    #
    #     for p in  all_splited_lines:
    #         print(p)
    datas = []
    with open("weather.jsonl",'r',encoding='utf8') as f:
        for l in f:
            data = json.loads(l)
            if(len(data["annotations"])>0):
                datas.append(data)

    label_datas = json2label(datas)
    print(label_datas)


    for label_data in label_datas:
        print("****"*20)
        for l in label_data:
            print(l)




