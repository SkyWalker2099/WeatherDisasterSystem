import os
import re

def tag_paragraph(content):
    content_taged = content
    lines = re.split('[，。！？；,?!;、]',content)
    visited = []
    for line in lines:
        old_line = line
        if(line.strip() in visited):
            continue
        if line.find("/AIAC") != -1:
            if line.find("受灾") == -1:
                print(">",line)
                line = line.replace("/AIAC","/AIAC2")
                content_taged = content_taged.replace(old_line,line)
        elif line.find("/ASAC") != -1:
            if line.find("成灾") == -1:
                print(">",line)
                line = line.replace("/ASAC","/AIAC2")
                content_taged = content_taged.replace(old_line, line)
        elif line.find("/ATAC") != -1:
            if line.find("绝收") == -1:
                print(">",line)
                line = line.replace("/ATAC","/AIAC2")
                content_taged = content_taged.replace(old_line, line)
        visited.append(line.strip())
    return content_taged

def tag_paragraph2(content):
    content_taged = content
    lines = re.split('[，。！？；,?!;、]',content)
    visited = []
    for line in lines:
        old_line = line
        if(line.strip() in visited):
            continue
        if line.find("/AHC") != -1:
            if line.find("倒塌") == -1 and line.find("垮塌") == -1:
                print(">",line)
                line = line.replace("/AHC","/AHC2")
                content_taged = content_taged.replace(old_line,line)
        visited.append(line.strip())
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
                    content_taged = tag_paragraph2(content)
                    g.write(content_taged)

if __name__ == '__main__':
    tag_dir("G:\BUPT\zaihai\文档标注2\面积\雪灾灾害\已标注","G:\BUPT\zaihai\文档标注2\面积\雪灾灾害\已标注_改")