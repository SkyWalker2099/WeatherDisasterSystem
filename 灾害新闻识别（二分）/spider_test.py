# -*- coding:utf-8 -*-
# editor: zzh

import requests
from bs4 import BeautifulSoup
from re_assisstant import *
from syn_bert_classifer import *

sbwr = SynBertWeatherRecongizer(config_path,checkpoint_path,dict_path,max_len=400)

# sbwr.bert.load_weights("MODELS\\syn_model\\bert_short.h5")
# sbwr.lstm_model.load_weights("MODELS\\syn_model\\lstm_model.h5")

# sbwr.bert.load_weights("MODELS\\all_data\\classifier_003.h5")
# sbwr.lstm_model.load_weights("MODELS\\all_data\\lstm_029.h5")

sbwr.bert.load_weights("MODELS\\balanced\\classifier_003.h5")

classificitions = ["disaster","not_disaster"]

dir = "爬取测试\\test2"
head = "https://www.baidu.com"

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/75.0.3770.100 Safari/537.36'
           }

classifications = ['低温灾害', '冰雹灾害', '台风灾害', '地质灾害', '城市内涝', '大雾灾害', '大风灾害', '干旱灾害', '暴雨洪涝', '森林火灾', '雪灾灾害', '雷电灾害']


def predict(title,contents):
    if(type(contents) != list):
        contents = [contents]
    index = sbwr.predict_str(title + "".join(contents))
    contents.append(title)
    has_info = has_disaster_info(contents)
    if not has_info:
        index = 1
    return index

def get_response(link):
    re = requests.get(link,headers = headers)
    if(re.status_code == 200):
        realencoding = requests.utils.get_encodings_from_content(re.text)
        re.encoding = realencoding[0]
    return re.text

def get_all_contents_page(start_page, max_page = None):
    """获取所有目录网页"""
    html_text = get_response(start_page)
    # print(html_text)
    soup = BeautifulSoup(html_text, features='lxml')
    html_texts = []
    html_texts.append(html_text)

#接下来一部分如果要用在不同网站上就要修改了
    xp = 'div[class="page-inner"] a'
    bns = soup.select(xp)
    bn = bns[-1]
    while(bn.text == '下一页 >'):
        next_link = head + bn['href']
        print(next_link)
        html_text = get_response(next_link)
        html_texts.append(html_text)
        if max_page and len(html_texts) > max_page:
            break
        soup = BeautifulSoup(html_text, features='lxml')
        bns = soup.select(xp)
        bn = bns[-1]
#这里返回的是相应的目录的html文件

    return html_texts

def get_article_links_from_contents_page(page_htmls):
    """根据目录html，获取其中所有的文章连接并读取html"""
    # head = "http://www.qizhiwang.org.cn/GB/433033/433035"
    xp = 'div[class="result-op c-container xpath-log new-pmd"]'

    results = []

    for page_html in page_htmls:
        soup = BeautifulSoup(page_html,features='lxml')
        arcs = soup.select(xp)
        print(len(arcs))
        for arcn in arcs:
            try:
                arc = arcn.select("a")[0]

                print(arc['href'])
                arc_html = get_response(arc['href'])
                results.append((arc_html,arc['href']))
                #html正文,url
            except IndexError as e:
                # e.with_traceback()
                pass

    return results

def get_all_arcs_contents(htmls):

    titlex = 'div[class="index-module_headerWrap_j_uQR"] h2'

    timex = 'div[class="index-module_articleSource_2dw16"] span'

    contentx = 'div[class="index-module_textWrap_3ygOc"] p'

    for html,url in htmls:

        contents = []

        title = ""
        time = ""

        try:
            # print(html)
            soup = BeautifulSoup(html, features='lxml')

            titlexs = soup.select(titlex)
            if(len(titlexs) != 0):
                title = titlexs[0].text

            timexs = soup.select(timex)
            if (len(timexs) != 0):
                time = timexs[0].text

            texts = soup.select(contentx)
            for t in texts:
                contents.append(t.text)

            source = '百家号'
            title = title.replace('\n', '').replace(' ', '').replace('\r', '').replace("|", '')


            # index,result = sbwr.predict_str(title + "".join(contents))
            index, result = sbwr.predict_short(title + "".join(contents))
            # index = predict(title,contents)

            print(title,index,result)
            try:
                with open(dir + '\\' + classificitions[index] + '\\' + title + ".txt", 'w', encoding='utf8') as f:
                    f.write(url + '\n')
                    f.write(title + '\n')
                    f.write(time + '\n')
                    f.write(source + '\n')
                    f.write('\n'.join(contents))
                    f.close()
            except Exception as e:
                # e.with_traceback()
                pass

        except Exception as e:
            # e.with_traceback()
            print('error',e.__repr__())

if __name__ == '__main__':
    for classification in classifications:
        s = """https://www.baidu.com/s?ie=utf-8&medium=2&wd={}&tn=news"""
        key = classification.replace("灾害","")
        # key = classification
        pages = get_all_contents_page(s.format(key),10)
        articles = get_article_links_from_contents_page(pages)
        get_all_arcs_contents(articles)